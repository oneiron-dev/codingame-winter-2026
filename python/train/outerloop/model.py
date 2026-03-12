from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TinyHybridNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        scalar_features: int,
        conv_channels: int,
        birds_per_player: int = 4,
        actions_per_bird: int = 5,
        num_conv_layers: int = 2,
    ) -> None:
        super().__init__()
        self.birds_per_player = birds_per_player
        self.actions_per_bird = actions_per_bird
        self.conv1 = nn.Conv2d(input_channels, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1) if num_conv_layers >= 3 else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        feature_dim = conv_channels + scalar_features
        self.policy_head = nn.Linear(feature_dim, birds_per_player * actions_per_bird)
        self.value_head = nn.Linear(feature_dim, 1)

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv1(grid))
        x = torch.relu(self.conv2(x))
        if self.conv3 is not None:
            x = torch.relu(self.conv3(x))
        pooled = self.pool(x).flatten(1)
        if scalars.ndim == 1:
            scalars = scalars.unsqueeze(0)
        features = torch.cat([pooled, scalars], dim=1)
        policy = self.policy_head(features).view(-1, self.birds_per_player, self.actions_per_bird)
        value = torch.tanh(self.value_head(features))
        return policy, value


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block with reduction ratio."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))  # global avg pool -> [B, C]
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, 1, 1)


class ResBlock(nn.Module):
    """Pre-activation residual block: conv3x3 -> BN -> ReLU -> conv3x3 -> BN + SE + skip -> ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(x))
        out = self.conv1(out)
        out = torch.relu(self.bn2(out))
        out = self.conv2(out)
        out = self.se(out)
        return torch.relu(out + residual)


class TeacherHybridNet(nn.Module):
    """Larger teacher network for distillation. Not exported to Rust.

    Architecture: stem conv(19->channels) -> N SE-res blocks -> policy+value heads.
    Default: 128 channels, 8 res blocks (~2-3M params).
    """

    def __init__(
        self,
        input_channels: int,
        scalar_features: int,
        conv_channels: int = 128,
        num_res_blocks: int = 8,
        birds_per_player: int = 4,
        actions_per_bird: int = 5,
    ) -> None:
        super().__init__()
        self.birds_per_player = birds_per_player
        self.actions_per_bird = actions_per_bird
        self.stem = nn.Conv2d(input_channels, conv_channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(conv_channels)
        self.blocks = nn.Sequential(*[ResBlock(conv_channels) for _ in range(num_res_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        feature_dim = conv_channels + scalar_features
        self.policy_fc1 = nn.Linear(feature_dim, 128)
        self.policy_fc2 = nn.Linear(128, birds_per_player * actions_per_bird)
        self.value_fc1 = nn.Linear(feature_dim, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.stem_bn(self.stem(grid)))
        x = self.blocks(x)
        pooled = self.pool(x).flatten(1)
        if scalars.ndim == 1:
            scalars = scalars.unsqueeze(0)
        features = torch.cat([pooled, scalars], dim=1)
        policy = self.policy_fc2(torch.relu(self.policy_fc1(features)))
        policy = policy.view(-1, self.birds_per_player, self.actions_per_bird)
        value = torch.tanh(self.value_fc2(torch.relu(self.value_fc1(features))))
        return policy, value
