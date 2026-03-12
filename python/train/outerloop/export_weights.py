from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from python.train.outerloop.model import TinyHybridNet


def export_weights(model_path: Path, config_path: Path, output_path: Path) -> dict:
    training_config = json.loads(config_path.read_text(encoding="utf-8"))
    num_conv_layers = int(training_config.get("num_conv_layers", 2))
    model = TinyHybridNet(
        input_channels=int(training_config["input_channels"]),
        scalar_features=int(training_config["scalar_features"]),
        conv_channels=int(training_config["conv_channels"]),
        num_conv_layers=num_conv_layers,
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    payload = {
        "version": 2 if num_conv_layers >= 3 else 1,
        "input_channels": int(training_config["input_channels"]),
        "scalar_features": int(training_config["scalar_features"]),
        "board_height": int(training_config["board_height"]),
        "board_width": int(training_config["board_width"]),
        "conv1": export_conv(model.conv1),
        "conv2": export_conv(model.conv2),
        "policy": export_linear(model.policy_head),
        "value": export_linear(model.value_head),
    }
    if model.conv3 is not None:
        payload["conv3"] = export_conv(model.conv3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def export_conv(layer: torch.nn.Conv2d) -> dict:
    return {
        "out_channels": layer.out_channels,
        "in_channels": layer.in_channels,
        "kernel_size": layer.kernel_size[0],
        "weights": layer.weight.detach().cpu().reshape(-1).tolist(),
        "bias": layer.bias.detach().cpu().reshape(-1).tolist(),
    }


def export_linear(layer: torch.nn.Linear) -> dict:
    return {
        "out_features": layer.out_features,
        "in_features": layer.in_features,
        "weights": layer.weight.detach().cpu().reshape(-1).tolist(),
        "bias": layer.bias.detach().cpu().reshape(-1).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = export_weights(args.model, args.training_config, args.out)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
