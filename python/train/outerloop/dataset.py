from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from python.train.dataset import resolve_dataset_paths


class HybridSelfPlayDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: str | Path, max_samples: int = 0) -> None:
        self.paths = resolve_dataset_paths(path)
        self.rows: list[dict] = []
        for dataset_path in self.paths:
            with dataset_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "hybrid_grid" not in row:
                        continue
                    self.rows.append(row)
                    if max_samples and len(self.rows) >= max_samples:
                        break
            if max_samples and len(self.rows) >= max_samples:
                break
        if not self.rows:
            raise ValueError(f"hybrid dataset is empty: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            "grid": torch.tensor(row["hybrid_grid"], dtype=torch.float32),
            "scalars": torch.tensor(row.get("scalars", []), dtype=torch.float32),
            "value": torch.tensor([row["value"]], dtype=torch.float32),
            "policy_targets": torch.tensor(row.get("policy_targets", [-100, -100, -100, -100]), dtype=torch.long),
            "weight": torch.tensor([row.get("weight", 1.0)], dtype=torch.float32),
        }


def dedup_rows(rows: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("encoded_view_hash", f"row-{len(deduped)}"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


class HybridDistillDataset(HybridSelfPlayDataset):
    """Extends HybridSelfPlayDataset with teacher soft targets for distillation."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        item = {
            "grid": torch.tensor(row["hybrid_grid"], dtype=torch.float32),
            "scalars": torch.tensor(row.get("scalars", []), dtype=torch.float32),
            "value": torch.tensor([row["value"]], dtype=torch.float32),
            "policy_targets": torch.tensor(row.get("policy_targets", [-100, -100, -100, -100]), dtype=torch.long),
            "weight": torch.tensor([row.get("weight", 1.0)], dtype=torch.float32),
        }
        if "teacher_policy_logits" in row:
            item["teacher_policy_logits"] = torch.tensor(row["teacher_policy_logits"], dtype=torch.float32)
        if "teacher_value" in row:
            item["teacher_value"] = torch.tensor([row["teacher_value"]], dtype=torch.float32)
        return item


def grouped_split_indices(rows: list[dict], train_split: float, seed: int) -> tuple[list[int], list[int]]:
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(int(row.get("seed", 0)), str(row.get("game_id", "game-0")))].append(index)

    groups = list(grouped.items())
    rng = random.Random(seed)
    rng.shuffle(groups)

    target_train = max(1, int(len(rows) * train_split))
    train_indices: list[int] = []
    valid_indices: list[int] = []
    for _, indices in groups:
        bucket = train_indices if len(train_indices) < target_train else valid_indices
        bucket.extend(indices)
    if not valid_indices and train_indices:
        valid_indices.append(train_indices.pop())
    if not train_indices and valid_indices:
        train_indices.append(valid_indices.pop())
    return train_indices, valid_indices
