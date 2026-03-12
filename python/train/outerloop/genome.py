from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python.train.java_smoke import stable_hash_bytes

REPO_ROOT = Path(__file__).resolve().parents[3]


DEFAULT_MODEL = {
    "enabled": False,
    "architecture": "tiny_hybrid_v1",
    "conv_channels": 8,
    "num_conv_layers": 2,
    "prior_mix": 0.0,
    "leaf_mix": 0.0,
    "value_scale": 48.0,
    "prior_depth_limit": 0,
    "leaf_depth_limit": 0,
    "epochs": 4,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "device_preference": "mps",
    "executor": "local",
    "gpu": "L40S",
    "max_samples": 0,
    "training_mode": "standard",
    "distill_alpha": 0.5,
    "distill_temperature": 3.0,
    "teacher_model_path": None,
    "teacher_conv_channels": 128,
    "teacher_num_res_blocks": 8,
}

DEFAULT_DATA = {
    "dataset_path": "python/train/data/selfplay.jsonl",
    "generate_dataset": False,
    "shared_dataset_id": None,
    "seed_start": 1,
    "seed_count": 64,
    "league": 4,
    "workers": 4,
    "max_turns": 120,
    "extra_nodes_after_root": 5000,
    "executor": "local",
}


@dataclass(frozen=True)
class CandidateGenome:
    payload: dict[str, Any]

    @property
    def kind(self) -> str:
        return str(self.payload.get("kind", "search"))

    @property
    def search(self) -> dict[str, Any]:
        return self.payload["search"]

    @property
    def eval(self) -> dict[str, Any]:
        return self.payload["eval"]

    @property
    def model(self) -> dict[str, Any]:
        return self.payload["model"]

    @property
    def data(self) -> dict[str, Any]:
        return self.payload["data"]

    @property
    def metadata(self) -> dict[str, Any]:
        return self.payload.setdefault("metadata", {})

    @property
    def semantic_hash(self) -> str:
        return semantic_hash(self.payload)

    @property
    def candidate_id(self) -> str:
        return f"{self.kind}-{self.semantic_hash[:12]}"


def load_genome(path: str | Path) -> CandidateGenome:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CandidateGenome(normalize_genome(payload))


def dump_genome(path: str | Path, genome: CandidateGenome | dict[str, Any]) -> None:
    payload = genome.payload if isinstance(genome, CandidateGenome) else normalize_genome(genome)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def semantic_hash(payload: dict[str, Any]) -> str:
    canonical = {
        "kind": payload.get("kind", "search"),
        "search": payload["search"],
        "eval": payload["eval"],
        "model": payload["model"],
        "data": payload["data"],
    }
    return stable_hash_bytes(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def normalize_genome(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(payload)
    normalized.setdefault("kind", "search")
    normalized.setdefault("search", {})
    normalized.setdefault("eval", {})
    normalized.setdefault("model", deepcopy(DEFAULT_MODEL))
    normalized.setdefault("data", deepcopy(DEFAULT_DATA))
    normalized.setdefault("metadata", {})

    model = deepcopy(DEFAULT_MODEL)
    model.update(normalized["model"])
    normalized["model"] = model

    data = deepcopy(DEFAULT_DATA)
    data.update(normalized["data"])
    normalized["data"] = data
    return normalized


def genome_from_bot_config(path: str | Path, *, kind: str = "search") -> CandidateGenome:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    genome = normalize_genome(
        {
            "kind": kind,
            "search": payload["search"],
            "eval": payload["eval"],
            "model": deepcopy(DEFAULT_MODEL),
            "data": deepcopy(DEFAULT_DATA),
            "metadata": {
                "source_config": str(Path(path)),
                "source_name": payload.get("name", Path(path).stem),
            },
        }
    )
    hybrid = payload.get("hybrid")
    if hybrid:
        genome["model"]["enabled"] = True
        genome["model"]["prior_mix"] = hybrid.get("prior_mix", 0.0)
        genome["model"]["leaf_mix"] = hybrid.get("leaf_mix", 0.0)
        genome["model"]["value_scale"] = hybrid.get("value_scale", 48.0)
    return CandidateGenome(genome)


def materialize_bot_config(
    genome: CandidateGenome,
    output_path: str | Path,
    *,
    name: str,
    weights_path: str | None = None,
) -> dict[str, Any]:
    output = {
        "name": name,
        "eval": deepcopy(genome.eval),
        "search": deepcopy(genome.search),
    }
    if genome.model.get("enabled") and weights_path:
        output["hybrid"] = {
            "weights_path": weights_path,
            "prior_mix": float(genome.model.get("prior_mix", 0.0)),
            "leaf_mix": float(genome.model.get("leaf_mix", 0.0)),
            "value_scale": float(genome.model.get("value_scale", 48.0)),
            "prior_depth_limit": int(genome.model.get("prior_depth_limit", 0)),
            "leaf_depth_limit": int(genome.model.get("leaf_depth_limit", 0)),
        }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
