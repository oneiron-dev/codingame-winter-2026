from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from python.train.outerloop.launch_modal import launch_modal

REPO_ROOT = Path(__file__).resolve().parents[3]


def build_dataset(spec: dict) -> dict:
    executor = str(spec.get("executor", "local"))
    if executor == "modal-selfplay":
        return launch_modal("selfplay", spec, preserve_selfplay_blob=True)

    command = [
        "python3",
        "-m",
        "python.train.parallel_selfplay",
        "--seed-start",
        str(spec["seed_start"]),
        "--seed-count",
        str(spec["seed_count"]),
        "--league",
        str(spec["league"]),
        "--workers",
        str(spec["workers"]),
        "--max-turns",
        str(spec["max_turns"]),
        "--extra-nodes-after-root",
        str(spec["extra_nodes_after_root"]),
        "--config-path",
        str(spec["config_path"]),
        "--dataset-path",
        str(spec["dataset_path"]),
        "--output-dir",
        str(spec["output_dir"]),
    ]
    output = subprocess.check_output(command, cwd=REPO_ROOT, text=True)
    return json.loads(output)


def build_shared_dataset(spec: dict) -> dict:
    """Generate a shared self-play dataset for all candidates in a run.

    Uses a larger seed count (500+) and writes to a well-known Volume path
    so all parallel candidates can reuse the same dataset.
    """
    executor = str(spec.get("executor", "modal-selfplay"))
    run_id = spec["run_id"]

    shared_spec = {
        "seed_start": int(spec.get("seed_start", 1)),
        "seed_count": int(spec.get("seed_count", 500)),
        "league": int(spec.get("league", 4)),
        "workers": int(spec.get("workers", 8)),
        "max_turns": int(spec.get("max_turns", 120)),
        "extra_nodes_after_root": int(spec.get("extra_nodes_after_root", 5000)),
        "config_path": spec["config_path"],
        "dataset_path": spec.get("dataset_path", f"python/train/data/shared_{run_id}.jsonl"),
        "output_dir": spec.get("output_dir", f"python/train/data/shared_{run_id}_artifacts"),
        "executor": executor,
        "shared_run_id": run_id,
    }
    if "config_json" in spec:
        shared_spec["config_json"] = spec["config_json"]
    if "maps_path" in spec:
        shared_spec["maps_path"] = spec["maps_path"]

    if executor == "modal-selfplay":
        payload = launch_modal("selfplay", shared_spec, preserve_selfplay_blob=False)
    else:
        payload = build_dataset(shared_spec)

    payload["shared_dataset_id"] = run_id
    payload["shared_dataset_path"] = shared_spec["dataset_path"]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--shared", action="store_true", help="Build a shared dataset for all candidates")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if args.shared:
        print(json.dumps(build_shared_dataset(spec), indent=2, sort_keys=True))
    else:
        print(json.dumps(build_dataset(spec), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
