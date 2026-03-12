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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    print(json.dumps(build_dataset(spec), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
