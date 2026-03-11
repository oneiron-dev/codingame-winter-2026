from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from python.train.experiment import EXPERIMENT
from python.train.results import append_result, check_gates, compute_composite
from python.train.train_value import train


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=128)
    parser.add_argument("--league", type=int, default=4)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--games", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--search-ms", type=int, default=2)
    parser.add_argument("--maps-path", type=Path, default=REPO_ROOT / "python/train/artifacts/maps_l4.jsonl")
    parser.add_argument("--dataset-path", type=Path, default=REPO_ROOT / EXPERIMENT["dataset_path"])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / EXPERIMENT["output_dir"])
    parser.add_argument("--results-db", type=Path, default=REPO_ROOT / EXPERIMENT["results_db"])
    parser.add_argument("--name", type=str, default="parallel_selfplay_value")
    parser.add_argument("--reuse-maps", action="store_true")
    parser.add_argument("--train", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, stdout=None) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=stdout)


def ensure_java_oracle() -> Path:
    run(
        [
            "mvn",
            "-q",
            "-DskipTests",
            "test-compile",
            "dependency:build-classpath",
            "-Dmdep.outputFile=cp.txt",
        ]
    )
    classpath = (REPO_ROOT / "cp.txt").read_text(encoding="utf-8").strip()
    return Path(f"target/classes:target/test-classes:{classpath}")


def build_exporter() -> Path:
    run(["cargo", "build", "-q", "-p", "snakebot-bot", "--bin", "selfplay_export"])
    return REPO_ROOT / "target/debug/selfplay_export"


def dump_maps(args: argparse.Namespace, classpath: Path) -> None:
    if args.reuse_maps and args.maps_path.exists():
        return
    args.maps_path.parent.mkdir(parents=True, exist_ok=True)
    with args.maps_path.open("w", encoding="utf-8") as handle:
        run(
            [
                "java",
                "-cp",
                str(classpath),
                "com.codingame.game.MapDumpCli",
                str(args.seed_start),
                str(args.seed_count),
                str(args.league),
            ],
            stdout=handle,
        )


def export_shards(args: argparse.Namespace, exporter_bin: Path) -> list[Path]:
    shard_dir = args.dataset_path.parent / f"{args.dataset_path.stem}_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    limit = args.games if args.games > 0 else args.seed_count
    workers = max(1, min(args.workers, limit))
    procs: list[tuple[int, subprocess.Popen[str], Path]] = []
    shard_paths: list[Path] = []

    for shard_index in range(workers):
        shard_path = shard_dir / f"selfplay-{shard_index:03d}.jsonl"
        shard_paths.append(shard_path)
        cmd = [
            str(exporter_bin),
            "--maps",
            str(args.maps_path),
            "--out",
            str(shard_path),
            "--limit",
            str(limit),
            "--max-turns",
            str(args.max_turns),
            "--search-ms",
            str(args.search_ms),
            "--shard-index",
            str(shard_index),
            "--num-shards",
            str(workers),
        ]
        procs.append(
            (
                shard_index,
                subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ),
                shard_path,
            )
        )

    for shard_index, proc, shard_path in procs:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"shard {shard_index} failed for {shard_path.name}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
        if stderr.strip():
            print(stderr.strip(), file=sys.stderr)

    return shard_paths


def merge_shards(dataset_path: Path, shard_paths: list[Path]) -> int:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    with dataset_path.open("w", encoding="utf-8") as out:
        for shard_path in shard_paths:
            if not shard_path.exists():
                continue
            with shard_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    out.write(line)
                    sample_count += 1
    return sample_count


def run_training(args: argparse.Namespace, sample_count: int) -> dict:
    config = deepcopy(EXPERIMENT)
    config["name"] = args.name
    config["dataset_path"] = str(args.dataset_path)
    config["output_dir"] = str(args.output_dir)
    config["results_db"] = str(args.results_db)
    metrics = train(config)
    metrics["dataset_samples_generated"] = sample_count
    gate_result = check_gates(metrics)
    status = "accepted" if gate_result.passed else "rejected"
    append_result(
        args.results_db,
        name=args.name,
        status=status,
        description="parallel self-play pipeline",
        metrics=metrics,
        failures=gate_result.failures,
    )
    return {
        "status": status,
        "composite_score": compute_composite(metrics),
        "failures": gate_result.failures,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    classpath = ensure_java_oracle()
    exporter_bin = build_exporter()
    dump_maps(args, classpath)
    shard_paths = export_shards(args, exporter_bin)
    sample_count = merge_shards(args.dataset_path, shard_paths)

    payload: dict[str, object] = {
        "maps_path": str(args.maps_path),
        "dataset_path": str(args.dataset_path),
        "workers": min(max(1, args.workers), args.games if args.games > 0 else args.seed_count),
        "sample_count": sample_count,
        "shards": [str(path) for path in shard_paths],
    }

    if args.train:
        payload["training"] = run_training(args, sample_count)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
