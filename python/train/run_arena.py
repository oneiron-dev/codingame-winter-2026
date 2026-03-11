from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from python.train.java_smoke import artifact_hash, behavior_hash, run_java_smoke
from python.train.results import (
    CURRENT_ACCEPTANCE_VERSION,
    append_result,
    check_gates,
    compute_composite,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE_VERSION = CURRENT_ACCEPTANCE_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--incumbent-config",
        type=Path,
        default=REPO_ROOT / "rust/bot/configs/incumbent_current.json",
    )
    parser.add_argument(
        "--anchor-config",
        type=Path,
        default=REPO_ROOT / "rust/bot/configs/anchor_root_only.json",
    )
    parser.add_argument(
        "--heldout-suite",
        type=Path,
        default=REPO_ROOT / "config/arena/heldout_v1.txt",
    )
    parser.add_argument(
        "--shadow-suite",
        type=Path,
        default=REPO_ROOT / "config/arena/shadow_v1.txt",
    )
    parser.add_argument("--league", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--results-db", type=Path, default=REPO_ROOT / "python/train/results.sqlite")
    parser.add_argument("--name", type=str, default="arena_eval")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def build_release_arena() -> Path:
    run(["cargo", "build", "--release", "-q", "-p", "snakebot-bot", "--bin", "arena"])
    return REPO_ROOT / "target/release/arena"


def run_arena_binary(
    arena_bin: Path,
    candidate_config: Path,
    opponent_config: Path,
    suite: Path,
    league: int,
    jobs: int,
) -> dict:
    output = subprocess.check_output(
        [
            str(arena_bin),
            "--bot-a-config",
            str(candidate_config),
            "--bot-b-config",
            str(opponent_config),
            "--suite",
            str(suite),
            "--league",
            str(league),
            "--jobs",
            str(jobs),
        ],
        cwd=REPO_ROOT,
        text=True,
    )
    return json.loads(output)


def main() -> None:
    args = parse_args()
    candidate_artifact_hash = artifact_hash(args.candidate_config)
    incumbent_artifact_hash = artifact_hash(args.incumbent_config)
    anchor_artifact_hash = artifact_hash(args.anchor_config)
    candidate_behavior_hash = behavior_hash(args.candidate_config)
    incumbent_behavior_hash = behavior_hash(args.incumbent_config)
    anchor_behavior_hash = behavior_hash(args.anchor_config)

    if candidate_behavior_hash == anchor_behavior_hash:
        raise ValueError("candidate config must differ from anchor config")

    if candidate_behavior_hash == incumbent_behavior_hash:
        metrics = {
            "acceptance_version": ACCEPTANCE_VERSION,
            "candidate_config_artifact_hash": candidate_artifact_hash,
            "candidate_config_behavior_hash": candidate_behavior_hash,
            "incumbent_config_artifact_hash": incumbent_artifact_hash,
            "incumbent_config_behavior_hash": incumbent_behavior_hash,
            "anchor_config_artifact_hash": anchor_artifact_hash,
            "anchor_config_behavior_hash": anchor_behavior_hash,
            "heldout_suite": str(args.heldout_suite),
            "heldout_suite_name": args.heldout_suite.stem,
            "shadow_suite": str(args.shadow_suite),
            "shadow_suite_name": args.shadow_suite.stem,
            "noop_reason": "candidate_behavior_matches_incumbent",
        }
        status = "informational"
        append_result(
            args.results_db,
            name=args.name,
            status=status,
            description="candidate behavior matches incumbent; skipped arena",
            metrics=metrics,
            failures=[],
            acceptance_version=ACCEPTANCE_VERSION,
        )
        payload = {
            "status": status,
            "reason": "candidate behavior matches incumbent",
            "composite_score": compute_composite(metrics),
            "metrics": metrics,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if candidate_artifact_hash == incumbent_artifact_hash:
        print("warning: candidate config matches incumbent artifact hash", file=sys.stderr)

    arena_bin = build_release_arena()
    heldout = run_arena_binary(
        arena_bin,
        args.candidate_config,
        args.incumbent_config,
        args.heldout_suite,
        args.league,
        args.jobs,
    )
    shadow = run_arena_binary(
        arena_bin,
        args.candidate_config,
        args.anchor_config,
        args.shadow_suite,
        args.league,
        args.jobs,
    )
    smoke = run_java_smoke(
        league=args.league,
        seed_file=REPO_ROOT / "config/arena/smoke_v1.txt",
        boss_count=4,
        mirror_count=4,
        candidate_config=args.candidate_config,
    )

    heldout_win_margin = (heldout["wins"] - heldout["losses"]) / max(heldout["matches"], 1)
    metrics = {
        "acceptance_version": ACCEPTANCE_VERSION,
        "candidate_config_artifact_hash": candidate_artifact_hash,
        "candidate_config_behavior_hash": candidate_behavior_hash,
        "incumbent_config_artifact_hash": incumbent_artifact_hash,
        "incumbent_config_behavior_hash": incumbent_behavior_hash,
        "anchor_config_artifact_hash": anchor_artifact_hash,
        "anchor_config_behavior_hash": anchor_behavior_hash,
        "heldout_suite": str(args.heldout_suite),
        "heldout_suite_name": heldout["suite_name"],
        "shadow_suite": str(args.shadow_suite),
        "shadow_suite_name": shadow["suite_name"],
        "heldout_body_diff": heldout["average_body_diff"],
        "heldout_win_margin": heldout_win_margin,
        "shadow_body_diff": shadow["average_body_diff"],
        "opening_move_max_ms": heldout["side_a"]["opening_move_max_ms"],
        "opening_move_p95_ms": heldout["side_a"]["opening_move_p95_ms"],
        "later_turn_p95_ms": heldout["side_a"]["later_move_p95_ms"],
        "later_turn_p99_ms": heldout["side_a"]["later_move_p99_ms"],
        "java_smoke_passed": float(smoke["passed"]),
        "java_smoke_embedded_artifact_hash": smoke["embedded_config_artifact_hash"],
        "java_smoke_embedded_behavior_hash": smoke["embedded_config_behavior_hash"],
    }
    gate_result = check_gates(metrics)
    status = (
        "accepted"
        if heldout["average_body_diff"] > 0.0
        and heldout["wins"] >= heldout["losses"]
        and shadow["average_body_diff"] >= 0.0
        and smoke["passed"]
        and gate_result.passed
        else "rejected"
    )

    append_result(
        args.results_db,
        name=args.name,
        status=status,
        description="arena + java smoke evaluation",
        metrics=metrics,
        failures=gate_result.failures,
        acceptance_version=ACCEPTANCE_VERSION,
    )
    payload = {
        "status": status,
        "composite_score": compute_composite(metrics),
        "failures": gate_result.failures,
        "metrics": metrics,
        "heldout": heldout,
        "shadow": shadow,
        "java_smoke": smoke,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
