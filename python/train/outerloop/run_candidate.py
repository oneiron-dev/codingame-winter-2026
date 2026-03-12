from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from python.train.java_smoke import artifact_hash, behavior_hash
from python.train.outerloop.genome import CandidateGenome, dump_genome, genome_from_bot_config, load_genome, materialize_bot_config
from python.train.outerloop.patch_llm import apply_patch, maybe_generate_patch
from python.train.outerloop.registry import ensure_run_manifest, iso_now, register_candidate, write_stage_result
from python.train.outerloop.workspace import create_worktree, remove_worktree


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", type=Path, default=None)
    parser.add_argument("--base-config", type=Path, default=REPO_ROOT / "rust/bot/configs/submission_current.json")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--stage", choices=("all", "stage0", "stage1", "stage2"), default="all")
    parser.add_argument("--program", type=str, default="automation/outerloop/outerloop.prose")
    parser.add_argument("--incumbent-config", type=Path, default=REPO_ROOT / "rust/bot/configs/incumbent_current.json")
    parser.add_argument("--anchor-config", type=Path, default=REPO_ROOT / "rust/bot/configs/anchor_root_only.json")
    parser.add_argument("--smoke-suite", type=Path, default=REPO_ROOT / "config/arena/smoke_v1.txt")
    parser.add_argument("--heldout-suite", type=Path, default=REPO_ROOT / "config/arena/heldout_v1.txt")
    parser.add_argument("--shadow-suite", type=Path, default=REPO_ROOT / "config/arena/shadow_v1.txt")
    parser.add_argument("--results-db", type=Path, default=REPO_ROOT / "python/train/results.sqlite")
    parser.add_argument("--arena-bin", type=Path, default=None)
    parser.add_argument(
        "--stage1-executor",
        choices=("local", "modal-arena-screen"),
        default=None,
    )
    parser.add_argument("--keep-worktree", action="store_true")
    return parser.parse_args()


def run_json(command: list[str], *, cwd: Path = REPO_ROOT, stdin_text: str | None = None) -> dict[str, Any]:
    output = subprocess.check_output(command, cwd=cwd, text=True, input=stdin_text)
    return json.loads(output)


def cargo_build(*args: str, cwd: Path = REPO_ROOT) -> None:
    subprocess.run(["cargo", "build", "--release", "-q", *args], cwd=cwd, check=True)


def repo_relative_string(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def stage0(
    *,
    run_id: str,
    candidate_id: str,
    genome: CandidateGenome,
    program: str,
) -> tuple[dict[str, Any], Path, Path]:
    ensure_run_manifest(run_id, program=program)
    candidate_dir = register_candidate(
        run_id,
        candidate_id,
        genome_hash=genome.semantic_hash,
        kind=genome.kind,
    )
    dump_genome(candidate_dir / "genome.json", genome)
    started_at = iso_now()
    worktree = create_worktree(run_id, candidate_id)
    config_path = candidate_dir / "candidate_config.json"
    materialize_bot_config(genome, config_path, name=candidate_id)

    failure_reason = "initial candidate materialization"
    patch_path = maybe_generate_patch(
        allowed_files=[
            "rust/bot/src/search.rs",
            "rust/bot/src/eval.rs",
            "rust/bot/src/features.rs",
            "python/train/outerloop/train_model.py",
        ],
        failure_reason=failure_reason,
        genome_hash=genome.semantic_hash,
        candidate_dir=candidate_dir,
    )
    if patch_path is not None:
        apply_patch(worktree, patch_path)

    payload = {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "candidate_dir": str(candidate_dir),
        "stage": "stage0",
        "status": "ready",
        "executor": "local",
        "started_at": started_at,
        "finished_at": iso_now(),
        "artifact_hash": artifact_hash(config_path),
        "behavior_hash": behavior_hash(config_path),
        "genome_hash": genome.semantic_hash,
        "config_path": str(config_path),
        "worktree": str(worktree),
    }
    write_stage_result(run_id, candidate_id, "stage0", payload)
    return payload, candidate_dir, worktree


def maybe_train_hybrid(
    genome: CandidateGenome,
    *,
    candidate_dir: Path,
    config_path: Path,
    worktree: Path,
) -> dict[str, Any] | None:
    if not genome.model.get("enabled"):
        return None

    modal_dir = worktree / ".outerloop_modal" / candidate_dir.name
    modal_dir.mkdir(parents=True, exist_ok=True)
    modal_config_path = modal_dir / "candidate_config.json"
    shutil.copy2(config_path, modal_config_path)

    dataset_spec = {
        "executor": genome.data.get("executor", "local"),
        "seed_start": genome.data["seed_start"],
        "seed_count": genome.data["seed_count"],
        "league": genome.data["league"],
        "workers": genome.data["workers"],
        "max_turns": genome.data["max_turns"],
        "extra_nodes_after_root": genome.data["extra_nodes_after_root"],
        "config_path": str(modal_config_path),
        "config_json": modal_config_path.read_text(encoding="utf-8"),
        "dataset_path": str(modal_dir / "dataset.jsonl"),
        "output_dir": str(modal_dir / "dataset_artifacts"),
    }
    if genome.data.get("generate_dataset"):
        dataset_spec_path = candidate_dir / "dataset_spec.json"
        dataset_spec_path.write_text(json.dumps(dataset_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        dataset_payload = run_json(
            ["python3", "-m", "python.train.outerloop.build_dataset", "--spec", str(dataset_spec_path)],
            cwd=worktree,
        )
        dataset_path = dataset_payload["dataset_path"]
    else:
        dataset_path = str((REPO_ROOT / genome.data["dataset_path"]).resolve())

    model_dir = candidate_dir / "model"
    train_spec = {
        "dataset_path": dataset_path,
        "output_dir": str(model_dir),
        "device_preference": genome.model.get("device_preference", "mps"),
        "gpu": genome.model.get("gpu", "L40S"),
        "epochs": genome.model.get("epochs", 4),
        "batch_size": genome.model.get("batch_size", 128),
        "learning_rate": genome.model.get("learning_rate", 1e-3),
        "weight_decay": genome.model.get("weight_decay", 1e-4),
        "max_samples": genome.model.get("max_samples", 0),
        "conv_channels": genome.model.get("conv_channels", 8),
        "seed": 42,
        "policy_loss_weight": 1.0,
    }
    if "dataset_jsonl_gz_b64" in dataset_payload:
        train_spec["dataset_jsonl_gz_b64"] = dataset_payload["dataset_jsonl_gz_b64"]
    executor = genome.model.get("executor", "local")
    train_spec_path = candidate_dir / "train_spec.json"
    train_spec_path.write_text(json.dumps(train_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if executor == "modal-train":
        train_payload = run_json(
            ["python3", "-m", "python.train.outerloop.launch_modal", "--task", "train", "--spec", str(train_spec_path)],
            cwd=worktree,
        )
        metrics = train_payload["metrics"]
        weights_path = model_dir / "hybrid_weights.json"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_text(train_payload["weights_json"], encoding="utf-8")
    else:
        metrics = run_json(
            ["python3", "-m", "python.train.outerloop.train_model", "--dataset-path", dataset_path, "--output-dir", str(model_dir),
             "--device-preference", str(train_spec["device_preference"]),
             "--epochs", str(train_spec["epochs"]),
             "--batch-size", str(train_spec["batch_size"]),
             "--learning-rate", str(train_spec["learning_rate"]),
             "--weight-decay", str(train_spec["weight_decay"]),
             "--max-samples", str(train_spec["max_samples"]),
             "--conv-channels", str(train_spec["conv_channels"]),
             "--seed", str(train_spec["seed"]),
             "--policy-loss-weight", str(train_spec["policy_loss_weight"])],
            cwd=worktree,
        )
        weights_path = model_dir / "hybrid_weights.json"
        run_json(
            [
                "python3",
                "-m",
                "python.train.outerloop.export_weights",
                "--model",
                str(metrics["model_path"]),
                "--training-config",
                str(metrics["training_config_path"]),
                "--out",
                str(weights_path),
            ],
            cwd=worktree,
        )

    materialize_bot_config(genome, config_path, name=config_path.stem, weights_path=str(weights_path))
    return {
        "dataset_path": dataset_path,
        "weights_path": str(weights_path),
        "metrics": metrics,
    }


def run_stage1(
    *,
    run_id: str,
    candidate_id: str,
    candidate_dir: Path,
    config_path: Path,
    genome_hash: str,
    stage1_executor: str,
    args: argparse.Namespace,
    worktree: Path,
) -> dict[str, Any]:
    started_at = iso_now()
    executor = stage1_executor
    if executor == "modal-arena-screen":
        suite_text = None
        try:
            relative_suite = args.smoke_suite.resolve().relative_to(REPO_ROOT.resolve())
            suite_path = str(relative_suite)
        except ValueError:
            suite_path = None
            suite_text = args.smoke_suite.read_text(encoding="utf-8")
        modal_spec: dict[str, Any] = {
            "candidate_config_json": config_path.read_text(encoding="utf-8"),
            "incumbent_config_path": repo_relative_string(args.incumbent_config),
            "anchor_config_path": repo_relative_string(args.anchor_config),
            "suite_path": suite_path,
            "suite_name": args.smoke_suite.stem,
            "name": f"{run_id}_{candidate_id}_stage1",
        }
        if suite_text is not None:
            modal_spec["suite_text"] = suite_text
        payload = run_json(
            [
                "python3",
                "-m",
                "python.train.outerloop.launch_modal",
                "--task",
                "arena-screen",
                "--spec",
                "/dev/stdin",
            ],
            cwd=worktree,
            stdin_text=json.dumps(modal_spec),
        )
    else:
        payload = run_json(
            [
                "python3",
                "-m",
                "python.train.run_arena",
                "--candidate-config",
                str(config_path),
                "--incumbent-config",
                str(args.incumbent_config),
                "--anchor-config",
                str(args.anchor_config),
                "--heldout-suite",
                str(args.smoke_suite),
                "--shadow-suite",
                str(args.smoke_suite),
                "--results-db",
                str(args.results_db),
                "--name",
                f"{run_id}_{candidate_id}_stage1",
                "--evaluation-mode",
                "screening",
                "--skip-java-smoke",
            ]
            + (["--arena-bin", str(args.arena_bin)] if args.arena_bin else []),
            cwd=worktree,
        )
    result = {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "candidate_dir": str(candidate_dir),
        "stage": "stage1",
        "status": payload["status"],
        "executor": executor,
        "started_at": started_at,
        "finished_at": iso_now(),
        "artifact_hash": artifact_hash(config_path),
        "behavior_hash": behavior_hash(config_path),
        "genome_hash": genome_hash,
        "result": payload,
    }
    write_stage_result(run_id, candidate_id, "stage1", result)
    return result


def run_stage2(
    *,
    run_id: str,
    candidate_id: str,
    candidate_dir: Path,
    config_path: Path,
    genome_hash: str,
    args: argparse.Namespace,
    worktree: Path,
) -> dict[str, Any]:
    started_at = iso_now()
    payload = run_json(
        [
            "python3",
            "-m",
            "python.train.run_arena",
            "--candidate-config",
            str(config_path),
            "--incumbent-config",
            str(args.incumbent_config),
            "--anchor-config",
            str(args.anchor_config),
            "--heldout-suite",
            str(args.heldout_suite),
            "--shadow-suite",
            str(args.shadow_suite),
            "--results-db",
            str(args.results_db),
            "--name",
            f"{run_id}_{candidate_id}_stage2",
            "--evaluation-mode",
            "authoritative",
        ]
        + (["--arena-bin", str(args.arena_bin)] if args.arena_bin else []),
        cwd=worktree,
    )
    result = {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "candidate_dir": str(candidate_dir),
        "stage": "stage2",
        "status": payload["status"],
        "executor": "local",
        "started_at": started_at,
        "finished_at": iso_now(),
        "artifact_hash": artifact_hash(config_path),
        "behavior_hash": behavior_hash(config_path),
        "genome_hash": genome_hash,
        "result": payload,
        "promotable": payload["status"] == "accepted",
    }
    write_stage_result(run_id, candidate_id, "stage2", result)
    return result


def main() -> None:
    args = parse_args()
    genome = load_genome(args.genome) if args.genome else genome_from_bot_config(args.base_config)
    candidate_id = genome.candidate_id
    stage0_payload, candidate_dir, worktree = stage0(
        run_id=args.run_id,
        candidate_id=candidate_id,
        genome=genome,
        program=args.program,
    )
    config_path = Path(stage0_payload["config_path"])
    try:
        cargo_build("-p", "snakebot-bot", "--bin", "snakebot-bot", cwd=worktree)
        cargo_build("-p", "snakebot-bot", "--bin", "arena", cwd=worktree)
        training = maybe_train_hybrid(genome, candidate_dir=candidate_dir, config_path=config_path, worktree=worktree)
        if training is not None:
            stage0_payload["training"] = training
            write_stage_result(args.run_id, candidate_id, "stage0", stage0_payload)
        if args.stage in ("all", "stage1"):
            run_stage1(
                run_id=args.run_id,
                candidate_id=candidate_id,
                candidate_dir=candidate_dir,
                config_path=config_path,
                genome_hash=genome.semantic_hash,
                stage1_executor=args.stage1_executor or str(genome.metadata.get("stage1_executor", "local")),
                args=args,
                worktree=worktree,
            )
        if args.stage in ("all", "stage2"):
            run_stage2(
                run_id=args.run_id,
                candidate_id=candidate_id,
                candidate_dir=candidate_dir,
                config_path=config_path,
                genome_hash=genome.semantic_hash,
                args=args,
                worktree=worktree,
            )
    finally:
        if not args.keep_worktree:
            remove_worktree(worktree)


if __name__ == "__main__":
    main()
