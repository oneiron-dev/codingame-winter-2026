from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import modal


def _detect_repo_root() -> Path:
    resolved = Path(__file__).resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "automation").exists() and (candidate / "python").exists() and (candidate / "rust").exists():
            return candidate
    return resolved.parent


REPO_ROOT = _detect_repo_root()
REMOTE_REPO = Path("/root/repo")

IGNORED_TOP_LEVEL = {
    ".git",
    ".prose",
    ".serena",
    "submission",
    "target",
    "worktrees",
}


def _ignore_repo_path(path: Path) -> bool:
    relative = path if path.is_absolute() else path
    parts = relative.parts
    return any(part in IGNORED_TOP_LEVEL for part in parts)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "build-essential",
        "curl",
        "default-jdk-headless",
        "git",
        "libssl-dev",
        "maven",
        "pkg-config",
    )
    .run_commands(
        "curl https://sh.rustup.rs -sSf | sh -s -- -y",
        "ln -sf /root/.cargo/bin/cargo /usr/local/bin/cargo",
        "ln -sf /root/.cargo/bin/rustc /usr/local/bin/rustc",
        "cargo --version",
        "mvn -version",
        "java -version",
    )
    .pip_install("torch")
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=str(REMOTE_REPO),
        copy=True,
        ignore=_ignore_repo_path,
    )
)

app = modal.App("snakebot-outerloop")
vol = modal.Volume.from_name("snakebot-datasets", create_if_missing=True)


def _repo_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REMOTE_REPO)
    env["SNAKEBOT_SELFPLAY_USE_CARGO_RUN"] = "1"
    env.setdefault("SNAKEBOT_GIT_SHA", "modal")
    return env


def _repo_relative_remote(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        return str(REMOTE_REPO / path)
    if path.resolve().is_relative_to(REMOTE_REPO.resolve()):
        return str(path)
    return str(path)


def _train_impl(spec_json: str) -> dict:
    sys.path.insert(0, str(REMOTE_REPO))
    spec = json.loads(spec_json)
    spec = dict(spec)

    with tempfile.TemporaryDirectory(prefix="snakebot-train-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        output_dir = tmpdir_path / "model"
        spec["output_dir"] = str(output_dir)
        if "volume_dataset_path" in spec:
            # Prefer reading dataset from shared Volume (avoids base64 serialisation).
            vol.reload()
            vol_src = Path(spec.pop("volume_dataset_path"))
            dataset_path = tmpdir_path / "dataset.jsonl"
            dataset_path.write_bytes(vol_src.read_bytes())
            spec.pop("dataset_jsonl_gz_b64", None)
            spec["dataset_path"] = str(dataset_path)
        elif "dataset_jsonl_gz_b64" in spec:
            dataset_path = tmpdir_path / "dataset.jsonl"
            dataset_path.write_bytes(
                gzip.decompress(base64.b64decode(spec.pop("dataset_jsonl_gz_b64")))
            )
            spec["dataset_path"] = str(dataset_path)
        else:
            spec["dataset_path"] = _repo_relative_remote(spec["dataset_path"])

        from python.train.outerloop.export_weights import export_weights
        from python.train.outerloop.train_model import train_distill_from_spec, train_from_spec

        training_mode = spec.get("training_mode", "standard")
        if training_mode == "distill":
            metrics = train_distill_from_spec(spec)
        else:
            metrics = train_from_spec(spec)
        weights_path = output_dir / "hybrid_weights.json"
        weights_payload = export_weights(
            Path(metrics["model_path"]),
            Path(metrics["training_config_path"]),
            weights_path,
        )
        metrics["model_path"] = "modal://hybrid_model.pt"
        metrics["training_config_path"] = "modal://training_config.json"
        return {
            "task": "train",
            "metrics": metrics,
            "weights_json": json.dumps(weights_payload, indent=2, sort_keys=True),
        }


def _selfplay_impl(spec_json: str) -> dict:
    spec = json.loads(spec_json)
    spec = dict(spec)

    with tempfile.TemporaryDirectory(prefix="snakebot-selfplay-") as tmpdir:
        tmp = Path(tmpdir)
        dataset_path = tmp / "selfplay.jsonl"
        output_dir = tmp / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        if "config_json" in spec:
            config_path = tmp / "candidate_config.json"
            config_path.write_text(spec.pop("config_json"), encoding="utf-8")
        else:
            config_path = Path(_repo_relative_remote(spec["config_path"]))

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
            str(config_path),
            "--dataset-path",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--merge-output",
        ]
        if spec.get("games", 0):
            command.extend(["--games", str(spec["games"])])
        if spec.get("search_ms", 0):
            command.extend(["--search-ms", str(spec["search_ms"])])
        if spec.get("maps_path"):
            maps_path = _repo_relative_remote(spec["maps_path"])
            if maps_path is None:
                raise ValueError("modal self-play requires maps_path to resolve inside the repo")
            command.extend(["--maps-path", maps_path, "--reuse-maps"])

        completed = subprocess.run(
            command,
            cwd=REMOTE_REPO,
            env=_repo_env(),
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "modal self-play failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        payload = json.loads(completed.stdout)
        dataset_bytes = dataset_path.read_bytes()
        compressed = base64.b64encode(gzip.compress(dataset_bytes)).decode("ascii")
        payload["task"] = "selfplay"
        payload["dataset_jsonl_gz_b64"] = compressed
        payload["dataset_path"] = spec["dataset_path"]

        # Also persist dataset to shared Volume for reliable cross-function access.
        run_id = spec.get("run_id", "unknown")
        candidate_id = spec.get("candidate_id", "unknown")
        vol_dataset_path = f"/data/{run_id}/{candidate_id}/dataset.jsonl"
        Path(vol_dataset_path).parent.mkdir(parents=True, exist_ok=True)
        Path(vol_dataset_path).write_bytes(dataset_bytes)
        vol.commit()
        payload["volume_dataset_path"] = vol_dataset_path

        return payload


def _write_temp_suite(tmpdir: Path, spec: dict) -> Path:
    if "suite_text" in spec:
        suite_path = tmpdir / f"{spec.get('suite_name', 'suite')}.txt"
        suite_path.write_text(spec["suite_text"], encoding="utf-8")
        return suite_path
    suite_path = _repo_relative_remote(spec.get("suite_path"))
    if suite_path is None:
        raise ValueError("arena-screen requires suite_path or suite_text")
    return Path(suite_path)


def _arena_screen_impl(spec_json: str) -> dict:
    spec = json.loads(spec_json)
    spec = dict(spec)

    with tempfile.TemporaryDirectory(prefix="snakebot-arena-screen-") as tmpdir:
        tmp = Path(tmpdir)
        candidate_config_payload = json.loads(spec["candidate_config_json"])
        if "weights_json" in spec:
            weights_path = tmp / spec.get("weights_filename", "hybrid_weights.json")
            weights_path.write_text(spec["weights_json"], encoding="utf-8")
            hybrid = candidate_config_payload.get("hybrid")
            if hybrid is not None:
                hybrid["weights_path"] = str(weights_path)
        candidate_config_path = tmp / "candidate_config.json"
        candidate_config_path.write_text(
            json.dumps(candidate_config_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        suite_path = _write_temp_suite(tmp, spec)
        results_db = tmp / "screening.sqlite"
        incumbent_config = _repo_relative_remote(spec["incumbent_config_path"])
        anchor_config = _repo_relative_remote(spec["anchor_config_path"])

        command = [
            "python3",
            "-m",
            "python.train.run_arena",
            "--candidate-config",
            str(candidate_config_path),
            "--incumbent-config",
            str(incumbent_config),
            "--anchor-config",
            str(anchor_config),
            "--heldout-suite",
            str(suite_path),
            "--shadow-suite",
            str(suite_path),
            "--results-db",
            str(results_db),
            "--name",
            str(spec.get("name", "modal_arena_screen")),
            "--evaluation-mode",
            "screening",
            "--skip-java-smoke",
        ]
        if spec.get("league") is not None:
            command.extend(["--league", str(spec["league"])])
        if spec.get("jobs") is not None:
            command.extend(["--jobs", str(spec["jobs"])])

        completed = subprocess.run(
            command,
            cwd=REMOTE_REPO,
            env=_repo_env(),
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "modal arena screening failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        payload = json.loads(completed.stdout)
        payload["task"] = "arena-screen"
        return payload


@app.function(
    image=image,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    timeout=60 * 60,
    max_containers=16,
    volumes={"/data": vol},
)
def train_l40s(spec_json: str) -> dict:
    return _train_impl(spec_json)


# Non-destructive note:
# We are intentionally keeping the alternative GPU entrypoints commented out.
# Right now the active training path is standardized on L40S to reduce ops noise
# while the hybrid loop is still being stabilized. If a later experiment really
# needs a different GPU tier, these definitions can be restored directly.
#
# @app.function(
#     image=image,
#     gpu="H100",
#     cpu=6.0,
#     memory=49152,
#     timeout=60 * 60,
#     max_containers=8,
# )
# def train_h100(spec_json: str) -> dict:
#     return _train_impl(spec_json)
#
#
# @app.function(
#     image=image,
#     gpu="H200",
#     cpu=6.0,
#     memory=65536,
#     timeout=60 * 60,
#     max_containers=4,
# )
# def train_h200(spec_json: str) -> dict:
#     return _train_impl(spec_json)
#
#
# @app.function(
#     image=image,
#     gpu="B200",
#     cpu=6.0,
#     memory=65536,
#     timeout=60 * 60,
#     max_containers=4,
# )
# def train_b200(spec_json: str) -> dict:
#     return _train_impl(spec_json)
#
#
# @app.function(
#     image=image,
#     gpu="A100",
#     cpu=6.0,
#     memory=49152,
#     timeout=60 * 60,
#     max_containers=8,
# )
# def train_a100(spec_json: str) -> dict:
#     return _train_impl(spec_json)


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=60 * 60,
    max_containers=32,
    volumes={"/data": vol},
)
def run_selfplay(spec_json: str) -> dict:
    return _selfplay_impl(spec_json)


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=60 * 60,
    max_containers=32,
    volumes={"/data": vol},
)
def run_arena_screen(spec_json: str) -> dict:
    return _arena_screen_impl(spec_json)


def _train_teacher_impl(spec_json: str) -> dict:
    sys.path.insert(0, str(REMOTE_REPO))
    spec = json.loads(spec_json)
    spec = dict(spec)

    with tempfile.TemporaryDirectory(prefix="snakebot-teacher-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        output_dir = tmpdir_path / "teacher"
        spec["output_dir"] = str(output_dir)
        # Resolve dataset from Volume or fallback
        if "volume_dataset_path" in spec:
            vol.reload()
            vol_src = Path(spec.pop("volume_dataset_path"))
            dataset_path = tmpdir_path / "dataset.jsonl"
            dataset_path.write_bytes(vol_src.read_bytes())
            spec["dataset_path"] = str(dataset_path)
        else:
            spec["dataset_path"] = _repo_relative_remote(spec["dataset_path"])

        from python.train.outerloop.train_model import train_teacher_from_spec

        metrics = train_teacher_from_spec(spec)
        # Save teacher model to Volume for later use
        model_path = Path(metrics["model_path"])
        config_path = Path(metrics["training_config_path"])
        run_id = spec.get("run_id", "default")
        vol_teacher_dir = Path(f"/data/{run_id}/teacher")
        vol_teacher_dir.mkdir(parents=True, exist_ok=True)
        (vol_teacher_dir / "teacher_model.pt").write_bytes(model_path.read_bytes())
        (vol_teacher_dir / "teacher_training_config.json").write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        vol.commit()
        metrics["volume_model_path"] = str(vol_teacher_dir / "teacher_model.pt")
        metrics["volume_config_path"] = str(vol_teacher_dir / "teacher_training_config.json")
        return {"task": "train-teacher", "metrics": metrics}


def _generate_soft_targets_impl(spec_json: str) -> dict:
    sys.path.insert(0, str(REMOTE_REPO))
    spec = json.loads(spec_json)
    spec = dict(spec)

    with tempfile.TemporaryDirectory(prefix="snakebot-softtargets-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        vol.reload()
        # Resolve teacher model
        if "volume_teacher_model_path" in spec:
            teacher_model_path = tmpdir_path / "teacher_model.pt"
            teacher_model_path.write_bytes(Path(spec["volume_teacher_model_path"]).read_bytes())
            spec["teacher_model_path"] = str(teacher_model_path)
        if "volume_teacher_config_path" in spec:
            teacher_config_path = tmpdir_path / "teacher_training_config.json"
            teacher_config_path.write_text(
                Path(spec["volume_teacher_config_path"]).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            spec["teacher_training_config_path"] = str(teacher_config_path)
        # Resolve dataset
        if "volume_dataset_path" in spec:
            dataset_path = tmpdir_path / "dataset.jsonl"
            dataset_path.write_bytes(Path(spec.pop("volume_dataset_path")).read_bytes())
            spec["dataset_path"] = str(dataset_path)
        else:
            spec["dataset_path"] = _repo_relative_remote(spec["dataset_path"])

        output_path = tmpdir_path / "augmented_dataset.jsonl"
        spec["output_path"] = str(output_path)

        from python.train.outerloop.train_model import generate_soft_targets

        result = generate_soft_targets(spec)
        # Write augmented dataset to Volume
        run_id = spec.get("run_id", "default")
        vol_out = Path(f"/data/{run_id}/augmented/dataset.jsonl")
        vol_out.parent.mkdir(parents=True, exist_ok=True)
        vol_out.write_bytes(output_path.read_bytes())
        vol.commit()
        result["volume_augmented_path"] = str(vol_out)
        result["task"] = "generate-soft-targets"
        return result


@app.function(
    image=image,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    timeout=60 * 60,
    max_containers=8,
    volumes={"/data": vol},
)
def train_teacher_l40s(spec_json: str) -> dict:
    return _train_teacher_impl(spec_json)


@app.function(
    image=image,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    timeout=60 * 60,
    max_containers=8,
    volumes={"/data": vol},
)
def generate_soft_targets_l40s(spec_json: str) -> dict:
    return _generate_soft_targets_impl(spec_json)


def _decode_dataset_payload(payload: dict, *, preserve_blob: bool = False) -> dict:
    output = dict(payload)
    dataset_blob = output["dataset_jsonl_gz_b64"]
    dataset_path = Path(output["dataset_path"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_bytes(gzip.decompress(base64.b64decode(dataset_blob)))
    if not preserve_blob:
        output.pop("dataset_jsonl_gz_b64")
    return output


def _train_function_for_gpu(gpu_name: str) -> Callable[[str], dict]:
    normalized = gpu_name.strip().upper()
    if normalized in {"L40S", "L40"}:
        return train_l40s
    # Non-destructive note:
    # Other GPU tiers are intentionally disabled for now. Restore the commented
    # branches above if we later want to reopen H100/H200/B200/A100 experiments.
    # if normalized == "H100":
    #     return train_h100
    # if normalized == "H200":
    #     return train_h200
    # if normalized == "B200":
    #     return train_b200
    # if normalized == "A100":
    #     return train_a100
    raise ValueError(f"unsupported modal gpu type: {gpu_name}")


@app.local_entrypoint()
def main(task: str, spec_json: str) -> str:
    spec = json.loads(spec_json)
    if task == "train":
        gpu_name = str(spec.get("gpu", "L40S"))
        result = _train_function_for_gpu(gpu_name).remote(spec_json)
        return json.dumps(result, indent=2, sort_keys=True)
    if task == "selfplay":
        result = run_selfplay.remote(spec_json)
        return json.dumps(_decode_dataset_payload(result), indent=2, sort_keys=True)
    if task == "arena-screen":
        result = run_arena_screen.remote(spec_json)
        return json.dumps(result, indent=2, sort_keys=True)
    if task == "train-teacher":
        result = train_teacher_l40s.remote(spec_json)
        return json.dumps(result, indent=2, sort_keys=True)
    if task == "generate-soft-targets":
        result = generate_soft_targets_l40s.remote(spec_json)
        return json.dumps(result, indent=2, sort_keys=True)
    raise ValueError(f"unsupported modal task: {task}")
