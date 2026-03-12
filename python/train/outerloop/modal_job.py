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
        if "dataset_jsonl_gz_b64" in spec:
            dataset_path = tmpdir_path / "dataset.jsonl"
            dataset_path.write_bytes(
                gzip.decompress(base64.b64decode(spec.pop("dataset_jsonl_gz_b64")))
            )
            spec["dataset_path"] = str(dataset_path)
        else:
            spec["dataset_path"] = _repo_relative_remote(spec["dataset_path"])

        from python.train.outerloop.export_weights import export_weights
        from python.train.outerloop.train_model import train_from_spec

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
        compressed = base64.b64encode(gzip.compress(dataset_path.read_bytes())).decode("ascii")
        payload["task"] = "selfplay"
        payload["dataset_jsonl_gz_b64"] = compressed
        payload["dataset_path"] = spec["dataset_path"]
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
        candidate_config_path = tmp / "candidate_config.json"
        candidate_config_path.write_text(spec["candidate_config_json"], encoding="utf-8")
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
)
def train_l40s(spec_json: str) -> dict:
    return _train_impl(spec_json)


@app.function(
    image=image,
    gpu="H100",
    cpu=6.0,
    memory=49152,
    timeout=60 * 60,
    max_containers=8,
)
def train_h100(spec_json: str) -> dict:
    return _train_impl(spec_json)


@app.function(
    image=image,
    gpu="H200",
    cpu=6.0,
    memory=65536,
    timeout=60 * 60,
    max_containers=4,
)
def train_h200(spec_json: str) -> dict:
    return _train_impl(spec_json)


@app.function(
    image=image,
    gpu="B200",
    cpu=6.0,
    memory=65536,
    timeout=60 * 60,
    max_containers=4,
)
def train_b200(spec_json: str) -> dict:
    return _train_impl(spec_json)


@app.function(
    image=image,
    gpu="A100",
    cpu=6.0,
    memory=49152,
    timeout=60 * 60,
    max_containers=8,
)
def train_a100(spec_json: str) -> dict:
    return _train_impl(spec_json)


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=60 * 60,
    max_containers=32,
)
def run_selfplay(spec_json: str) -> dict:
    return _selfplay_impl(spec_json)


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=60 * 60,
    max_containers=32,
)
def run_arena_screen(spec_json: str) -> dict:
    return _arena_screen_impl(spec_json)


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
    if normalized == "H100":
        return train_h100
    if normalized == "H200":
        return train_h200
    if normalized == "B200":
        return train_b200
    if normalized == "A100":
        return train_a100
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
    raise ValueError(f"unsupported modal task: {task}")
