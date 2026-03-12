from __future__ import annotations

import fcntl
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

from python.train.outerloop import OUTERLOOP_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]
PROGRAM_PATH = "automation/outerloop/outerloop.prose"
T = TypeVar("T")


def iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_root(run_id: str) -> Path:
    return REPO_ROOT / "artifacts" / "outerloop" / "runs" / run_id


def _default_manifest(run_id: str, *, program: str) -> dict[str, Any]:
    timestamp = iso_now()
    return {
        "schema_version": OUTERLOOP_SCHEMA_VERSION,
        "run_id": run_id,
        "program": program,
        "status": "created",
        "created_at": timestamp,
        "updated_at": timestamp,
        "active_stage": None,
        "promoted_candidate_id": None,
        "candidates": {},
    }


def _manifest_path(run_id: str) -> Path:
    return run_root(run_id) / "manifest.json"


def _lock_path(run_id: str) -> Path:
    return run_root(run_id) / "manifest.lock"


def _load_manifest_unlocked(run_id: str, *, program: str) -> dict[str, Any]:
    manifest_path = _manifest_path(run_id)
    if not manifest_path.exists():
        return _default_manifest(run_id, program=program)
    raw = manifest_path.read_text(encoding="utf-8").strip()
    if not raw:
        return _default_manifest(run_id, program=program)
    manifest = json.loads(raw)
    manifest.setdefault("schema_version", OUTERLOOP_SCHEMA_VERSION)
    manifest.setdefault("run_id", run_id)
    manifest.setdefault("program", program)
    manifest.setdefault("candidates", {})
    manifest.setdefault("created_at", iso_now())
    manifest.setdefault("updated_at", iso_now())
    manifest.setdefault("active_stage", None)
    manifest.setdefault("promoted_candidate_id", None)
    manifest.setdefault("status", "created")
    return manifest


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def _with_manifest_lock(
    run_id: str,
    *,
    program: str,
    mutator: Callable[[dict[str, Any]], T],
) -> T:
    root = run_root(run_id)
    root.mkdir(parents=True, exist_ok=True)
    lock_file = _lock_path(run_id).open("a+", encoding="utf-8")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    try:
        manifest = _load_manifest_unlocked(run_id, program=program)
        result = mutator(manifest)
        manifest["updated_at"] = iso_now()
        write_json(_manifest_path(run_id), manifest)
        return result
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def ensure_run_manifest(run_id: str, *, program: str) -> Path:
    def _ensure(manifest: dict[str, Any]) -> Path:
        manifest.setdefault("schema_version", OUTERLOOP_SCHEMA_VERSION)
        manifest.setdefault("run_id", run_id)
        manifest.setdefault("program", program)
        manifest.setdefault("status", "created")
        manifest.setdefault("created_at", iso_now())
        manifest.setdefault("updated_at", iso_now())
        manifest.setdefault("active_stage", None)
        manifest.setdefault("promoted_candidate_id", None)
        manifest.setdefault("candidates", {})
        return _manifest_path(run_id)

    return _with_manifest_lock(run_id, program=program, mutator=_ensure)


def candidate_dir(run_id: str, candidate_id: str) -> Path:
    path = run_root(run_id) / "candidates" / candidate_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def register_candidate(
    run_id: str,
    candidate_id: str,
    *,
    genome_hash: str,
    behavior_hash: str | None = None,
    artifact_hash: str | None = None,
    kind: str,
) -> Path:
    candidate_path = candidate_dir(run_id, candidate_id)

    def _register(manifest: dict[str, Any]) -> Path:
        manifest.setdefault("candidates", {})[candidate_id] = {
            "candidate_id": candidate_id,
            "candidate_dir": str(candidate_path),
            "genome_hash": genome_hash,
            "behavior_hash": behavior_hash,
            "artifact_hash": artifact_hash,
            "kind": kind,
            "status": "created",
            "stages": {},
            "updated_at": iso_now(),
        }
        return candidate_path

    return _with_manifest_lock(run_id, program=PROGRAM_PATH, mutator=_register)


def write_stage_result(
    run_id: str,
    candidate_id: str,
    stage: str,
    payload: dict[str, Any],
) -> Path:
    candidate_path = candidate_dir(run_id, candidate_id)
    result_path = candidate_path / f"{stage}.json"
    write_json(result_path, payload)

    def _apply_stage(manifest: dict[str, Any]) -> Path:
        entry = manifest.setdefault("candidates", {}).setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "candidate_dir": str(candidate_path),
                "genome_hash": payload.get("genome_hash"),
                "behavior_hash": payload.get("behavior_hash"),
                "artifact_hash": payload.get("artifact_hash"),
                "kind": payload.get("kind"),
                "status": "created",
                "stages": {},
                "updated_at": iso_now(),
            },
        )
        entry["status"] = payload.get("status", "unknown")
        entry["updated_at"] = iso_now()
        entry.setdefault("stages", {})[stage] = str(result_path)
        entry["behavior_hash"] = payload.get("behavior_hash", entry.get("behavior_hash"))
        entry["artifact_hash"] = payload.get("artifact_hash", entry.get("artifact_hash"))
        manifest["active_stage"] = stage
        manifest["status"] = payload.get("status", manifest.get("status", "running"))
        return result_path

    return _with_manifest_lock(run_id, program=PROGRAM_PATH, mutator=_apply_stage)


def mark_promoted(run_id: str, candidate_id: str) -> None:
    def _promote(manifest: dict[str, Any]) -> None:
        manifest["status"] = "promoted"
        manifest["promoted_candidate_id"] = candidate_id
        if candidate_id in manifest.get("candidates", {}):
            manifest["candidates"][candidate_id]["status"] = "promoted"

    _with_manifest_lock(run_id, program=PROGRAM_PATH, mutator=_promote)
