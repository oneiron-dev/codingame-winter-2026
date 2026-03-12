from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import modal.exception

from python.train.outerloop import modal_job


def _retry_remote(fn, *args, max_retries: int = 3, base_delay: float = 5.0):
    """Wrap a Modal ``.remote()`` call with exponential backoff + jitter."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn.remote(*args)
        except (ConnectionError, OSError, TimeoutError, modal.exception.Error) as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
            print(f"[launch_modal] .remote() attempt {attempt + 1} failed ({exc!r}), retrying in {delay:.1f}s …")
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]  # unreachable but keeps mypy happy


def launch_modal(task: str, spec: dict, *, preserve_selfplay_blob: bool = False) -> dict:
    with modal_job.app.run():
        spec_json = json.dumps(spec)
        if task == "train":
            gpu_name = str(spec.get("gpu", "L40S"))
            return _retry_remote(modal_job._train_function_for_gpu(gpu_name), spec_json)
        if task == "selfplay":
            payload = _retry_remote(modal_job.run_selfplay, spec_json)
            return modal_job._decode_dataset_payload(payload, preserve_blob=preserve_selfplay_blob)
        if task == "arena-screen":
            return _retry_remote(modal_job.run_arena_screen, spec_json)
        if task == "train-teacher":
            return _retry_remote(modal_job.train_teacher_l40s, spec_json)
        if task == "generate-soft-targets":
            return _retry_remote(modal_job.generate_soft_targets_l40s, spec_json)
    raise ValueError(f"unsupported modal task: {task}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("train", "selfplay", "arena-screen", "train-teacher", "generate-soft-targets"), required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.spec.read_text(encoding="utf-8"))
    result = launch_modal(args.task, payload)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
