from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def stable_hash_bytes(data: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in data:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def artifact_hash(path: Path) -> str:
    return stable_hash_bytes(path.read_bytes())


def behavior_hash(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = {
        "eval": payload["eval"],
        "search": payload["search"],
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return stable_hash_bytes(raw)


def build_release_bot(config_path: Path) -> tuple[Path, dict]:
    config_path = config_path.resolve()
    env = os.environ.copy()
    env["SNAKEBOT_CONFIG_PATH"] = str(config_path)
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-q",
            "-p",
            "snakebot-bot",
            "--bin",
            "snakebot-bot",
            "--bin",
            "show_embedded_config",
        ],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )
    info = json.loads(
        subprocess.check_output(
            [str(REPO_ROOT / "target/release/show_embedded_config")],
            cwd=REPO_ROOT,
            text=True,
        )
    )
    expected_artifact_hash = artifact_hash(config_path)
    expected_behavior_hash = behavior_hash(config_path)
    if info["artifact_hash"] != expected_artifact_hash:
        raise RuntimeError(
            "embedded config artifact hash mismatch: "
            f"expected {expected_artifact_hash}, got {info['artifact_hash']}"
        )
    if info["behavior_hash"] != expected_behavior_hash:
        raise RuntimeError(
            "embedded config behavior hash mismatch: "
            f"expected {expected_behavior_hash}, got {info['behavior_hash']}"
        )
    return REPO_ROOT / "target/release/snakebot-bot", info


def load_seeds(path: Path, count: int) -> list[int]:
    seeds: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        seeds.append(int(line))
        if len(seeds) >= count:
            break
    return seeds


def run_java_smoke(
    *,
    league: int,
    seed_file: Path,
    boss_count: int,
    mirror_count: int,
    candidate_config: Path = REPO_ROOT / "rust/bot/configs/submission_current.json",
) -> dict:
    classpath = ensure_java_oracle()
    bot_path, embedded = build_release_bot(candidate_config)
    strict_bot = f"env SNAKEBOT_STRICT_RECONCILE=1 {bot_path}"
    boss = "python3 config/Boss.py"
    seeds = load_seeds(seed_file, boss_count + mirror_count)
    if len(seeds) < boss_count + mirror_count:
        raise RuntimeError(f"not enough smoke seeds in {seed_file}")

    matches: list[dict] = []
    failed: list[dict] = []

    for seed in seeds[:boss_count]:
        payload = run_single_match(classpath, league, seed, strict_bot, boss)
        matches.append(payload)
        if not payload["passed"]:
            failed.append(payload)

    for seed in seeds[boss_count : boss_count + mirror_count]:
        payload = run_single_match(classpath, league, seed, strict_bot, strict_bot)
        matches.append(payload)
        if not payload["passed"]:
            failed.append(payload)

    return {
        "league": league,
        "candidate_config": str(candidate_config),
        "candidate_config_artifact_hash": artifact_hash(candidate_config),
        "candidate_config_behavior_hash": behavior_hash(candidate_config),
        "embedded_config_artifact_hash": embedded["artifact_hash"],
        "embedded_config_behavior_hash": embedded["behavior_hash"],
        "embedded_config_name": embedded["name"],
        "matches": len(matches),
        "passed": not failed,
        "failures": failed,
        "results": matches,
    }


def run_single_match(classpath: Path, league: int, seed: int, agent_a: str, agent_b: str) -> dict:
    output = subprocess.check_output(
        [
            "java",
            "-cp",
            str(classpath),
            "com.codingame.game.RunnerCli",
            "--league",
            str(league),
            "--seed",
            str(seed),
            "--agent-a",
            agent_a,
            "--agent-b",
            agent_b,
        ],
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return json.loads(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=int, default=4)
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=REPO_ROOT / "config/arena/smoke_v1.txt",
    )
    parser.add_argument("--boss-count", type=int, default=4)
    parser.add_argument("--mirror-count", type=int, default=4)
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=REPO_ROOT / "rust/bot/configs/submission_current.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_java_smoke(
        league=args.league,
        seed_file=args.seed_file,
        boss_count=args.boss_count,
        mirror_count=args.mirror_count,
        candidate_config=args.candidate_config,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
