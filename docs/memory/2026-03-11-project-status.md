# WinterChallenge 2026 Project Status

Date: 2026-03-11

## Summary

The repository now has:

- a Java-parity Rust simulator
- a stronger Rust search bot with corrected depth-2 refinement and embedded submission config
- a release-mode Rust arena harness with frozen seed suites
- a real Java-referee smoke canary for live I/O/reconciliation checks on the exact built candidate artifact
- a deterministic self-play/export path with Rust-generated seeds by default and a grouped Python value-training pipeline
- semantic behavior hashes alongside raw artifact hashes for config identity and promotion logic
- split opening/later-turn arena timing so only later turns are hard-gated

The newest local commit after the original simulator/bot work is:

- `43d7d1c` `Add arena evaluation and deterministic search upgrades`
- plus the current uncommitted follow-up for search-fix/config-discipline cleanup
- plus the current uncommitted follow-up for behavior-hash, timing-gate, and ledger cleanup

The most important current checks that passed are:

```bash
cargo run -q -p snakebot-engine --bin java_diff -- --init-source rust --seeds 50 --turns 20
```

```bash
python3 -m python.train.java_smoke --boss-count 1 --mirror-count 1
```

```bash
python3 -m python.train.run_arena --heldout-suite /tmp/arena_smoke_2.txt --shadow-suite /tmp/arena_smoke_2.txt --jobs 2 --name arena_smoke
```

## Read This Folder

Use [README.md](./README.md) as the index.

Detailed follow-up docs:

- [Engine, Search, and Evaluation Harness](./2026-03-11-engine-search-and-eval.md)
- [Training and Data Pipeline](./2026-03-11-training-and-data-pipeline.md)

## Local Commits Added

- `f00bb51` `Add Rust simulator, bot, and oracle tooling`
- `d0a63f8` `Fix Java parity in simulator state updates`
- `8b811b1` `Add parallel self-play data pipeline`
- `4f6d769` `Add project status memory note`
- `43d7d1c` `Add arena evaluation and deterministic search upgrades`

## Current Repo Shape

- Rust engine in `rust/engine`
- Rust bot/search/arena/export in `rust/bot`
- Python training/eval wrappers in `python/train`
- Java oracle and runner helpers in `src/test/java/com/codingame/game`

## What Changed Most Recently

- Engine semantics are split cleanly: natural Java-parity `is_game_over()` stays separate from contest `is_terminal(200)` and `final_result(200)`.
- Live search now uses an embedded submission config, while arena/self-play/training operate on explicit candidate/incumbent/anchor JSONs.
- Depth-2 refinement now recomputes refined root worst/mean scores and exported root values instead of only downgrading branches.
- Java smoke now rebuilds the candidate artifact, verifies the embedded artifact and behavior hashes, and then runs the referee canary on that exact build.
- Arena now distinguishes raw artifact identity from semantic behavior identity, short-circuits behavior self-matches, and records opening/later timing buckets separately.
- Self-play now defaults to Rust-generated seeds, records explicit budget type/value plus both config hashes, and can train directly from shard directories without mandatory merging.

## Recommended Next Read

- Read [2026-03-11 Engine, Search, and Evaluation Harness](./2026-03-11-engine-search-and-eval.md) if you need to work on live strength or referee parity.
- Read [2026-03-11 Training and Data Pipeline](./2026-03-11-training-and-data-pipeline.md) if you need to work on export, datasets, or model training.

## Notes

- `.serena/` exists locally and is untracked. It was intentionally not committed.
- `python/train/artifacts/` and `python/train/data/` are being used for local generated assets and are ignored.
- This file is now the short status entry point, not the full detail dump.
