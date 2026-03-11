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
- a staged release-mode search sweep helper with smoke filtering and heldout/shadow finalist evaluation
- stage-1 sweep screening that no longer pollutes authoritative acceptance results
- a generated single-file submission artifact at `submission/flattened_main.rs`
- a first true promoted winner: `submission_current.json` is now the breadth-heavier `6/8/3/3` finalist, while `incumbent_current.json` preserves the prior `6/6/4/4` baseline

The newest local commits after the original simulator/bot work are:

- `43d7d1c` `Add arena evaluation and deterministic search upgrades`
- `0eec4d7` `Tighten evaluation hashes and arena timing gates`
- `e93927b` `Add staged search sweep and regression fixtures`
- `be7cb69` `Fix sweep screening status and arena build reuse`
- `c6aa257` `Add flattened submission generator and artifact`

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

```bash
python3 -m python.train.sweep_search --smoke-suite /tmp/arena_smoke_2.txt --heldout-suite /tmp/arena_smoke_2.txt --shadow-suite /tmp/arena_smoke_2.txt --top-my-values 4,6 --top-opp-values 6 --child-my-values 3 --child-opp-values 4 --later-turn-values 38,40 --smoke-top-k 2
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
- `0eec4d7` `Tighten evaluation hashes and arena timing gates`

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
- Arena results now carry heldout/shadow tiebreak win rates, and the search regression is frozen on a dedicated test config instead of the live embedded submission config.
- The repo now includes a staged release-mode sweep runner plus a shared Rust/Python hash fixture test so config hashing and sweep evaluation can evolve without drifting silently.
- Stage-1 sweep runs are now explicitly `screening`, not `accepted`/`rejected`, and the sweep reuses one prebuilt release `arena` binary instead of rebuilding it per candidate.
- Self-play now defaults to Rust-generated seeds, records explicit budget type/value plus both config hashes, and can train directly from shard directories without mandatory merging.
- The repo now also includes `tools/generate_flattened_submission.py`, which emits the pasteable `submission/flattened_main.rs` artifact and should be rerun after live bot/config changes.
- A clean rerun of the first apparently winning `tmy4/topp4/cmy5/copp5` finalist family did not hold up on authoritative confirmation. The `lat38`, `lat40`, and `lat42` variants all passed Java smoke and easily beat the weak anchor, but all three still lost to the current incumbent on `heldout_v1`, so no promotion happened.
- The next distinct `tmy6/topp8` finalist family did produce a real winner. `sweep_tmy6_topp8_cmy3_copp3_lat40` was accepted on authoritative rerun and promoted into `submission_current.json`.

## Latest Evaluation Read

- Current live CodinGame result reported from the submitted bot: global rank `168 / 1108`, Bronze rank `168 / 1108`
- Clean authoritative confirmations run after the interrupted sweep:
  - `sweep_tmy4_topp4_cmy5_copp5_lat40_lat38`: rejected, heldout body diff `-0.830078125`
  - `sweep_tmy4_topp4_cmy5_copp5_lat40`: rejected, heldout body diff `-0.716796875`
  - `sweep_tmy4_topp4_cmy5_copp5_lat40_lat42`: rejected, heldout body diff `-0.642578125`
- Clean authoritative confirmations run for the next `tmy6/topp8` finalists:
  - `sweep_tmy6_topp8_cmy3_copp3_lat40`: accepted, heldout body diff `+0.08984375`, heldout win margin `0.0`, shadow body diff `+9.33203125`, later-turn `p99` `41 ms`
  - `sweep_tmy6_topp8_cmy3_copp4_lat40`: rejected, heldout body diff `-0.435546875`, later-turn `p99` `62 ms`
  - `sweep_tmy6_topp8_cmy5_copp5_lat40`: rejected, heldout body diff `+0.7109375`, later-turn `p99` `62 ms`
- Conclusion: the first true promotion has now happened. `submission_current.json` is the promoted `6/8/3/3` breadth-heavy winner, and `incumbent_current.json` remains the old `6/6/4/4` baseline for future comparisons.

## Recommended Next Read

- Read [2026-03-11 Engine, Search, and Evaluation Harness](./2026-03-11-engine-search-and-eval.md) if you need to work on live strength or referee parity.
- Read [2026-03-11 Training and Data Pipeline](./2026-03-11-training-and-data-pipeline.md) if you need to work on export, datasets, or model training.
- The immediate next operational step is to submit the promoted config, monitor live rank movement, and only then decide whether to run a narrower follow-up sweep or reopen self-play quality work.

## Notes

- `.serena/` exists locally and is untracked. It was intentionally not committed.
- `python/train/artifacts/` and `python/train/data/` are being used for local generated assets and are ignored.
- This file is now the short status entry point, not the full detail dump.
