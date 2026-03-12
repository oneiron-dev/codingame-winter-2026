# WinterChallenge 2026 Project Status

Date: 2026-03-11

## Summary

The repository now has:

- a Java-parity Rust simulator
- a stronger Rust search bot with corrected depth-2 refinement and embedded submission config
- a release-mode Rust arena harness with frozen seed suites
- a real Java-referee smoke canary for live I/O/reconciliation checks on the exact built candidate artifact
- a deterministic self-play/export path with Rust-generated seeds by default and a grouped Python value-training pipeline
- a Prose-first outer-loop scaffold with candidate manifests, git worktrees, staged evaluation, and promotion helpers
- a tiny hybrid policy+value branch with identity-preserving features, local/Modal training hooks, Rust-side prior/leaf integration, and weight export
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

The current important limitation is:

- the generated single-file submission artifact is still search-only
- hybrid-enabled configs can now be trained and evaluated locally, but `tools/generate_flattened_submission.py` will fail explicitly until one-file hybrid weight embedding exists

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
- The repo now also includes a Prose-controlled outer-loop lab under `automation/outerloop` and `python/train/outerloop`, with run manifests under `artifacts/outerloop/runs/<run_id>/manifest.json` and candidate stage artifacts under `artifacts/outerloop/runs/<run_id>/candidates/<candidate_id>/`.
- The hybrid branch is now present end to end: self-play exports `hybrid_grid` and `policy_targets`, Python can train/export a tiny hybrid model, and Rust can blend that model into action ordering and leaf evaluation.
- One local screening smoke run for the outer loop succeeded end to end and produced a real candidate artifact tree. The hybrid candidate itself was extremely slow and weak, which is expected at this stage and should be treated as plumbing validation, not as a viable submission result.
- A real native Prose VM smoke run also succeeded under `.prose/runs/20260312-144527-prose01/`: planner and worker/evaluator sessions were executed from `outerloop.prose`, a search finalist cleared a tiny stage-2 rerun with Java smoke, and the final verdict correctly refused promotion because the run used temporary smoke suites rather than the real frozen heldout/shadow sets.
- A second native Prose VM run then targeted real optimization around the promoted `6/8/3/3` submission under `.prose/runs/20260312-164032-a2d348/`. That run tested `6/9/3/3`, `7/8/3/3`, and `7/9/3/3 @ 42 ms` on the real smoke suite. All three were catastrophically over the later-turn budget (`164-173 ms` p99) and none showed a positive heldout signal, so the evaluator advanced no finalists and the run terminated without stage 2 or promotion.
- Modal execution is now real in the outer loop, not just planned. The current validated split is:
  - Modal CPU for self-play shard generation
  - Modal GPU for tiny hybrid training
  - Local CPU for authoritative heldout/shadow arena plus Java smoke
- The candidate runner now also supports a Modal CPU stage-1 screening path via `modal-arena-screen`. A smoke run under `artifacts/outerloop/runs/modal-stage1-smoke-2/` screened `search-a3734068c5ae` on a 4-seed temporary suite and wrote normal candidate artifacts with `executor = modal-arena-screen`, heldout body diff `-1.25`, shadow body diff `+2.625`, and later-turn `p99 = 42 ms`. The important part is that stage 1 can now leave the laptop while stage 2 remains authoritative and local.
- The hybrid branch also now has its first sane result shape. A root-prior-only 12-channel candidate (`hybrid-e86e007498ec`) screened positively on a tiny suite (`heldout_body_diff = +1.125`, `later_turn_p99 = 41 ms`) but was rejected on a full authoritative rerun after the runtime inference bug was fixed. Final authoritative result: heldout body diff `-1.337890625`, shadow body diff `+7.603515625`, later-turn `p99 = 41 ms`, Java smoke passed. This matters because the remaining blocker is now strength, not plumbing or timing explosions.
- A clean rerun of the first apparently winning `tmy4/topp4/cmy5/copp5` finalist family did not hold up on authoritative confirmation. The `lat38`, `lat40`, and `lat42` variants all passed Java smoke and easily beat the weak anchor, but all three still lost to the current incumbent on `heldout_v1`, so no promotion happened.
- The next distinct `tmy6/topp8` finalist family did produce a real winner. `sweep_tmy6_topp8_cmy3_copp3_lat40` was accepted on authoritative rerun and promoted into `submission_current.json`.

## Latest Evaluation Read

- Pre-promotion live CodinGame result from the prior submission: global rank `168 / 1108`, Bronze rank `168 / 1108`
- Post-promotion live CodinGame result for the promoted `6/8/3/3` submission: global rank `147 / 1108`, Bronze rank `147 / 1108`, score `29.57`
- Net live change after the first real promotion: `+21` leaderboard places and `29.57` as the recorded comparison score for this submission
- Clean authoritative confirmations run after the interrupted sweep:
  - `sweep_tmy4_topp4_cmy5_copp5_lat40_lat38`: rejected, heldout body diff `-0.830078125`
  - `sweep_tmy4_topp4_cmy5_copp5_lat40`: rejected, heldout body diff `-0.716796875`
  - `sweep_tmy4_topp4_cmy5_copp5_lat40_lat42`: rejected, heldout body diff `-0.642578125`
- Clean authoritative confirmations run for the next `tmy6/topp8` finalists:
  - `sweep_tmy6_topp8_cmy3_copp3_lat40`: accepted, heldout body diff `+0.08984375`, heldout win margin `0.0`, shadow body diff `+9.33203125`, later-turn `p99` `41 ms`
  - `sweep_tmy6_topp8_cmy3_copp4_lat40`: rejected, heldout body diff `-0.435546875`, later-turn `p99` `62 ms`
  - `sweep_tmy6_topp8_cmy5_copp5_lat40`: rejected, heldout body diff `+0.7109375`, later-turn `p99` `62 ms`
- Conclusion: the first true promotion has now happened. `submission_current.json` is the promoted `6/8/3/3` breadth-heavy winner, and `incumbent_current.json` remains the old `6/6/4/4` baseline for future comparisons.
- New optimization read after the second native Prose VM run: simply pushing root breadth beyond `6/8/3/3` is not the next gain. The next promising search direction is to reduce root cost or make breadth more selective, because naive breadth increases are currently blowing the later-turn gate by roughly `4x`.
- New hybrid read: root-prior-only hybrids are now timing-safe enough to evaluate seriously, but the best current candidate still loses to the incumbent on full heldout despite passing Java smoke. The correct next hybrid move is more candidate volume and better self-play/training diversity, not more infrastructure work.

## Recommended Next Read

- Read [2026-03-11 Engine, Search, and Evaluation Harness](./2026-03-11-engine-search-and-eval.md) if you need to work on live strength or referee parity.
- Read [2026-03-11 Training and Data Pipeline](./2026-03-11-training-and-data-pipeline.md) if you need to work on export, datasets, or model training.
- Read [2026-03-12 Prose Outer Loop and Hybrid Lab](./2026-03-12-prose-outerloop-and-hybrid-lab.md) if you need to work on Prose orchestration, candidate artifacts, the hybrid branch, or Helios-facing contracts.
- The immediate next operational step is no longer infrastructure. It is to use the new Modal-backed outer loop to generate more hybrid candidates cheaply at stage 1, keep authoritative reruns local, and see whether a root-prior-only hybrid can produce the first real heldout-positive promotion.

## Notes

- `.serena/` exists locally and is untracked. It was intentionally not committed.
- `python/train/artifacts/` and `python/train/data/` are being used for local generated assets and are ignored.
- This file is now the short status entry point, not the full detail dump.
