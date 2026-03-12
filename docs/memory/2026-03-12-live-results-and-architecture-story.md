# 2026-03-12 Live Results and Architecture Storyline

This note is a lightweight narrative memory surface for future writeups.

Use it when we want to answer:

- what the current bot architecture actually is
- which agent/config is currently live
- how the leaderboard moved over time
- what the key turning points were

## Current architecture

The current stack is:

- Java-parity Rust engine for rules, gravity, collisions, mapgen, and exact forward simulation
- Rust contest bot with embedded build-time submission config
- heuristic search bot with exhaustive root evaluation plus selective depth-2 refinement
- release-mode Rust arena harness for heldout/shadow comparisons
- Java smoke canary for exact built-artifact verification against the real referee loop
- Rust-seed self-play/export pipeline and a paused Python value-training path
- generated single-file submission artifact at `submission/flattened_main.rs`

The current search philosophy is still search-first, not ML-first:

- exact simulator
- robust arena evaluation
- search config tuning
- only then consider richer self-play / ML work

## Current agent lineup

### Live submission

File:

- `rust/bot/configs/submission_current.json`

Current promoted search shape:

- `deepen_top_my = 6`
- `deepen_top_opp = 8`
- `deepen_child_my = 3`
- `deepen_child_opp = 3`
- `later_turn_ms = 40`

Interpretation:

- broader root and opponent coverage
- shallower child follow-up search
- timing-safe under the current later-turn gate

### Prior incumbent

File:

- `rust/bot/configs/incumbent_current.json`

Preserved prior baseline:

- `deepen_top_my = 6`
- `deepen_top_opp = 6`
- `deepen_child_my = 4`
- `deepen_child_opp = 4`
- `later_turn_ms = 40`

Interpretation:

- more balanced root/child allocation
- was the first stable baseline that looked genuinely competitive

### Weak anchor

File:

- `rust/bot/configs/anchor_root_only.json`

Shape:

- all deepening disabled
- `extra_nodes_after_root = 0`

Purpose:

- deliberately weaker comparison point
- useful for catching regressions that still beat nothing

## Key offline turning point

The first apparently strong `tmy4/topp4/cmy5/copp5` family did **not** survive authoritative reruns.

The first true promotion came from:

- `sweep_tmy6_topp8_cmy3_copp3_lat40`

Authoritative acceptance evidence:

- heldout body diff `+0.08984375`
- heldout win margin `0.0`
- shadow body diff `+9.33203125`
- later-turn `p99 = 41 ms`
- Java smoke passed

Important nearby failures:

- `tmy6/topp8/cmy3/copp4` lost heldout and failed timing
- `tmy6/topp8/cmy5/copp5` had stronger raw heldout body diff but failed timing badly

The lesson from that promotion:

- broader top-level coverage beat deeper child search
- timing gate mattered more than raw heldout body diff alone

## Live leaderboard history

Known checkpoints:

| Date | Config | Rank | League | Score | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-03-11 | Pre-promotion baseline (`6/6/4/4`) | `168 / 1108` | Bronze | not recorded | First solid live baseline |
| 2026-03-12 | Promoted breadth-heavy winner (`6/8/3/3`) | `147 / 1108` | Bronze | `29.57` | First live post-promotion checkpoint |
| 2026-03-13 | Same (`6/8/3/3`) | `108 / 1388` | Silver | — | League promotion to Silver (108th / 763 Silver) |

Current comparison delta:

- rank improvement from first promotion: `168 → 108` (global), Bronze → Silver
- league promotion achieved with search-only 6/8/3/3 bot

## Blog-post fuel

If we write about this later, the interesting story is probably:

1. build exact local truth first
2. stop trusting pretty sweep winners until heldout + timing + smoke all agree
3. discover that the first flashy winner was fake
4. get the first true promotion from a modest but robust breadth-heavy config
5. watch the live leaderboard move in the same direction as the local evidence

Useful headline themes:

- “Why the first winning sweep result was wrong”
- “Breadth beat depth in our first real Winter Challenge promotion”
- “From exact simulator parity to the first live rank bump”

## Architecture evolution (2026-03-13)

Key changes committed in the distillation pipeline batch:

1. **Rust inference 2-3x speedup**: Flattened `Vec<Vec<Vec<f32>>>` to flat `Vec<f32>` with stride indexing in `hybrid.rs`. Eliminates triple-pointer indirection. 8ch inference estimated ~0.9ms instead of ~2.7ms.
2. **3rd conv layer support**: Optional `conv3` in `TinyHybridWeights` (backward compatible). Student receptive field 5x5 → 7x7, covering ~30% of the 23-tile short dimension. Only +0.3ms overhead.
3. **1x1 kernel support**: Kernel offset now uses `kernel_size/2` instead of hardcoded `1`. Enables bottleneck architectures later.
4. **Schema v2**: Rust accepts both v1 (2-layer) and v2 (3-layer) weight files.
5. **Modal Volume migration**: Datasets transferred via `modal.Volume("snakebot-datasets")` instead of gzip+base64 JSON blobs. Fixes aiohttp/SSL/broken pipe errors on 40MB-2.3GB transfers.
6. **Retry logic**: All `.remote()` calls wrapped with exponential backoff + jitter (3 retries, catching ConnectionError/OSError/TimeoutError).
7. **Shared dataset generation**: `build_shared_dataset()` generates one big self-play dataset (500+ seeds) reused by all parallel candidates. Eliminates 8x redundant self-play from same config.
8. **Teacher-student distillation pipeline**: TeacherHybridNet (128ch, 8 SE-res blocks, ~2-3M params) → soft target generation → KL+MSE distillation training for student. Loss: `T²·KL(student/T, teacher/T) + 1.5·MSE(value) + α·CE(hard_targets)`.

## What to append next

Add a new row whenever any of these happen:

- a new CodinGame live score/rank checkpoint
- a new promoted submission config
- a materially different architecture phase, like opponent-pool self-play or policy/value integration

Keep this file narrative-friendly and compact. The deeper implementation details should stay in the other memory docs.
