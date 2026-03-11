# 2026-03-11 Engine, Search, and Evaluation Harness

## Engine semantics

The Rust engine now keeps Java-parity natural game end separate from contest terminal logic.

- `GameState::is_game_over()` remains the Java-parity natural end check.
- `GameState::is_terminal(max_turns)` adds the contest turn cap.
- `GameState::final_result(max_turns)` exposes explicit body scores, final scores, losses, body diff, loss diff, winner, and terminal reason.

This avoids muddying the parity tools while still giving search, arena, and self-play the right contest semantics.

## Parity status

The important parity fixes already in place are:

- Java RNG bounded-range parity in `java_random.rs`
- per-turn bird direction reset parity in `state.rs`

The strongest recent parity check that passed was:

```bash
cargo run -q -p snakebot-engine --bin java_diff -- --init-source rust --seeds 50 --turns 20
```

## Search behavior

The Rust search path now has:

- live budgets of `850 ms` on turn `0` and `40 ms` afterwards
- deterministic offline budget mode: exhaustive root first, then `extra_nodes_after_root`
- heuristic ordering for friendly actions and opponent replies
- incumbent-based early cutoff inside the opponent loop
- depth-2 top-k follow-up search after the root pass
- corrected deepening backup: second ply now uses `max_my min_opp`
- refined root reranking: deepened reply values recompute each root action's worst score, mean score, and exported root value before the final root selection

The live bot no longer relies on an in-code default config for submission behavior. It now embeds a build-time config artifact, with `submission_current.json` as the default source and an override path for local candidate promotion runs.

Config identity is now split in two:

- artifact hash: raw file bytes, used for build provenance and Java smoke verification
- behavior hash: canonicalized `eval + search`, used for arena equality checks and promotion logic

This prevents renamed-but-identical configs from quietly masquerading as distinct strategies.

The fixed deepening regression is no longer tied to the live submission config. It now uses the dedicated frozen config:

- `rust/bot/configs/test_search_regression_v1.json`

The current implementation intentionally does **not** include a transposition table yet. The current order of work is:

1. ordering
2. cutoff
3. depth-2 top-k
4. TT only if profiling shows it is worth the extra complexity

## Arena harness

There is now a release-mode Rust arena binary:

- `rust/bot/src/bin/arena.rs`

It:

- runs two configs against each other in-process
- swaps seats automatically
- parallelizes across seeds
- reports average body diff, W/D/L, tiebreak win rate, average turns, node counts, opening-move timing, later-turn timing, and both artifact/behavior hashes

Frozen committed suites:

- `config/arena/smoke_v1.txt`
- `config/arena/heldout_v1.txt`
- `config/arena/shadow_v1.txt`

## Live-path canary

There is also now a small Java-referee smoke canary:

- Java CLI: `src/test/java/com/codingame/game/RunnerCli.java`
- Python wrapper: `python/train/java_smoke.py`

The canary runs the real compiled Rust bot through the actual referee runner and can force hard failure on reconciliation fallback via:

```bash
SNAKEBOT_STRICT_RECONCILE=1
```

It now also verifies that the built bot artifact embeds the same candidate artifact and behavior hashes that the arena evaluation used.

## Flattened submission artifact

There is now a mechanical single-file submission generator:

- `tools/generate_flattened_submission.py`

It emits:

- `submission/flattened_main.rs`

This artifact is intended for CodinGame paste submissions. It is compile-checked locally with plain `rustc` and should be regenerated whenever either of these change:

- `rust/bot/configs/submission_current.json`
- any live bot or engine module used by the contest binary

The compiled check binary at `submission/flattened_main` is local-only and ignored.

## Recent runtime checks

Short checks that passed after the behavior-hash/timing split landed:

- tiny 2-seed release-mode arena run against the root-only anchor: later-turn `p99` around `31 ms`
- Java smoke on boss + mirror matches: passed with zero reported runner error counts
- small staged sweep smoke run: stage-1 wrote `screening` results only, stage-2 ran authoritative heldout/shadow evaluation, and the sweep reused one prebuilt release `arena` binary for the full run

## Latest confirmation results

After the interrupted sweep produced an apparently accepted `tmy4/topp4/cmy5/copp5` family winner, that family was rerun cleanly on full `heldout_v1 + shadow_v1 + Java smoke`:

- `sweep_tmy4_topp4_cmy5_copp5_lat40_lat38`
  - heldout body diff `-0.830078125`
  - shadow body diff `+6.783203125`
  - later-turn `p99` `33 ms`
  - Java smoke passed
  - final result: rejected
- `sweep_tmy4_topp4_cmy5_copp5_lat40`
  - heldout body diff `-0.716796875`
  - shadow body diff `+6.970703125`
  - later-turn `p99` `33 ms`
  - Java smoke passed
  - final result: rejected
- `sweep_tmy4_topp4_cmy5_copp5_lat40_lat42`
  - heldout body diff `-0.642578125`
  - shadow body diff `+6.994140625`
  - later-turn `p99` `37 ms`
  - Java smoke passed
  - final result: rejected

Interpretation:

- this topology family is healthy and clearly stronger than the weak anchor
- timing and live-path correctness are fine
- but none of the three variants beat the current incumbent on heldout
- so no promotion was made from that family

The highest-signal next search action is to confirm other distinct finalists from the interrupted sweep rather than rerunning the same family or promoting on the earlier partial result.

## Live benchmark note

The submitted bot was also reported at:

- global rank `168 / 1108`
- Bronze rank `168 / 1108`

That is a useful sanity check that the incumbent/base is already a real competitive baseline, not just a locally overfit config.

## Still intentionally basic

- No TT yet
- No policy prior or network-guided search yet
- Current arena acceptance logic is still simple body-diff/WDL based, not a full Elo framework
- `submission_current.json` is now the first promoted winner from the sweep process, with a breadth-heavier `6/8/3/3` search shape
- `incumbent_current.json` now preserves the prior `6/6/4/4` baseline for future authoritative comparisons

## First real promotion

The next distinct `tmy6/topp8` finalist confirmations produced the first true promoted winner.

- `sweep_tmy6_topp8_cmy3_copp3_lat40`
  - heldout body diff `+0.08984375`
  - heldout win margin `0.0`
  - shadow body diff `+9.33203125`
  - later-turn `p99` `41 ms`
  - Java smoke passed
  - final result: accepted and promoted
- `sweep_tmy6_topp8_cmy3_copp4_lat40`
  - heldout body diff `-0.435546875`
  - shadow body diff `+7.76953125`
  - later-turn `p99` `62 ms`
  - Java smoke passed
  - final result: rejected on heldout and timing
- `sweep_tmy6_topp8_cmy5_copp5_lat40`
  - heldout body diff `+0.7109375`
  - shadow body diff `+9.130859375`
  - later-turn `p99` `62 ms`
  - Java smoke passed
  - final result: rejected on timing despite positive heldout score

Interpretation:

- the winning move was not “more child depth”; it was broader root/opponent coverage with shallower child follow-ups
- two nearby finalists had stronger raw heldout body diff but failed the later-turn time gate badly, so they were not promotable
- the promoted winner is the first search config that is both stronger than the prior incumbent and still safe under the current acceptance contract
