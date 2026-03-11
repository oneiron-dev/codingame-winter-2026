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

## Recent runtime checks

Short checks that passed after the behavior-hash/timing split landed:

- tiny 2-seed release-mode arena run against the root-only anchor: later-turn `p99` around `31 ms`
- Java smoke on boss + mirror matches: passed with zero reported runner error counts

## Still intentionally basic

- No TT yet
- No policy prior or network-guided search yet
- Current arena acceptance logic is still simple body-diff/WDL based, not a full Elo framework
- `incumbent_current.json` is still intentionally behavior-identical to `submission_current.json` until the first real sweep winner is promoted
