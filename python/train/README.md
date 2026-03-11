# Local Training Scaffold

This directory is the local, W&B-free training side of the Rust search/value pipeline.

## Expected dataset format

`SelfPlayDataset` reads JSONL where each line has:

```json
{
  "schema_version": 3,
  "config_artifact_hash": "....",
  "config_behavior_hash": "....",
  "seed": 1,
  "game_id": "seed-1-league-4",
  "turn": 0,
  "owner": 0,
  "raw_state_hash": "....",
  "encoded_view_hash": "....",
  "grid": [[[0.0, 1.0], [1.0, 0.0]]],
  "scalars": [0.0, 1.0, 0.0, 1.0, 0.1, 0.9],
  "value": 0.25,
  "weight": 1.0,
  "chosen_action_id": 3,
  "joint_action_count": 27,
  "root_values": [0.4, 0.1, -0.2],
  "budget_type": "extra_nodes_after_root",
  "budget_value": 64,
  "search_stats": {"elapsed_ms": 12, "root_pairs": 729, "extra_nodes": 64, "cutoffs": 8}
}
```

- `grid` is `[channels][height][width]`
- `scalars` is an auxiliary feature vector
- `value` is the normalized final score target in `[-1, 1]`
- `encoded_view_hash` is the dedup key used before grouped train/validation splitting
- `config_artifact_hash` tracks the exact file bytes used to build/export a run
- `config_behavior_hash` tracks only `eval + search` semantics and is the strategy identity key
- `chosen_action_id`, `joint_action_count`, and `root_values` are the compact search targets for later policy distillation

## Main entry points

- `python -m python.train.train_value`
- `python -m python.train.run_local_experiment`
- `python -m python.train.run_arena`
- `python -m python.train.java_smoke`

## Optional Java oracle map dumps

Generate exact initial states from the Java referee:

```bash
mvn -q -DskipTests test-compile dependency:build-classpath -Dmdep.outputFile=cp.txt
java -cp "target/classes:target/test-classes:$(cat cp.txt)" \
  com.codingame.game.MapDumpCli 1 1000 4 > python/train/artifacts/maps_l4.jsonl
```

Each line is:

```json
{
  "seed": 1,
  "leagueLevel": 4,
  "state": { "... exact initial referee state ..." }
}
```

Rust can load these dump records through `snakebot_engine::load_dump_records`.

## Rust self-play export

Export training samples directly from Rust-generated seeds:

```bash
cargo run --release -q -p snakebot-bot --bin selfplay_export -- \
  --seed-start 1 \
  --seed-count 200 \
  --league 4 \
  --out python/train/data/selfplay.jsonl \
  --limit 200 \
  --max-turns 120 \
  --extra-nodes-after-root 5000 \
  --config rust/bot/configs/submission_current.json \
  --git-sha "$(git rev-parse HEAD)"
```

Notes:

- Offline export is deterministic by default: it always finishes the root pass, then spends the requested extra node budget on deepening.
- `--search-ms` is still available as an explicit override for debugging, but it is no longer the default.
- `--maps` remains available as an explicit oracle/debug input mode, but it is no longer the default hot path.
- The exporter writes owner-relative samples for both players on each turn.
- The current tensor shape is `8 x 23 x 42` with `6` scalar features.

## Parallel local pipeline

Use the local M4 Max for CPU-parallel self-play generation, then train on `mps`:

```bash
python -m python.train.parallel_selfplay \
  --seed-count 512 \
  --workers 8 \
  --games 512 \
  --max-turns 120 \
  --extra-nodes-after-root 5000 \
  --train
```

Notes:

- The script builds release binaries and shards Rust-seed self-play across `--workers`.
- Java build/dump work only happens if you explicitly pass `--maps-path`.
- Shards are written under `python/train/data/<dataset>_shards/`. Merging into a single file is optional via `--merge-output`.
- Training stays mostly single-run on `mps`; parallelism is intended for map dumping and self-play export.

## Arena and live-path smoke

Run heldout Rust-vs-Rust arena matches:

```bash
python -m python.train.run_arena \
  --candidate-config rust/bot/configs/submission_current.json \
  --incumbent-config rust/bot/configs/incumbent_current.json \
  --anchor-config rust/bot/configs/anchor_root_only.json
```

Run the Java referee smoke canary directly:

```bash
python -m python.train.java_smoke \
  --boss-count 4 \
  --mirror-count 4 \
  --candidate-config rust/bot/configs/submission_current.json
```

Notes:

- Arena is the strength authority. Java smoke is the real I/O canary.
- Java smoke now verifies that the built bot artifact embeds the same candidate artifact and behavior hashes that arena evaluated.
- `run_arena` treats candidate/incumbent behavior self-matches as informational no-ops instead of pretending they are meaningful arena results.
- Both scripts use release-mode Rust binaries.

## Notes

- PyTorch will prefer `mps` on Apple Silicon when available.
- `results.sqlite` is the local experiment ledger.
- Training-only runs are stored as `informational`. Arena plus Java smoke is the acceptance gate.
