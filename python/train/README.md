# Local Training Scaffold

This directory is the local, W&B-free training side of the Rust search/value pipeline.

## Expected dataset format

`SelfPlayDataset` reads JSONL where each line has:

```json
{
  "grid": [[[0.0, 1.0], [1.0, 0.0]]],
  "scalars": [0.0, 1.0, 0.0, 1.0, 0.1, 0.9],
  "value": 0.25,
  "weight": 1.0
}
```

- `grid` is `[channels][height][width]`
- `scalars` is an auxiliary feature vector
- `value` is the normalized final score target in `[-1, 1]`
- `weight` is optional

## Main entry points

- `python -m python.train.train_value`
- `python -m python.train.run_local_experiment`

## Java oracle map dumps

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

Export training samples from Java-generated initial states:

```bash
cargo run -q -p snakebot-bot --bin selfplay_export -- \
  --maps python/train/artifacts/maps_l4.jsonl \
  --out python/train/data/selfplay.jsonl \
  --limit 200 \
  --max-turns 120 \
  --search-ms 2
```

Notes:

- `--search-ms 0` means unlimited root search and is much slower.
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
  --search-ms 2 \
  --train
```

Notes:

- The script builds the Java oracle once, builds the Rust exporter once, then shards self-play across `--workers`.
- Shards are written under `python/train/data/<dataset>_shards/` and merged into the final dataset path.
- Training stays mostly single-run on `mps`; parallelism is intended for map dumping and self-play export.

## Notes

- PyTorch will prefer `mps` on Apple Silicon when available.
- `results.sqlite` is the local experiment ledger.
- The intent is that Rust self-play/export will later produce the JSONL samples consumed here.
