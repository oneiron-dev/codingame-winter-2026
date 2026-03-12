# 2026-03-11 Training and Data Pipeline

## Current export path

The self-play export path is now release-mode and deterministic by default.

Main pieces:

- `rust/bot/src/bin/selfplay_export.rs`
- `rust/bot/src/selfplay.rs`
- `python/train/parallel_selfplay.py`

Key behavior:

- Rust-generated seeds are now the default hot path.
- Java map dumps are still available, but only as an explicit oracle/debug mode.
- Offline export now defaults to `extra_nodes_after_root` instead of wall-clock search.
- Release binaries are used for self-play/export instead of debug builds.
- Export rows now also carry `hybrid_grid` and `policy_targets` for the hybrid branch.

## Training row schema

Rows now include more than just tensors and a scalar value target.

Important fields:

- schema version
- git SHA
- config artifact hash
- config behavior hash
- seed
- game id
- turn
- owner
- raw state hash
- encoded-view hash
- chosen joint-action id
- joint-action count
- aligned root value vector
- budget type and budget value
- search stats

This is meant to support:

- grouped splits
- dedup checks
- later policy distillation
- cleaner separation between build provenance and strategy identity
- hybrid teacher-student training without inventing a second export path

## Training split and dedup

The Python training path now:

- deduplicates by encoded-view hash
- groups by `(seed, game_id)`
- splits by grouped games instead of random rows

This is a guard against the earlier leakage issue where validation rows duplicated training positions.

## Hybrid training branch

There is now a new hybrid path under `python/train/outerloop`:

- `dataset.py`
- `model.py`
- `train_model.py`
- `export_weights.py`

Current behavior:

- reads the new `hybrid_grid` + `policy_targets` schema
- trains a tiny policy+value CNN
- exports a small JSON weight bundle that Rust can consume directly

The Rust side now has:

- `rust/bot/src/hybrid.rs`

That module loads the exported weights and can supply:

- action-ordering priors
- leaf-value bonus

Current status:

- local train/export/inference smoke passed
- the first end-to-end screening candidate using this path was far too slow and weak
- treat the hybrid path as new lab infrastructure, not as a promotable branch yet

## Outer-loop artifacts and ledger

`python/train/results.py` is still the local SQLite ledger, but its meaning is now stricter:

- training-only runs are `informational`
- arena plus Java smoke is the acceptance gate
- acceptance semantics are now versioned so older accepted rows can be migrated to `legacy` when the arena gate meaning changes

Wrapper scripts:

- `python/train/run_local_experiment.py`
- `python/train/run_arena.py`
- `python/train/sweep_search.py`
- `python/train/outerloop/run_candidate.py`

The outer loop now writes candidate artifacts under:

- `artifacts/outerloop/runs/<run_id>/manifest.json`
- `artifacts/outerloop/runs/<run_id>/candidates/<candidate_id>/`

The Helios-facing contract is:

- one run manifest per run
- one candidate directory per candidate
- `genome.json` plus `stage*.json` files
- stage result JSONs carry run/candidate linkage, hashes, executor, and timestamps

## Recent smoke runs

Examples that passed after the pipeline update:

```bash
python3 -m python.train.parallel_selfplay \
  --seed-count 4 \
  --workers 2 \
  --games 4 \
  --max-turns 20 \
  --extra-nodes-after-root 64 \
  --train
```

That exported `160` rows and completed an `mps` training smoke.

## Distillation pipeline (2026-03-13)

New training modes added to the hybrid path:

### Training modes

- **`standard`** (default): Direct supervised training on self-play data. Cross-entropy for policy, SmoothL1 for value. Existing behavior.
- **`teacher`**: Train a larger TeacherHybridNet (128ch, 8 SE-res blocks, ~2-3M params) with cosine LR schedule. Never exported to Rust. Used only for generating soft targets.
- **`distill`**: Train student (TinyHybridNet) using combined KL+MSE loss from teacher soft targets plus hard CE loss from ground truth.

### Distillation loss

```
L = T² · KL(softmax(student/T), softmax(teacher/T))
  + 1.5 · MSE(student_value, teacher_value)
  + α · CE(student_policy, hard_targets)
```

Hyperparameter ranges: T ∈ {2, 3, 5}, α ∈ {0.3, 0.5, 0.7}.

### Soft target generation

`generate_soft_targets()` runs teacher inference over the full dataset and writes augmented JSONL with `teacher_policy_logits` (shape [4][5]) and `teacher_value` (float) per row. Uses `HybridDistillDataset` which extends `HybridSelfPlayDataset` to also return these teacher fields.

### Student improvements

- Optional 3rd conv layer (`num_conv_layers=3`): receptive field 5x5 → 7x7
- Flat-array Rust inference: `Vec<f32>` with stride indexing instead of `Vec<Vec<Vec<f32>>>`
- 1x1 kernel support in Rust for future bottleneck architectures
- Schema v2 for 3-layer models; Rust accepts both v1 and v2

### Pipeline flow (orchestrated by outerloop.prose)

1. Generate shared self-play dataset (500+ seeds, one per run)
2. Train teacher on shared dataset (Modal GPU)
3. Generate soft targets from teacher (Modal GPU)
4. Train N distilled students in parallel (Modal GPU)
5. Screen students via arena (Modal CPU)
6. Authoritative stage-2 locally

### Modal functions

- `train_teacher_l40s`: trains teacher, saves checkpoint to Volume
- `generate_soft_targets_l40s`: loads teacher from Volume, writes augmented dataset to Volume
- `train_l40s`: detects `training_mode: “distill”` and dispatches to distillation path

## Current caveats

- Small smoke datasets can still produce tiny validation sets, so training metrics from those runs are only sanity checks.
- The current promoted submission is still search-only.
- The new hybrid path uses identity-preserving features, but there is no single-file hybrid submission path yet.
- The first meaningful hybrid milestone is “beats the incumbent locally without breaking timing,” not “promote immediately.”
- The sweep helper is still the preferred way to search the current Rust baseline; hybrid work should stay behind the same authoritative gate.
