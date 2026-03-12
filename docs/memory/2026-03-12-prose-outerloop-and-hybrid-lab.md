# 2026-03-12 Prose Outer Loop and Hybrid Lab

## What was added

The repo now has the first real Prose-first research lab scaffold.

Control plane:

- `automation/outerloop/program.md`
- `automation/outerloop/outerloop.prose`
- focused subprograms for offspring generation, screening, hybrid training, authoritative evaluation, and promotion

Worker/runtime layer:

- `python/train/outerloop/genome.py`
- `python/train/outerloop/registry.py`
- `python/train/outerloop/workspace.py`
- `python/train/outerloop/run_candidate.py`
- `python/train/outerloop/train_model.py`
- `python/train/outerloop/export_weights.py`
- `python/train/outerloop/build_dataset.py`
- `python/train/outerloop/launch_modal.py`
- `python/train/outerloop/modal_job.py`
- `python/train/outerloop/promote.py`

Rust hybrid hooks:

- `rust/bot/src/hybrid.rs`
- hybrid config support in `rust/bot/src/config.rs`
- hybrid-guided ordering / leaf blending in `rust/bot/src/search.rs`
- identity-preserving feature export in `rust/bot/src/features.rs`

## Architecture split

The intended ownership is now:

- Prose: workflow logic
- Python/Rust CLIs: deterministic work
- Helios: optional external cockpit/UI

This is deliberate. Helios is not supposed to own selection, mutation, arena logic, or promotion semantics. It should read the repo artifacts and drive the repo CLIs.

## Candidate artifacts

The outer loop now writes candidate artifacts under:

- `artifacts/outerloop/runs/<run_id>/manifest.json`
- `artifacts/outerloop/runs/<run_id>/candidates/<candidate_id>/`

The intended candidate directory contract is:

- `genome.json`
- `stage0.json`
- `stage1.json`
- `stage2.json` when authoritative evaluation runs
- optional `patch.diff`
- optional `model/`

Stage result JSONs now carry:

- `run_id`
- `candidate_id`
- `candidate_dir`
- `stage`
- `status`
- `artifact_hash`
- `behavior_hash`
- `genome_hash`
- `executor`
- `started_at`
- `finished_at`

The run manifest tracks:

- run id
- program path
- current status
- active stage
- candidate map
- promoted candidate id if one exists

## Hybrid branch

The hybrid branch is now present but still experimental.

Important current design:

- search remains the controller
- the tiny network is only for action ordering / value blending
- the export path now includes `hybrid_grid` and `policy_targets`
- the Python side can train a tiny conv net and export JSON weights
- the Rust side can load those weights and produce priors / leaf bonuses

Current representation:

- walls
- apples
- a support/fall-risk channel
- per-bird head/body channels for all four friendly and four enemy bots
- the existing six scalar features

## Verified so far

Verified locally in this phase:

- `cargo test --workspace`
- `python3 tools/generate_flattened_submission.py`
- small Rust-seed self-play export with the new hybrid schema
- tiny local hybrid train smoke on `mps`
- Rust-consumable weight export
- one `run_candidate` screening smoke that produced a real outer-loop run manifest and candidate stage artifacts

The hybrid screening smoke was intentionally bad in strength/timing terms. That is not a regression result to optimize around; it only proves the candidate runner and artifact contracts work end to end.

## Native Prose VM smoke run

After the scaffold landed, the repo also got a real native Prose VM smoke run.

Run state:

- `.prose/runs/20260312-144527-prose01/`

Important files:

- `.prose/runs/20260312-144527-prose01/state.md`
- `.prose/runs/20260312-144527-prose01/bindings/offspring.md`
- `.prose/runs/20260312-144527-prose01/bindings/finalists.md`
- `.prose/runs/20260312-144527-prose01/bindings/verdict.md`

What the VM did:

1. interpreted `automation/outerloop/outerloop.prose` inside the agent session
2. spawned a planner session
3. spawned three worker screening sessions in parallel
4. spawned an evaluator session to pick one finalist
5. spawned one worker authoritative-eval session
6. spawned a final evaluator session to issue the verdict

The three offspring were:

- `search-37c207806429`
- `search-02d93fb0e8a4`
- `hybrid-c6941dcf7906`

The hybrid candidate behaved exactly like an early hybrid smoke should: extremely slow and weak. The two search candidates both screened positively on the tiny temporary suites, and the evaluator chose `search-02d93fb0e8a4` for one stage-2 rerun.

Stage-2 result on the tiny temporary suites:

- candidate: `search-02d93fb0e8a4`
- heldout body diff: `+5.0`
- shadow body diff: `+11.75`
- later-turn `p99`: `28 ms`
- Java smoke: passed

Final verdict:

- `promising_smoke_only`
- `promote: false`

Reason:

- the VM run was intentionally a tiny-suite runtime proof, not a real `heldout_v1` / `shadow_v1` promotion attempt
- the correct next step for that exact candidate is a normal authoritative rerun on the real frozen suites

This matters because it proves the Prose runtime model is actually viable in this repo:

- the session can act as the VM
- `.prose/runs/...` can hold durable execution state
- the existing repo CLIs are sufficient worker surfaces
- Helios can treat that filesystem state as UI data instead of reimplementing workflow logic

## Native Prose VM optimization run

After the VM smoke run, the same in-session Prose model was used for one real optimization pass against the promoted live submission.

Run state:

- `.prose/runs/20260312-164032-a2d348/`

Base target:

- current submission `6/8/3/3`
- search-only, not hybrid
- hypothesis: a little more top-level breadth might beat the incumbent if timing stayed safe

Planner output:

- `search-37c207806429` = `6/9/3/3`
- `search-cfb01bf6ba43` = `7/8/3/3`
- `search-cf0e0f45226f` = `7/9/3/3 @ 42 ms`

All three candidates were screened on the real smoke suite through `python.train.outerloop.run_candidate`.

Screening result:

- `search-37c207806429`
  - heldout body diff `-3.078125`
  - later-turn `p99 = 171 ms`
- `search-cfb01bf6ba43`
  - heldout body diff `-0.15625`
  - later-turn `p99 = 173 ms`
- `search-cf0e0f45226f`
  - heldout body diff `-3.0625`
  - later-turn `p99 = 164 ms`

Evaluator verdict:

- no finalists
- no stage-2 authoritative rerun
- no promotion

The important lesson is not just “these candidates lost.” It is:

- broadening the current `6/8/3/3` winner further is not a marginal tradeoff
- in the current implementation it explodes later-turn cost far past the `45 ms` gate
- the next search optimization should reduce root cost or schedule breadth more selectively, not simply add more top-level branches

## Important limitation

The single-file CodinGame submission artifact is intentionally still search-only.

`tools/generate_flattened_submission.py` now fails explicitly if the chosen submission config enables the hybrid model. That is safer than pretending a hybrid submission is supported when the one-file generator does not yet embed weights.

So the current state is:

- local hybrid training/evaluation: supported
- hybrid search guidance in Rust: supported
- hybrid single-file CodinGame submission: not supported yet

## Dev ergonomics change

`workspace.py` now overlays the current repo’s uncommitted changes into candidate worktrees. That matters for local iteration because plain `git worktree add HEAD` would otherwise omit new outer-loop files until after a commit.

This is a development convenience, not a change to the authoritative promotion path.

## What to do next

The next high-value milestone is not “turn on RL.”

It is:

1. keep the current promoted search bot as the live baseline
2. use the new outer loop to search hybrid/search variants in a controlled way
3. only consider a hybrid submission path once a hybrid candidate is actually strong enough and the one-file embedding problem is solved

## Modal executor split now works

The repo now has a validated three-way execution split:

- Modal CPU: self-play shard generation
- Modal GPU: tiny hybrid training
- Local CPU: authoritative heldout/shadow arena plus Java smoke

This keeps the expensive synthetic-data and training loops off the laptop without weakening the promotion gate.

There is now also a dedicated stage-1 screening executor:

- `modal-arena-screen`

That executor runs screening arena jobs on Modal CPU while leaving stage 2 local and authoritative.

Validated smoke run:

- run id: `modal-stage1-smoke-2`
- candidate: `search-a3734068c5ae`
- executor: `modal-arena-screen`
- suite: temporary 4-seed subset of `smoke_v1`
- result: `screening`
- heldout body diff: `-1.25`
- shadow body diff: `+2.625`
- later-turn `p99`: `42 ms`

The result itself is not strategically important. What matters is that:

- the candidate runner can now offload stage 1 to Modal CPU
- the run manifest/stage artifact contract still works
- authoritative stage 2 and Java smoke remain local only

## Current hybrid read

The first hybrid branch result that looked structurally sane is now clear:

- tree-wide hybrid guidance was the wrong integration and caused catastrophic timing blowups
- root-prior-only hybrid guidance is viable on timing

Best screening signal so far:

- candidate: `hybrid-e86e007498ec`
- shape: 12-channel tiny model, root-only prior, `prior_mix = 0.12`, no leaf mix
- tiny screen result: `heldout_body_diff = +1.125`, `later_turn_p99 = 41 ms`

But the authoritative rerun on the real frozen suites was still rejected:

- heldout body diff: `-1.337890625`
- heldout win margin: `-0.091796875`
- shadow body diff: `+7.603515625`
- later-turn `p99`: `41 ms`
- Java smoke: passed

So the hybrid branch is now in the right failure mode:

- plumbing works
- timing is under control
- the remaining problem is strength, not integration correctness
