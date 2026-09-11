# Sokoban state modeling

Single-response RLVR training for a VLM that receives one VAGEN Sokoban RGB frame and an environment-provided block of 1–3 actions. It emits the current symbolic board in `<perception>` and one final board in `<prediction>`.

The authoritative specification is [DATASET_PLAN.md](DATASET_PLAN.md). The original multi-turn `Sokoban` environment is unchanged.

## Generate and audit data

Run a small smoke build before the formal 50,000/2,000/10,000 splits:

```bash
python -m sokoban_state_modeling.scripts.generate_dataset \
  --output-dir sokoban_state_modeling/data/smoke_v9_1000 --split train --num-tasks 1000
```

Generation commits deterministic 100-row chunks under `.<split>.staging`. If a run is
interrupted, invoke the same command with `--resume`; split, size, versions, and the full
sampler profile must match the checkpoint. At most the uncommitted current chunk is
recomputed.

For the formal dataset, use a separate output root and generate all three manifests:

```bash
python -m sokoban_state_modeling.scripts.generate_dataset --output-dir sokoban_state_modeling/data/formal_v9 --split train --num-tasks 50000
python -m sokoban_state_modeling.scripts.generate_dataset --output-dir sokoban_state_modeling/data/formal_v9 --split validation --num-tasks 2000
python -m sokoban_state_modeling.scripts.generate_dataset --output-dir sokoban_state_modeling/data/formal_v9 --split test --num-tasks 10000
```

Only manifests whose row-level `generator_version` matches the package are accepted.
The current published smoke outputs are `data/smoke_v9_1000` and
`data/smoke_v9_eval`; earlier versioned smoke and `data/legacy_v*` directories are
diagnostic artifacts only. The v1 archive is the only one that contains PNG-backed
tasks.

The command writes `manifests/<split>.jsonl`, a uint64 little-endian `.idx`, and `audits/<split>.json`. RGB files are not stored: every reset renders the fixed `current_state` in memory with the versioned VAGEN Sokoban renderer. Generation fails instead of relaxing an unavailable quota. Files are built and audited in a staging directory; the manifest is published last only after the audit passes.

After all three splits exist, verify that their map, current-state and task hashes do not overlap:

```bash
python -m sokoban_state_modeling.scripts.audit_splits \
  --train /absolute/path/train.jsonl \
  --validation /absolute/path/validation.jsonl \
  --test /absolute/path/test.jsonl \
  --output /absolute/path/split_leakage.json
```

`configs/eval_qwen35_4b.yaml` covers the 2,000-row validation manifest;
`configs/test_qwen35_4b.yaml` covers the independent 10,000-row test manifest. Offline
`scripts/evaluate.py` rejects missing or duplicate task indices by default. Its
`--allow-partial` option is only for explicitly labelled development reports.

## Train Qwen3.5-4B with ordinary GAE

Set the four absolute manifest/index paths, optionally set `SOKOBAN_STATE_REWARD_PROFILE=exact_only`, then run:

```bash
bash sokoban_state_modeling/scripts/train_qwen35_4b.sh
```

The launcher explicitly selects `algorithm.adv_estimator=gae`, enables the critic, uses `trainer.harness=concat`, and sets `max_turns: 1`. No bi-level or VAGEN custom GAE is used.
Before allocating the run it also verifies the exact 50,000/2,000 manifest sizes, schema
compatibility, the VERL checkout, PyTorch, and visible CUDA devices. This check is
expected to fail on the current CPU-only development machine and pass on the later GPU
training platform.

## Test

```bash
pytest -q sokoban_state_modeling/tests
```
