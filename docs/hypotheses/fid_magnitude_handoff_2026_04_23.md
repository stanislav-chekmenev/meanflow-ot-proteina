# FID magnitude debug — session handoff (2026-04-23)

## TL;DR for next session

Bug 1 (val/fid re-emitted 137 times into WandB) is **fixed** and merged into the
current `refactor` branch. FIDCallback now logs via
`logger.experiment.log({...}, commit=False)` on rank 0 only — bypasses
Lightning's `_ResultCollection._forward_cache` stickiness. 18/18 tests green.

The remaining open question is Bug 2: **FID values are 25K–150K, which is
~50–500× larger than Proteína paper numbers**. The user wants to diagnose
whether this is (a) under-trained 1-step generator, (b) train-vs-eval
distribution mismatch in the reference features, or (c) something else.
Start here — don't re-derive Bug 1.

## Observed FID trajectory (from wandb)

Two runs, both 2-GPU, `fid.n_samples=5000`, `fid.lengths=[64,100,128,150,192,200,250,256]`,
`fid.real_features_path=pdb_eval_ca_features.pth` (15,357 PDB eval structures):

| Step | BASE (`dzjxwk1t`) | OT-EUCL (`9dxsa6wa`) |
|-----:|------------------:|---------------------:|
| 10000 | 156,356 | 65,821 |
| 20000 | 56,372 | **11,782** ← best |
| 30000 | 27,150 | 12,643 |
| 40000 | 33,248 | 34,885 |

Non-monotone; both runs converged to ~35K at step 40K despite very different
train losses (OT `raw_loss_mf_step` hit 0.73 at step 36K; BASE stayed ~2–27).
User's instinct: "OT train loss is way lower than BASE, how can the FIDs be
the same?"

Train loss is extremely spiky in both runs (sample `raw_loss_mf_step` jumping
2 → 15,000 between adjacent 2k-step buckets), i.e. bad-batch outliers are
reaching the optimizer (`skip_nan_grad=False`). This is a separate
instability concern but not the FID-magnitude question.

## Leading hypothesis (DO THIS FIRST, 10 lines of code, 2 min)

`fid.real_features_path` points at the full PDB eval set (15,357 structures).
Training is on a **filtered subset**: monomers ≤ length 256, `fraction=0.008`
≈ 2,740 rows per memory `project_pdb_train_prep_2026_04_20.md`. The FID is
measuring "generator ↔ *full-PDB-eval*" while the generator only ever saw a
narrow slice. Even a model that perfectly reproduces the training subset
would get a large FID against the eval reference because the distributions
differ.

**Probe to verify / falsify.** Feed the training PDBs themselves as "fake"
into the existing metric_factory and compute FID against the same eval
reference:

```python
# 1-GPU, interactive
from proteinfoundation.metrics.metric_factory import (
    GenerationMetricFactory, generation_metric_from_list,
)
import glob

mf = GenerationMetricFactory(
    metrics=["FID"], ca_only=True, reset_real_features=False,
    real_features_path="./data/metric_factory/features/pdb_eval_ca_features.pth",
    ckpt_path="./data/metric_factory/model_weights/gearnet_ca.pth",
).cuda()

# The training subset staged on netscratch — exact path in memory
# project_dataset_staging_layout.md
train_pdbs = sorted(glob.glob(
    "/netscratch/schekmenev/ot_mf_proteina_ds/pdb_train/**/*.pdb",
    recursive=True,
))[:5000]
print(generation_metric_from_list(train_pdbs, mf))
```

Possible outcomes:
- Prints **~5K–30K** → the floor is the train/eval distribution gap, models
  are near the achievable bound. Action: dump a train-subset reference via
  `GenerationMetricFactory.dump_real_dataset_features` and point
  `fid.real_features_path` at that for training runs; keep the PDB-eval
  reference for publication numbers only.
- Prints **<100 (paper-scale)** → the reference and train distribution are
  actually comparable; the models really are off-manifold. Move to the
  multi-step diagnostic below.

## If the probe says "model is genuinely off-manifold": multi-step FM diagnostic

Infrastructure already exists. **Do not write a new integrator.**

`Proteina.generate_fm_euler(nsamples, n, nsteps=100)` at
[proteinfoundation/proteinflow/model_trainer_base.py:787-827](proteinfoundation/proteinflow/model_trainer_base.py#L787-L827)
forces `h=0` on the network (i.e. asks for instantaneous velocity, `r=t`)
and does standard Euler integration from t=1 to t=0. Written as a diagnostic
for exactly this situation.

Cheapest options (pick one):

1. **Interactive notebook / python -c** using the existing `last.ckpt`. Zero
   new code. Load checkpoint → call `generate_fm_euler(..., nsteps=200)` →
   feed results into `GenerationMetricFactory` via `_build_pyg_batch`
   ([proteinfoundation/callbacks/fid_callback.py:84-108](proteinfoundation/callbacks/fid_callback.py#L84-L108)
   has the reference implementation). Compare against the 1-step FID at the
   same checkpoint. ≈15 min of wall time for 5k samples × 200 steps on one
   80 GB GPU.
2. **FIDCallback flag** `fid.use_fm_euler: bool` with a one-line branch that
   picks `generate_fm_euler` over `generate`. Logs `val/fid_fm_euler`
   alongside `val/fid` in the *same* training run. Raise
   `fid.eval_every_n_steps` or lower `fid.n_samples` on the multi-step path
   to absorb the ~100× generation cost.

The user explicitly asked whether a standalone script is needed. Answer: no —
option 1 above is sufficient for a first data point.

## What is definitively correct (don't re-verify)

- FID aggregation. `ProteinFrechetInceptionDistance` at
  [proteinfoundation/metrics/fid.py:121-169](proteinfoundation/metrics/fid.py#L121-L169)
  accumulates `features.sum(0)` and `features.T @ features` per update call,
  then forms mean/cov once in `compute()`. Matches the user's spec. CA-only
  end-to-end: sample path (`samples_to_atom37` puts CA at openfold index 1)
  + callback `_build_pyg_batch` (`mask[:, 1]=True`) + GearNet
  `ca_only=True` path. Identical pipeline to `inference_fid_ca.yaml` +
  `generation_metric_from_list`.
- DDP. `compute()` is called on all ranks, torchmetrics
  `dist_reduce_fx="sum"` all-reduces features; rank-zero-only
  `load_real_dataset_features` + sum-reduce gives correct aggregate
  (rank-0's loaded tensor + rank-1's zeros sum to the loaded tensor).
- 5K samples actually get through the pipeline per firing (verified: ~45 s
  at step 10000, 2500 chunks/rank × 16/chunk × 8 lengths).
- Bug 1 fix is verified — regression test
  `test_val_fid_emitted_exactly_once_per_firing` in
  [tests/callbacks/test_fid_callback.py](tests/callbacks/test_fid_callback.py)
  asserts 1 `experiment.log` call across 11 consecutive train batches.

## Files / runs / memories to be aware of

- **Fixed callback**: [proteinfoundation/callbacks/fid_callback.py:192-206](proteinfoundation/callbacks/fid_callback.py#L192-L206)
- **Tests**: [tests/callbacks/test_fid_callback.py](tests/callbacks/test_fid_callback.py) — 18 passing
- **Bug-1 memory**: `project_fid_log_stickiness_fix_2026_04_23.md`
- **Prior FID refactor memory**: `project_rmsd_chirality_to_fid_refactor_2026_04_23.md`
  — gives the full FID callback design rationale
- **Training subset**: memory `project_pdb_train_prep_2026_04_20.md`
  (fraction=0.008 → ~2740 rows) and
  `project_dataset_staging_layout.md` (where the PDBs are on disk)
- **Training instability context**: memories
  `project_lr_schedule_bottleneck_2026_04_22.md`,
  `project_val_metrics_noise_2026_04_22.md`
- **Wandb runs**: `dzjxwk1t` (BASE meanflow), `9dxsa6wa` (OT-EUCL) — both
  still `state=running` at last check; compare their FID/loss trajectories
  rather than re-running
- **Reference features**:
  `./data/metric_factory/features/pdb_eval_ca_features.pth` — 15,357 PDB
  structures, 512-dim GearNet-CA features, `real_features_sum` mean
  ≈ 11917 (per-feature-per-sample ≈ 0.78)

## What NOT to re-investigate

- Do NOT re-diagnose the val/fid duplication — fix is merged, tests pass.
- Do NOT inspect Lightning's `_ResultCollection` internals — already
  reverse-engineered the `_forward_cache` stickiness path.
- Do NOT re-verify FID aggregation correctness — confirmed.
- Do NOT write a new multi-step integrator — `generate_fm_euler` exists.
- Do NOT start by launching a new training run to see what happens — the
  diagnostic probe above is cheaper.

## Suggested first tool calls for next session

1. Read this file and `project_fid_log_stickiness_fix_2026_04_23.md` in
   memory.
2. Run the train-subset-as-fake probe (the ~10-line snippet above) on a
   1-GPU interactive node. Use `.venv/bin/python` + `PYTHONPATH=.`.
3. Depending on the probe result, branch to either (a) dump a new
   train-subset reference .pth, or (b) run the multi-step FM diagnostic on
   the latest checkpoint.
