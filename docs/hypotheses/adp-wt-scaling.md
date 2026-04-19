# Hypothesis: `adp-wt-scaling`

Branch: `hyp/adp-wt-scaling` (off `main @ 52eb5c1`)
Spec: `docs/superpowers/specs/2026-04-18-rmsd-plateau-hypothesis-sweep-design.md @ 07e5b14`
Memory: `/mnt/labs/home/schekmenev/.claude/projects/-mnt-labs-home-schekmenev-projects-meanflow-ot-proteina/memory/project_debug_meanflow_1ubq_analysis.md`

## Pitch

### Hypothesis
Apply the adaptive weighting `adp_wt = (loss_mf.detach() + eps)^norm_p` (currently used only on the MeanFlow loss at `model_trainer_base.py:252-259`) to ALL loss components including the chirality hinge, via a three-point norm_p sweep (`norm_p ∈ {0, 1, 2}`) to test whether chirality is being crushed under `norm_p > 0` when it sits outside the `adp_wt` normalization.

### Mechanism
At `norm_p > 0`, the MF loss is divided by `(loss_mf.detach() + eps)^norm_p`, producing an effective MF gradient of magnitude `O((loss_mf)^(1 - norm_p))`. The chirality hinge is NOT currently rescaled by `adp_wt`, so its raw magnitude is added on top — at `norm_p > 0`, MF and chirality gradients operate at different scales (chirality grows unbounded relative to MF), causing the model to prioritize chirality corrections that interfere with MF learning. Applying the SAME `adp_wt` to chirality restores scale-consistency across losses: both shrink together as MF loss shrinks. A three-point sweep (`norm_p ∈ {0, 1, 2}`) tests the full range from "no scaling at all" through "strong scaling", producing a `norm_p` invariance curve.

Memory §3 documents the direct observation: at `p=1` the chirality term not being adaptively normalized makes it dominate MF.

### Prediction
- At `norm_p=0`, runs are numerically identical to baseline (adp_wt ≡ 1).
- At `norm_p=1` (canonical), reflected-RMSD separates from the 2.7 Å mirror plateau by `trainer/global_step ≈ 5K` and is stable below 2.7 Å at `gs ≥ 10K`.
- At `norm_p=2`, the scaling is stronger; if the mechanism holds, the effect from `p=1` should appear earlier but may overshoot (adp_wt < 1 at late training amplifies chirality).

### Primary signal
`eval/rmsd_reflected_mf1` stable below 2.7 Å at `trainer/global_step ≥ 10K` for the `norm_p=1` run. The same sbatch at `norm_p=0` (control) should plateau at ~2.7 Å like the baseline sweep.

### Secondary signals (false-positive guards)
1. `eval/chirality_mf1` (now a mean over `nsamples=16` in `[-1, +1]`) climbs away from 0 toward +1 at `norm_p=1`. If reflected-RMSD improves but chirality stays near 0, the improvement is from some unrelated effect (e.g. LR drift) and the hypothesis is NOT validated.
2. `train/raw_loss_chirality` stays bounded away from 0 (i.e. the hinge is still firing) while reflected-RMSD improves. If `raw_loss_chirality` collapsed to 0, the improvement would instead be a trivial "chirality loss turned itself off" artifact.
3. `train/raw_adp_wt_mean` (newly logged) is visibly `<1` at late training under `norm_p=1`, confirming `adp_wt` is amplifying (not suppressing) gradients as MF loss drops — the MECHANISM itself is active.
4. `norm_p=0` control run on this SAME branch should reproduce baseline behavior: reflected-RMSD plateau ~2.7 Å, chirality 1-bit noisy. That rules out this branch accidentally introducing an unrelated improvement.

## Verified file:line anchors (frozen @07e5b14)
- `adp_wt` formula: `proteinfoundation/proteinflow/model_trainer_base.py:252-259` (the `adaptive_loss` method).
- Chirality branch entry: `model_trainer_base.py:370-387`. The current code adds `chirality_loss_weight * loss_chir` raw, with no `adp_wt` scaling.
- Chirality hinge implementation: `proteinfoundation/proteinflow/chirality_loss.py:36-42, 93-97`.
- Eval chirality reducer: `proteinfoundation/callbacks/protein_eval.py:75-78, 269, 291` (the `_chirality_sign` helper and call site inside `_generate_and_save`).

## Implementation summary

### One file touched (the intervention's core):
- `proteinfoundation/proteinflow/model_trainer_base.py`
  - Added `_compute_adp_wt(self, loss)` (~line 260): returns the same `(loss.detach() + eps)^p` formula as `adaptive_loss` but exposes it as a tensor instead of dividing in place. Returns `ones_like(loss)` when `norm_p` is absent so the code path stays backward-compatible.
  - In `_compute_single_noise_loss`:
    - Compute `adp_wt_mf = self._compute_adp_wt(loss_mf)` ONCE, then call `self.adaptive_loss(loss_mf)` (which internally recomputes the same tensor, bit-identical — see note below).
    - In the chirality branch, divide the scalar `loss_chir` by `adp_wt_mf.mean()` before adding to the combined loss. At `norm_p=0`, `adp_wt_mf ≡ 1`, so the division is a no-op and the code is bit-identical to the baseline.
    - Return a new scalar `raw_adp_wt_mean` (for logging the `adp_wt` magnitude over training).
  - Thread `raw_adp_wt_mean` through both the `K=1` and `K>1` paths in `training_step` and log as `train/raw_adp_wt_mean`.

### Shape-matching decision (document per pitch)
`loss_mf` is per-sample (`[B]`), so `adp_wt_mf` is also `[B]`. `chirality_hinge_loss` returns a scalar (it reduces internally via `.sum()/valid_count`). We rescale the scalar `loss_chir` by `adp_wt_mf.mean()` — i.e. we use the batch-mean of the per-sample `adp_wt` as a single scalar divisor for the chirality term. This preserves the guarantee at `norm_p=0` that `adp_wt_mf.mean() == 1` exactly (no-op). At `norm_p > 0` it uses the same detached, loss-derived magnitude that the MF branch uses, modulo reducing `[B]` → scalar by `.mean()`. Taking the mean is the natural choice given chirality is already a batch-scalar hinge.

Note: `adaptive_loss` still re-computes `adp_wt` internally (it consumes per-sample `loss_mf`, divides, then `.mean()`s later). We deliberately do NOT refactor it — both computations are deterministic functions of the same tensor, so they produce bit-identical values. Keeping the change minimal (per spec's "Keep the change minimal. Do NOT refactor the adp_wt computation.") was chosen over deduplicating the two computations.

### Shared (guardrail) changes in this branch
These are SHARED across all 3 hypothesis branches per the spec's guardrail §2–3:
- `proteinfoundation/callbacks/protein_eval.py`: Added `nsamples: int = 1` kwarg. When `nsamples > 1`, the callback generates N proteins per eval mode and averages RMSD / reflected-RMSD. Chirality is now the MEAN of per-sample signs in `[-1, +1]`, reducing the 1-bit artifact at `nsamples=1`. The first generated PDB is still saved for WandB 3-D visualization.
- `proteinfoundation/train.py`: Wire `eval.nsamples` from config to the callback constructor (default 1).
- `configs/experiment_config/training_ca_debug.yaml`: Added `nsamples: 1` default under `eval:`.
- `proteinfoundation/utils/training_analysis_utils.py`: New `TrainLoaderStatsCallback` logs `len(train_dataloader)` and estimated per-epoch optim-step count at fit start (guardrail #3). `proteinfoundation/train.py` adds it to the callbacks list.

### Chirality-reducer guardrail #2 verification
Spec guardrail #2 requires verifying that the `chirality_mf1` metric reduces as an average, not a majority vote, before relying on it at `nsamples=16`. Quoting the relevant lines from `proteinfoundation/callbacks/protein_eval.py`:

```python
for b in range(atom37_np.shape[0]):
    gen_ca = atom37_np[b, :, _CA_INDEX, :]  # [n, 3]
    n_common = min(len(gt_ca), len(gen_ca))
    if n_common <= 0:
        continue
    gen_c = gen_ca[:n_common]
    gt_c = gt_ca[:n_common]
    rmsds.append(_ca_rmsd(gen_c, gt_c))
    rmsds_refl.append(_ca_rmsd_with_reflection(gen_c, gt_c))
    chirs.append(_chirality_sign(gen_c, gt_c))
if rmsds:
    rmsd = float(np.mean(rmsds))
    rmsd_reflected = float(np.mean(rmsds_refl))
    # Mean chirality sign in [-1, +1]; preserves the single-sample
    # semantics at nsamples=1 (still ±1).
    chirality = float(np.mean(chirs))
```

This is an arithmetic MEAN, not a majority vote. At `nsamples=16` a model that flips ~50/50 gives `chirality_mf1 ≈ 0`, and a model that correctly places handedness gives `chirality_mf1 → +1` — exactly the continuous signal the spec requires.

## Unit tests

All tests in `tests/test_hyp_adp_wt_scaling.py`, runnable via `PYTHONPATH=. pytest tests/test_hyp_adp_wt_scaling.py -v`. Six tests:

1. `test_norm_p_zero_leaves_chirality_identical_to_pre_adp_wt` — at `p=0`, combined loss delta between chirality-on/off equals `weight * raw_chir` exactly (backward-compat).
2. `test_norm_p_one_scales_chirality_by_inv_loss_mf` — at `p=1`, chirality contribution is `raw_chir / adp_wt_mean` (synthetic mirrored target).
3. `test_norm_p_two_scales_chirality_by_inv_loss_mf_squared` — at `p=2`, chirality contribution is `raw_chir / adp_wt_mean_p2`.
4. `test_gradient_respects_adp_wt_scaling_on_chirality_branch` — grads through the chirality inputs scale exactly by `1/adp_wt` (closed-form toy example).
5. `test_compute_adp_wt_reads_config_norm_p` — `_compute_adp_wt` reads from `training.meanflow.norm_p` / `norm_eps` and returns ones when absent.
6. `test_regression_adp_wt_scaling_is_active_at_norm_p_one` — FAILS if the chirality adp_wt division is reverted. This is the guard against silent refactors.

### FAIL-then-PASS evidence
With the chirality `/ adp_wt_chir` division temporarily removed, tests 2, 3, 6 FAIL with the intervention-specific assertion messages (e.g. `"Intervention appears inactive at norm_p=1: chir_p0=2.747..., chir_p1=2.747..."`). Restoring the division makes all six pass. The other existing tests (`test_selfcond_chirality.py`, `test_loss_accumulation.py`) also pass after the 5-tuple return update (`raw_adp_wt_mean` added).

## sbatch
`scripts/hyp_adp_wt_scaling.sbatch`.
- Copy of `scripts/train_debug.sbatch` (untouched).
- Guardrail overrides: `opt.accumulate_grad_batches=1`, `eval.nsamples=16`; `datamodule.repeat=2`, `batch_size=2`, `max_epochs=20000` are kept.
- Hypothesis knob: `training.meanflow.norm_p=$MEANFLOW_NORM_P` (default 1; sweep `{0, 2}` off the same branch).
- Chirality on: `training.chirality_loss.enabled=True weight=1.0 margin_alpha=0.1 stride=1`.
- Self-cond off: `training.self_cond=False self_cond_prob=0.0`.
- `TrainLoaderStatsCallback` logs `len(train_dataloader)` + est. optim-steps/epoch at fit start.
- Runtime logs exposed: `train/raw_loss_chirality` (existing), `train/raw_adp_wt_mean` (new), `eval/chirality_mf1` (now mean over nsamples=16, continuous).

## False-positive ruling (summary)
The intervention is confirmed VALID only if all of:
1. `norm_p=1` run shows reflected-RMSD < 2.7 Å stable at `gs ≥ 10K` (primary).
2. `eval/chirality_mf1` climbs toward +1 over time at `norm_p=1` (secondary).
3. `train/raw_loss_chirality` stays > ~0 throughout (not collapsing to 0).
4. `train/raw_adp_wt_mean < 1` at late training at `norm_p=1` (mechanism active).
5. `norm_p=0` control run (same branch, same sbatch with MEANFLOW_NORM_P=0) reproduces baseline plateau — ruling out branch-side artifacts.

If (1) holds but (2) or (3) fails, the improvement is not from the adp_wt-chirality coupling — reject.
If (5) fails (norm_p=0 control also improves), the branch has an unrelated fix hiding in the shared changes (`nsamples`, startup callback) — investigate before attributing to the intervention.
