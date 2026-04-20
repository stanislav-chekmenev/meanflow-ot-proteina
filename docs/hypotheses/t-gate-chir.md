# Hypothesis: t-gate on the chirality hinge loss

Branch: `hyp/t-gate-chir`
Spec SHA: `07e5b14` (docs/superpowers/specs/2026-04-18-rmsd-plateau-hypothesis-sweep-design.md)
sbatch: `scripts/hyp_t_gate_chir.sbatch`
Unit tests: `tests/test_hyp_t_gate_chir.py`

## Hypothesis

Introduce a t-gate on the chirality hinge loss so that the chirality term is
only active for small diffusion times t (near clean data), preventing the
hinge from firing on noisy intermediate predictions where `T_pred` is
uninformative and masking the MeanFlow gradient.

## Mechanism

At large t, `x_1_pred = z − t·v_pred` is a noisy extrapolation dominated by
the z noise term; the triple products computed on it carry little signal
about the model's handedness learning. Including the chirality hinge at all
t averages high-variance/low-signal gradient noise into the optimizer update,
crowding out the MeanFlow signal that the model actually uses to fit the
vector field. A t-gate of the form

```
chirality_loss *= (t < t_gate_max).float()    # hard
chirality_loss *= sigmoid((t_gate_max - t) / t_sharpness)   # soft
```

restricts chirality pressure to low-t where `x_1_pred ≈ x_1` and `T_pred` is
meaningful. The MeanFlow loss itself is *not* gated; only the auxiliary
chirality term is gated.

## Prediction

With `t_gate_max=0.3` (hard), the chirality hinge fires only in the ~15–30%
of batches where t is small. We expect:

- Reflected-RMSD at gs ≥ 10K drops below 2.7 Å and stays there (primary
  success criterion).
- `chir_sign_agreement` improves from ~0.5 (mixed basin) toward +1 by gs ~
  5K as the gradient the model now receives from chirality is less noisy.
- MF loss trajectory matches an ungated baseline within noise at early
  epochs (chirality updates at low t are closer to the hinge's natural
  operating regime, so the "pressure" on MF drops).

## Primary signal

`eval/rmsd_reflected_mf1` (and `eval/rmsd_reflected_mf10x`) at
`trainer/global_step ≥ 10000` must average below 2.7 Å over a sliding window
of ~20 eval points, and must not degrade past that.

## Secondary signals (false-positive ruling)

1. `eval/chirality_mf1` at `nsamples=16` (mean across samples, real-valued
   in [-1, +1]) must trend positive rather than hovering near 0. Without the
   guardrail, a 1-bit per eval step can appear "correct" purely by luck; the
   averaged estimator rules this out.
2. `train/raw_loss_chirality` must drop meaningfully relative to the
   ungated baseline at the same optimizer step — this confirms that the
   hinge is firing on the low-t batches it can actually correct, rather
   than being degenerate everywhere.
3. `eval/fm_loss_mirror_abs_diff` should remain ~0 throughout — the network
   is O(3)-equivariant, so the FM loss itself cannot discriminate chirality.
   A spurious positive on primary RMSD caused by EMA-swap artefacts would
   also move this diagnostic; its continued flatness is evidence the
   improvement is real.
4. `len(train_dataloader)` and optimizer-steps-per-epoch logged at startup
   by `StartupInfoCallback` confirm 1 step = 1 epoch (rules out
   accumulate_grad_batches confusion mimicking a schedule shift).

## Proposed change (implemented)

Verified file:line anchors (at the HEAD commit of this branch):

- `proteinfoundation/proteinflow/model_trainer_base.py:373-405` — chirality
  branch now:
  1. Computes per-sample hinge via new
     `chirality_hinge_loss_per_sample` (see below).
  2. Builds a per-sample `gate` in [0, 1] from `t` and `chirality_t_gate_max`
     / `chirality_t_gate_mode`.
  3. Reduces the gated per-sample loss by `.mean()` and adds
     `weight * loss_chir` to the combined loss.
- `proteinfoundation/proteinflow/chirality_loss.py:57-81` — new
  `chirality_hinge_loss_per_sample` returns a `[B]`-shaped tensor
  (per-sample mean of hinges / per-sample valid count). Original
  `chirality_hinge_loss` preserved.
- `proteinfoundation/proteinflow/proteina.py:128-131` — plumbs
  `t_gate_max` (default 1.0, identity gate) and `t_gate_mode` (default
  `"hard"`) out of `cfg_exp.training.chirality_loss` onto
  `self.chirality_t_gate_max` / `self.chirality_t_gate_mode`.
- `configs/experiment_config/training_ca.yaml` and
  `configs/experiment_config/training_ca_debug.yaml` — both add the two
  new keys under `training.chirality_loss` with default values that
  preserve current behaviour (`t_gate_max=1.0`, `t_gate_mode="hard"`).

Defaults are chosen so any previous config that omits the keys gets an
identity gate — backwards-compat is preserved.

## sbatch summary (diff from train_debug.sbatch)

| Env var                     | baseline | t-gate-chir |
|-----------------------------|---------:|------------:|
| `ACCUMULATE_GRAD_BATCHES`   | 10       | 1           |
| `EVAL_NSAMPLES`             | N/A      | 16          |
| `CHIRALITY_ENABLED`         | False    | True        |
| `CHIRALITY_WEIGHT`          | 1.0      | 1.0         |
| `CHIRALITY_MARGIN_ALPHA`    | 0.1      | 0.1         |
| `CHIRALITY_STRIDE`          | 1        | 1           |
| `CHIRALITY_T_GATE_MAX`      | —        | 0.3         |
| `CHIRALITY_T_GATE_MODE`     | —        | `hard`      |
| `MEANFLOW_NORM_P`           | 1        | 0           |
| `SELF_COND`                 | False    | False       |
| `SELF_COND_PROB`            | 0.5      | 0.0         |
| `EXP_NAME`                  | meanflow-base-p0-bs2 | hyp-t-gate-chir |

`MEANFLOW_NORM_P=0` isolates this run from the norm_p slot (a separate
hypothesis sweeps `norm_p` scaling). `DATAMODULE_REPEAT=2`,
`DATAMODULE_BATCH_SIZE=2`, `MAX_EPOCHS=20000` are unchanged per spec
guardrail #4.

## Guardrail #2: chirality reducer verification

`proteinfoundation/callbacks/protein_eval.py:_generate_and_save` aggregates
per-sample chirality flags by averaging:

```python
chiralities.append(_chirality_sign(gen_c, gt_c))
...
chirality = float(np.mean(chiralities))
```

`_chirality_sign` returns ±1 per sample (see `protein_eval.py:75-78`), so the
averaged `chirality` is a real value in [-1, +1] — a proper low-variance
estimator of the probability that the model generates the correct
handedness. With `nsamples=1` this collapses back to the original 1-bit
Bernoulli value; bumping to `nsamples=16` (guardrail) gives ~4x lower
variance, which is what we need to tell a slow chirality drift apart from
noise.

The callback previously hardcoded `nsamples=1` at `protein_eval.py:265,269`.
This branch exposes `nsamples` through the constructor / `eval.nsamples`
config key, threaded via `proteinfoundation/train.py:256-270`. The same
exposure is applied on every hypothesis branch (shared guardrail change).

## Test plan

`tests/test_hyp_t_gate_chir.py` asserts:

1. **Hard gate zeroes chirality when all t > t_gate_max** — per-sample gate
   kills every contribution, `raw_loss_chir == 0.0`.
2. **Hard gate equals ungated when all t < t_gate_max** — per-sample gate
   is 1 everywhere, so the result matches
   `chirality_hinge_loss_per_sample(...).mean()`.
3. **Hard gate on mixed batch zeroes only above-cutoff samples** — exact
   equality with `(per_sample * gate).mean()`.
4. **Soft gate at t ≫ t_gate_max** — chirality ≈ `sigmoid((t_gate_max - t) /
   t_gate_max) * ungated`, checked to 1e-4 relative tolerance.
5. **Soft gate at t ≪ t_gate_max** — chirality ≈ `sigmoid(0.99) * ungated`.
6. **Default (`t_gate_max=1.0`, hard) is identity** — matches ungated loss
   exactly on a seeded mirror batch.
7. **Config plumbing** — `Proteina(cfg)` sets `chirality_t_gate_max` and
   `chirality_t_gate_mode` from `cfg.training.chirality_loss`, with the
   documented defaults when the keys are absent.
8. **MF loss is NOT gated** — gating chirality off (either by
   `t_gate_max=0.0` or `enabled=False`) yields identical MF and FM losses,
   so the gate cannot leak into the MeanFlow gradient path.

Run with `PYTHONPATH=. pytest tests/test_hyp_t_gate_chir.py -v`.
