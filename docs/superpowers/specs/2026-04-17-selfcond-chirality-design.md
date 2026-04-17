# Self-conditioning + chirality hinge loss for MeanFlow training

Date: 2026-04-17
Branch: `debug_run`

## Motivation

The chirality-diagnostic work (commit `dd66260`) established that on the 1ubq debug setup the MeanFlow loss is effectively O(3)-invariant: training on a single L-chiral protein gives the model no signal to prefer GT handedness over its mirror image. Two levers were identified as likely fixes:

1. **Self-conditioning** — feeds a detached previous-step prediction into the network through coordinate-dependent features (`x_sc`, `x_sc_pair_dists`). This couples consecutive predictions and breaks strict O(3) invariance of the loss. It was disabled in MeanFlow training because of assumed incompatibility with `torch.func.jvp`.
2. **Chirality-sensitive loss** — an explicit penalty on predicting the wrong handedness, computed from signed triple products of CA coordinates.

Both are added as opt-in toggles so we can A/B them on the debug run before committing to them in full training.

## Non-goals

- Rewriting the JVP path or loss-accumulation logic.
- Changing the sampling/inference code paths.
- Supporting motif conditioning together with self-cond (still asserted off at top of `training_step`).
- Making chirality loss apply to anything other than C-alpha coordinates.

## Part A — Self-conditioning in MeanFlow training

### JVP safety

The concern was that `torch.func.jvp` cannot tolerate dropout-style branching across passes. The resolution is that `x_sc` is a **closed-over constant** inside the JVP primal function — detached, carries no tangent, and is identical across the three jvp call evaluations. This is the same pattern already used for `mask` and `cath_code` in `u_func` (`model_trainer_base.py:301-308`). The legacy 50%-probability dropout is applied *outside* the JVP (one decision per training step, per K-pass), not inside it.

### Mechanism

1. **Config-driven activation.** In `Proteina.__init__` read:
   - `cfg_exp.training.self_cond` (bool, existing key)
   - `cfg_exp.training.self_cond_prob` (float, default 0.5, new key)

   Remove the warning at `proteina.py:119-123` (`"self_cond=True has no effect in MeanFlow training."`).

2. **Per-step decision.** In `ModelTrainerBase.training_step`, after sampling `(t, r)` and before the K-loop:
   ```python
   use_sc = (
       self.training
       and self.cfg_exp.training.get("self_cond", False)
       and not val_step
       and random.random() < self.self_cond_prob
   )
   ```
   Validation always uses `use_sc=False`.

3. **Per-K-pass warmup.** All of this lives inside `_compute_single_noise_loss`. When `use_sc=True` (passed in as a kwarg):
   - After sampling `x_0` and constructing `z = (1-t_ext)*x_1 + t_ext*x_0` (existing code, `model_trainer_base.py:292`),
   - Run a `torch.no_grad()` forward using the same `u_func`-style call, with `x_sc` absent (feature factory defaults to zeros — `feature_factory.py:426-432`, `560-574`),
   - Capture `u_warmup = u_func_nograd(z, t_ext, r_ext)`, detach, apply mask + zero-COM,
   - Interpret as a clean-coord prediction: with `u` being the MeanFlow avg velocity, `x_1 ≈ z - t_ext * u` (exact when `r=0`, an approximation at general `r` — this matches how the FM sub-pass already uses `v_pred`).
   - Bind as a local `x_sc` (detached), scoped to this function call only.

4. **JVP pass with x_sc.** In `u_func`, extend:
   ```python
   batch_nn = {"x_t": z_in, "t": t_flat, "h": h, "mask": mask}
   if x_sc is not None:
       batch_nn["x_sc"] = x_sc  # detached, no tangent
   if "cath_code" in batch: ...
   ```
   `x_sc` is closed over the same way as `mask`; JVP treats it as constant.

5. **FM sub-pass with x_sc.** The FM-loss call at `model_trainer_base.py:333` (`v_pred = u_func(z, t, t)`) receives the same `x_sc` (detached).

6. **Signature change.** `_compute_single_noise_loss` gains a kwarg `use_sc: bool = False`. The warmup happens inside the helper, so each K pass (with its own `x_0`) gets a fresh `x_sc`. `training_step` just computes `use_sc` once per step and passes the same value to each K invocation.

### What the sc-warmup costs

One extra no-grad forward per noise pass when `use_sc=True`. At `p_sc=0.5` and `K=1`, amortized 0.5× extra forward. At `K>1`, 0.5× extra per accumulation step. Acceptable for debug; worth flagging for full-scale runs.

### Config surface

Both `configs/experiment_config/training_ca.yaml` and `configs/experiment_config/training_ca_debug.yaml`:

```yaml
training:
  self_cond: False            # existing, now wired up
  self_cond_prob: 0.5         # new: probability of activating x_sc per step
```

### NN feature wiring

Verified already wired in all `ca_af3_*` NN configs: `feats_init_seq` contains `"x_sc"` and `feats_pair_repr` contains `"x_sc_pair_dists"`. The debug run uses `ca_af3_60M_notri` (via `caflow.yaml`) which has both. No config changes needed on the NN side.

### Risk: JVP + x_sc numerical stability

If `x_sc` comes from a randomly initialized network early in training, it is close to noise. The gradient signal still comes from the JVP primary pass; the warmup is a no-grad constant. Unlike the legacy flow-matching trainer, there's no risk of self-cond "leaking" gradient back through the warmup. The main risk is subtle feature-statistics shifts early in training, which the 0.5 dropout partially mitigates (the network sees both sc and no-sc inputs).

## Part B — Chirality hinge loss

### Descriptor

Per-residue signed triple products from CA coords `x ∈ R^{B, n, 3}` with mask `m ∈ {0,1}^{B, n}` and stride `k` (default 1):

```
e1_i = x_{i+k}  - x_i
e2_i = x_{i+2k} - x_i
e3_i = x_{i+3k} - x_i
T_i  = det([e1_i, e2_i, e3_i])           # signed volume, nm^3
```

Valid indices: `i ∈ [0, n - 3k)`. Validity mask: `valid_i = m_i * m_{i+k} * m_{i+2k} * m_{i+3k}`.

Vectorized: stack edges `→ [B, n-3k, 3, 3]`, apply `torch.linalg.det`.

### Margin

Per batch, from GT `x_1`:
```
T_gt      = triple_products(x_1, stride=k)    # [B, n-3k]
m_T_batch = alpha * (|T_gt| * valid).sum() / valid.sum()
```
`alpha` is configurable (default `0.1`). `m_T_batch` is a scalar, recomputed per step — cheap (one det per step on a small tensor) and adapts to batch content. In the single-protein debug run this is effectively constant across steps.

### Loss

```
signed_agreement = sign(T_gt) * T_pred    # positive if same handedness
L_chir           = mean over valid i of relu(m_T_batch - signed_agreement_i)
```

Zero when predictions match GT handedness with triple-product magnitude ≥ margin; positive and linear in shortfall otherwise.

### Where `x_pred` comes from

Inside `_compute_single_noise_loss`, after the FM sub-pass computes `v_pred = u_func(z, t, t)` (line 333), reconstruct:
```
x_1_pred = z - t_ext * v_pred              # instantaneous-velocity form (r=t)
x_1_pred = self.fm._mask_and_zero_com(x_1_pred, mask)
```
This reuses the FM-pass forward — no extra network call. `v_pred` already requires grad, so chirality loss participates in the backward graph.

### Combining with existing loss

```
loss_total = (1 - mf_ratio) * loss_fm + mf_ratio * loss_mf + w_chir * loss_chir
```
where `w_chir = cfg_exp.training.chirality_loss.weight` and gating is:
```
if self.chirality_loss_enabled and w_chir > 0:
    ... compute L_chir ...
    combined_adp_loss = combined_adp_loss + w_chir * loss_chir
```

Adaptive weighting (`self.adaptive_loss`) is **not** applied to `L_chir` — the hinge already has the right scale by construction.

### Logging

```
train/raw_loss_chirality         # mean hinge loss (pre-weight)
train/chirality_margin_m_T       # per-step margin scalar
train/chirality_signed_agreement # mean(signed_agreement / |T_gt|)  — in [-1, 1], tracks handedness
```

### Config surface

Both training yamls:
```yaml
training:
  chirality_loss:
    enabled: False
    weight: 1.0
    margin_alpha: 0.1
    stride: 1
```

## sbatch changes

In `scripts/train_debug.sbatch`, add variables (near the existing `EXP_NAME`, `LR`, etc. block):

```bash
SELF_COND=False
SELF_COND_PROB=0.5
CHIRALITY_ENABLED=False
CHIRALITY_WEIGHT=1.0
CHIRALITY_MARGIN_ALPHA=0.1
CHIRALITY_STRIDE=1
```

And to the `--exp_overrides` block:
```
training.self_cond=$SELF_COND \
training.self_cond_prob=$SELF_COND_PROB \
training.chirality_loss.enabled=$CHIRALITY_ENABLED \
training.chirality_loss.weight=$CHIRALITY_WEIGHT \
training.chirality_loss.margin_alpha=$CHIRALITY_MARGIN_ALPHA \
training.chirality_loss.stride=$CHIRALITY_STRIDE \
```

## Tests

New tests in `tests/test_chirality_diagnostics.py` (or a new `test_selfcond_chirality.py`):

1. **Triple-product sign flip under reflection.** `triple_products(Q · x) == -triple_products(x)` elementwise. Proves the descriptor is chirality-sensitive.
2. **Hinge loss zero at identity.** `chirality_hinge_loss(x_gt, x_gt, mask, alpha=0.1) == 0`.
3. **Hinge loss large at mirror.** `chirality_hinge_loss(Q · x_gt, x_gt, mask, alpha=0.1) > 0` and ≥ margin scale.
4. **Margin scale.** Recomputed margin `m_T_batch` for a known synthetic protein matches `alpha * mean(|T_gt|)` within tolerance.
5. **Self-cond plumbing.** Unit test on a fake NN (a small `nn.Module` returning `coors_pred`): call `_compute_single_noise_loss` twice, once with `use_sc=False` and once with `use_sc=True`. Assert (a) the fake NN is called twice when `use_sc=True` (warmup + JVP) and once when `use_sc=False`; (b) the second invocation when `use_sc=True` receives `x_sc` in its batch dict; (c) the returned loss is finite in both cases.
6. **Chirality-loss gradient flows.** With a fake NN that returns a learnable-parameter-scaled `coors_pred`, compute `loss_chir` and call `.backward()`; assert non-zero grads on the fake NN parameter.

## File impact

| File | Change |
|------|--------|
| `proteinfoundation/proteinflow/proteina.py` | Read new config keys; remove self_cond warning; store `self_cond_prob`, chirality hyperparams. |
| `proteinfoundation/proteinflow/model_trainer_base.py` | Add self-cond warmup + `x_sc` plumbing in `_compute_single_noise_loss` and `training_step`; add `_chirality_hinge_loss` helper (or import from a new module); log new metrics. |
| `configs/experiment_config/training_ca.yaml` | Add `self_cond_prob`, `chirality_loss.*` keys. |
| `configs/experiment_config/training_ca_debug.yaml` | Same additions. |
| `configs/experiment_config/model/nn/*.yaml` (audit) | Ensure `x_sc` is in `feats_init_seq` where not already present. |
| `scripts/train_debug.sbatch` | New bash variables + Hydra overrides. |
| `tests/test_chirality_diagnostics.py` (or new file) | Six new tests above. |

## Rollout

1. Implement + tests pass locally (unit tests only — no GPU required for the new logic beyond what already exists).
2. One debug run on 1ubq with both features off — confirm no regression vs. `dd66260` baseline.
3. One debug run with `self_cond=True` only.
4. One debug run with `chirality_loss.enabled=True` only (alpha=0.1, weight=1.0).
5. One run with both on.
6. Compare `eval/chirality_*` diagnostic on all four — expect handedness signal to appear in (3)–(5).
