# Protein Eval Callback — Design Spec

## Goal

Add periodic protein structure evaluation during debug training runs. Every N training steps, generate a protein using the current model weights, write it as a PDB file, and log it to WandB as a `wandb.Molecule` for 3D interactive visualization. Also log the ground truth 1UBQ structure for visual comparison.

## Approach

Lightning Callback (`ProteinEvalCallback`) that hooks into `on_train_batch_end`.

## Architecture

### New file: `proteinfoundation/callbacks/protein_eval.py`

```
ProteinEvalCallback(pl.Callback):
    __init__(eval_every_n_steps, n_residues, ground_truth_pdb_path)
    on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % eval_every_n_steps != 0 or trainer.global_step == 0:
            return
        with torch.no_grad():
            samples = pl_module.generate(nsamples=1, n=n_residues, nsteps=1)
            atom37 = pl_module.samples_to_atom37(samples)  # [1, n, 37, 3]
        Write atom37 to a temp PDB via write_prot_to_pdb()
        Log wandb.Molecule(pdb_path) as "eval/generated_protein"
        On first call, also log ground truth as "eval/ground_truth_protein"
        Clean up temp PDB file
```

### Modified: `proteinfoundation/train.py`

- Import `ProteinEvalCallback`
- Read eval config from `cfg_exp.eval` (with defaults/guard for when it's absent)
- Instantiate callback and append to `callbacks` list when `cfg_exp.eval.enabled` is True and WandB logging is active

### Modified: `configs/experiment_config/training_ca_debug.yaml`

Add eval section:
```yaml
eval:
  enabled: true
  every_n_steps: 500
  n_residues: 76
  ground_truth_pdb: ${data_path}/debug/raw/1ubq.cif
```

## Data Flow

```
global_step % 500 == 0
  -> model.generate(nsamples=1, n=76, nsteps=1)
     -> Gaussian noise [1, 76, 3] in nm
     -> fm.meanflow_sample() with 1-step
     -> CA coords [1, 76, 3] in nm
  -> model.samples_to_atom37()
     -> atom37 [1, 76, 37, 3] in Angstroms (only CA populated)
  -> write_prot_to_pdb(atom37[0].cpu().numpy(), tmp_pdb_path, overwrite=True, no_indexing=True)
     -> /tmp/eval_protein_step_500.pdb
  -> wandb.log({"eval/generated_protein": wandb.Molecule(tmp_pdb_path)}, step=global_step)
  -> (first time) wandb.log({"eval/ground_truth_protein": wandb.Molecule(gt_pdb_path)})
  -> os.remove(tmp_pdb_path)
```

## Key Decisions

1. **Callback, not inline**: Keeps training logic clean. Easy to toggle via config.
2. **wandb.Molecule from PDB file**: Uses WandB's built-in 3D molecular viewer. PDB format is already supported by the existing `write_prot_to_pdb()` utility.
3. **Ground truth logged once**: The 1UBQ reference structure is static, so log it on the first eval step only.
4. **Temp file cleanup**: PDB files are written to `/tmp/` and deleted after logging to avoid disk buildup.
5. **Model stays in train mode**: The callback uses `torch.no_grad()` but does not toggle `model.eval()` to avoid disrupting batch norm / dropout state (the current model uses no dropout or batch norm, but this is safer).
6. **EMA awareness**: The existing `EMA` callback swaps in EMA weights during validation. Since this callback runs on `on_train_batch_end` (not during validation), it uses the live training weights, not EMA. This is intentional for debug — we want to see what the training weights produce.

## Files Changed

| File | Change |
|------|--------|
| `proteinfoundation/callbacks/__init__.py` | New file (empty or with import) |
| `proteinfoundation/callbacks/protein_eval.py` | New file — ProteinEvalCallback class |
| `proteinfoundation/train.py` | Import callback, instantiate from config, add to callbacks list |
| `configs/experiment_config/training_ca_debug.yaml` | Add `eval` config section |

## Testing

Run the debug training script and verify:
- WandB dashboard shows `eval/generated_protein` molecules at steps 500, 1000, etc.
- WandB dashboard shows `eval/ground_truth_protein` molecule (logged once)
- Training loss is unaffected (no gradient contamination)
- No temp PDB files accumulate on disk
