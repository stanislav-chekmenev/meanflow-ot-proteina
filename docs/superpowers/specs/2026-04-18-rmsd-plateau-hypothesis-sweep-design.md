# Multi-agent hypothesis sweep: MeanFlow 1-step RMSD plateau

Date: 2026-04-18
Base branch: `main` (includes merge `d5cfa31`)
Problem owner: user (Stanislav)

## Motivation

On the 1ubq single-protein debug regime, the MeanFlow 1-step model's reflected-RMSD plateaus around 2.7 Å and sometimes degrades past `trainer/global_step ≈ 10K`. The 2026-04-18 root-cause analysis (saved in auto-memory as `project_debug_meanflow_1ubq_analysis.md`) corrected several prior assumptions:

- `accumulate_grad_batches=10` is a silent no-op in this config because Lightning 2.5 forces an optimizer step on the final batch of every epoch; with 1 micro-batch per epoch, 1 optim step = 1 epoch.
- The prior LR-peak-at-epoch-10K hypothesis for the degradation is **refuted** — LR peaks at epoch 1000 and is in deep decay (~0.09·base_lr) by epoch 10K.
- Eval chirality is a 1-bit Bernoulli sample (`nsamples=1`), so the metric cannot diagnose slow drift.
- FM loss is O(3)-invariant on a single chiral protein, so reflected-RMSD is the only quantity that can discriminate handedness failures from generic modeling failures.

The remaining candidate causes are listed in the memory under section 4. This spec operationalizes a multi-agent process to turn those candidates into three concrete, runnable, testable hypothesis branches.

## Goal

Produce **three `hyp/<slug>` branches off `main`**, each containing:

1. A focused code change implementing one hypothesized fix.
2. Unit tests that pass on the change.
3. An sbatch script in `scripts/` modeled on `train_debug.sbatch`, with the known-confound guardrails applied (see §Guardrails).
4. A short `docs/hypotheses/<slug>.md` note with: hypothesis one-liner, mechanism, predicted metric delta, primary/secondary success signals, and how to rule out false positives.

The user runs the sbatch scripts. Subagents do not submit jobs.

## Non-goals

- Solving the plateau — this spec only stages testable interventions.
- Touching the Lightning `accumulate_grad_batches` behavior itself. Every sbatch sets `accumulate_grad_batches=1` and logs `len(train_dataloader)` at startup.
- Revisiting the sampling/inference paths beyond what a specific hypothesis requires.
- Changing eval chirality from `nsamples=1` as part of a hypothesis change — **this is mandated as a guardrail on all branches** so hypotheses are evaluable.
- Running the sbatch jobs inside subagents.

## Architecture: agent roles

### Two main agents (debate partners, stateless between rounds)

Main agents are stateless `Agent` dispatches; continuity across rounds comes from the orchestrator embedding prior-round artifacts (pitches, rulings, diffs) in each dispatch prompt. "Debate partners" means role-consistent, not stateful.

- **Architect agent** (`subagent_type: general-purpose`, model: opus)
  - Responsibility: feasibility, codebase integration, test strategy, change localization.
  - Grounded in the file:line anchors from the memory file:
    - Adaptive loss: `proteinfoundation/proteinflow/model_trainer_base.py:252-259`
    - Chirality branch: `model_trainer_base.py:370-387` (enable-guard at `:372`; no t-gate exists yet — any t-gating is a hypothesis-scope change)
    - LR scheduler: `model_trainer_base.py:59-86`
    - Chirality hinge math: `proteinfoundation/proteinflow/chirality_loss.py:36-42, 93-97`
    - Eval chirality: `proteinfoundation/callbacks/protein_eval.py:75-78, 269, 291`
    - Self-cond warmup/JVP: `model_trainer_base.py:311-326, 338-339`
    - Self-cond inference zero fallback: `proteinfoundation/nn/feature_factory.py:429-432, 575-578`
  - Approval criteria:
    1. Change is localized (no incidental refactors).
    2. Does not silently alter another knob the analysis depends on.
    3. Unit-testable in isolation.
    4. sbatch script differs from baseline by exactly the named levers.

- **ML scientist agent** (`subagent_type: general-purpose`, model: opus)
  - Responsibility: mechanistic plausibility against the known facts.
  - Must answer: *why does this intervention move reflected-RMSD?* — not just *why is current code wrong?*
  - Approval criteria:
    1. Mechanism is specific (names the pathway from change → loss landscape → reflected-RMSD).
    2. Prediction is falsifiable (states what metric moves, how much, by when in training).
    3. Success criterion is genuinely discriminative (reflected-RMSD < 2.7 Å stable at gs ≥ 10K, plus at least one secondary signal that rules out trivial confounds like EMA swap artifacts).
    4. Does not duplicate another approved hypothesis's mechanism.

### Three hypothesis subagents (fresh per hypothesis)

Each subagent is a single general-purpose agent run whose prompt embeds:
- The full text of this spec (attach or inline).
- The full memory file `project_debug_meanflow_1ubq_analysis.md`.
- The approval verdict (what was approved, any revisions required).
- The guardrails (§Guardrails).

Subagents are instructed to pitch one hypothesis they think is most likely to move reflected-RMSD. They must cite mechanism, prediction, and primary/secondary signals. Parallel dispatches naturally isolate them; main agents deduplicate in Round 2.

## Data flow

### Round 1 — Independent pitch (parallel)

Three hypothesis subagents are dispatched in parallel. Each returns a structured pitch:

```
## Hypothesis: <one-liner>
## Mechanism: <2-4 sentences — specific pathway>
## Prediction: <what metric moves, by how much, by gs N>
## Primary signal: reflected-RMSD < 2.7 Å stable at gs ≥ 10K
## Secondary signals: <≥1 signal that rules out a false positive>
## Proposed change: <file:line-level sketch>
## Unit test sketch: <what invariant the test asserts>
## sbatch diff from train_debug.sbatch: <named overrides>
```

### Round 2 — Debate + approval (main agents see all three pitches)

Architect and ML scientist are each dispatched once with all three pitches. Each returns per-pitch rulings:

```
Pitch N: APPROVE | REVISE(<reason + required change>) | REJECT(<reason>)
Dedup notes: <if two pitches overlap, state which survives>
```

Both main agents must independently APPROVE the same pitches. Discrepancy handling:
- APPROVE + REVISE → subagent revises, re-pitches, architect + ML scientist re-rule on just that pitch. **Cap: 1 revision round per pitch.** If still not APPROVE/APPROVE after revision, treat as tiebreaker case.
- APPROVE + REJECT → orchestrator dispatches a **tiebreaker subagent** (`subagent_type: general-purpose`, model: opus) with both rulings, the pitch, and the memory file. Tiebreaker returns a ruling with stated rationale; orchestrator adopts it. No user intervention.
- REVISE + REVISE → orchestrator merges required changes (union of both) and re-dispatches subagent with combined revision list. Then re-rules.
- REVISE + REJECT → treat as REJECT (tiebreaker).
- Both REJECT → subagent given ONE chance to propose a new hypothesis; if the new pitch is also rejected by either main agent, tiebreaker decides whether to keep/drop the slot. **Max 1 restart per slot.**

Target: exactly three approved, deduplicated hypotheses.

### Round 3 — Implement (parallel, one subagent per approved hypothesis)

Orchestrator pre-creates each branch on `main` (`git branch hyp/<slug> main`) and then dispatches an implementation subagent with `Agent` tool's `isolation: "worktree"` parameter pointing at that branch. The subagent inherits a worktree already on `hyp/<slug>`; no branch rename happens. Subagents do not run `git checkout` or `git branch` themselves.

Each approved subagent (fresh dispatch, prompt includes approval + any revisions + branch name) performs:

1. Verify `git rev-parse --abbrev-ref HEAD` equals `hyp/<slug>` before any edits. If not, abort and report to orchestrator.
2. Implement the change. Localized commits with descriptive messages. **No "Co-Authored-By" lines per user memory.**
3. Write unit tests under `tests/`. Tests run with `PYTHONPATH=. pytest tests/<file> -v` per user memory.
4. Run unit tests; iterate until green.
5. Write `scripts/hyp_<slug>.sbatch` (copy of `train_debug.sbatch`, overrides per pitch, plus guardrails).
6. Write `docs/hypotheses/<slug>.md` with the pitch template filled in.
7. Commit locally. Report branch name + test output + sbatch path back to orchestrator.
8. **Orchestrator**, not subagent, pushes the branch to `origin` after Round 4 sign-off.

### Round 4 — Review (main agents review each branch)

Architect and ML scientist are dispatched once per branch with the diff vs. `main` plus the hypothesis doc:

- Architect: code review (localization, test coverage, sbatch parity).
- ML scientist: sbatch success-criterion review (will this run discriminate the hypothesis, or is the predicted signal drowned in noise?).

Both must sign off. If either requests changes, the subagent is re-dispatched with the review. **Cap: max 2 review iterations per branch.** If sign-off not achieved after 2 iterations, orchestrator dispatches tiebreaker subagent (same mechanism as Round 2) for a final ruling; orchestrator adopts it.

Terminal state: three branches on disk, each with commits + tests + sbatch + hypothesis doc, all reviewed, all pushed to `origin`.

## Guardrails (mandatory on every branch)

1. sbatch script sets `opt.accumulate_grad_batches=1` (removes the known no-op). Memory's fix options 2 (`raise datamodule.repeat ≥ 20`) and 3 (`loss_accumulation_steps=K`) are **out of scope** for this sweep — do not pick them even though memory lists them as valid.
2. sbatch script sets eval `nsamples=16`. Relies on existing chirality aggregation at `protein_eval.py:291`; subagent MUST verify that reducer averages `chirality_mf1` rather than majority-voting before relying on it, and include the verification in the hypothesis doc. If the reducer is not an average, subagent must add an averaging emit alongside the existing one (not replace it) — architect sign-off required in Round 2.
3. sbatch script logs `len(train_dataloader)` and per-epoch optimizer-step count at startup.
4. sbatch script keeps `datamodule.repeat=2`, `batch_size=2`, `max_epochs=20000` **unchanged**. Changes to these three knobs are forbidden — any hypothesis that requires different values must be rejected in Round 2 and re-pitched.
5. Branch does NOT modify `accumulate_grad_batches` semantics in code — guardrail #1 handles it via config.
6. Branch must not bundle unrelated changes (e.g., refactors, cosmetic fixes).
7. No `--no-verify` on commits; no force-push; no PR creation (branches stay local + pushed to origin for user).

## Git hygiene

- Orchestrator pre-creates each branch via `git branch hyp/<slug> main`, then dispatches implementation subagents with the `Agent` tool's `isolation: "worktree"` parameter targeting that branch. No rename happens.
- Each subagent works inside its worktree only. No cross-worktree file reads.
- On successful Round 4 sign-off, orchestrator pushes the branch to `origin` with `-u`. Worktrees are kept so user can inspect.
- Commits use normal `git commit -m` (no Claude co-author trailer, per user memory).

## Testing

### Unit tests per hypothesis

Each subagent writes at least one unit test that:
- Is runnable via `PYTHONPATH=. pytest tests/<file> -v`.
- Directly exercises the new code path (not just "model.forward still runs").
- Asserts the mechanistic invariant the hypothesis relies on (example: "with chirality scaled by `(loss_mf.detach()+eps)^norm_p`, the ratio chirality/mf in the total-loss gradient is independent of `norm_p`" — this is a concrete assertion, not a smoke test).

### Sbatch success criterion review (ML scientist, Round 4)

For each sbatch, ML scientist checks:
- Does the script log the hypothesis's secondary signal?
- Is the run length sufficient for the predicted effect to show (≥10K optim steps = epochs)?
- Would the predicted metric delta be visible above the run-to-run noise floor?

## Known candidate hypotheses (seed list — subagents are NOT forced to pick from this, but it frames what "plausible" means)

From memory §4 (remaining causes after LR-peak refutation) and §6 (loss formulation changes):

- **H-A (pin-chirality-scaling):** Put chirality term inside `adaptive_loss`, or multiply chirality by `(loss_mf.detach()+eps)^norm_p`. Mechanism: preserves chirality/MF ratio under norm_p>0, preventing chirality from being crushed as MF drops. Prediction: at norm_p=1, chirality gradient no longer vanishes; reflected-RMSD separates from mirror plateau by gs 5K.
- **H-B (introduce a t-gate for chirality and bias toward large t):** Currently the chirality hinge runs at whatever `t_ext` is sampled for the MeanFlow batch (no dedicated t-gate exists at `model_trainer_base.py:370-387`). Add a `chirality.t_min` config (e.g., 0.5) so the hinge only fires when `t_ext >= t_min`, OR resample a separate large-t for chirality. Mechanism: handedness is decided at large t (near noise); at small t, `x_1_pred ≈ x_1` and the hinge has near-zero gradient. **This is a scope expansion**: subagent must add the config key and the gate together in one localized change.
- **H-C (global signed-volume term):** Add a global signed-volume term (not per-window margin). Mechanism: fixes the near-planar margin collapse and gives a basin-global chirality gradient.
- **H-D (eval on clean MF prediction):** Change eval to use clean MF prediction (r=0 path) and increase `nsamples ≥ 16`. Mechanism: reduces eval variance; the plateau may partly be a measurement artifact, not a training artifact.
- **H-E (2-pass self-cond at inference):** Run inference with one warmup pass feeding `x_sc` into a refinement pass. Mechanism: fixes train/infer `x_sc` distribution shift (train sees warmup-noisy `x_sc`, infer sees zeros).
- **H-F (fixed-scale margin):** Replace `α·|T_gt|` margin with a fixed scale independent of `|T_gt|`. Mechanism: avoids margin collapse on near-planar CA windows.

These are seeds. Subagents may propose others if they cite a mechanism grounded in the memory file.

## Orchestration (what I do)

1. Dispatch a **spec-review subagent** (`subagent_type: general-purpose`, model: opus) with this spec and the memory file. Returns: ambiguities, contradictions, or gaps that would cause downstream agents to drift. Apply fixes inline.
2. **Freeze spec version.** After applying spec-review fixes, commit and capture the commit SHA. All downstream agent prompts (Round 1–4) cite this SHA and are told: "if you read a different version of this file, report back and stop." If later edits are required, abort and restart Round 1.
3. Dispatch 3 hypothesis subagents in parallel (Round 1).
4. Dispatch architect + ML scientist in parallel with all three pitches (Round 2).
5. Reconcile verdicts; tiebreaker subagent for any split; loop REVISE slots (max 1 revision, max 1 restart per slot).
6. Pre-create `hyp/<slug>` branches on main; dispatch 3 implementation subagents in parallel (Round 3), each in its own worktree.
7. Dispatch architect + ML scientist to review each branch (Round 4, parallelizable per branch; max 2 review iterations per branch before tiebreaker).
8. Push branches to origin with `-u`.
9. Report: three branch names + sbatch paths + hypothesis docs.

## Risks and mitigations

- **Main-agent disagreement stalls progress.** → Tiebreaker subagent (Round 2). No user intervention required.
- **Subagents duplicate hypotheses.** → Round 2 dedup by main agents. If dedup collapses slots below 3, orchestrator re-dispatches to fill.
- **Subagent writes tests that smoke-test but do not assert the mechanism.** → ML scientist's Round 4 review explicitly checks the test asserts the mechanistic invariant, not just "doesn't crash."
- **Worktrees pile up if orchestration aborts mid-flow.** → Orchestrator tracks worktree names and cleans up on explicit user ask; otherwise `keep` so no work is lost.
- **sbatch scripts drift from the baseline shape.** → Architect's Round 4 review diffs against `train_debug.sbatch`.

## Success criteria (for this spec, not for the hypotheses)

- All three branches exist, have passing unit tests, have sbatch scripts that pass architect review, have hypothesis docs.
- User can run any sbatch script with a single `sbatch scripts/hyp_<slug>.sbatch` invocation.
- No branch bundles more than one hypothesis.
- The hypothesis docs together form a coherent set — no two hypotheses test the same mechanism.
