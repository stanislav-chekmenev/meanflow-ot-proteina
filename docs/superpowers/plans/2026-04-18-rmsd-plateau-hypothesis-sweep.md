# MeanFlow RMSD Plateau Hypothesis Sweep — Orchestration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Orchestrate a multi-agent flow that produces three `hyp/<slug>` branches off `main`, each containing one focused intervention aimed at breaking the 2.7 Å reflected-RMSD plateau, with unit tests and a runnable sbatch script per branch. The spec for this plan is `docs/superpowers/specs/2026-04-18-rmsd-plateau-hypothesis-sweep-design.md` (frozen at commit `07e5b14`).

**Architecture:** Orchestrator (this Claude session) dispatches one spec-review subagent (already done), three hypothesis-pitch subagents, an architect main agent, an ML scientist main agent, an optional tiebreaker subagent, three implementation subagents (in isolated worktrees), and the two main agents again for final review. All downstream prompts cite the frozen spec SHA and cap iteration counts per the spec.

**Tech Stack:** `Agent` tool (general-purpose, opus), `Agent` tool's `isolation: "worktree"` parameter, git worktrees, pytest (run with `PYTHONPATH=. pytest -v`), Slurm sbatch based on `scripts/train_debug.sbatch`.

---

## Agent Permission Scopes

Enforced at prompt level — each dispatched agent receives explicit "you may / you may not" rules in its prompt. The `Agent` tool does not support per-dispatch tool allowlists, so scope enforcement is behavioural (the prompt forbids violations) plus structural (reviewer agents operate outside worktrees so they have nothing to modify that matters).

**Reviewer agents (READ-ONLY):**
- Spec-review subagent (already ran).
- Architect main agent (Round 2 + Round 4).
- ML Scientist main agent (Round 2 + Round 4).
- Tiebreaker subagent (when dispatched).

Reviewer agents are explicitly prompted: *"You are READ-ONLY. Do not run Edit, Write, or NotebookEdit. Do not run any `git` command that writes state (commit, push, branch, checkout, merge, rebase, reset, add). Read and Grep and reading `git log`/`git diff`/`git status` are permitted. If you need to suggest a change, return it as bullet text in your ruling — do not apply it yourself."*

**Implementation subagents (WRITE within worktree only):**
- Three Round-3 implementation subagents.
- Round 3.5 fixup re-dispatches (Task 9.5).

Implementation subagents are prompted: *"You have write access ONLY within your assigned worktree on branch hyp/{SLUG}. You MUST NOT: commit to main, checkout main, merge to main, push to origin (orchestrator handles push after review), modify the spec or plan files. You MAY: edit code, add/remove files, create commits on hyp/{SLUG}, run tests, read any file in the repo."*

**Orchestrator (this Claude session) — privileged:**
- Creates branches off main (read-only use of main).
- Pushes approved branches to origin after Round 4 sign-off.
- Never commits directly to main during this plan.
- Never merges hypothesis branches into main during this plan (integration is a separate user decision).

**`main` branch protection (enforced by orchestrator + subagent prompts):**
- No task in this plan commits to `main`. The only writes to `main` in this plan are the spec commit + this plan commit, both done during brainstorming/writing-plans before agent dispatch begins.
- Task 1 step 2 verifies working tree is clean on `main` before dispatch starts.
- Implementation subagents work in worktrees on `hyp/<slug>`, not on `main`.
- If any subagent reports a git state with HEAD = main, orchestrator aborts that subagent's work immediately.

---

## File Structure

The plan creates orchestration state (mostly in-memory / in-prompts) and four types of artifacts on disk:

**Orchestration tracking (this session only):**
- Pitches from Round 1 — in this session's context, passed verbatim to Round 2.
- Round 2 rulings — in this session's context, passed to Round 3.
- Round 4 review outputs — in this session's context, gate on push.

**Per-hypothesis artifacts (one set per approved hypothesis, on branch `hyp/<slug>`):**
- `docs/hypotheses/<slug>.md` — hypothesis doc (one-liner, mechanism, prediction, signals).
- `scripts/hyp_<slug>.sbatch` — runnable sbatch (copy of `train_debug.sbatch` with per-hypothesis overrides + guardrails).
- `tests/test_hyp_<slug>.py` — unit test asserting the mechanistic invariant.
- Code changes localized to whatever files the hypothesis requires (always documented in the hypothesis doc).

**One shared change applied to every branch before the hypothesis diff:**
- `proteinfoundation/callbacks/protein_eval.py` — expose `nsamples` as a config-driven parameter (currently hardcoded to 1 at `:265, :269`). This is the guardrail-#2 enablement; doing it once per branch rather than in a separate PR is a deliberate scope choice to keep each branch self-contained.

---

## Task 1: Confirm frozen spec SHA and baseline state

**Files:**
- Read: `docs/superpowers/specs/2026-04-18-rmsd-plateau-hypothesis-sweep-design.md`

- [ ] **Step 1: Verify spec commit SHA matches frozen value**

Run:
```bash
git log -1 --format="%H" -- docs/superpowers/specs/2026-04-18-rmsd-plateau-hypothesis-sweep-design.md
```

Expected output: `07e5b14dc603b4c9e4c9ecef8f1ba29c7eb73748`

If SHA differs, the spec was edited after freeze — abort and return to brainstorming to re-freeze.

- [ ] **Step 2: Verify branch is `main` and working tree is clean**

Run:
```bash
git rev-parse --abbrev-ref HEAD
git status --short
```

Expected: branch `main`, no output from `git status --short` (or only untracked files that are irrelevant).

- [ ] **Step 3: Verify `train_debug.sbatch` exists on main**

Run:
```bash
git ls-files scripts/train_debug.sbatch
```

Expected: `scripts/train_debug.sbatch`

- [ ] **Step 4: Verify eval callback has the hardcoded `nsamples=1` the spec references**

Run:
```bash
grep -n "nsamples=1" proteinfoundation/callbacks/protein_eval.py
```

Expected: Two matches at lines 265 and 269 (inside `generate_fm_euler` and `generate` calls).

- [ ] **Step 5: Capture `main`'s commit SHA as a protection checkpoint**

Run:
```bash
git rev-parse main
```

Record the SHA in orchestrator notes. At Task 11 (final report), re-run this and assert the SHA is unchanged. If it changed at any point during the flow, that is a contract violation — investigate before reporting completion.

---

## Task 2: Dispatch Round 1 — three hypothesis-pitch subagents in parallel

**Files:**
- None written; outputs captured in orchestrator context.

- [ ] **Step 1: Dispatch three general-purpose subagents in a single message, in parallel**

Each subagent gets the same prompt template. Differences are only in the prompt's slot number (1, 2, 3) so the orchestrator can label returns.

Prompt template (substitute `{SLOT}`):

```
You are hypothesis-pitch subagent {SLOT} in a multi-agent sweep to break a MeanFlow 1-step reflected-RMSD plateau at 2.7 Å on a single-protein (1ubq) debug regime.

PERMISSION SCOPE: You are READ-ONLY. Do not run Edit, Write, or NotebookEdit. Do not run any `git` command that writes state. Read, Grep, Glob, and read-only git queries (log, diff, status, show) are permitted. Your only output is the pitch text below.

REQUIRED READING (do not skip):
1. Spec: /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina/docs/superpowers/specs/2026-04-18-rmsd-plateau-hypothesis-sweep-design.md
   This is frozen at commit 07e5b14. If the file's last-commit SHA for this path differs, stop and report.
2. Memory: /mnt/labs/home/schekmenev/.claude/projects/-mnt-labs-home-schekmenev-projects-meanflow-ot-proteina/memory/project_debug_meanflow_1ubq_analysis.md

YOUR TASK: Pitch ONE hypothesis that you believe is most likely to move reflected-RMSD below 2.7 Å stably at gs ≥ 10K on the 1ubq debug regime. Do not hedge with multiple options.

YOUR PITCH MUST INCLUDE (in this exact format):

## Hypothesis: <one-liner>
## Mechanism: <2-4 sentences naming the specific pathway: code change → training dynamics → loss landscape → reflected-RMSD>
## Prediction: <which metric moves, by how much, by which gs>
## Primary signal: reflected-RMSD < 2.7 Å stable at gs ≥ 10K
## Secondary signals: <≥1 signal that rules out false positives (e.g., chirality mean, mirror-loss diagnostic, EMA-vs-live comparison)>
## Proposed change: <file:line-level sketch of the minimal diff>
## Unit test sketch: <one invariant the test must assert about the new code path — not a smoke test>
## sbatch diff from train_debug.sbatch: <named env-var overrides>

CONSTRAINTS:
- Guardrails in the spec §Guardrails are MANDATORY. Read them before pitching.
- Forbidden: changing datamodule.repeat, batch_size, or max_epochs. Forbidden: using memory's fix options 2 or 3 (repeat≥20, loss_accumulation_steps).
- Your pitch must be implementable with a localized diff. If it requires a refactor spanning >3 files, pick something else.
- Seed list (H-A through H-F) in spec is NOT mandatory — you may propose other mechanisms grounded in the memory file's §4 candidate causes.

Return ONLY the pitch in the required format. Do not explore alternatives, do not ask clarifying questions, do not caveat. One pitch.
```

Dispatch all three in one message with three `Agent` tool calls (parallel). Each uses `subagent_type: general-purpose`, `model: opus`.

- [ ] **Step 2: Collect the three pitches verbatim**

Copy each returned pitch into orchestrator notes labeled `Pitch 1`, `Pitch 2`, `Pitch 3`. If any subagent returned something other than the required format, re-dispatch that slot with the same prompt plus "your last return did not match the required format; return only the template filled in."

---

## Task 3: Dispatch Round 2 — architect and ML scientist review all three pitches in parallel

**Files:**
- None written.

- [ ] **Step 1: Dispatch architect agent and ML scientist agent in parallel (single message, two Agent calls)**

Architect prompt:

```
You are the ARCHITECT main agent in a multi-agent hypothesis sweep. You and the ML Scientist agent independently rule on three pitches; both must APPROVE for a pitch to advance.

PERMISSION SCOPE: You are READ-ONLY. Do not run Edit, Write, or NotebookEdit. Do not run any `git` command that writes state (commit, push, branch, checkout, merge, rebase, reset, add, stash). You may run: Read, Grep, Glob, Bash for read-only git queries (log, diff, status, show, ls-files). If you need to suggest a change, return it as bullet text in your ruling — do not apply it.

REQUIRED READING:
1. Spec (frozen @07e5b14): /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina/docs/superpowers/specs/2026-04-18-rmsd-plateau-hypothesis-sweep-design.md
2. Memory: /mnt/labs/home/schekmenev/.claude/projects/-mnt-labs-home-schekmenev-projects-meanflow-ot-proteina/memory/project_debug_meanflow_1ubq_analysis.md
3. The codebase at /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina (currently on branch main).

YOUR APPROVAL CRITERIA (spec §Architecture/Architect):
1. Change is localized (no incidental refactors; ≤3 files touched by the hypothesis diff, excluding the shared nsamples-exposure change and test file).
2. Does not silently alter another knob the analysis depends on (chirality aggregation, adaptive-loss math unless explicitly the hypothesis, JVP closure, EMA semantics).
3. Unit-testable in isolation — the proposed unit test asserts a concrete mechanistic invariant, not "doesn't crash."
4. sbatch script differs from baseline by exactly the named env-var overrides plus guardrails.

GROUND YOUR REVIEW in the file:line anchors from spec §Architecture/Architect.

THREE PITCHES FOLLOW. For each, return:

Pitch N: APPROVE | REVISE(<reason + required change as bullet list>) | REJECT(<reason>)
Dedup notes: <if two pitches overlap mechanistically, state which survives and which does not>

Return only the rulings. Do not ask questions.

---
<PASTE Pitch 1 verbatim>
---
<PASTE Pitch 2 verbatim>
---
<PASTE Pitch 3 verbatim>
---
```

ML scientist prompt (same structure, swap role):

```
You are the ML SCIENTIST main agent in a multi-agent hypothesis sweep. You and the Architect agent independently rule on three pitches; both must APPROVE for a pitch to advance.

PERMISSION SCOPE: You are READ-ONLY. Do not run Edit, Write, or NotebookEdit. Do not run any `git` command that writes state. You may run: Read, Grep, Glob, Bash for read-only git queries (log, diff, status, show). If you want to suggest a change, return it as bullet text in your ruling.

REQUIRED READING:
1. Spec (frozen @07e5b14): /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina/docs/superpowers/specs/2026-04-18-rmsd-plateau-hypothesis-sweep-design.md
2. Memory: /mnt/labs/home/schekmenev/.claude/projects/-mnt-labs-home-schekmenev-projects-meanflow-ot-proteina/memory/project_debug_meanflow_1ubq_analysis.md

YOUR APPROVAL CRITERIA (spec §Architecture/ML scientist):
1. Mechanism is specific — names the pathway from code change → loss landscape → reflected-RMSD.
2. Prediction is falsifiable — states what metric moves, how much, by which gs.
3. Success criterion is discriminative — reflected-RMSD < 2.7 Å stable at gs ≥ 10K, plus ≥1 secondary signal that rules out trivial confounds (EMA swap, chirality 1-bit noise, FM loss O(3)-invariance).
4. No mechanistic duplication with another approved hypothesis.

KEY FACTS TO CHECK AGAINST (from memory):
- 1 optim step = 1 epoch on this regime (accumulate=1 guardrail enforces this).
- FM loss is O(3)-invariant on single chiral protein — reflected-RMSD is the ONLY discriminative RMSD metric.
- Eval chirality is 1-bit Bernoulli at nsamples=1 unless the branch raises nsamples (all branches do, per guardrail #2).
- LR peaks at epoch 1000, not 10K. Late-training degradation is NOT LR-scheduler-driven.

THREE PITCHES FOLLOW. For each, return:

Pitch N: APPROVE | REVISE(<reason + required change as bullet list>) | REJECT(<reason>)
Dedup notes: <if two pitches test the same mechanism, state which survives>

Return only the rulings.

---
<PASTE Pitch 1 verbatim>
---
<PASTE Pitch 2 verbatim>
---
<PASTE Pitch 3 verbatim>
---
```

- [ ] **Step 2: Reconcile verdicts per pitch**

For each pitch, cross-tabulate the architect and ML-scientist rulings. Apply the discrepancy table from spec §Data flow/Round 2:
- APPROVE + APPROVE → advance.
- APPROVE + REVISE → Task 4 (revision loop).
- APPROVE + REJECT → Task 5 (tiebreaker).
- REVISE + REVISE → Task 4 (merge revisions).
- REVISE + REJECT → treat as REJECT → Task 5 (tiebreaker).
- REJECT + REJECT → Task 6 (restart slot).

Record the decision table in orchestrator notes.

---

## Task 4: Revision loop (only if any pitch received REVISE)

**Files:**
- None written.

- [ ] **Step 1: For each revision-requiring pitch, re-dispatch the original hypothesis subagent**

Prompt:

```
You are hypothesis-pitch subagent {SLOT} returning for revision. Your original pitch received the following required changes:

<combined architect revision bullets + ML scientist revision bullets — if REVISE+REVISE, merge both lists>

Revise your pitch to address ALL required changes. Do not change the mechanism unless a revision explicitly requires it. Return the revised pitch in the same template as before. One revision only — this is the cap.

Original pitch for your reference:
<paste original pitch verbatim>
```

- [ ] **Step 2: Re-dispatch Round 2 rulings for revised pitches only**

Same Round 2 prompt structure as Task 3, but include only the revised pitches. Both main agents must now rule APPROVE for the pitch to advance. Any remaining REVISE or REJECT → Task 5 (tiebreaker). **Cap enforced: this is the final revision round.**

---

## Task 5: Tiebreaker (only if any pitch has APPROVE+REJECT or unresolved post-revision)

**Files:**
- None written.

- [ ] **Step 1: Dispatch tiebreaker subagent per contested pitch**

Prompt:

```
You are a TIEBREAKER subagent resolving a disagreement between an Architect agent and an ML Scientist agent on hypothesis pitch {SLOT}.

PERMISSION SCOPE: You are READ-ONLY. Do not write, edit, or modify any file. Do not run any `git` command that writes state. You may Read, Grep, Glob, and run read-only git queries.

REQUIRED READING:
1. Spec (frozen @07e5b14): /mnt/labs/home/schekmenev/projects/meanflow-ot-proteina/docs/superpowers/specs/2026-04-18-rmsd-plateau-hypothesis-sweep-design.md
2. Memory: /mnt/labs/home/schekmenev/.claude/projects/-mnt-labs-home-schekmenev-projects-meanflow-ot-proteina/memory/project_debug_meanflow_1ubq_analysis.md

THE PITCH:
<paste pitch verbatim>

ARCHITECT RULING:
<paste architect ruling verbatim>

ML SCIENTIST RULING:
<paste ML scientist ruling verbatim>

YOUR TASK: Return exactly one of:
- APPROVE: <rationale citing which main agent's concerns are overridden and why>
- REJECT: <rationale citing which main agent's concerns are decisive and why>

You MUST cite specific spec §s or memory findings in your rationale. Do not invent new criteria.
```

Dispatch `subagent_type: general-purpose`, `model: opus`.

- [ ] **Step 2: Adopt tiebreaker rulings**

Record final verdict per contested pitch. APPROVE → advance. REJECT → Task 6 (restart slot).

---

## Task 6: Restart rejected slot (max 1 restart per slot)

**Files:**
- None written.

- [ ] **Step 1: For each rejected slot, re-dispatch Round 1 with a blocklist**

Prompt adds:

```
Your previous pitch was rejected with this reasoning:
<paste final rejection rationale>

Do NOT re-propose the same mechanism. Pick a different intervention from the spec's seed list or a memory-grounded alternative. Same format requirements as before.
```

- [ ] **Step 2: Re-run Round 2 on new pitch**

Same as Task 3 for just the restarted slot. If this second pitch also fails to achieve APPROVE+APPROVE after Round 2 + optional revision + optional tiebreaker, the slot is **dropped**. Final count may be 2 hypotheses instead of 3 — acceptable under spec §Risks. Do not initiate a second restart.

---

## Task 7: Pre-create branches for all approved hypotheses

**Files:**
- No file writes; git state only.

- [ ] **Step 1: For each approved hypothesis, assign a slug**

Slug format: short-kebab-case derived from the hypothesis one-liner. Examples: `pin-chir-scaling`, `t-gate-chir`, `clean-eval-nsamples`. Record the slug → hypothesis mapping in orchestrator notes.

- [ ] **Step 2: Create each branch from main**

Run (one command per hypothesis):
```bash
git branch hyp/<slug> main
```

- [ ] **Step 3: Verify branches exist and point at main**

Run:
```bash
git branch --list 'hyp/*' -v
```

Expected: one line per hypothesis, each showing the `main` commit SHA.

---

## Task 8: Dispatch Round 3 — implementation subagents in parallel worktrees

**Files per branch (created by subagent):**
- Create: `docs/hypotheses/<slug>.md`
- Create: `scripts/hyp_<slug>.sbatch`
- Create: `tests/test_hyp_<slug>.py`
- Modify: `proteinfoundation/callbacks/protein_eval.py` (expose `nsamples`)
- Modify: whatever the hypothesis touches (documented in the hypothesis doc)

- [ ] **Step 1: Dispatch one implementation subagent per approved hypothesis, in parallel**

Each uses `Agent` with `isolation: "worktree"` and `subagent_type: general-purpose`, `model: opus`. The worktree is created on the pre-existing `hyp/<slug>` branch.

Prompt per subagent (substitute `{SLUG}`, `{PITCH}`, `{REVISIONS}`):

```
You are implementation subagent for hypothesis `{SLUG}` in a multi-agent sweep. You are running inside an isolated git worktree already checked out to branch `hyp/{SLUG}` off `main`.

PERMISSION SCOPE: You have WRITE access ONLY on branch `hyp/{SLUG}` inside your assigned worktree.

You MUST NOT:
- Commit to `main`. Your first action verifies HEAD is `hyp/{SLUG}`; if it is `main`, abort immediately and report.
- Run `git checkout main`, `git checkout <any-other-branch>`, `git switch`, `git merge`, `git rebase`, `git reset --hard`, or `git push` at any point.
- Modify the spec file `docs/superpowers/specs/2026-04-18-rmsd-plateau-hypothesis-sweep-design.md` or this plan file `docs/superpowers/plans/2026-04-18-rmsd-plateau-hypothesis-sweep.md`.
- Create or delete branches other than work on `hyp/{SLUG}`.
- Push to `origin`. Orchestrator handles push after Round 4 sign-off.
- Use `--no-verify` or skip hooks.
- Add `Co-Authored-By` or any Claude-attribution trailer to commits.

You MAY:
- Read any file in the repo.
- Edit, Write, or delete files that the hypothesis change touches, within your worktree.
- Run `git add`, `git commit` (on `hyp/{SLUG}` only), `git status`, `git diff`, `git log`.
- Run tests and any read-only shell command.

REQUIRED READING:
1. Spec (frozen @07e5b14): docs/superpowers/specs/2026-04-18-rmsd-plateau-hypothesis-sweep-design.md
2. Memory (path-reference, not a file in repo): /mnt/labs/home/schekmenev/.claude/projects/-mnt-labs-home-schekmenev-projects-meanflow-ot-proteina/memory/project_debug_meanflow_1ubq_analysis.md
3. Baseline sbatch you will copy: scripts/train_debug.sbatch

APPROVED PITCH:
{PITCH}

REVISIONS REQUIRED (may be empty):
{REVISIONS}

YOUR TASKS (execute in order):

1. Verify you are on the correct branch. Run: git rev-parse --abbrev-ref HEAD. Expected: hyp/{SLUG}. If not, STOP and report.

2. Apply the shared guardrail-#2 enablement: expose `nsamples` as a parameter in protein_eval.py (currently hardcoded to 1 at :265, :269). Thread it up through the callback constructor and through the eval config. This is a SHARED change — keep it minimal and clean, same in every branch.

3. Implement the hypothesis change per the approved pitch. Localize the diff. Do NOT refactor beyond what the pitch requires.

4. Write a unit test at `tests/test_hyp_{SLUG}.py` that asserts the mechanistic invariant stated in the pitch's "Unit test sketch" field. The test MUST:
   - Be runnable with: `PYTHONPATH=. pytest tests/test_hyp_{SLUG}.py -v`
   - Directly exercise the new code path (not `model.forward(...)` smoke).
   - Fail if the hypothesis change is reverted — verify this by temporarily reverting your implementation, running the test (should FAIL), then restoring the implementation (should PASS). Include the FAIL-then-PASS evidence in your report.

5. Run the full test file. Iterate until all tests pass.

6. Write `scripts/hyp_{SLUG}.sbatch` by copying `scripts/train_debug.sbatch` and applying:
   - `ACCUMULATE_GRAD_BATCHES=1` (guardrail #1)
   - `EVAL_NSAMPLES=16` — new env var; wire it into the --exp_overrides block as `eval.nsamples=$EVAL_NSAMPLES` (add the config key if not already present in training_ca_debug.yaml).
   - Keep `DATAMODULE_REPEAT=2`, `DATAMODULE_BATCH_SIZE=2`, `MAX_EPOCHS=20000` unchanged (guardrail #4).
   - Change `EXP_NAME` to match hypothesis: `hyp-{SLUG}`.
   - Apply your pitch's named env-var overrides.
   - Add a startup echo block logging `len(train_dataloader)` and per-epoch optimizer-step count. Prefer wiring this into a Lightning callback or a pre-run Python snippet rather than a shell hack.
   - `--job-name=hyp_{SLUG}` and `--output=logs/hyp_{SLUG}_%j.out`.

7. Write `docs/hypotheses/{SLUG}.md` filling in the pitch template verbatim plus:
   - **Primary file:line anchors the change touches** (so the architect can review quickly).
   - **How to rule out false positives** — concrete check using the secondary signal.
   - **Verification that `protein_eval.py`'s chirality reducer averages rather than majority-votes** (guardrail #2 requirement). Check by reading the reducer code and quoting it in the doc.

8. Commit in logical chunks: (a) shared nsamples exposure, (b) hypothesis implementation, (c) tests, (d) sbatch + hypothesis doc. Messages use conventional imperative style. NO Co-Authored-By lines. NO --no-verify.

9. Return a report with:
   - Final branch name.
   - List of commits (git log main..HEAD --oneline).
   - Test output (last pytest run).
   - Path to sbatch script.
   - FAIL-then-PASS evidence for the test (quoted pytest output under both states).
   - One-line summary of the hypothesis doc.

CONSTRAINTS:
- Do NOT submit sbatch (`sbatch ...`). Never.
- Do NOT push to origin. Orchestrator pushes after review.
- Do NOT merge to main, do NOT create PR.
- Do NOT modify scripts/train_debug.sbatch itself — copy it.
- Do NOT modify accumulate_grad_batches semantics in code — the sbatch env-var handles it.
- Do NOT change datamodule.repeat, batch_size, or max_epochs.
```

- [ ] **Step 2: Collect reports from all three implementation subagents**

Record each subagent's return (branch, commits, test output, sbatch path, doc summary) in orchestrator notes. Do not proceed to Round 4 until all three return successfully.

---

## Task 9: Dispatch Round 4 — architect and ML scientist review each branch in parallel

**Files:**
- None written by this task (but may trigger subagent re-dispatch that writes fixups).

- [ ] **Step 1: For each branch, dispatch architect code review and ML scientist sbatch review in parallel (single message per branch)**

Architect prompt (one per branch):

```
You are the ARCHITECT main agent reviewing branch hyp/{SLUG} before it is pushed to origin.

PERMISSION SCOPE: You are READ-ONLY. Do not Edit, Write, or modify files. Do not commit, push, or run any writing git command. You may Read, Grep, Glob, and run `git log`, `git diff`, `git show` against branch hyp/{SLUG}.

REQUIRED READING:
1. Spec (frozen @07e5b14): docs/superpowers/specs/2026-04-18-rmsd-plateau-hypothesis-sweep-design.md
2. Branch diff: git diff main..hyp/{SLUG}
3. Hypothesis doc: docs/hypotheses/{SLUG}.md on branch hyp/{SLUG}
4. Test output provided below: <paste subagent's pytest evidence>

YOUR CHECKS:
1. Change is localized — count files touched outside the shared nsamples exposure. ≤3 is the cap.
2. No silent alteration to other knobs (adaptive-loss math, JVP closure, chirality reducer, EMA, optimizer schedule unless explicitly the hypothesis).
3. The unit test asserts a mechanistic invariant, not a smoke. Verify FAIL-then-PASS evidence is concrete.
4. sbatch diff vs scripts/train_debug.sbatch contains ONLY the pitch's named overrides + guardrail #1 (accumulate=1) + nsamples=16 + startup logging + job-name/output changes.
5. No --no-verify, no Co-Authored-By, no PR created.

RETURN:
APPROVE | REVISE(<bullet list of required fixes>) | REJECT(<reason>)

REJECT should be rare — use REVISE for fixable issues.
```

ML scientist prompt (one per branch):

```
You are the ML SCIENTIST main agent reviewing sbatch script scripts/hyp_{SLUG}.sbatch on branch hyp/{SLUG} before it is pushed.

PERMISSION SCOPE: You are READ-ONLY. Do not Edit, Write, or modify files. Do not run writing git commands. You may Read, Grep, Glob, and run read-only git queries against branch hyp/{SLUG}.

REQUIRED READING:
1. Spec (frozen @07e5b14): docs/superpowers/specs/2026-04-18-rmsd-plateau-hypothesis-sweep-design.md
2. Memory: /mnt/labs/home/schekmenev/.claude/projects/-mnt-labs-home-schekmenev-projects-meanflow-ot-proteina/memory/project_debug_meanflow_1ubq_analysis.md
3. The sbatch: scripts/hyp_{SLUG}.sbatch on branch hyp/{SLUG}
4. The hypothesis doc: docs/hypotheses/{SLUG}.md on branch hyp/{SLUG}

YOUR CHECKS (spec §Testing/Sbatch success criterion review):
1. Does the script log the hypothesis's secondary signal? If not, REVISE.
2. Is the run length (20K epochs, accumulate=1 → 20K optim steps) sufficient for the predicted effect to show?
3. Is the predicted metric delta visible above the noise floor at nsamples=16?
4. Does the hypothesis doc verify the chirality reducer averages rather than majority-votes?
5. Does the doc list a concrete false-positive ruling check using the secondary signal?

RETURN:
APPROVE | REVISE(<bullet list of required fixes>) | REJECT(<reason>)
```

Dispatch both in parallel per branch. Three branches × 2 agents = 6 calls total, but can be split across 1-3 messages.

- [ ] **Step 2: Reconcile Round 4 verdicts per branch**

- Both APPROVE → branch is ready for push (Task 10).
- Any REVISE → Task 9.5 (revision loop, max 2 iterations per branch).
- Any REJECT → Task 9.5, treat as REVISE; if REJECT persists after 2 iterations, tiebreaker per spec §Data flow/Round 4.

---

## Task 9.5: Round 4 revision loop (only if any branch received REVISE/REJECT)

**Files:**
- None written by this task; subagent writes fixup commits.

- [ ] **Step 1: Re-dispatch the implementation subagent for each REVISE branch**

The subagent is a fresh dispatch but the worktree already has the branch state. Prompt includes the combined REVISE bullet list and instructs:

```
Your branch hyp/{SLUG} received the following required fixes from Round 4 review. Apply them, re-run tests, commit fixups, and report back.

PERMISSION SCOPE: Same as Round 3 implementation subagent. WRITE access ONLY on `hyp/{SLUG}` inside your worktree. MUST NOT commit to main, checkout main, push, merge, or modify spec/plan files. May Read anything; Edit/Write only within this hypothesis's files. Add fixup commits only — do not rebase or squash.

ARCHITECT REVISIONS (may be empty):
<paste>

ML SCIENTIST REVISIONS (may be empty):
<paste>

Apply all revisions. Do NOT change the core hypothesis. Do NOT rebase or squash — add fixup commits. Run tests after fixups. Report back with the new commit list and test output.
```

- [ ] **Step 2: Re-run Task 9 for revised branches**

Cap: max 2 revision iterations per branch. If a third would be required, dispatch tiebreaker subagent (same prompt pattern as Task 5) with the branch diff, both main-agent rulings, and the memory file; adopt tiebreaker ruling.

---

## Task 10: Push approved branches to origin

**Files:**
- None written; git network only.

- [ ] **Step 1: For each approved branch, push with `-u`**

Run (one command per approved branch):
```bash
git push -u origin hyp/<slug>
```

Expected: "new branch hyp/<slug> -> hyp/<slug>" with upstream set.

If the user has not authorized push in this session, ask first and list the branches that will be pushed. User's default instruction ("Executing actions with care") applies — confirm before push.

- [ ] **Step 2: Verify all branches pushed**

Run:
```bash
git branch -vv | grep 'hyp/'
```

Expected: each line shows `[origin/hyp/<slug>]`.

---

## Task 11: Final report to user

**Files:**
- None written; text-only summary.

- [ ] **Step 1: Produce concise summary report**

Format:

```
Hypothesis sweep complete. Spec: docs/superpowers/specs/2026-04-18-rmsd-plateau-hypothesis-sweep-design.md (frozen @07e5b14)

Branches produced (N):
1. hyp/<slug-1> — <one-liner>. Run: sbatch scripts/hyp_<slug-1>.sbatch. Doc: docs/hypotheses/<slug-1>.md
2. hyp/<slug-2> — <one-liner>. Run: sbatch scripts/hyp_<slug-2>.sbatch. Doc: docs/hypotheses/<slug-2>.md
3. hyp/<slug-3> — <one-liner>. Run: sbatch scripts/hyp_<slug-3>.sbatch. Doc: docs/hypotheses/<slug-3>.md

Each branch contains: hypothesis implementation + unit tests + sbatch + hypothesis doc.
All branches pushed to origin. All tests pass locally.

Total agent dispatches: <count>. Revisions triggered: <count>. Tiebreakers triggered: <count>.
```

If fewer than 3 branches survived (per Task 6's drop rule), report honestly — do not pad.

- [ ] **Step 1.5: Verify `main` commit SHA is unchanged from Task 1 step 5**

Run:
```bash
git rev-parse main
```

Expected: matches the SHA captured at Task 1 step 5. If it differs, a contract violation occurred — investigate the divergent commit (`git log <old-sha>..main`) before declaring success.

- [ ] **Step 2: Remind user of guardrails for running**

```
Guardrails applied to every sbatch (do not unset):
- accumulate_grad_batches=1 (memory's known PL 2.5 no-op bug)
- eval nsamples=16 (breaks the ±1 chirality 1-bit metric)
- datamodule.repeat=2, batch_size=2, max_epochs=20000 unchanged across branches

Primary success signal: reflected-RMSD < 2.7 Å stable at trainer/global_step ≥ 10K.
Each hypothesis doc names its secondary signal for false-positive ruling.
```

---

## Self-Review Checklist

Running inline before handoff:

**1. Spec coverage:**
- Spec §Goal (3 branches with code/tests/sbatch/doc) → Tasks 2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10 ✓
- Spec §Guardrails (all 7 items) → referenced in Task 8 prompt + Round 4 checks ✓
- Spec §Git hygiene (pre-create branch, worktree, no subagent checkout, orchestrator pushes) → Tasks 7, 8, 10 ✓
- Spec §Architecture/roles (architect + ML scientist + 3 pitch + 3 impl + tiebreaker) → all dispatched ✓
- Spec §Data flow/Round 2 discrepancy table (all 6 cases) → Task 3 step 2 + Tasks 4, 5, 6 ✓
- Spec §Orchestration step 2 (freeze SHA) → Task 1 step 1 ✓
- Spec §Risks/Round 4 cap (max 2 iterations, then tiebreaker) → Task 9.5 ✓

**2. Placeholder scan:** All prompts filled in with substitution markers only (`{SLOT}`, `{SLUG}`, `{PITCH}`, `{REVISIONS}`) that are explicitly substituted at dispatch time. No "TBD", no "implement later", no vague "add appropriate handling." ✓

**3. Type consistency:** Slug naming convention stated once (Task 7 step 1) and reused consistently. File paths (`docs/hypotheses/<slug>.md`, `scripts/hyp_<slug>.sbatch`, `tests/test_hyp_<slug>.py`) used consistently across Tasks 8, 9, 10, 11. ✓

No issues found.
