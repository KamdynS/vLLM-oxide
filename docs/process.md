# How work gets done

Claude writes most of the code. The owner decides, reviews, and must be able to defend every line that merges. The process below exists so that "Claude wrote it" never quietly becomes "nobody understands it."

## The loop (every PR)

1. **Plan mode.** Every PR starts in Claude Code plan mode. No code is written until the plan is approved.
2. **Grill the plan (`/grill-me`).** Claude interviews the owner in rounds over the design tree: each round asks every question whose prerequisites are settled, with Claude's recommended answer. Facts (hardware specs, model config, what the code currently does) are Claude's job to look up; decisions are the owner's. Done when nothing is left silently assumed.
3. **Write the plan.** Claude writes the plan with these sections:
   - **Goal**: what this PR changes and why now.
   - **Design**: the decisions from the grill, with the reasoning.
   - **Authorship**: which files Claude writes and which the owner writes.
   - **Tests**: what proves it's correct, written before the code.
   - **Measurement**: what gets recorded in `docs/results.md`, if anything.
   - **ADRs**: any durable decision gets its own record in `docs/decisions/`, drafted by Claude from the grill answers.
4. **Review the plan in plannotator.** Leaving plan mode opens the plan in plannotator (hook in `.claude/settings.json`). The owner annotates; Claude revises until approved. The approved plan is saved to `docs/plans/NNNN-<short-title>.md` and ships with the PR.
5. **Implement.** On a feature branch, following the plan's authorship split.
6. **Review the code in plannotator (`/plannotator-review`).** The owner annotates the diff; Claude revises until approved.
7. **Comprehension check (`/explain`).** Before the PR opens, Claude picks the 3–5 places in the diff most likely to hide a misunderstanding (`unsafe`, index math, constants, sync points, KV ownership) and the owner explains them. No hints, no recommended answers. Gaps become fixes, comments, tests, or follow-up tasks.
8. **PR and merge.** The PR description includes the authorship and what the owner explained (see below).

Small mechanical PRs (formatting, dependency bumps, typo fixes) may skip steps 2 and 7. Say so in the PR.

## Staying honest

Claude's job here is to keep the owner's understanding in step with the code, not just to ship.

- **Two modes of questioning, never mixed.** Planning grills give recommendations because they're about deciding. Comprehension checks (`/explain`, `/gate`) never do, because they're about knowing.
- **Authorship is recorded, not implied.** Every PR description has an `## Authorship` section, e.g. `scheduler.rs: Claude. Explained by owner: preemption order, refcount release.` "Not yet explained" is an allowed and honest entry.
- **Claude calls out rubber-stamping.** If a plan or diff touching core code is approved with no annotations, Claude asks one comprehension question before moving on.
- **Gates are the hard check.** `/gate M<n>` grills across everything built so far, one question at a time, no notes. A milestone isn't done until the owner passes it.
