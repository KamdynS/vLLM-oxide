# How work gets done

Claude writes most of the code. The owner decides, reviews, and must be able to defend every line that merges. The process below exists so that "Claude wrote it" never quietly becomes "nobody understands it."

## The loop (every PR)

1. **Plan mode.** Every PR starts in Claude Code plan mode. No code is written until the plan is approved.
2. **Grill the plan (`/grill-me`).** Claude interviews the owner in rounds over the **architecture** decisions (see [What gets grilled](#what-gets-grilled)), each with Claude's recommended answer. Facts (hardware specs, model config, what the code currently does) are Claude's job to look up. Architecture decisions are the owner's. Plumbing decisions are Claude's, listed in the plan. Done when no architecture decision is left silently assumed.
3. **Write the plan.** Claude writes the plan with these sections:
   - **Goal**: what this PR changes and why now.
   - **Design**: the decisions from the grill, with the reasoning.
   - **Claude's defaults**: every plumbing decision Claude made, one line each with a reason. The owner strikes or changes any of them in review; silence means accepted.
   - **Authorship**: which files Claude writes and which the owner writes.
   - **Tests**: what proves it's correct, written before the code.
   - **Measurement**: what gets recorded in `docs/results.md`, if anything.
   - **ADRs**: any durable decision gets its own record in `docs/decisions/`, drafted by Claude from the grill answers.
4. **Review the plan in plannotator.** Leaving plan mode opens the plan in plannotator (hook in `.claude/settings.json`). The owner annotates; Claude revises until approved. The approved plan is saved to `docs/plans/NNNN-<short-title>.md` and ships with the PR.
5. **Implement.** On a feature branch, following the plan's authorship split.
6. **Review the code in plannotator (`/plannotator-review`).** The owner annotates the diff; Claude revises until approved.
7. **Comprehension check (`/explain`).** Before the PR opens, Claude picks the 3–5 places in the diff most likely to hide a misunderstanding (`unsafe`, index math, constants, sync points, KV ownership) and the owner explains them. No hints, no recommended answers. Gaps become fixes, comments, tests, or follow-up tasks.
8. **PR and merge.** The PR description includes the authorship and what the owner explained (see below).

Small mechanical PRs (formatting, dependency bumps, typo fixes) and plumbing-only PRs (loading, build setup, glue with no architecture decision) may skip steps 2 and 7. Say so in the PR.

## What gets grilled

The grill exists so the owner designs the system, not so they sign off on file paths.

- **Architecture only.** A question is asked only if its answer changes serving behavior, performance or memory, or an interface another component depends on, *or* a senior inference engineer would plausibly ask it in an interview. Examples: the backend trait boundary, KV layout, who owns memory, prefill vs decode paths, GPU sync points, scheduling policy.
- **Claude decides plumbing without asking.** File locations, dependencies, fixtures, crate housekeeping, test mechanics, naming and error types go in the plan's **Claude's defaults** list, not the grill.
- **Testing is not a grill topic.** CLAUDE.md sets the bar; Claude meets it.
- **Teach, then ask.** Every question opens with 2–4 plain-English sentences: what the thing is and why it matters for serving. Then it asks one real question, ending in "?". Define jargon the first time it appears.
- **At most 4 questions per round**, ordered by how much the answer shapes the system.

## Who writes what

The owner writes **whole chunks, types included**. Data-structure and interface design is where the decisions live, so the owner never just fills in function bodies under Claude's types. Claude writes everything else.

**Owner chunks**, in build order (hours are estimates):

| Chunk | Milestone | Hours | Why it's the owner's |
| --- | --- | --- | --- |
| KV block manager + block tables | M2 | 6–8 | Paged KV is the core memory idea of serving |
| Scheduler / continuous batching policy | M2 | 10–15 | The core policy idea, and the biggest jump in learning |
| Concurrency boundary: tokio ↔ engine thread channels, backpressure, cancellation, streaming out | M3 | 6–10 | How a request actually moves through the system |
| One warm-up CUDA kernel, then the paged-attention kernel | M1 | 12–18 | The kernel story has to be first-person |
| Prefix cache | M4 | 6–8 | Feeds the router and the project's original question |

**Claude writes:** workspace and build setup, weight and tokenizer loading, the CPU backend and forward pass (KV layout decided in the grill), the CUDA backend and cuBLAS wrappers, elementwise kernels, the axum/OpenAI API surface, the load generator, metrics, and the HF reference tools.

**Always the owner's, every PR:** the back-of-envelope estimate before a measurement, and the write-up of the gap after it.

### How an owner chunk runs

- **The grill gives no recommended answers.** Claude asks the questions (invariants, failure modes, memory pressure, why not the alternative), the owner designs, and Claude pushes back only on what's actually wrong.
- **The owner writes the chunk and its basic tests. Then Claude writes adversarial property tests against the owner's API** as an independent check.
- **If Claude's code needs a chunk before it exists**, Claude codes against a minimal stub interface agreed in the grill. The owner's real types replace it and may change it.
- **When stuck (~45 min), the owner can ask for help in steps:** a hint, then pseudocode, then code. Whatever was used goes in the PR's Authorship section, e.g. `block_manager.rs: owner, with a hint on refcount release`.
- **During the launch crunch (mid-Dec to mid-Jan)**, the owner doesn't start new chunks; just review, estimates and `/explain`.

## Staying honest

Claude's job here is to keep the owner's understanding in step with the code, not just to ship.

- **Two modes of questioning, never mixed.** Planning grills give recommendations because they're about deciding. Comprehension checks (`/explain`, `/gate`) never do, because they're about knowing.
- **Authorship is recorded, not implied.** Every PR description has an `## Authorship` section, e.g. `scheduler.rs: Claude. Explained by owner: preemption order, refcount release.` "Not yet explained" is an allowed and honest entry.
- **Claude calls out rubber-stamping.** If a plan or diff touching core code is approved with no annotations, Claude asks one comprehension question before moving on.
- **Gates are the hard check.** `/gate M<n>` grills across everything built so far, one question at a time, no notes. A milestone isn't done until the owner passes it.
