# Grill log

One entry per grilling session, newest first. Missed questions become tasks or ADRs before the next session.

### 2026-11-07: code walkthrough, M0 PR2 commit 1 (`backend/paged.rs`)

- **Result:** Pass with follow-ups. The owner read sections 3–4 of the PagedAttention paper first.
- **Answered well:**
  - Positions 15 and 16 fall in different logical blocks.
  - A too-short block table is not "out of memory", and it means the scheduler got something wrong.
- **Missed or shaky:**
  - **Slot for position 20, table `[5, 2, 9]`, block size 16.** Said "the second block" and stopped at the *logical* block. Answer: logical block 1 → physical block `table[1]` = 2, offset 4, so slot 36. The logical→physical step is the point of paging.
  - **Why positions 15 and 16 aren't adjacent in the pool.** Explained it by the block boundary only. Answer: they aren't adjacent because the table maps their logical blocks to physical blocks 5 and 2. With table `[5, 6, …]` they would be.
  - **`BlockTableTooShort`.** Said "the block didn't have space." Answer: the *table* lists too few blocks to cover the tokens. The block manager failed to append one at a block boundary. That's a bug. An empty free pool is a separate *policy* case (preemption), handled before metadata is built.
- **Follow-ups:**
  - [ ] M2 grill: append-on-boundary and preempt-when-empty are separate responsibilities of the block manager and scheduler.

### 2026-11-07: design, M0 PR2 (backend trait, paged CPU forward pass)

- **Result:** Pass with follow-ups. This was the first architecture-only grill: 3 rounds, 10 questions, all architecture.
- **Answered well:**
  - Op-level trait, motivated by per-op observability. Note: real GPU per-op timing needs CUDA events, not wall-clock.
  - Paged KV from M0. Batched prefill with `context_len`. Argmax on the backend. Opt-in profiling. LM head on the last token only ("we just want to serve tokens").
  - Separate prefill and decode paths, a deliberate departure from vLLM V1. The ITL consequence is accepted and carried to the M2 scheduler grill (ADR 0004).
- **Missed or shaky:**
  - **Q2, KV ownership.** You said "paged attention means the engine has to hold KV." Right answer: paged attention is a *bookkeeping* idea. The engine holds block tables; the backend holds the bytes, which on a GPU the engine can't touch. You corrected this in the next round.
  - **Q9, activation memory.** You picked "allocate per op" because "no other system preallocates." The premise was wrong: llama.cpp (ggml) and TensorRT-LLM preallocate, and vLLM only *looks* like per-op allocation because PyTorch's caching allocator plus a startup profiling run sit underneath. You chose the per-step arena once that was explained.
  - **Comprehension check on the arena** (plan approved with no annotations). Question: "What goes wrong if the residual stream is allocated inside the layer scope?" Not answered. Answer: the scope reset rewinds the bump pointer below it, the next layer's `q` lands on the same bytes, and you get silent corruption (use-after-free). In Rust, `Tensor<'layer>` can't escape the scope's closure, so it's a compile error instead. That's now a `compile_fail` doctest in `arena.rs`.
- **Follow-ups:**
  - [x] ADR 0003 (trait, memory ownership, arena, sync) and ADR 0004 (prefill/decode split, paged interface).
  - [ ] M2 scheduler grill: prefill stalls decode (ADR 0004 consequence), with a number for it.
  - [ ] `/explain` on this PR should revisit the arena scope rule and the paged slot arithmetic.

### 2026-11-05 — design — M0 PR1 (config + weights)

- **Result:** Pass with follow-ups (the grill itself failed, not the owner)
- **Answered well:**
  - Hand-counted Llama 3.2 1B params: 1,235,814,400, missing only the 2,048-param final norm. KV: 64 KiB/token f32, 32 KiB bf16; ~61k tokens in 4 GB.
  - Picked Llama 3.2 1B (ADR 0002); strict config rejection; bf16→f32 at load; tied embedding stored once.
- **Missed or shaky:**
  - Peak RAM while loading: not answered, because the question was unclear (it said "the file" without saying which). Answer: ~2.5 GB file bytes + 4.9 GB f32 ≈ 7.4 GB.
  - Load-time estimate: "about a second". Checked against the measurement in results.md.
- **What went wrong with the grill:** four rounds and about an hour on plumbing (fixture paths, model dir, test mechanics, mmap) and none on architecture. One "question" had no question in it.
- **Follow-ups:**
  - [x] New rule: grill architecture only, teach then ask, max 4 per round; Claude lists plumbing as "Claude's defaults" (docs/process.md "What gets grilled").
  - [ ] Next PR (CPU forward pass + backend trait) is the first real architecture grill under the new rule.

## Template

### YYYY-MM-DD — <scope: design | code | gate M?> — <component>

- **Result:** Pass | Pass with follow-ups | Fail
- **Answered well:**
  -
- **Missed or shaky:**
  - Question — what I said — what the right answer is
- **Follow-ups:**
  - [ ] Task or ADR to write
