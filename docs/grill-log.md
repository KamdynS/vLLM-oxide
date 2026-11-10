# Grill log

One entry per grilling session, newest first. Missed questions become tasks or ADRs before the next session.

### 2026-11-10: code walkthrough, M0 PR2 commit 3 (`arena.rs`, `arena/ops.rs`)

- **Result:** Pass with follow-ups. Took three sessions. After the first read the owner said "I read the code, but I still don't _really_ understand it", so Claude rebuilt the arena from scratch in stages (Vec per op → one buffer + offsets → rewind → closure scopes → lifetime branding → nesting → in-place ops → arena id), each stage fixing the previous stage's bug. That landed.
- **Owner's review changes:**
  - "Why are we doing all the math from here downward in arena.rs?" The ops do no math, but the file mixed two ideas. Split into `arena.rs` (memory) and `arena/ops.rs` (the safe op API and the SAFETY block). It's a child module so `alloc` and `region` stay private.
  - Flagged a self-referential module doc. Replaced with a "Using it" paragraph.
  - Asked about `Option<RefCell<...>>`. Answer added as a field doc comment.
  - Asked whether every op really needs `unsafe`. Answer: unsafe is unavoidable at the device boundary; the design question is how small the trusted region is. Considered: safe trait ops with runtime overlap checks in every backend (cheap, but duplicated per backend, and it only moves the unsafe down a layer).
- **Answered well:**
  - Writing into `residual` from a block scope survives the rewind: its region is below the block's mark, and a rewind only frees what's above.
  - `Cell` despite `Arena::scope(&mut self)`: scopes hold the arena by `&`, so `top` needs interior mutability, and `Cell` suffices for a `usize`.
  - After the re-explanation, in their own words: "The stricter we have our borrow checker, the easier our code is."
- **Missed or shaky:**
  - **Overlap: what keeps fresh outputs and in-place ops from aliasing.** Didn't know at first. After teaching, said `Clone` would allow two handles but didn't name the in-place op as where it breaks. Answer: fresh outputs come from `top`, above every live tensor. In-place ops take `&mut Tensor`, and `Tensor` isn't `Clone`, so one region has one handle.
  - **Outer scope allocating inside an inner scope.** Didn't know. Answer: `Scope::scope(&mut self)` borrows the outer scope for the whole closure. Otherwise the inner rewind would free the outer allocation.
  - **`normed` moved to the step scope "to save memory."** Said it wouldn't compile because `residual` needs `&mut`. It compiles; the borrows don't overlap in time. Answer: it leaks. A bump arena only frees on rewind, so 16 dead `normed`s pile up (`ArenaExhausted` at T = 2048). The rewind is already the reuse: each layer's `normed` lands on the same offset.
  - **Capacity vs workspace (promise 1).** Pointed at `alloc`'s check, which compares against the arena's own `capacity`, and said `region` checks bounds (it checks arena identity). Answer: nothing compared the two numbers; it was a convention in `Llama::new`.
- **Design change from the quiz:** `Arena::new(&backend)` now reads the capacity from `backend.workspace_elems()`, so "capacity == workspace" holds by construction. A `debug_assert` in `Arena::scope` catches a mismatched backend.
- **Process note:** the owner said "I never could have written this", which feels strange. We discussed reading vs producing. Offered an optional "rebuild it from memory" step for new core concepts (arena, paged attention, CUDA kernels, scheduler). The owner kept the current loop for now because interview prep is the priority, and may do hard small projects in free time.
- **Follow-ups:**
  - [ ] `/gate M0`: re-ask "where is each of the three unsafe promises kept?" cold.
  - [ ] `/gate M0`: re-ask "a tensor goes out of use in a bump arena: when is its memory reused?"

### 2026-11-07: code walkthrough, M0 PR2 commit 2 (`backend/mod.rs`, `error.rs`)

- **Result:** Pass with follow-ups.
- **Owner's review change:** asked "should we have errors.rs?". That led to renaming `BackendError` → `EngineError`, since half its variants weren't backend errors.
- **Answered well:**
  - The trait doesn't prevent overlapping regions; the caller is trusted.
  - `argmax` returns to the engine and everything else stays on the device, so data isn't moved twice.
  - `write_kv` needs `&mut` because it writes the pool.
  - After correction, restated in their own words: "The engine decides the block, the backend stores the numbers."
- **Missed or shaky:**
  - **The unsafe contract.** Called it "standard bound checks". Answer: bounds is the easy promise (the CPU backend re-checks it). The key promise is no overlap between input and output regions, which no bounds check catches.
  - **Cost of `rmsnorm` returning a `Vec` on GPU.** Got the round trip, missed the sync. Answer: the CPU must wait for all queued GPU work before reading. That would happen 33 times per step instead of once, leaving the GPU idle in between (assessment Q9b).
  - **Pool vs prefix cache.** Said "prefill uses the prefix cache, which is immutable". Answer: prefill reads the KV pool (all KV memory, including keys written earlier in the same step). The prefix cache (M4) is a reuse policy over blocks in that same pool.
  - **Why `KvPool` is an associated type.** Said "the engine isn't the one who manages paged attention", which is backwards, and a repeat of grill Q2. Answer: the engine manages paging (tables, allocation). The backend stores the bytes, which on CUDA are device memory the engine can't touch. The OS analogy landed: kernel page tables vs RAM chips.
- **Follow-ups:**
  - [ ] Bookkeeping vs bytes has now been missed twice. `/gate M0` must re-ask it cold.
  - [ ] Pool vs prefix cache: revisit in the M4 grill.

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
