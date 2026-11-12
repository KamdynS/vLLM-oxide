# M0 PR2: backend trait, paged CPU forward pass, greedy match vs HF

## Context

PR1 (#2, merged `543fa9c`) loads Llama 3.2 1B into f32 `ModelWeights`. The owner wants larger PRs (2026-11-07), so this PR covers **everything left in M0**. When it merges, `/gate M0` can run: greedy output matches HF for 256 tokens on 20 prompts.

It's also the first real architecture PR. The grill (2026-11-07, 3 rounds) settled the backend trait, KV ownership, paged layout, the prefill/decode split, sampling, observability and activation memory. Those decisions shape M1 (CUDA backend), M2 (the owner's block manager and scheduler) and M4 (prefix cache).

Branch: `m0/02-cpu-forward` off `main`.

## Design (owner's decisions from the grill)

1. **Op-level backend trait (Q1).** The Llama model is written once in `engine`, as calls to about 12 trait ops. Tensors are opaque handles, so the engine never reads activation memory. The owner's reason is observability: every op is a named point to hook into. The CPU backend becomes the one-to-one oracle for every future CUDA kernel. *vLLM: same in spirit (model written once, pluggable attention backend).*
2. **KV: backend owns the bytes, engine owns the bookkeeping (Q2, Q5).** The backend allocates the KV pool once (`alloc_kv_pool(num_blocks, block_size) -> KvPool`). The engine holds block tables (`Vec<u32>`) and slot mappings and passes them into ops. *vLLM: same (scheduler holds the block pool; worker allocates the tensors).*
3. **Paged KV from M0 (Q3).** The pool is `[num_blocks, block_size, kv_heads, head_dim]` for K and for V, per layer. M0 builds block tables with a trivial "next free block" counter for one sequence. That counter is **not a block manager**: the block manager stays the owner's M2 chunk and replaces it. *vLLM: same.*
4. **Separate prefill and decode paths (Q4).** Two entry points, `Llama::prefill` and `Llama::decode`, with separate attention ops. The norm, MLP and projection code is shared through a per-layer helper; only attention and its metadata differ. *Departure from vLLM V1, which unified the two (chunked prefill, mixed batches). Closer to vLLM V0 and SGLang's EXTEND/DECODE modes.* **Known consequence, for the M2 scheduler grill:** a step is either all prefill or all decode, so a long prompt stalls every running sequence and their ITL spikes. Mixing them needs a third mode later.
5. **Batched prefill that can start partway into a sequence (Q6).** `prefill` takes several sequences, each with `context_len` (tokens already in KV) and new tokens. A query at position `p` attends to keys `0..=p`, including keys already cached. M0 always calls it with one sequence and `context_len = 0`. M2 batching and M4 prefix hits then need no signature change.
6. **Greedy argmax runs in the backend (Q7).** Only `[num_seqs]` u32 token ids come back to the engine; this is the one allowed sync per step. Logits are read only through an explicit `read_logits` debug call that the hot loop never uses.
7. **Per-op timing is opt-in (Q8).** Every op runs inside a `tracing` span. Timing is collected only when profiling is enabled, and it never adds a sync to the hot loop (on CUDA it will use async event readback). Off by default.
8. **LM head on the last token only (Q10).** Prefill gathers each sequence's last hidden state and runs the 128,256 × 2048 LM head on `[num_seqs, hidden]`. The logits buffer is sized by `max_seqs`. Prompt logprobs are out of scope.
9. **Activations come from a per-step arena (Q9).** At startup the backend reserves one arena sized from `max_batch_tokens`. Ops return tensors taken from it by bumping a pointer, and the arena resets at the end of each step, so nothing is allocated in the hot loop. Running out of arena space is an error (`ArenaExhausted`) meaning the size arithmetic was wrong. *Departure from vLLM, which uses PyTorch's caching allocator plus a profiling forward pass to size memory. Close to ggml's compute buffer.*
   - **Tensor lifetime (Claude's design, owner to check):** `Tensor<'step>` borrows the arena, so the borrow checker refuses a reset while any tensor is alive. The model nests scopes so one layer's temporaries are freed before the next layer starts. The residual stream lives in the outer scope.
   - **Size:** about 36k f32 per token for one layer's live temporaries (q, k, v, attn, o, normed, gate, up, act, down) ≈ 143 KB/token, so about 293 MB at `max_batch_tokens = 2048`, plus the residual stream and logits. The code computes this number and a test checks it against the measured high-water mark.

## Claude's defaults (strike any in review)

- **Block size 16.** `TODO(measure)`; it's vLLM's default and a reasonable placeholder until M2/M1 measure it.
- **CPU matmul** is a hand-written dot-product loop over HF's `[out, in]` rows, parallelized with `rayon` over output rows and written so the compiler can auto-vectorize it (AVX-512 on the 7700X). No BLAS: the CPU backend is the reference implementation, not the performance target.
- **Where things live:**
  - CPU backend: `crates/engine/src/backend/cpu/`.
  - Trait and shared types: `crates/engine/src/backend/mod.rs`.
  - Model: `crates/engine/src/model.rs`.
  - Arena: `crates/engine/src/backend/cpu/arena.rs`.
  - `kernels` crate: still empty.
- **Startup config struct (`EngineConfig`):**
  - `max_batch_tokens = 2048`
  - `max_seqs = 32` (the M2 gate number)
  - `max_model_len = 8192`
  - KV pool budget 2 GiB, i.e. 32,768 tokens or 2,048 blocks. The 30 GB box minus 4.9 GB of weights leaves plenty.
  - All marked as placeholders with their source.
- **RoPE:** precompute `inv_freq` with Llama 3 frequency scaling once, and a cos/sin table up to `max_model_len`. Rotate-half pairing, matching HF's `q_proj` layout (checked by the HF test, not assumed).
- **KV dtype** is f32 on CPU (ADR 0001).
- **Tokenizer:** the `tokenizers` crate loading `tokenizer.json`. Prepend BOS (128000). Base model, so no chat template. Stop on EOS (128001) or 256 tokens.
- **M0 driver:** `engine::generate` runs a single-sequence greedy loop: prefill, then decode until done. Plus `examples/generate.rs` (prompt in, text out, tok/s logged via `tracing`).
- **Reference tool:** `tools/reference/dump_logits.py` with HF transformers, **float32, CPU, greedy**, `prompts.jsonl` (20 prompts). It writes `tools/reference/out/<id>.safetensors`, which Rust already reads, holding:
  - the prompt ids
  - 256 generated ids
  - full logits at the last prompt position and at decode steps 1, 64 and 255
  - the top-1 minus top-2 logit gap at every step, to tell near-ties from bugs
- **Python env:** NixOS has no python on PATH, so add `tools/reference/shell.nix` (python3 with torch, transformers, safetensors). Run it with `nix-shell tools/reference/shell.nix --run "python …"`.
- **Prompts:** 20 prompts of 1 to ~300 tokens. Lengths are deliberately not multiples of 16, and one is exactly 16 and one exactly 17, so the paged indexing crosses block boundaries.
- **Errors:** `thiserror` `BackendError` (shape mismatch, arena exhausted, KV out of range). No `unwrap` in `prefill`, `decode` or `generate`.
- **ADRs, drafted by Claude from the answers above:**
  - **ADR 0003, backend trait and memory ownership:** op-level trait, opaque tensors, per-step arena, KV bytes in the backend and bookkeeping in the engine, argmax-only sync.
  - **ADR 0004, separate prefill/decode and the paged attention interface:** two paths, batched prefill with `context_len`, paged layout from M0, last-token LM head, and the ITL consequence. Records the vLLM V1 departure.
  - Update CLAUDE.md "Settled so far" with both.

## Trait sketch (exact names may change in implementation)

```rust
pub trait Backend {
    type Tensor<'s>;           // activation in the step arena
    type Weights;              // device-resident model weights
    type KvPool;               // [layers][K|V][num_blocks, block_size, kv_heads, head_dim]

    fn upload_weights(&self, w: ModelWeights) -> Result<Self::Weights>;
    fn alloc_kv_pool(&self, num_blocks: usize, block_size: usize) -> Result<Self::KvPool>;
    fn step<R>(&mut self, f: impl FnOnce(&Step<'_, Self>) -> R) -> R; // arena reset after f

    // ops on Step: embed, rmsnorm, matmul, rope, write_kv(slot_mapping),
    // prefill_attention(PrefillMeta), decode_attention(DecodeMeta),
    // silu_mul, add, gather_last, argmax -> Vec<u32> (the sync), read_logits (debug)
}
```

`PrefillMeta { cu_seqlens, context_lens, positions, block_tables, slot_mapping }` and `DecodeMeta { context_lens, positions, block_tables, slot_mapping }`. Shapes go in comments at every boundary.

## Authorship

- **Claude:** all code, the Python tool, the ADR drafts and the docs. Per `process.md`, the CPU forward pass and backend are Claude's; M0 has no owner chunks.
- **Owner:**
  - every design decision above (grill 2026-11-07)
  - the gap write-up
  - plannotator review
  - `/explain`

## Tests (written before the code)

- **Per-op unit tests** against naive formulas, with odd shapes (e.g. 7 tokens, 3 heads):
  - rmsnorm
  - rope (against the direct formula, including llama3 scaling of `inv_freq`)
  - matmul
  - silu_mul
  - gather_last
  - argmax, including ties (lowest index wins, matching torch)
- **Paged attention vs naive contiguous attention** on random data with a **shuffled, non-identity block table**, for both prefill (several sequences, `context_len > 0`) and decode. This is the test that catches paging bugs; the identity table in M0 would hide them.
- **Prefill/decode consistency:** prefill N tokens, versus prefill N−k then decode k. Final logits close, the same token chosen.
- **Arena:** exhaustion returns an error, reset reclaims, and the computed size ≥ the measured high-water mark at `max_batch_tokens`.
- **Real model (`#[ignore]`):**
  - all 20 prompts, greedy tokens exact for 256 steps (or until EOS)
  - logits at the stored steps within tolerance
  - on any token mismatch, the test prints the HF top-1/top-2 gap at that step
- **Tolerance:** stated after first measurement, from the observed max abs and relative logit difference with headroom, and justified in the test comment and in results.md. Both sides are f32, so the expectation is about 1e-4 relative. Exceeding that is investigated, never loosened silently.

## Measurement

_Changed 2026-11-07 (process.md): no owner estimates for CPU performance. The CPU backend is a correctness reference._

- **Recorded once, for test-runtime planning:** CPU decode tok/s and prefill time (`examples/generate`, release build). Closes ADR 0001's `TODO(measure)`.
- **Correctness numbers, recorded in `docs/results.md`:** max logit difference vs HF (this sets the tolerance), and the arena high-water mark vs `arena_elems`.

## Verification

1. `cargo fmt --all && cargo clippy --workspace -- -D warnings && cargo test --workspace`
2. `nix-shell tools/reference/shell.nix --run "python tools/reference/dump_logits.py --model models/Llama-3.2-1B --prompts tools/reference/prompts.jsonl"`
3. `cargo test -p engine --release -- --ignored`: 20/20 prompts exact for 256 tokens.
4. `cargo run --release --example generate -- models/Llama-3.2-1B "The capital of France is"` gives sensible text and logs tok/s.
5. Save this plan as `docs/plans/0002-cpu-forward.md`. Add the grill-log entry for 2026-11-07, recording the Q2 bytes-vs-bookkeeping confusion and the Q9 "no one preallocates" premise as learning points. Update the CLAUDE.md Commands section with the reference and generate commands.
6. Then `/plannotator-review`, `/explain`, open the PR, and `/gate M0`.

## Implementation notes (where the code departs from the plan above)

- **The arena is generic, in `crates/engine/src/arena.rs`, not in `backend/cpu/`.** Offsets work the same on any device, so the bump allocator, scopes and lifetime rules are written once. The backend only allocates one workspace and maps `Region { offset, rows, cols }` (first called `Desc`) to memory.
- **Trait op methods are `unsafe fn`.** They take raw workspace offsets, and the backend trusts that outputs don't overlap inputs. `arena::Scope` is the only caller and upholds this, with one shared SAFETY argument. Each tensor carries an arena id, so mixing arenas is a runtime error instead of aliasing.
- **Arena is 184 MB, not ~293 MB.** Attention and the MLP get separate nested scopes, and `silu_mul` works in place (`gate = silu(gate) * up`). Peak = `T·h + max(attn, mlp, final)`. A test checks that the measured high-water mark equals this exactly.
- **The trait has no "upload activation" op.** Tests put host data into the arena by `embed`-ing an uploaded table.
- **`.cargo/config.toml` sets `target-cpu=native`** so the 16-lane dot product compiles to AVX-512. The binaries aren't portable; this is a single-box project.
- **The `compile_fail` doctest** in `arena.rs` turns the scope-escape rule into a checked fact.
- **Test diagnostics** in `tests/hf_match.rs` use `println!` (shown with `--nocapture`). The `tracing` rule applies to library and binary code.

### Changes from the owner's code review (2026-11-07)

- **Attention takes plain per-row arrays, not a closure.** The owner flagged `keys_for: impl Fn(usize) -> (&'m [u32], usize) + Sync` as a smell. Both metadata structs now carry `row_seq` and `row_num_keys`, and `paged_attention` takes them as arrays: the same shape of arguments the M1 CUDA kernel will receive.
- **Paged addressing has its own file**, `backend/paged.rs` (`KvLayout`, `slot_of`, `PrefillMeta`, `DecodeMeta`), separate from the trait in `backend/mod.rs`. That makes it reviewable as one concept.
- **Readability pass:**
  - `Desc` → `Region`, `cu_seqlens` → `seq_row_starts`, decode `seq_lens` → `row_num_keys`.
  - Model weights renamed to HF's `q_proj` … `down_proj`.
  - No one-letter locals or C-style abbreviations.
- **Review process:** the PR is walked through in about 6 concept-sized commits in reading order, each annotated by the owner before the next.
- **`BackendError` → `EngineError`, in its own `src/error.rs`** (owner review). Half its variants come from the arena, the model and the metadata, so the old name hid what it was. One enum for the crate is enough until a caller needs to match on which layer failed.
