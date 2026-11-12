# 0003: Op-level backend trait; backend owns bytes, engine owns bookkeeping; per-step arena

- **Status:** Accepted
- **Date:** 2026-11-07
- **Milestone:** M0

## Context

ADR 0001 put the engine behind a backend trait so the CPU work done now survives the CUDA backend coming in M1. This ADR decides what that trait looks like. There are four parts: what crosses it, who owns KV memory, how activations get memory, and what comes back to the host. Every later component builds on these: the M1 kernels, the M2 block manager and scheduler, and the M4 prefix cache.

## Options

**What crosses the trait (grill Q1)**

| Option | For | Against |
| --- | --- | --- |
| **Op-level**: ~12 primitives (matmul, rmsnorm, rope, attention, …), model written once in `engine` | One model implementation; every CUDA kernel gets a 1:1 CPU oracle; every op is a hook for observability | Fusing ops means adding trait methods |
| Model-level: `forward(tokens) -> logits` per backend | Each backend free to fuse | Model written twice, so CPU and GPU drift structurally, which is the drift ADR 0001 exists to prevent |

**Who owns KV memory (Q2, Q5)**

| Option | For | Against |
| --- | --- | --- |
| **Backend owns the bytes, engine owns the bookkeeping** (block tables, slot mappings passed into ops) | Block manager and scheduler stay device-agnostic and testable without a GPU | Every attention op takes metadata |
| Backend owns both ("append for seq 7") | Simple op signatures | Scheduling policy leaks into the backend |
| Engine owns both as host memory | Simplest on CPU | Impossible once KV is on a GPU |

**Activation memory (Q9)**

| Option | For | Against |
| --- | --- | --- |
| Allocate per op | Reads like PyTorch | On CUDA it needs a caching allocator and a profiling run to size the KV pool |
| Destination-passing into named buffers | Zero-cost, explicit | Every call site manages buffers |
| **Per-step bump arena**, ops return tensors | Reads like per-op allocation; nothing is allocated after startup; the size is arithmetic, not profiling | Lifetimes in the API; arena size must be computed correctly |

**Sampling (Q7):** argmax in the backend, returning only `[num_seqs]` u32 (chosen), vs copying the logits to the host (500 KB per sequence per step).

**Observability (Q8):** opt-in per-op timing (chosen) vs always on.

## Decision

- An **op-level trait** with opaque handles. The engine never reads activation or KV memory.
- The **backend allocates the KV pool** once. The engine passes block tables and slot mappings.
- **Activations come from a per-step bump arena** over one workspace allocated at startup. Scopes are nested, and lifetimes stop a tensor from outliving its scope.
- **Argmax runs on the backend.** It is the one device→host sync per step. Logits are read only through a debug call.
- **Per-op timing is opt-in** and never adds a sync.

## Evidence

- **Arena size is computed, not profiled.** `model::arena_elems` gives `T·h + max(attn, mlp, final)`. For Llama 3.2 1B with `max_batch_tokens = 2048` and `max_seqs = 32`, that's 46.1 M f32 = **184 MB**, with the MLP scope dominating. `arena_high_water_equals_formula` checks that the measured high-water mark equals the formula exactly.
- **Op-level launch overhead on CUDA** is about 5 µs × ~10 ops × 16 layers ≈ 1 ms/step. That's acceptable at M1, and CUDA graphs remove it later (M5 fork).
- **The arena's safety argument is checked by the compiler.** A `compile_fail` doctest in `arena.rs` shows that returning a tensor from an inner scope doesn't build.

## Comparison (after the fact)

- **vLLM** writes the model once in PyTorch with a pluggable attention backend. That's the same idea as op-level, but pluggable only at attention. Its scheduler-side block pool plus worker-allocated KV tensors is the same split as ours.
- **vLLM activations** use PyTorch's caching allocator, plus a profiling forward pass at `max_num_batched_tokens` to learn peak activation memory before sizing KV. **This is our departure:** we compute the peak instead of profiling it.
- **llama.cpp (ggml)** plans the graph and allocates one compute buffer up front, which is close to our arena.

## Consequences

- **Easier:** M1 is "implement the trait for CUDA" plus kernels, and every kernel is tested against its CPU counterpart. M2's block manager is pure bookkeeping. The KV budget is whatever is left after weights and an exactly known workspace.
- **Harder:** a fused kernel needs a new trait method and a CPU reference for it. Changing the scope structure in `model.rs` means updating `arena_elems`; the test catches it if you forget.
- **Revisit if:** launch overhead dominates decode before CUDA graphs land, or a backend needs memory the arena can't express (e.g. cuBLAS workspace, which should become a separate fixed allocation).

## Grill questions this must survive

- Why not let each backend implement the whole forward pass and fuse freely?
- Why is the engine allowed to hold block tables but not KV bytes?
- How do you know the arena is big enough without profiling? What if it isn't?
- What stops activation memory being reused while still live? (Lifetimes; see the `compile_fail` doctest.)
- Where is the one device→host sync, and why only there?
