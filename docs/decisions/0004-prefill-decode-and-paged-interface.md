# 0004: Separate prefill and decode paths over a paged-KV interface

- **Status:** Accepted
- **Date:** 2026-11-07
- **Milestone:** M0

## Context

The attention interface is the signature most likely to be changed by later milestones:
- M2 batches many sequences and needs paged KV.
- M4 prefix caching starts a prefill partway into a sequence.

M0 only needs one sequence. If the M0 interface fits only that case, M2 and M4 both rewrite the most important op in the system. This ADR fixes the interface now.

## Options

**Paged KV in M0 (grill Q3)**

| Option | For | Against |
| --- | --- | --- |
| **Paged from M0**: pool of `[num_blocks, block_size, kv_heads, head_dim]`, ops take block tables and slot mappings; M0 uses a trivial counter, not an allocator | The signature is fixed once; the M0 HF match also validates paged indexing | One extra level of indexing now |
| Contiguous per sequence, replaced in M2 | Simpler today | The core op changes in the milestone with the most moving parts |

**Prefill and decode (Q4)**

| Option | For | Against |
| --- | --- | --- |
| **Two entry points and two attention ops** | Matches the conceptual split (compute-bound prefill, memory-bound decode); each can be specialized | A step is all-prefill or all-decode; mixing needs a third mode |
| One varlen path for both (vLLM V1) | Mixed batches and chunked prefill come free | Ragged attention everywhere; the conceptual split is hidden |

**Prefill shape (Q6):** batched sequences, each with `context_len` and new tokens (chosen), vs one sequence starting at position 0.

**LM head (Q10):** last token per sequence only (chosen), vs all positions.

## Decision

- **Paged KV from M0**, block size 16 (`TODO(M1)`: measure).
- **Separate `prefill` and `decode` paths and attention ops.** Everything except attention and the final row selection is shared code.
- **Prefill takes a batch of sequences**, each with `context_len` (tokens already cached) and new tokens.
- **The LM head runs only on each sequence's last token.**

## Evidence

- **LM head cost:** 128,256 × 2048 ≈ 263 M multiply-adds per row. On a 500-token prompt, computing it for every row would waste ~130 GFLOP and a 256 MB logits buffer. Last-token-only bounds the logits by `max_seqs`, giving 16 MB at 32 sequences.
- **Paged indexing is tested without relying on M0's identity table.** `paged_*_attention_matches_naive_with_shuffled_blocks` uses interleaved physical blocks, `context_len > 0`, and lengths on and off block boundaries.
- **Cached context works in prefill.** `prefill_with_context_matches_full_prefill` checks the M4 prefix-hit path today.
- **No-op batching is bitwise.** `batched_prefill_is_batch_invariant` checks that batching changes no logit, because no op mixes rows.

## Comparison (after the fact)

- **vLLM V1** unified prefill and decode: the scheduler assigns tokens per request, chunked prefill is always on, and prefill and decode share a batch. **This is our departure.** We are closer to vLLM V0 (separate prefill and decode metadata) and to SGLang's `EXTEND` / `DECODE` forward modes. SGLang's `EXTEND` is exactly our "prefill with `context_len`".
- Paged KV and block size 16 are the same as vLLM.

## Consequences

- **Known cost, carried into the M2 scheduler grill:** with separate step types, a long prompt arriving while 31 sequences decode stalls all of them for the whole prefill, so their ITL spikes. The fixes are chunked prefill (bounded prefill per step) or a third "mixed" mode. Measure it in M3 before choosing.
- **Easier:** M2 and M4 need no change to the attention signatures. The decode op can be specialized (one query per sequence) without touching prefill.
- **Revisit if:** M3 measurements show ITL spikes from prefill stalls that chunked prefill alone can't fix.

## Grill questions this must survive

- vLLM unified prefill and decode. Why didn't you, and what does it cost you?
- A 4,000-token prompt arrives while 31 sequences are decoding. What happens to their inter-token latency?
- Why build paged KV before there's a second sequence?
- How does a prefix-cache hit flow through `prefill`?
- Why compute logits for only one row per sequence?
