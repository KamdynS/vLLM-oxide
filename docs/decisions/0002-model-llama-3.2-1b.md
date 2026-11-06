# 0002: Serve Llama 3.2 1B (base)

- **Status:** Accepted
- **Date:** 2026-11-05
- **Milestone:** M0

## Context

Every later piece is sized from the model: KV bytes per token, block pool size, kernel shapes (head_dim, GQA ratio), and the vLLM/SGLang baselines in M5. The project says "Llama-family". Per ADR 0001, M0 and M2 run on CPU in f32, on a 30 GB RAM box with no swap. The NVIDIA card arriving ~2026-11-30 is a ~$600 consumer card (likely 12–16 GB VRAM).

## Options

| Option | For | Against |
| --- | --- | --- |
| Llama 3.2 1B | Real Llama architecture (GQA, llama3 RoPE scaling, tied embeddings); 4.9 GB in f32 fits CPU; 2.5 GB bf16 leaves most of a consumer GPU for KV; well supported by vLLM/SGLang for baselines | Gated download (license); small model makes decode latency-bound on launch overhead rather than bandwidth on a big GPU |
| Llama 3.2 3B | More realistic compute per token | ~12.8 GB in f32 on CPU makes M0's 20 × 256-token check slow; 6.4 GB bf16 eats half the VRAM |
| Llama 3.1 8B | The model people actually benchmark | 32 GB in f32 does not fit 30 GB RAM; 16 GB bf16 does not fit the card |
| Qwen2.5-0.5B | Already cached locally, 2× faster on CPU | Not Llama: QKV biases, different RoPE config; baselines and write-up become about a less common model |

## Decision

Serve Llama 3.2 1B, base (not Instruct), for M0 through M5.

## Evidence

Owner's hand count before any code (2026-11-05): 1,235,814,400 params (embeddings 262,668,288 counted once because tied; 60,821,504 per layer × 16; final norm 2,048). KV cache: 2 × 16 layers × 8 KV heads × 64 × 4 B = 64 KiB/token in f32, 32 KiB in bf16. So 4 GB of f32 KV holds about 61k tokens.

Base rather than Instruct: M0 prompts are raw text completion, so no chat template is involved in matching HF. Instruct shares the architecture, so it can be swapped in for M3 demos without code changes.

## Consequences

- **Easier:** CPU f32 fits with room to spare (peak ~7.4 GB while loading). The GPU has plenty of headroom for a large KV pool, which is what makes batching and prefix caching interesting.
- **Harder:** at 1B, GPU decode for small batches will be dominated by kernel launch overhead, not memory bandwidth. Profiles have to account for that, and CUDA graphs may matter earlier than for a bigger model.
- **Revisit if:** the chosen card has ≥ 24 GB and the write-up needs a model people benchmark, in which case 8B in bf16.

## Grill questions this must survive

- Why not 8B, which is what everyone benchmarks?
- At batch 1 on your GPU, is decode for this model bandwidth-bound or launch-bound, and how would you tell?
- What does tying the embeddings save, and where does that tensor get read twice in a forward pass?
