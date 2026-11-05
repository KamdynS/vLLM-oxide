# vllm-oxide

A single-node LLM serving engine in Rust with hand-written CUDA kernels: paged KV cache, continuous batching, chunked prefill, prefix caching, and a cache-aware router behind an OpenAI-compatible API.

Built as a learning project: every design decision is recorded in [`docs/decisions/`](docs/decisions/), and every performance claim has a row in [`docs/results.md`](docs/results.md).

## Status

See the milestone table in [CLAUDE.md](CLAUDE.md#milestones).

## Layout

- `crates/kernels` — CUDA kernels and Rust launch wrappers
- `crates/engine` — model, KV block manager, scheduler, prefix cache, sampler
- `crates/server` — OpenAI-compatible HTTP + SSE
- `crates/router` — multi-replica routing
- `crates/bench` — open-loop load generator
- `tools/reference` — HF reference outputs for correctness checks
- `docs/` — decisions, grill log, results
