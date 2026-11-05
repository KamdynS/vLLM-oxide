# CLAUDE.md

Rust LLM inference server with its own CUDA kernels. Single node, Llama-family model, OpenAI-compatible API, paged KV cache, continuous batching, prefix caching, cache-aware router.

**The point of this project is that the owner can defend every line.** It is a learning and portfolio project, not a product. Optimize for understanding first, performance second, speed of delivery last.

Full plan (goals, architecture, roadmap, question bank): https://claude.ai/artifact/6Va2PppzfSq87aoymCWTH3

## Working agreement

Claude writes most of the code. The owner decides, reviews, and must understand everything that merges. Claude's job is to keep the owner honest about that, not just to ship.

- **Follow the PR loop in [`docs/process.md`](docs/process.md).** Plan mode → `/grill-me` → plan → plannotator plan review → implement → `/plannotator-review` → `/explain` → PR.
- **Authorship is decided in the plan and recorded in the PR.** The owner may take any file; Claude writes the rest. Never blur who wrote what.
- **Explain before editing.** For any non-trivial change, state the problem, the options, and the recommended choice before touching code.
- **No unexplained magic numbers.** Every constant (block size, tile size, thread count, tolerance, chunk size) gets a comment saying where it came from: a measurement, a hardware limit, or a placeholder marked `// TODO(measure)`.
- **Don't copy from vLLM / SGLang / nano-vllm.** Design from first principles in the grill; compare against them afterward and record the comparison.
- **Push back.** If the owner's design is wrong or will not scale, say so directly with the reason. Agreement is not the job.
- **Call out rubber-stamping.** If the owner approves core code without engaging, ask a comprehension question before moving on.

## Repo layout

```
crates/
  kernels/     CUDA C++ sources in cuda/, compiled to PTX by build.rs; safe Rust launch wrappers
  engine/      model loading, forward pass, KV block manager, scheduler, prefix cache, sampler, engine loop
  server/      axum HTTP server: OpenAI API, SSE streaming, cancellation, metrics
  router/      multi-replica router: load- and cache-aware placement
  bench/       open-loop load generator and result reporting
tools/
  reference/   Python scripts that dump HF transformers logits and hidden states for comparison
docs/
  process.md   the per-PR workflow
  plans/       one approved plan per PR
  decisions/   ADRs for durable decisions (criteria in its README)
  grill-log.md record of every grilling session
  results.md   benchmark results log, one row per measured change
```

Layout is a starting point. Changing it is a decision and gets an ADR.

## Commands

```bash
cargo build --release
cargo test --workspace                     # unit + property tests
cargo test -p kernels -- --test-threads=1  # kernel tests share the GPU
cargo clippy --workspace -- -D warnings
cargo fmt --all

compute-sanitizer --tool memcheck target/release/<test-binary>   # every kernel change
nsys profile -o prof target/release/engine-bench ...              # timeline
ncu --set full -k <kernel_name> target/release/<binary>           # single-kernel deep dive

python tools/reference/dump_logits.py --model <path> --prompts tools/reference/prompts.jsonl
```

Fill in exact binary names and flags as they come to exist. Keep this section accurate.

## Architecture invariants

- **One engine thread owns the CUDA context.** Async (tokio) handlers never call into CUDA. They talk to the engine over channels.
- **The engine loop is `schedule → prepare inputs → forward → sample → emit`.** One forward pass per step, one new token per running sequence per decode step.
- **The hot loop does not sync with the GPU** except to read sampled token ids. Any new sync point needs a comment justifying it.
- **KV memory is a fixed pool of blocks** allocated once at startup from whatever memory remains after weights and workspace. No allocation in the hot loop.
- **Block ownership:** every block is either free or referenced by ≥1 block table, refcount equals the number of references, and nothing is freed while a running sequence uses it. These are checked by property tests.
- **Batch invariance:** a request's greedy output is identical whether it runs alone or in a batch. Regressions here are bugs, not noise.

## Code conventions

- Every `unsafe` block has a `// SAFETY:` comment naming the invariant it relies on and who upholds it.
- No `unwrap()` / `expect()` in the engine loop or server request path. Use `thiserror` for library errors, `anyhow` only in binaries and tools.
- `tracing` for logs, with a request id span. No `println!`.
- Kernel launch wrappers validate shapes, dtypes and alignment before launching, and return `Result`.
- CUDA kernels: one kernel per file in `crates/kernels/cuda/`, a header comment with the math, the launch config and why, and the expected bottleneck (memory or compute).
- Tensor shapes are written in comments as `[batch, seq, heads, head_dim]` at every function boundary.
- Prefer clear code over clever code until a profile says otherwise.

## Testing and correctness

- **No optimization without a test before and a measurement after.** Record the measurement in `docs/results.md`.
- Every kernel has a test against a naive CPU or reference implementation, including odd and non-power-of-two shapes, and passes compute-sanitizer.
- Model-level correctness is checked against `tools/reference` output: logits within a stated tolerance, greedy tokens exact for 256 tokens.
- Block manager and scheduler use `proptest` for invariants. Add a property whenever a bug is found.
- Floating-point tolerances are stated and justified, never loosened silently to make a test pass.

## Performance work

- Profile first (nsys for the timeline, ncu for one kernel). State the bottleneck before proposing a fix.
- For each kernel, record achieved bandwidth or FLOPs as a fraction of the GPU's peak.
- Benchmarks are open-loop with Poisson arrivals; report p50/p90/p99 TTFT and ITL, throughput, and goodput at a stated SLO. Record GPU, driver, CUDA version, commit and config with every run.

## Decisions

Every decision is recorded somewhere. Most go in the PR's plan doc (`docs/plans/`). Ones that later PRs depend on, that someone would reasonably question, or that are expensive to reverse get an ADR in `docs/decisions/` (criteria in its README). If Claude notices a decision in a diff that isn't recorded in either place, it raises it before approving.

Settled so far (update as ADRs land):

- Model: _TBD (leaning Llama 3.2 1B)_
- Backend: CPU-first behind a backend trait; CUDA backend added when the NVIDIA card arrives (~2026-11-30). The CPU backend stays as the kernel reference. AMD card unused. ([ADR 0001](docs/decisions/0001-cpu-first-backend.md))
- Kernels: CUDA C++ compiled to PTX in build.rs, launched from Rust _(confirm in ADR)_
- GEMM: cuBLAS _(own GEMM as a side exercise only)_

## Grilling

Two kinds of questioning, kept separate:

- **Planning (`/grill-me`)**: before every PR, in plan mode. Rounds of questions with Claude's recommended answers; the owner decides. Grill like a skeptical senior inference engineer: arithmetic intensity, bytes moved, failure modes, why not the alternative, what happens under memory pressure. Push for numbers.
- **Comprehension (`/explain`, `/gate`)**: one question at a time, no hints or recommended answers. `/explain <path>` before every PR; `/gate M<n>` at each milestone (see `.claude/commands/gate.md`).

Log grills and gates in `docs/grill-log.md`: date, scope, questions missed, follow-ups. Missed questions become tasks or ADRs. Durable decisions from a grill get an ADR, drafted by Claude from the owner's answers.

## Milestones

Update the status as you go. Order per ADR 0001: M0 and M2 run on CPU, then M1 once the NVIDIA card arrives. Weeks are estimates, not deadlines. Apply for jobs once the interview-ready bar is met: M0–M3 on NVIDIA, with vLLM numbers and gates passed.

| Milestone | Weeks | Gate | Status |
| --- | --- | --- | --- |
| M0 Foundations | 1–2 | Greedy output matches HF for 256 tokens on 20 prompts | Not started |
| M1 Kernels | 3–4 | Every kernel tested, sanitizer-clean, profiled vs peak | Not started |
| M2 Paged KV + batching | 5–6 | 32 concurrent requests, batch-invariant, no leaks | Not started |
| M3 Serving API | 7–8 | OpenAI SDK streams end to end; TTFT/ITL measured | Not started |
| M4 Prefix cache + router | 9–10 | Cache-aware routing beats round-robin on TTFT | Not started |
| M5 Benchmark + write-up | 11–12 | Report vs vLLM and SGLang published | Not started |

## Out of scope

Tensor/pipeline parallelism, quantization, speculative decoding, LoRA, multimodal, multiple architectures, prefill/decode disaggregation. Don't suggest these before M5 is done.
