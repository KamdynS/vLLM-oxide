# 0001: CPU-first backend trait; CUDA deferred until the NVIDIA card arrives

- **Status:** Accepted
- **Date:** 2026-11-05
- **Milestone:** M0

## Context

I don't have an NVIDIA card yet (expected around 2026-11-30, about a $600 budget, model not chosen). I have plenty of work I can start without one: the scheduler, the KV block manager, and a CPU reference for paged attention. The risk is drift: the "model" on CPU won't be the same as it will be once NVIDIA hardware arrives.

My AMD card (RX 6700 or RX 7700 XT, unconfirmed) drives my display and will not be used.

_Claude's addition:_ drift comes in two kinds, and they need different fixes:
- **Structural:** engine code that assumes host memory (indexing a `Vec<f32>` of KV directly, allocating mid-step) would force a rewrite of the scheduler and block manager. The backend trait prevents this.
- **Numerical:** CPU f32 and GPU bf16 logits differ, and greedy tokens can flip. No trait prevents this. It is handled by a stated tolerance and tests against the CPU reference.

## Options

| Option | For | Against |
| --- | --- | --- |
| CPU-first behind a backend trait, CUDA when the card arrives | Start now on device-agnostic work; CPU backend stays as the kernel test oracle | Kernels come last; CPU performance numbers are only meaningful in relative terms |
| Use the AMD card (ROCm/HIP) now | Real GPU numbers sooner | Consumer RDNA has patchy ROCm support and weak profilers; I'd learn workarounds, not inference |
| Borrowed Triton kernels | Fastest route to GPU numbers | Code I didn't design and can't defend; Python/Triton plumbing in a Rust build |
| Wait for the card | Nothing wasted | Weeks lost on work that doesn't need a GPU |

## Decision

Build CPU-first behind a backend trait, and add a CUDA backend when the NVIDIA card arrives.

Consequences for the plan:
- **Milestone order:** M0 (match HF) and M2 (paged KV + batching) on CPU, then M1 (kernels) on NVIDIA. M3–M5 as planned. Exit criteria are unchanged. M3 server work may start on CPU if the card is late.
- **Dates are estimates.** Instead of a calendar go/no-go, I apply once the interview-ready bar is met: M0–M3 working on NVIDIA, with vLLM numbers and the gates passed.

## Evidence

No measurements yet; this is a sequencing decision, not a performance one. The supporting fact: M2's core logic (block ownership, refcounts, scheduling) is bookkeeping and does not depend on where the math runs.

- `TODO(measure)`: CPU decode tok/s for the chosen model at batch 1. This sets how slow the M0 256-token × 20-prompt check will be.

## Consequences

- **Easier:** work starts today, and the property tests for the block manager and scheduler run without a GPU. Every future kernel has a CPU reference to test against.
- **Harder:** the trait has to be designed before we know the CUDA backend's needs, so it may change at M1. Kernels, the most interview-visible piece, come last.
- **Revisit if:** the card slips past about month 2 (then rent cloud NVIDIA for M1), or the trait turns out to force host-device syncs in the hot loop.

## Grill questions this must survive

- Why not use the AMD card you already have?
- What in the engine would have to change when CUDA arrives, and why is that list short?
- How do you tell a numerical difference between CPU and GPU apart from a bug?
