# Decision records

Most decisions live in the PR plan that made them (`docs/plans/`, "Design" section). An ADR is for the smaller set that outlives a single PR.

## What gets an ADR

Write one when a decision meets any of these:

- **Later PRs depend on it.** Other code will assume it (e.g. KV cache layout, model choice).
- **Someone would reasonably question it.** A reviewer or interviewer would ask "why not X?"
- **Reversing it is expensive.** Changing it later means touching many files or re-measuring.

Everything else stays in the plan doc.

## How

Copy `0000-template.md` to `NNNN-short-title.md`, numbered in order. Claude drafts it from the owner's grill answers, and the owner reviews it with the PR. Write it before or alongside the code, not after. If a decision is reversed, keep the old record, mark it superseded, and write a new one.

## Index

| # | Decision | Status | Plan |
| --- | --- | --- | --- |
| [0001](0001-cpu-first-backend.md) | CPU-first backend trait; CUDA deferred until NVIDIA card; milestone reorder | Accepted | — |
| [0002](0002-model-llama-3.2-1b.md) | Serve Llama 3.2 1B (base) | Accepted | [0001](../plans/0001-config-and-weights.md) |

## Known forks (starting point, not a complete list)

These are the big architectural forks from the original project plan. Each milestone's first grill maps that milestone's full decision tree, and this list gets replaced as it does.

- **M0:** ~~model~~ (settled: ADR 0002); kernel language (CUDA C++ via nvcc vs Rust-on-GPU vs Triton); CUDA bindings (cudarc vs bindgen over the driver API)
- **M1:** GEMM (cuBLAS vs own)
- **M2:** KV block size; preemption (recompute vs swap)
- **M3:** sampling location (GPU vs CPU)
- **M4:** prefix cache structure (block hash vs radix tree); router cache state (shadow tree vs replica reports)
- **M5:** CUDA graphs for decode
