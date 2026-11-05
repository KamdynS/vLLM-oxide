# Decision records

One file per decision someone could reasonably question. Copy `0000-template.md` to `NNNN-short-title.md`, numbered in order.

Write the ADR before or alongside the code, not after. If a decision is reversed, keep the old record, mark it superseded, and write a new one.

## Open decisions (from the project plan)

| # | Decision | Decide by |
| --- | --- | --- |
| 0001 | Model (Llama 3.2 1B vs Qwen2.5 0.5B/1.5B vs TinyLlama) | M0 |
| 0002 | Kernel language (CUDA C++ via nvcc vs Rust-on-GPU vs Triton) | M0 |
| 0003 | CUDA bindings (cudarc vs bindgen over the driver API) | M0 |
| 0004 | GEMM (cuBLAS vs own) | M1 |
| 0005 | KV block size | M2 |
| 0006 | Preemption (recompute vs swap) | M2 |
| 0007 | Sampling location (GPU vs CPU) | M3 |
| 0008 | Prefix cache structure (block hash vs radix tree) | M4 |
| 0009 | Router cache state (shadow tree vs replica reports) | M4 |
| 0010 | CUDA graphs for decode | M5 |
