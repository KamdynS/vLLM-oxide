# Results log

Every measured change gets a row: before and after, same workload, same machine. Newest first.

## Environment

| Field      | Value                                                                        |
| ---------- | ---------------------------------------------------------------------------- |
| CPU        | AMD Ryzen 7 7700X (Zen 4, 8C/16T, AVX-512), 32 GB DDR5, no swap (2026-11-06) |
| Disk       | Samsung 990 PRO 2TB NVMe                                                     |
| GPU        | none yet (ADR 0001)                                                          |
| GPU memory |                                                                              |
| Driver     |                                                                              |
| CUDA       |                                                                              |
| Model      | Llama 3.2 1B base (ADR 0002)                                                 |
| dtype      | bf16 on disk, f32 in memory                                                  |

Update this table and note the date if anything changes; old rows keep their original environment.

## Load results

Loading `model.safetensors` (2.47 GB) into f32 `ModelWeights`. Cold means the file was evicted from page cache first (`dd iflag=nocache count=0`; Cached dropped 2.4 GB). Three runs each.

| Date       | Commit      | Change                        | Cold load (s) | Warm load (s) | Peak RSS           | Owner estimate           | Notes                                                                                                                                                                                                                                                               |
| ---------- | ----------- | ----------------------------- | ------------- | ------------- | ------------------ | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-11-06 | b110143+PR1 | `fs::read` + bf16→f32 collect | 2.34–2.40     | 2.17–2.20     | 7.24 GiB (7.42 GB) | ~1 s; peak not estimated | Peak matches 2.47 GB file + 4.94 GB f32. Cold − warm is only ~0.2 s, so the disk is not the bottleneck. Gap explanation: I did not account for how long it would take for the OS to actually give us the memory. That added about a second on top of our estimates. |

## Correctness results

Model-level agreement with HF transformers (`tests/hf_match.rs`, reference from `tools/reference/dump_logits.py`). Both sides are f32 on CPU, and HF uses eager attention.

| Date | Commit | Check | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-11-07 | M0 PR2 | Greedy tokens, 20 prompts × 256 (or to EOS) | **20/20 exact** | Prompts 1–238 tokens, including 16 and 17 tokens to cross a block boundary. HF's smallest top-1/top-2 gap was 9e-4 (p01), so the run included near-ties. |
| 2026-11-07 | M0 PR2 | Max \|logit diff\| at steps 0, 1, 64, 255 | **5.4e-5** worst, typically 1–2e-5 | From summation order over 16 layers. Tolerance set to 2e-4: ~4× headroom, still below the smallest gap. |
| 2026-11-07 | M0 PR2 | Arena peak vs `arena_elems` | Equal (unit test); 21.4 MB at a 238-token prefill | The formula gives 184 MB at `max_batch_tokens = 2048`. |

## CPU reference speed (test-runtime planning only)

Not a performance target, and not estimated or optimized (process.md). Recorded so test runtimes are predictable.

| Date | Commit | Workload | Prefill | Decode | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-11-07 | M0 PR2 | 238-token prompt, 128 tokens out, batch 1, f32, 16 rayon threads | 149 tok/s (1.6 s) | 10.0 tok/s (100 ms/token) | Matmul 94% of op time, decode attention 4%. 4.94 GB / 100 ms ≈ 49 GB/s of weight reads, close to this box's DRAM bandwidth, so decode is memory-bound. The M0 gate test takes 7.6 min. |

## Kernel results

| Date | Commit | Kernel | Shape | Time (µs) | Achieved GB/s or TFLOPs | % of peak | Bottleneck | Notes |
| ---- | ------ | ------ | ----- | --------- | ----------------------- | --------- | ---------- | ----- |

## End-to-end results

| Date | Commit | Change | Workload | Rate (req/s) | TTFT p50/p99 (ms) | ITL p50/p99 (ms) | Throughput (tok/s) | Notes |
| ---- | ------ | ------ | -------- | ------------ | ----------------- | ---------------- | ------------------ | ----- |

## Baselines

| Date | Engine + version | Workload | Rate (req/s) | TTFT p50/p99 (ms) | ITL p50/p99 (ms) | Throughput (tok/s) | Config |
| ---- | ---------------- | -------- | ------------ | ----------------- | ---------------- | ------------------ | ------ |
