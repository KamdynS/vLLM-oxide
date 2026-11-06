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

## Kernel results

| Date | Commit | Kernel | Shape | Time (µs) | Achieved GB/s or TFLOPs | % of peak | Bottleneck | Notes |
| ---- | ------ | ------ | ----- | --------- | ----------------------- | --------- | ---------- | ----- |

## End-to-end results

| Date | Commit | Change | Workload | Rate (req/s) | TTFT p50/p99 (ms) | ITL p50/p99 (ms) | Throughput (tok/s) | Notes |
| ---- | ------ | ------ | -------- | ------------ | ----------------- | ---------------- | ------------------ | ----- |

## Baselines

| Date | Engine + version | Workload | Rate (req/s) | TTFT p50/p99 (ms) | ITL p50/p99 (ms) | Throughput (tok/s) | Config |
| ---- | ---------------- | -------- | ------------ | ----------------- | ---------------- | ------------------ | ------ |
