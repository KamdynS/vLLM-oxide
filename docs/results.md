# Results log

Every measured change gets a row: before and after, same workload, same machine. Newest first.

## Environment

| Field | Value |
| --- | --- |
| GPU | |
| GPU memory | |
| Driver | |
| CUDA | |
| Model | |
| dtype | |

Update this table and note the date if anything changes; old rows keep their original environment.

## Kernel results

| Date | Commit | Kernel | Shape | Time (µs) | Achieved GB/s or TFLOPs | % of peak | Bottleneck | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## End-to-end results

| Date | Commit | Change | Workload | Rate (req/s) | TTFT p50/p99 (ms) | ITL p50/p99 (ms) | Throughput (tok/s) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Baselines

| Date | Engine + version | Workload | Rate (req/s) | TTFT p50/p99 (ms) | ITL p50/p99 (ms) | Throughput (tok/s) | Config |
| --- | --- | --- | --- | --- | --- | --- | --- |
