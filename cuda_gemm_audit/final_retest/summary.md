# Final retest summary

## Protocol and evidence

This round used commit `19546861d594697543c78f98c17c3b0430cad147`, fixed GPU 0
(NVIDIA A100-SXM4-40GB, SM80), seed 1234, 10 warm-up iterations, 50 CUDA Event
measurements, and three independent rounds. All 70/70 records are measured and
pass correctness. The measured rows are in `results.csv`; error metrics are in
`correctness.csv`.

## Representative IDs

- cuBLAS FP32 reference: kernel 0
- naive: kernel 1
- shared-memory tiling: kernel 3
- 2D register tiling: kernel 5
- warp tiling: kernel 10
- existing double buffering: kernels 11 and 12 were both tested

There is no single shape-independent double-buffer winner: kernel 11 is faster
than 12 on 4096³ (15.1819 vs 15.0962 TFLOP/s), while kernel 12 is faster on
128×4096×4096 (3.0328 vs 1.8420 TFLOP/s). The LLM-like shape falls back to the
naive path for kernels 3/5/10/11/12, so it cannot select a native optimized
double-buffer winner.

## Performance ladder (median across the three rounds)

| Shape | Naive | Shared | Reg2D | Warp | Best double buffer | cuBLAS |
|---|---:|---:|---:|---:|---:|---:|
| 4096³ | 0.2921 | 5.3818 | 13.9696 | 16.4233 | 15.1819 (k11) | 17.8291 TFLOP/s |
| 128×4096×4096 | 0.2775 | 3.7787 | 2.7943 | 3.1092 | 3.0328 (k12) | 13.0664 TFLOP/s |
| 1000×1536×768 | 0.2800 | 2.3728* | 2.3753* | 2.3753* | 2.3753* | 8.0559 TFLOP/s |

`*` means `fallback_naive=1`; these values are not native performance of the
named optimized kernel.

For 4096³, kernel 10 reaches 16.4233 TFLOP/s, 56.22× naive, and 92.12% of
cuBLAS. The prior 13.2104 TFLOP/s result is not reproduced under this round's
protocol; the new result is faster, with different timing/toolkit conditions.
For 128×4096×4096, kernel 10 reaches 3.1092 TFLOP/s and only 22.09% of cuBLAS,
confirming the pronounced medium-M degradation.

## Boundary path

For 127×4097×1025, kernels 3, 5, 10, 11, and 12 all report
`fallback_naive=1` and `pass=1`. Kernels 0 and 1 execute natively. Thus the
optimized kernels were not demonstrated to support arbitrary non-divisible
dimensions; this shape exercises the explicit fallback path.
