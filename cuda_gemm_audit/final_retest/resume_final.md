# Resume-ready final facts

All figures below are medians across three independent rounds from
`final_retest/results.csv`.

- Implemented and validated an FP32 SGEMM optimization ladder on NVIDIA A100
  SM80: naive (k1), shared-memory tiling (k3), 2D register tiling (k5), and
  warp tiling (k10), using cuBLAS FP32 (k0) as the reference.
- On 4096³, warp tiling (k10) reached **16.4233 TFLOP/s**, **56.22×** the
  naive kernel and **92.12%** of cuBLAS, with `pass=1` in all three rounds.
- On 128×4096×4096, k10 reached **3.1092 TFLOP/s** and **22.09%** of cuBLAS,
  documenting the medium-M utilization/throughput drop rather than hiding it.

Boundary correctness was also checked on 127×4097×1025: optimized kernels
reported `fallback_naive=1`, so these results must not be described as native
arbitrary-dimension support.
