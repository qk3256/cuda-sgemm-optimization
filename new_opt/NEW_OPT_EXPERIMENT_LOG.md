# New Optimization Experiment Log

All experiments began on 2026-10-01 in branch `codex/new-opt-20261001`. One principal variable should change per experiment. A result is kept only after correctness and both target shapes are considered.

## Experiment 0 — trustworthy baseline and inventory

- Current bottleneck evidence: historical labels and reports were not trustworthy for the new run; an old NCU report had captured a cuBLAS `ampere_sgemm` symbol instead of the custom kernel.
- Hypothesis: a profile-only executable plus strict comparison protocol will identify the real starting point.
- Modification: added `new_opt_runner`, fixed-seed correctness, pedantic FP32 cuBLAS, CUDA Event samples, median/mean/p95, native-shape safety checks, and a profile mode that launches exactly one custom kernel and never invokes cuBLAS.
- Expected metric: not an optimization; establishes valid timing and symbol attribution.
- Correctness: all 26 `(13 kernels) x (2 shapes)` runs passed.
- Performance:
  - square winner kernel 9: 15.55 TFLOP/s stable confirmation, 87.2% of strict cuBLAS
  - small-M winner kernel 4: 4.47 TFLOP/s, 34.6% of strict cuBLAS
- Profiler change: N/A; baseline reports awaiting counter permission.
- Conclusion: **keep**. This is the experimental foundation.

## Experiment 1 — pending baseline NCU diagnosis

- Current bottleneck evidence: pending `baseline_k9_4096.ncu-rep` and `baseline_k4_smallm.ncu-rep`.
- Hypothesis: pending evidence. Static inspection alone shows 102 registers/thread for kernel 9 and 46 for kernel 4; this is not enough to choose the next change.
- Modification: none yet.
- Expected profiler metric: to be chosen from achieved occupancy, issue active, eligible/active warps, FP32 utilization, memory throughput, and stall breakdown.
- Correctness: pending.
- Latency / TFLOP/s / % cuBLAS: pending.
- Profiler change: pending.
- Conclusion: pending.

