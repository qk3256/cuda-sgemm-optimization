# Resume finalization status

## Decision

Do not publish new numeric claims from `final_retest/`: the standardized
10-warmup/50-measurement/3-round protocol was not executed because the current
Codex runtime namespace lacks GPU device access.

## Previously supported wording

The earlier representative measurements can still be used if their protocol
is stated exactly:

- 4096³ cuBLAS FP32: 14.8012 TFLOP/s from the existing representative CSV;
- 4096³ warp-tiling kernel 10: 13.2104 TFLOP/s from the same CSV;
- NCU kernel-10 evidence: 97.804% SM throughput and 9.031% DRAM throughput
  for the existing 4096³ profile.

These are not final-retest numbers.

## Missing final evidence

- 6 representative kernels × 3 performance shapes × 3 rounds;
- 50 measurements per round;
- final cross-kernel speedup-vs-naive and ratio-vs-cuBLAS values;
- final non-divisible execution-path verification under this protocol.

