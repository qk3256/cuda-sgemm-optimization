# Final retest summary

## Protocol requested

The planned protocol was fixed seed 1234, fixed GPU 0, at least 10 warmup
iterations, 50 CUDA Event measurements, and three independent rounds for each
representative kernel/shape pair. The four shapes were:

- 4096×4096×4096;
- 128×4096×4096;
- existing LLM-like rectangular benchmark shape 1000×1536×768 (no model mapping
  is asserted);
- existing non-divisible smoke shape 127×4097×1025.

The representative kernel IDs were 1 (naive), 3 (shared-memory tiling), 5
(2D register tiling), 10 (warp tiling), 11 (the double-buffer stage with an
existing historical benchmark row), and 0 (cuBLAS FP32 reference). Kernel 12
is the async/barrier double-buffer variant, but no existing evidence proves it
is faster than kernel 11, so it was not mislabeled as the best double-buffer
stage.

## Outcome

No final-retest workload could be launched from the current Codex execution
namespace because the GPU device nodes are absent. `results.csv` and
`correctness.csv` therefore contain explicit `NOT_RUN_GPU_UNAVAILABLE` rows,
not guessed latency or error values. Existing prior logs remain available under
`profiles/` and are not treated as this round's results.

## Resume consequence

This round cannot promote the project from “representative evidence exists” to
“standardized 10×50×3 retest completed.” The existing resume numbers remain
usable only with their original protocol/source, not as final_retest numbers.

