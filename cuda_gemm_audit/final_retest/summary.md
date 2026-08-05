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

The confirmed representative IDs are 1 (naive), 3 (shared-memory tiling), 5
(2D register tiling), 10 (warp tiling), and 0 (cuBLAS FP32 reference). Kernel
11 and kernel 12 are both present double-buffer implementations, but the
checked-in benchmark set contains no valid 11-vs-12 comparison. The helper
`scripts/run_final_retest.py` therefore measures both and selects the faster
one from this standardized round; neither is labeled “best” before that
comparison exists.

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
