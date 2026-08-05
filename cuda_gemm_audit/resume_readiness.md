# Resume readiness

## Decision: B — can write the implementation and methodology, but avoid broad performance claims

The project satisfies the code-location and implementation requirements for a
resume bullet. The custom kernels, cuBLAS reference, CUDA Event harness, NCU
reports, and A100/SM80 build are all directly inspectable. Correctness is
confirmed for the measured representative/smoke rows, but the full requested
matrix and the required 50-iteration three-round audit re-test are not present
as independent evidence. The benchmark CSV and autotune search are also
partial.

## Safe claims

- Implemented the listed CUDA optimization stages from naive indexing through
  shared-memory/register/warp tiling and double-buffer variants.
- Built a deterministic FP32 cuBLAS-referenced correctness and CUDA Event
  measurement harness with arbitrary dimensions, fixed seed/device, and
  explicit fallback reporting.
- Collected NCU reports on A100 SM80 for kernel 10 at 4096³ and
  128×4096×4096, with directly measured SOL and resource metrics.
- Reported representative measured throughput numbers with their exact shape,
  iteration count, and source CSV/log.

## Claims not yet safe

- “All 13 kernels pass all seven requested shapes.”
- “Global best autotune configuration.”
- “Kernel 10 is universally compute-bound.”
- Any relative-to-naive speedup, because the audit subset does not contain a
  same-protocol naive baseline for both headline shapes.

