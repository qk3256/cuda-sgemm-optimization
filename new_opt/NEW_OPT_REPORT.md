# CUDA FP32 SGEMM New Optimization Report

Experiment start: 2026-10-01  
Branch: `codex/new-opt-20261001`  
Worktree: `/home/why/workspace/lab/cuda-sgemm/SGEMM_NEW_OPT_20261001`

This report is intentionally separate from the historical SGEMM results. No old result file is overwritten.

## Reproducibility contract

- GPU: NVIDIA A100-SXM4-40GB, compute capability 8.0, 108 SMs
- Driver: 570.133.20
- CUDA compiler: 12.4, build 12.4.131
- Inputs, output, and accumulation: FP32
- cuBLAS reference: `CUBLAS_PEDANTIC_MATH`, `CUBLAS_COMPUTE_32F_PEDANTIC`, `CUBLAS_GEMM_DEFAULT`
- Timing: warm-up followed by per-launch CUDA Event timing; median is the headline latency
- Correctness: fixed seed, strict cuBLAS reference, finite check, max-absolute and relative-L2 checks
- Shapes: `4096x4096x4096` and small-M `128x4096x4096`

## Initial baseline

Stable confirmation runs used 10 warm-ups and 50 iterations for the square case, and 10 warm-ups and 100 iterations for small-M.

| Implementation | M,N,K | Median latency | TFLOP/s | % strict cuBLAS |
|---|---:|---:|---:|---:|
| strict FP32 cuBLAS | 4096,4096,4096 | 7.711 ms | 17.824 | 100% |
| initial best custom, kernel 9 | 4096,4096,4096 | 8.839 ms | 15.549 | 87.2% |
| CUTLASS SIMT `128x64x8`, 2 stages | 4096,4096,4096 | 7.313 ms | 18.799 | 105.5% |
| strict FP32 cuBLAS | 128,4096,4096 | 0.333 ms | 12.906 | 100% |
| initial best custom, kernel 4 | 128,4096,4096 | 0.962 ms | 4.467 | 34.6% |
| CUTLASS SIMT `128x64x8`, 2 stages | 128,4096,4096 | 0.525 ms | 8.187 | 63.4% |

The full 13-kernel inventory is in `NEW_OPT_RESULTS.csv`. The old project label “warp-tiled best” did not survive the controlled re-baseline: kernel 10 measured 13.22 TFLOP/s while kernel 9 measured 15.36 TFLOP/s in the same 20-iteration sweep.

## Initial kernel anatomy

### Square-shape winner: kernel 9

- CTA tile: `BM=128, BN=128, BK=16`
- thread tile: `TM=8, TN=8`
- threads/block: 256
- compiler resources: 102 registers/thread, 16 KiB static shared memory/block
- data flow: vectorized cooperative global loads to shared memory, then shared-to-register outer products

### Historical warp-tiled kernel 10

- CTA tile: `128x128x16`
- warp tile: `64x64`
- thread tile: `8x4`, with four N sub-iterations
- threads/block: 128
- compiler resources: 167 registers/thread, 16 KiB static shared memory/block

The much higher register count is a concrete warning sign, but it is not yet sufficient to claim an occupancy bottleneck. The NCU report must confirm achieved occupancy, eligible warps, issue activity, and stalls.

### Small-M winner: kernel 4

- CTA tile: `64x64x8`
- one-dimensional register tile: eight outputs/thread
- threads/block: 512
- compiler resources: 46 registers/thread, 4 KiB static shared memory/block

Its smaller `BM` creates more CTAs than the 128-row kernels, which is useful when M is only 128. NCU is required to separate grid-level underfill from instruction- and memory-level inefficiency.

## CUTLASS reference design

The fastest available CUTLASS FP32 SIMT candidate is `cutlass_simt_sgemm_128x64_8x2_nn_align1`:

- CTA tile `128x64x8`
- warp tile `64x32x8`
- 2x2 warps = 128 threads/block
- scalar SIMT instruction shape `1x1x1`
- two pipeline stages
- FP32 accumulator and `OpMultiplyAdd`

The generated operator is architecture-generic SIMT with minimum CC 5.0 rather than a Tensor Core kernel; it executes on Ampere without TF32. The useful ideas to transfer are the CTA/warp/thread hierarchy, balanced per-thread tile, cooperative iterators, shared-memory layouts, and staged mainloop—not the surrounding template machinery.

## Current status

The baseline, CUTLASS comparison, and three generations of NCU-guided custom work are complete. The current square best is kernel 17 at 17.13 TFLOP/s, 96.16% of same-session strict cuBLAS. The current small-M best is kernel 19 at 4.72 TFLOP/s, up 5.6% from the initial shape-specific best but still far below cuBLAS and CUTLASS.

The most important transferred ideas were not merely CUTLASS's tile constants. Performance improved when the implementation adopted:

- a balanced CTA/warp/thread hierarchy;
- same-thread register prefetch before current-tile compute;
- double-buffered shared storage;
- a vector-load-compatible A shared layout;
- split-column, coalesced C stores;
- explicit register-residency control.

Conversely, a producer/consumer double buffer and a scalar shared permutation both improved selected low-level metrics but lost end-to-end performance through extra instructions or synchronization.

## Final best and remaining gap

Current square best: kernel 17, CTA `128x128x16`, thread tile `8x8`, two stages, 256 threads, 128 registers/thread, 32 KiB shared/block, 17.13 TFLOP/s and 96.16% of strict cuBLAS. Profiling of kernel 17 is pending final extraction before the next optimization decision.

