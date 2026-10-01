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
| CUTLASS profiler SIMT `128x64x8`, 2 stages | 4096,4096,4096 | 7.313 ms | 18.799 | 105.5%* |
| strict FP32 cuBLAS | 128,4096,4096 | 0.333 ms | 12.906 | 100% |
| initial best custom, kernel 4 | 128,4096,4096 | 0.962 ms | 4.467 | 34.6% |
| CUTLASS profiler SIMT `128x64x8`, 2 stages | 128,4096,4096 | 0.525 ms | 8.187 | 63.4%* |

`*` The standalone CUTLASS profiler uses a different timing loop. It is useful for selecting an operator, but its square result is not used for the final percentage. The exact generated Universal operator was later embedded in `new_opt_runner` and measured with the identical per-launch Event protocol.

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

## Final controlled comparison

The final 200-iteration sandwich used the same process-independent runner, fixed data, correctness reference, warm-up count, and per-launch CUDA Events:

| Implementation | Shape | Median ms | TFLOP/s | % strict cuBLAS |
|---|---:|---:|---:|---:|
| strict FP32 cuBLAS | 4096x4096x4096 | 7.712–7.714 | 17.817–17.822 | 100% |
| **hand-written kernel 26** | 4096x4096x4096 | **7.755** | **17.723** | **99.45–99.47%** |
| exact CUTLASS Universal SIMT | 4096x4096x4096 | 7.851 | 17.506 | 98.23–98.25% |
| strict FP32 cuBLAS | 128x4096x4096 | 0.334 | 12.866 | 100% |
| hand-written kernel 26 | 128x4096x4096 | 0.635 | 6.765 | 52.6% |
| exact CUTLASS Universal SIMT | 128x4096x4096 | 0.525 | 8.176 | 63.6% |

The hand-written result is not a CUTLASS call. Kernel 22 is the separately named CUTLASS-backed reference; kernels 23–26 are independently compiled CUDA kernels.

The most important transferred ideas were not merely CUTLASS's tile constants. Performance improved when the implementation adopted:

- a balanced CTA/warp/thread hierarchy;
- same-thread register prefetch before current-tile compute;
- double-buffered shared storage;
- a vector-load-compatible A shared layout;
- split-column, coalesced C stores;
- explicit register-residency control.

Conversely, a producer/consumer double buffer and a scalar shared permutation both improved selected low-level metrics but lost end-to-end performance through extra instructions or synchronization.

## Final best kernel

Kernel 26 uses:

- CTA tile `128x64x8`
- warp tile `64x32x8`, arranged as a 2x2 warp grid
- thread tile `8x8`
- 128 threads/block
- two shared-memory stages
- 128 registers/thread, zero compiler stack bytes after `unroll=2`
- 12 KiB shared memory/block
- same-thread global-to-register prefetch, then register-to-shared commit
- vectorized global loads and coalesced split-column stores
- A shared layout that is simultaneously bank-aware for compute reads and cooperative writes
- CUTLASS-style CTA group swizzle of 8 to improve cross-CTA cache locality

The decisive changes were not larger tiles. The original `128x64x8` hand-written kernel generated shared-memory excess during A tile stores because different K groups mapped distinct addresses to the same bank. XORing the K-group bit into the shared row permutation raised square performance from 16.44 to 17.14 TFLOP/s and small-M from 4.72 to 5.50 TFLOP/s. A two-way mainloop unroll then removed the compiler stack frame and raised the results to 17.57 and 6.67 TFLOP/s. Finally, a CTA group size of 8 improved square performance to 17.72 TFLOP/s without changing per-CTA math or resources.

Final NCU comparison explains why the compact hand-written implementation slightly exceeds the exact CUTLASS SIMT operator in the same runner. CUTLASS has an almost perfect shared layout (0.16% wavefront overhead versus 7.4% above ideal for custom) and higher issue activity, but executes 2.780B warp instructions. Custom executes 2.378B, **14.5% fewer**, while sustaining 98.67% FP32 FMA-pipe utilization versus CUTLASS's 97.57%. The result is an explicit tradeoff: simpler address/iterator machinery beats a cleaner shared-memory micro-metric for this configuration.

The remaining 4096-square gap is only about 0.55%, so clock and scheduling variation matters. A wider square sweep found equality at 2048 and 96.9–99.8% elsewhere; apparent short-run wins at 3072 and 6144 did not remain consistently ahead in 200-iteration repeats. Accordingly, this report does **not** claim a stable cuBLAS win. The much larger small-M gap is structural: one 128-row CTA strip provides too few waves and cannot match cuBLAS's shape-specific strategy. The next valuable work would be a dedicated small-M family (smaller BM and/or split-K), not a misleading single-shape timing claim.

The native fast path assumes `M` is divisible by 128, `N` by 512 for group-8 scheduling, and `K` by 8. A production dispatcher should route tails or unsupported shapes to a boundary-safe kernel. This constraint is stated explicitly rather than hidden by the benchmark harness.

