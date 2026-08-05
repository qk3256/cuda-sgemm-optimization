# Implementation facts

## Kernel implementation audit

| Optimization | Actually implemented | Code location | Core logic | Experiment evidence |
|---|---|---|---|---|
| Naive GEMM | Yes | `src/kernels/1_naive.cuh:15` | One output element per thread; direct global-memory loads | Current correctness CSV has measured rows; representative benchmark does not include kernel 1 |
| Global-memory coalescing | Yes | `src/kernels/2_kernel_global_mem_coalesce.cuh:9` | 1D 32×32 block and consecutive thread indexing for loads | Correctness rows are measured; no stable standalone audit benchmark row |
| Shared-memory tiling | Yes | `src/kernels/3_kernel_shared_mem_blocking.cuh:19-53` | `As`/`Bs` shared tiles, barriers around load/compute | Smoke CSV/logs and current correctness rows |
| Register tiling | Yes | `src/kernels/4_kernel_1D_blocktiling.cuh`, `5_kernel_2D_blocktiling.cuh` | Per-thread `threadResults` arrays; operands reused in registers | Current correctness rows; no complete independent performance table |
| Vectorized memory access | Yes | `src/kernels/6_kernel_vectorize.cuh:47-55`, output at `:85-95` | `float4` global loads/stores and transposed shared-memory A | Current correctness rows; compiler resource data available via cuobjdump |
| Bank-conflict variants | Yes | `src/kernels/7_kernel_resolve_bank_conflicts.cuh:57-74`, `8_kernel_bank_extra_col.cuh:26-74` | Reordered/padded shared-memory B layouts | Current correctness rows; no complete per-stage benchmark table |
| Warp tiling | Yes | `src/kernels/10_kernel_warptiling.cuh:100-186` | Warp placement, warp subtiles, shared tiles, register accumulators | Direct 4096³ and 128×4096×4096 timings plus NCU |
| Double buffering/software pipelining | Yes | `src/kernels/11_kernel_double_buffering.cuh:106-198` | Two shared-memory buffers with alternating producer/consumer paths | Implemented; no complete independent performance claim |
| Async/barrier double buffering | Yes | `src/kernels/12_kernel_double_buffering.cuh:108-201` | `cuda::barrier`, `cuda::memcpy_async`, two shared buffers | Implemented; no complete independent performance claim |
| Tensor Core/WMMA custom kernel | No evidence | No `wmma`, `mma_sync`, `__half`, or BF16 custom kernel found | Not part of the custom-kernel ladder | Do not claim custom Tensor Core implementation |

## Numerical and layout facts

1. Data type: custom kernels and the publishable reference are FP32. The
   source also contains unused/reference helper functions for BF16 and TF32
   cuBLAS modes (`src/runner.cu:134-149`); the audited kernel IDs use the strict
   FP32 cuBLAS path.
2. Layout: host logical matrices are row-major `A[M,K]`, `B[K,N]`, `C[M,N]`.
   cuBLAS receives `B` then `A` with `CUBLAS_OP_N/CUBLAS_OP_N`, leading
   dimensions `N`, `K`, `N`, implementing `(B^T A^T)^T = A B`.
3. Tiles: concrete block/tile values are set in `src/runner.cu`; examples are
   32×32 shared tiles (kernel 3), 64/128 register tiles (kernels 4-8), and
   kernel-10 default `BM=BN=128`, `BK=16`, `WM=WN=64`, `TM=8`, `TN=4`, 128
   threads.
4. Registers cache per-thread output accumulators and short operand vectors;
   shared memory caches A/B block tiles. Kernel 10 explicitly declares
   `threadResults`, `regM`, and `regN` at lines 142-146.
5. Boundary handling: the original optimized kernels do not generally guard
   edge tiles. `experiment_runner.cu:21-29` detects exact-safe shapes and
   substitutes a safe naive kernel for unsafe custom-kernel shapes, reporting
   `fallback_naive=1`. This is harness behavior, not proof that the original
   optimized kernels are independently edge-safe.
6. Reference: strict FP32 cuBLAS `cublasGemmEx` at `experiment_runner.cu:31-32`.
7. Timing: `experiment_runner.cu:51-55` preallocates buffers, warms up, records
   CUDA Events without synchronization inside the measurement loop, synchronizes
   once after the sequence, then reports median/mean/P95/std.
8. FLOPs: `2*M*N*K`; TFLOP/s is computed from median milliseconds at
   `experiment_runner.cu:55`.

