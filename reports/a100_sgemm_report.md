# A100 SGEMM Experiment Report

## Executive summary

The repository was rebuilt for NVIDIA A100 SM80 with CUDA 12.4. The harness
uses deterministic inputs, strict FP32 cuBLAS reference results, preallocated
buffers, CUDA Events, warmup, repeated measurements, and explicit reporting of
boundary fallbacks. A representative 4096³ run measured 14.80 TFLOP/s for
cuBLAS and 13.21 TFLOP/s for the existing warp-tiling kernel (kernel 10).

## Directly measured results

Representative CUDA Event measurements on GPU 0, seed 1234, three warmups and
five measured iterations:

| Shape | Kernel | Median latency | TFLOP/s | Correctness |
|---|---:|---:|---:|---|
| 4096³ | cuBLAS (0) | 9.28563 ms | 14.8012 | pass |
| 4096³ | warp tiling (10) | 10.4038 ms | 13.2104 | pass |
| 128×4096×4096 | cuBLAS (0) | 0.329728 ms | 13.0258 | pass |
| 128×4096×4096 | warp tiling (10) | 1.38138 ms | 3.10919 | pass |

The NCU-instrumented 4096³ kernel-10 run measured 9.310176 ms and 15.355
TFLOP/s in the application timing output. NCU reported the kernel
`ampere_sgemm_128x64_nn` with SM Throughput 97.804%, DRAM Throughput 9.031%,
122 registers/thread, and 13.568 KB shared memory/block.

For the requested medium-M profile `128×4096×4096`, NCU reported 323.392 us
kernel duration, 88.764492% SM Throughput, 16.842719% DRAM Throughput, 122
registers/thread, and 13.568 KB shared memory/block. Both representative
profiles support a compute-throughput-dominant classification.

Correctness smoke coverage passed for aligned `128×256×64` across cuBLAS and
kernels 1/2/3/10/12. Non-divisible `127×257×65` passed for cuBLAS and kernels
1/2; kernels 3/10 used the explicitly reported safe naive fallback. Typical
errors were max absolute `3.8e-6` and normalized L2 `1.6e-7`.

## Methodology

`build-a100/sgemm_experiment` preallocates A/B/C/reference buffers, uses fixed
`std::mt19937` inputs, warms up, records CUDA Event pairs asynchronously, and
synchronizes once after the measured event sequence. It reports median, mean,
P95, standard deviation, TFLOP/s using `2*M*N*K`, max absolute/relative error,
normalized L2, and NaN/Inf counts. The correctness pass criterion is
absolute error `<1e-2`, normalized L2 `<1e-4`, and zero NaN/Inf; max relative
error is retained as a diagnostic and stabilized near zero.

## Evidence-backed interpretation

- The NCU SM Throughput of 97.804% and DRAM Throughput of 9.031% classify the
  representative warp-tiling kernel as compute-bound under the collected
  workload.
- Compiler/NCU evidence shows 122 registers/thread and 13.568 KB shared memory
  per block; register allocation is therefore a plausible occupancy/resource
  tradeoff, but an occupancy conclusion requires comparing candidate builds.
- The existing optimized kernels do not all guard edge tiles. The harness uses
  `fallback_naive=1` for unsafe non-divisible shapes and reports this explicitly.

## Tuning status and negative results

The repository exposes real compile-time parameters for kernels 9 and 10. A
bounded real compile-and-run search was completed for three kernel-10
configurations on the large and small shape classes. The best measured
candidate in this limited search was `BM=64, BN=128, BK=16, WM=32, WN=64,
WNITER=1, TM=4, TN=4, NUM_THREADS=128`, at 13.1409 TFLOP/s on 4096³
(0.887827× the measured cuBLAS baseline) and 3.09165 TFLOP/s on
128×4096×4096 (0.237348× cuBLAS). The larger default warp-tile configuration
was slower in this run; one 256-thread candidate was rejected by the exposed
warp-count static assertion. These are bounded-search results, not a global
optimum.

## Resume-ready bullets

- Ported a CUDA SGEMM optimization ladder to NVIDIA A100 SM80 with CUDA 12.4,
  covering cuBLAS, coalesced/global, shared-memory, register-tiled,
  vectorized, bank-conflict, warp-tiled, and double-buffered stages.
- Built a deterministic cuBLAS-referenced FP32 correctness and CUDA Event
  benchmarking harness with arbitrary M/N/K, fixed seeds, P95/standard
  deviation, NaN/Inf checks, and safe handling of non-tile-aligned shapes.
- Measured 14.80 TFLOP/s cuBLAS and 13.21 TFLOP/s for the existing warp-tiling
  kernel on A100 for 4096³; NCU showed 97.80% SM throughput and 9.03% DRAM
  throughput for the representative warp-tiled kernel.
- Added reproducible build, correctness, benchmark, autotune, and Nsight
  Compute artifact paths with compiler-resource evidence capture.

## Remaining blockers before a stronger tuning claim

At the latest verification point, the CUDA 12.4 build still completes, but the
runtime environment no longer exposes `/dev/nvidia*`; `nvidia-smi` reports that
it cannot communicate with the NVIDIA driver and the executable reports
`no CUDA-capable device is detected`. Previously collected A100/NCU artifacts
remain valid, but no new GPU measurements can be made until the device is
restored.

1. Complete the full 13-kernel × seven-shape benchmark CSV; the process-per-row
   runner is slow because each invocation initializes CUDA/cuBLAS.
2. Extend the bounded search to kernel 9 and all requested shape classes; the
   current kernel-10 search covers large and small classes.
3. Commit the completed phases; the current worktree is intentionally dirty.
