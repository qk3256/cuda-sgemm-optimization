# Resume draft

## Detailed version

**CUDA SGEMM implementation and A100 performance audit**

1. Implemented and mapped a CUDA FP32 SGEMM optimization ladder covering naive
   indexing, coalesced global loads, shared-memory tiling, 1D/2D register
   tiling, vectorized `float4` access, shared-memory bank-layout variants,
   warp tiling, and double buffering; each stage is traceable to a kernel
   header and dispatch ID.
2. Built a deterministic cuBLAS-referenced correctness and CUDA Event harness
   with arbitrary M/N/K, fixed seed/device, warmup, median/mean/P95/std latency,
   TFLOP/s from `2*M*N*K`, error metrics, and explicit safe fallback reporting
   for non-tile-aligned shapes.
3. Rebuilt for NVIDIA A100-SXM4-40GB SM80 with CUDA 12.4 and collected NCU
   evidence for warp tiling: 97.804% SM throughput / 9.031% DRAM throughput
   at 4096³ and 88.764% / 16.843% at 128×4096×4096; representative timing
   rows measured 14.8012 TFLOP/s cuBLAS and 13.2104 TFLOP/s kernel 10 at 4096³.

## Compressed version

1. Implemented a traceable CUDA FP32 SGEMM optimization ladder through
   shared-memory/register/warp tiling and double buffering, with deterministic
   cuBLAS correctness and CUDA Event measurement tooling.
2. On A100 SM80, measured 14.8012 TFLOP/s cuBLAS and 13.2104 TFLOP/s for the
   existing warp-tiled kernel at 4096³; NCU reported 97.804% SM throughput for
   the representative warp-tiled kernel.

