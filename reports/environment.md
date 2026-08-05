# A100 SGEMM Environment

## Host and toolchain

- GPU 0: NVIDIA A100-SXM4-40GB, compute capability 8.0, 40 GiB
- GPU 1: NVIDIA A100-SXM4-40GB, compute capability 8.0, 40 GiB
- Driver: 570.133.20
- CUDA Toolkit: 12.4.131 (`/usr/local/cuda-12.4`)
- CMake: 3.28.3 (user-local installation)
- Ninja: 1.13.0 (user-local installation)
- Nsight Compute: 2024.1.1.0
- Host compiler: GCC 9.4.0
- Selected device: GPU 0
- Build directory: `build-a100`
- CUDA architecture: `80`

The original system CUDA 10.1 and CMake 3.16 remain installed, so all A100
commands explicitly select CUDA 12.4 and the user-local CMake/Ninja binaries.

The GPU device was temporarily absent from one shell session, but was restored
by the host and verified with `nvidia-smi`; both A100 devices are currently
visible. At verification time each device reported approximately 33.7 GiB of
memory in use by other workloads, so measurements should be treated as
single-GPU runs with possible external contention.

## Git state

- Base commit: `5a7dcc513d951ba764d51bc9d587b3163f3a894d`
- Branch: `codex/a100-sgemm-experiment`
- Worktree contains the experiment implementation, scripts, reports, and
  generated artifacts; no destructive cleanup was performed.

## Matrix and cuBLAS semantics

Matrices are row-major: `A[M,K]`, `B[K,N]`, `C[M,N]`. cuBLAS is column-major
and is called with `CUBLAS_OP_N, CUBLAS_OP_N`, passing `B` then `A`, so
`(B^T A^T)^T = A B`. Leading dimensions are `ldb=N`, `lda=K`, `ldc=N`.
The reference uses `cublasGemmEx` with `CUBLAS_COMPUTE_32F`; TF32 is not
enabled as the publishable strict-FP32 reference.

## Kernel map

| ID | Stage |
|---:|---|
| 0 | cuBLAS FP32 reference |
| 1 | naive indexing |
| 2 | global-memory coalescing |
| 3 | shared-memory tiling |
| 4 | 1D block/register tiling |
| 5 | 2D block/register tiling |
| 6 | vectorized memory access |
| 7/8 | shared-memory bank-conflict variants |
| 9 | configurable register tiling |
| 10 | warp tiling |
| 11/12 | double-buffer variants |
