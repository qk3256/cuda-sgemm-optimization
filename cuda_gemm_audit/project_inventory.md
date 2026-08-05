# CUDA GEMM project inventory

## Scope and selected project

- Primary project: `/home/why/workspace/infra/cuda-sgemm/SGEMM_CUDA`
- Reason selected: it contains the complete CUDA SGEMM ladder, a cuBLAS
  reference, an A100 runner, benchmark/correctness scripts, and NCU reports.
- No second project with stronger directly attributable evidence was found in
  the current worktree. The audit did not expand the search into unrelated
  repositories.

## Git

- Branch: `codex/a100-sgemm-experiment`
- HEAD/base commit: `5a7dcc513d951ba764d51bc9d587b3163f3a894d`
- Recent history includes the upstream verification fix `1981307`.
- Worktree is dirty with experiment source, scripts, reports, results, and
  `build-a100/`; these changes predate this audit and are preserved.

## Main source files

- `src/runner.cu`: kernel dispatch, cuBLAS reference, compile-time tile choices.
- `src/kernels/1_naive.cuh` through `12_kernel_double_buffering.cuh`:
  kernel implementations.
- `experiment_runner.cu`: deterministic correctness and CUDA Event harness.
- `CMakeLists.txt`, `Makefile`: CUDA build configuration and targets.
- `cuBLAS_sgemm.cu`, `sgemm.cu`, `simplest_kernel.cu`: standalone/original
  runners.

## Kernel IDs

| ID | Implementation stage | Source/dispatch evidence |
|---:|---|---|
| 0 | cuBLAS FP32 reference | `src/runner.cu:124`, dispatch at `:506` |
| 1 | naive indexing | `src/kernels/1_naive.cuh`, `src/runner.cu:152` |
| 2 | global-memory coalescing | `src/kernels/2_kernel_global_mem_coalesce.cuh`, `:159` |
| 3 | shared-memory tiling | `src/kernels/3_kernel_shared_mem_blocking.cuh`, `:167` |
| 4 | 1D block/register tiling | `src/kernels/4_kernel_1D_blocktiling.cuh`, `:181` |
| 5 | 2D block/register tiling | `src/kernels/5_kernel_2D_blocktiling.cuh`, `:193` |
| 6 | vectorized memory access | `src/kernels/6_kernel_vectorize.cuh`, `:217` |
| 7 | bank-conflict layout variant | `src/kernels/7_kernel_resolve_bank_conflicts.cuh`, `:241` |
| 8 | extra-column bank variant | `src/kernels/8_kernel_bank_extra_col.cuh`, `:265` |
| 9 | configurable register tiling | `src/kernels/9_kernel_autotuned.cuh`, `:289` |
| 10 | warp tiling | `src/kernels/10_kernel_warptiling.cuh`, `:331` |
| 11 | double buffering | `src/kernels/11_kernel_double_buffering.cuh`, `:392` |
| 12 | `cuda::memcpy_async`/barrier double buffering | `src/kernels/12_kernel_double_buffering.cuh`, `:453` |

## Scripts and artifacts

- Correctness: `scripts/check_correctness.sh`.
- CUDA Event benchmark: `scripts/run_benchmarks.sh`.
- Real bounded kernel-10 compile/run search: `scripts/run_autotune_real.sh`.
- NCU collection/import: `scripts/run_ncu.sh`, `scripts/run_ncu_root.sh`.
- Existing reports: `reports/environment.md`, `reports/a100_sgemm_report.md`.
- Existing raw NCU reports: `profiles/raw/*.ncu-rep` for 4096³ and
  128×4096×4096.
- Existing structured data: `results/*.csv`, `results/*.json`.

## Build/run status

`build-a100` is configured for `/usr/local/cuda-12.4/bin/nvcc`, CUDA
architecture 80, and CMake 3.28.3. `cmake --build build-a100 --parallel`
currently reports `ninja: no work to do`. A100 runtime and NCU execution are
proven by the existing raw logs and the current correctness process; the
default non-elevated Codex namespace may not expose `/dev/nvidia*`, while the
user's node161 shell and elevated tool namespace do.

