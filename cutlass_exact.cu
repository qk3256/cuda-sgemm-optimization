#include <cuda_runtime.h>

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

// This is the exact template configuration emitted by CUTLASS's profiler as
// cutlass_simt_sgemm_128x64_8x2_nn_align1.  It remains a separate baseline:
// it is not presented as one of the hand-written kernels.
using CutlassExactSgemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<128, 64, 8>,
    cutlass::gemm::GemmShape<64, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAdd>;

cudaError_t runCutlassExactSgemm(int M, int N, int K, float alpha,
                                const float *A, const float *B, float beta,
                                float *C) {
  CutlassExactSgemm op;
  CutlassExactSgemm::Arguments args(
      {M, N, K}, {A, K}, {B, N}, {C, N}, {C, N}, {alpha, beta});
  return op(args) == cutlass::Status::kSuccess ? cudaSuccess
                                               : cudaErrorLaunchFailure;
}

// Exact kernel-level type from the CUTLASS profiler's generated source.
using CutlassExactUniversalKernel =
    typename cutlass::gemm::kernel::DefaultGemmUniversal<
        float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,
        float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,
        float, cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        cutlass::gemm::GemmShape<128, 64, 8>,
        cutlass::gemm::GemmShape<64, 32, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        2,
        cutlass::arch::OpMultiplyAdd>::GemmKernel;

using CutlassExactUniversal =
    cutlass::gemm::device::GemmUniversalAdapter<CutlassExactUniversalKernel>;

cudaError_t runCutlassExactUniversalSgemm(int M, int N, int K, float alpha,
                                         const float *A, const float *B,
                                         float beta, float *C) {
  CutlassExactUniversal op;
  // The generated operation exposes a column-major interface.  Reinterpret
  // row-major AB=C as the equivalent column-major B^T A^T=C^T.
  CutlassExactUniversal::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K},
      1,
      {alpha, beta},
      B, A, C, C,
      int64_t(K) * N, int64_t(M) * K, int64_t(M) * N, int64_t(M) * N,
      N, K, N, N);
  return op(args) == cutlass::Status::kSuccess ? cudaSuccess
                                               : cudaErrorLaunchFailure;
}
