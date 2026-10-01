#pragma once

#include <cuda_runtime.h>

namespace register_pipeline {

__device__ __forceinline__ int swizzle_a(int k_row, int m_col) {
  // Preserve each aligned float4 group while XORing higher k bits into the
  // shared-memory bank-select bits.
  return m_col ^ ((k_row >> 2) << 3);
}

template <int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void compute_tile(const float *As, const float *Bs,
                                             int read_idx, int c_row,
                                             int c_col, float *sum) {
#pragma unroll
  for (int kk = 0; kk < BK; ++kk) {
    float reg_a[TM];
    float reg_b[TN];

    *reinterpret_cast<float4 *>(&reg_a[0]) =
        *reinterpret_cast<const float4 *>(
            &As[(read_idx * BK + kk) * BM + swizzle_a(kk, c_row)]);
    *reinterpret_cast<float4 *>(&reg_a[4]) =
        *reinterpret_cast<const float4 *>(
            &As[(read_idx * BK + kk) * BM + swizzle_a(kk, c_row + 4)]);

    *reinterpret_cast<float4 *>(&reg_b[0]) =
        *reinterpret_cast<const float4 *>(
            &Bs[(read_idx * BK + kk) * BN + c_col]);
    *reinterpret_cast<float4 *>(&reg_b[4]) =
        *reinterpret_cast<const float4 *>(
            &Bs[(read_idx * BK + kk) * BN + c_col + 64]);

#pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
#pragma unroll
      for (int ni = 0; ni < TN; ++ni) {
        sum[mi * TN + ni] += reg_a[mi] * reg_b[ni];
      }
    }
  }
}

template <int BM = 128, int BN = 128, int BK = 16, int TM = 8, int TN = 8,
          int MIN_BLOCKS_PER_SM = 1, bool FAST_ALPHA1_BETA0 = false>
__global__ __launch_bounds__(256, MIN_BLOCKS_PER_SM)
void sgemmRegisterPipelineBeta0(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  const int load_a_row = tid / 4;
  const int load_a_col = (tid % 4) * 4;
  const int load_b_row = tid / 32;
  const int load_b_col = (tid % 32) * 4;

  const int c_row = warp_id * 16 + (lane_id / 16) * 8;
  const int c_col = (lane_id % 16) * 4;

  __shared__ __align__(16) float As[2 * BK * BM];
  __shared__ __align__(16) float Bs[2 * BK * BN];

  float sum[TM * TN] = {0.0f};

  float *a_ptr = A + (blockIdx.y * BM + load_a_row) * K + load_a_col;
  float *b_ptr = B + load_b_row * N + blockIdx.x * BN + load_b_col;

  float4 next_a0 = *reinterpret_cast<float4 *>(&a_ptr[0]);
  float4 next_a1 = *reinterpret_cast<float4 *>(&a_ptr[64 * K]);
  float4 next_b0 = *reinterpret_cast<float4 *>(&b_ptr[0]);
  float4 next_b1 = *reinterpret_cast<float4 *>(&b_ptr[8 * N]);

  As[(load_a_col + 0) * BM + swizzle_a(load_a_col + 0, load_a_row)] = next_a0.x;
  As[(load_a_col + 1) * BM + swizzle_a(load_a_col + 1, load_a_row)] = next_a0.y;
  As[(load_a_col + 2) * BM + swizzle_a(load_a_col + 2, load_a_row)] = next_a0.z;
  As[(load_a_col + 3) * BM + swizzle_a(load_a_col + 3, load_a_row)] = next_a0.w;
  As[(load_a_col + 0) * BM + swizzle_a(load_a_col + 0, load_a_row + 64)] = next_a1.x;
  As[(load_a_col + 1) * BM + swizzle_a(load_a_col + 1, load_a_row + 64)] = next_a1.y;
  As[(load_a_col + 2) * BM + swizzle_a(load_a_col + 2, load_a_row + 64)] = next_a1.z;
  As[(load_a_col + 3) * BM + swizzle_a(load_a_col + 3, load_a_row + 64)] = next_a1.w;
  *reinterpret_cast<float4 *>(&Bs[load_b_row * BN + load_b_col]) = next_b0;
  *reinterpret_cast<float4 *>(&Bs[(load_b_row + 8) * BN + load_b_col]) = next_b1;
  __syncthreads();

  int read_idx = 0;
  int write_idx = 1;
  for (int bk = BK; bk < K; bk += BK) {
    a_ptr += BK;
    b_ptr += BK * N;

    // Register prefetch.  The dependent shared stores occur only after the
    // current tile's FMA stream, allowing the global loads to overlap it.
    next_a0 = *reinterpret_cast<float4 *>(&a_ptr[0]);
    next_a1 = *reinterpret_cast<float4 *>(&a_ptr[64 * K]);
    next_b0 = *reinterpret_cast<float4 *>(&b_ptr[0]);
    next_b1 = *reinterpret_cast<float4 *>(&b_ptr[8 * N]);

    compute_tile<BM, BN, BK, TM, TN>(As, Bs, read_idx, c_row, c_col, sum);

    float *As_write = As + write_idx * BK * BM;
    float *Bs_write = Bs + write_idx * BK * BN;
    As_write[(load_a_col + 0) * BM + swizzle_a(load_a_col + 0, load_a_row)] = next_a0.x;
    As_write[(load_a_col + 1) * BM + swizzle_a(load_a_col + 1, load_a_row)] = next_a0.y;
    As_write[(load_a_col + 2) * BM + swizzle_a(load_a_col + 2, load_a_row)] = next_a0.z;
    As_write[(load_a_col + 3) * BM + swizzle_a(load_a_col + 3, load_a_row)] = next_a0.w;
    As_write[(load_a_col + 0) * BM + swizzle_a(load_a_col + 0, load_a_row + 64)] = next_a1.x;
    As_write[(load_a_col + 1) * BM + swizzle_a(load_a_col + 1, load_a_row + 64)] = next_a1.y;
    As_write[(load_a_col + 2) * BM + swizzle_a(load_a_col + 2, load_a_row + 64)] = next_a1.z;
    As_write[(load_a_col + 3) * BM + swizzle_a(load_a_col + 3, load_a_row + 64)] = next_a1.w;
    *reinterpret_cast<float4 *>(&Bs_write[load_b_row * BN + load_b_col]) = next_b0;
    *reinterpret_cast<float4 *>(&Bs_write[(load_b_row + 8) * BN + load_b_col]) = next_b1;

    __syncthreads();
    read_idx ^= 1;
    write_idx ^= 1;
  }

  compute_tile<BM, BN, BK, TM, TN>(As, Bs, read_idx, c_row, c_col, sum);

  float *c_ptr = C + (blockIdx.y * BM + c_row) * N + blockIdx.x * BN + c_col;
#pragma unroll
  for (int mi = 0; mi < TM; ++mi) {
    if constexpr (FAST_ALPHA1_BETA0) {
      *reinterpret_cast<float4 *>(&c_ptr[mi * N]) =
          *reinterpret_cast<float4 *>(&sum[mi * TN]);
      *reinterpret_cast<float4 *>(&c_ptr[mi * N + 64]) =
          *reinterpret_cast<float4 *>(&sum[mi * TN + 4]);
      continue;
    }
    float4 out0 = make_float4(alpha * sum[mi * TN + 0], alpha * sum[mi * TN + 1],
                              alpha * sum[mi * TN + 2], alpha * sum[mi * TN + 3]);
    float4 out1 = make_float4(alpha * sum[mi * TN + 4], alpha * sum[mi * TN + 5],
                              alpha * sum[mi * TN + 6], alpha * sum[mi * TN + 7]);
    if (beta != 0.0f) {
      const float4 old0 = *reinterpret_cast<float4 *>(&c_ptr[mi * N]);
      const float4 old1 = *reinterpret_cast<float4 *>(&c_ptr[mi * N + 64]);
      out0.x += beta * old0.x; out0.y += beta * old0.y;
      out0.z += beta * old0.z; out0.w += beta * old0.w;
      out1.x += beta * old1.x; out1.y += beta * old1.y;
      out1.z += beta * old1.z; out1.w += beta * old1.w;
    }
    *reinterpret_cast<float4 *>(&c_ptr[mi * N]) = out0;
    *reinterpret_cast<float4 *>(&c_ptr[mi * N + 64]) = out1;
  }
}

} // namespace register_pipeline
