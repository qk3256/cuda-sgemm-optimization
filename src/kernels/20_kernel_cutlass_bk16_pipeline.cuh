#pragma once

#include <cuda_runtime.h>

namespace cutlass_bk16_pipeline {

__device__ __forceinline__ int permute_m(int row) {
  const int tile = row >> 3;
  const int half = (row >> 2) & 1;
  const int elem = row & 3;
  return (tile & 7) * 4 + elem + half * 32 + (tile >> 3) * 64;
}

template <int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void compute_tile(const float *As, const float *Bs,
                                             int stage, int c_row, int c_col,
                                             float *sum) {
#pragma unroll
  for (int kk = 0; kk < BK; ++kk) {
    float reg_a[TM];
    float reg_b[TN];
    const float *As_k = As + (stage * BK + kk) * BM;
    const float *Bs_k = Bs + (stage * BK + kk) * BN;
    *reinterpret_cast<float4 *>(&reg_a[0]) =
        *reinterpret_cast<const float4 *>(&As_k[permute_m(c_row)]);
    *reinterpret_cast<float4 *>(&reg_a[4]) =
        *reinterpret_cast<const float4 *>(&As_k[permute_m(c_row + 4)]);
    *reinterpret_cast<float4 *>(&reg_b[0]) =
        *reinterpret_cast<const float4 *>(&Bs_k[c_col]);
    *reinterpret_cast<float4 *>(&reg_b[4]) =
        *reinterpret_cast<const float4 *>(&Bs_k[c_col + 16]);
#pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
#pragma unroll
      for (int ni = 0; ni < TN; ++ni) {
        sum[mi * TN + ni] += reg_a[mi] * reg_b[ni];
      }
    }
  }
}

template <int BM = 128, int BN = 64, int BK = 16, int TM = 8, int TN = 8>
__global__ __launch_bounds__(128, 3)
void sgemmCutlassBk16Pipeline(int M, int N, int K,
                             const float *__restrict__ A,
                             const float *__restrict__ B,
                             float *__restrict__ C) {
  constexpr int THREADS = 128;
  constexpr int A_VECS = BM * BK / (THREADS * 4);
  constexpr int B_VECS = BK * BN / (THREADS * 4);
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int c_row = (warp_id >> 1) * 64 + (lane >> 2) * 8;
  const int c_col = (warp_id & 1) * 32 + (lane & 3) * 4;

  __shared__ __align__(16) float As[2 * BK * BM];
  __shared__ __align__(16) float Bs[2 * BK * BN];
  float sum[TM * TN] = {0.0f};
  float4 next_a[A_VECS];
  float4 next_b[B_VECS];

  const float *A_block = A + blockIdx.y * BM * K;
  const float *B_block = B + blockIdx.x * BN;

  auto load_next = [&](int bk) {
#pragma unroll
    for (int v = 0; v < A_VECS; ++v) {
      const int vec = tid + v * THREADS;
      const int row = vec / (BK / 4);
      const int col = (vec % (BK / 4)) * 4;
      next_a[v] = *reinterpret_cast<const float4 *>(&A_block[row * K + bk + col]);
    }
#pragma unroll
    for (int v = 0; v < B_VECS; ++v) {
      const int vec = tid + v * THREADS;
      const int row = vec / (BN / 4);
      const int col = (vec % (BN / 4)) * 4;
      next_b[v] = *reinterpret_cast<const float4 *>(&B_block[(bk + row) * N + col]);
    }
  };

  auto store_stage = [&](int stage) {
    float *As_write = As + stage * BK * BM;
    float *Bs_write = Bs + stage * BK * BN;
#pragma unroll
    for (int v = 0; v < A_VECS; ++v) {
      const int vec = tid + v * THREADS;
      const int row = vec / (BK / 4);
      const int col = (vec % (BK / 4)) * 4;
      const float *x = reinterpret_cast<const float *>(&next_a[v]);
#pragma unroll
      for (int e = 0; e < 4; ++e)
        As_write[(col + e) * BM + permute_m(row)] = x[e];
    }
#pragma unroll
    for (int v = 0; v < B_VECS; ++v) {
      const int vec = tid + v * THREADS;
      const int row = vec / (BN / 4);
      const int col = (vec % (BN / 4)) * 4;
      *reinterpret_cast<float4 *>(&Bs_write[row * BN + col]) = next_b[v];
    }
  };

  load_next(0);
  store_stage(0);
  __syncthreads();

  int read_stage = 0;
  int write_stage = 1;
  for (int bk = BK; bk < K; bk += BK) {
    load_next(bk);
    compute_tile<BM, BN, BK, TM, TN>(As, Bs, read_stage, c_row, c_col, sum);
    store_stage(write_stage);
    __syncthreads();
    read_stage ^= 1;
    write_stage ^= 1;
  }
  compute_tile<BM, BN, BK, TM, TN>(As, Bs, read_stage, c_row, c_col, sum);

  float *c_ptr = C + (blockIdx.y * BM + c_row) * N + blockIdx.x * BN + c_col;
#pragma unroll
  for (int mi = 0; mi < TM; ++mi) {
    *reinterpret_cast<float4 *>(&c_ptr[mi * N]) =
        *reinterpret_cast<float4 *>(&sum[mi * TN]);
    *reinterpret_cast<float4 *>(&c_ptr[mi * N + 16]) =
        *reinterpret_cast<float4 *>(&sum[mi * TN + 4]);
  }
}

} // namespace cutlass_bk16_pipeline
