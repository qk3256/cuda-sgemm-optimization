#pragma once

#include <cuda_runtime.h>

namespace cutlass_register_pipeline {

// Logical rows are grouped as [8-row thread tile][two float4 halves].  The
// low bank bits enumerate the eight thread rows in steps of four, while the
// half and upper tile bit select distinct storage above the bank bits.
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

template <int BM = 128, int BN = 64, int BK = 8, int TM = 8, int TN = 8>
__global__ __launch_bounds__(128, 4)
void sgemmCutlassRegisterPipeline(int M, int N, int K, const float *A,
                                 const float *B, float *C) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int warp_row = warp_id >> 1;
  const int warp_col = warp_id & 1;

  const int c_row = warp_row * 64 + (lane >> 2) * 8;
  const int c_col = warp_col * 32 + (lane & 3) * 4;

  // Two A float4 vectors and one B float4 vector per thread cover a stage.
  const int a_row0 = tid >> 1;
  const int a_row1 = a_row0 + 64;
  const int a_col = (tid & 1) * 4;
  const int b_row = tid >> 4;
  const int b_col = (tid & 15) * 4;

  __shared__ __align__(16) float As[2 * BK * BM];
  __shared__ __align__(16) float Bs[2 * BK * BN];
  float sum[TM * TN] = {0.0f};

  const float *a_ptr0 = A + (blockIdx.y * BM + a_row0) * K + a_col;
  const float *a_ptr1 = A + (blockIdx.y * BM + a_row1) * K + a_col;
  const float *b_ptr = B + b_row * N + blockIdx.x * BN + b_col;

  float4 next_a0 = *reinterpret_cast<const float4 *>(a_ptr0);
  float4 next_a1 = *reinterpret_cast<const float4 *>(a_ptr1);
  float4 next_b = *reinterpret_cast<const float4 *>(b_ptr);

  float *As_write = As;
  As_write[a_col * BM + permute_m(a_row0) + 0] = next_a0.x;
  As_write[(a_col + 1) * BM + permute_m(a_row0) + 0] = next_a0.y;
  As_write[(a_col + 2) * BM + permute_m(a_row0) + 0] = next_a0.z;
  As_write[(a_col + 3) * BM + permute_m(a_row0) + 0] = next_a0.w;
  As_write[a_col * BM + permute_m(a_row1) + 0] = next_a1.x;
  As_write[(a_col + 1) * BM + permute_m(a_row1) + 0] = next_a1.y;
  As_write[(a_col + 2) * BM + permute_m(a_row1) + 0] = next_a1.z;
  As_write[(a_col + 3) * BM + permute_m(a_row1) + 0] = next_a1.w;
  *reinterpret_cast<float4 *>(&Bs[b_row * BN + b_col]) = next_b;
  __syncthreads();

  int read_stage = 0;
  int write_stage = 1;
  for (int bk = BK; bk < K; bk += BK) {
    a_ptr0 += BK;
    a_ptr1 += BK;
    b_ptr += BK * N;
    next_a0 = *reinterpret_cast<const float4 *>(a_ptr0);
    next_a1 = *reinterpret_cast<const float4 *>(a_ptr1);
    next_b = *reinterpret_cast<const float4 *>(b_ptr);

    compute_tile<BM, BN, BK, TM, TN>(As, Bs, read_stage, c_row, c_col, sum);

    As_write = As + write_stage * BK * BM;
    float *Bs_write = Bs + write_stage * BK * BN;
    As_write[a_col * BM + permute_m(a_row0)] = next_a0.x;
    As_write[(a_col + 1) * BM + permute_m(a_row0)] = next_a0.y;
    As_write[(a_col + 2) * BM + permute_m(a_row0)] = next_a0.z;
    As_write[(a_col + 3) * BM + permute_m(a_row0)] = next_a0.w;
    As_write[a_col * BM + permute_m(a_row1)] = next_a1.x;
    As_write[(a_col + 1) * BM + permute_m(a_row1)] = next_a1.y;
    As_write[(a_col + 2) * BM + permute_m(a_row1)] = next_a1.z;
    As_write[(a_col + 3) * BM + permute_m(a_row1)] = next_a1.w;
    *reinterpret_cast<float4 *>(&Bs_write[b_row * BN + b_col]) = next_b;

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

} // namespace cutlass_register_pipeline
