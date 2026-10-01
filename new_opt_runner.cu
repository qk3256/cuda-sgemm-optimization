#include "src/runner.cuh"

#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::cerr << "CUDA error: " << cudaGetErrorString(e) << '\n'; return 2; } } while (0)
#define CUBLAS_CHECK(call) do { cublasStatus_t e = (call); if (e != CUBLAS_STATUS_SUCCESS) { \
  std::cerr << "cuBLAS error: " << int(e) << '\n'; return 2; } } while (0)

struct Args {
  int kernel = 10, M = 4096, N = 4096, K = 4096;
  int seed = 1234, device = 0, warmup = 10, iters = 50;
  std::string mode = "benchmark";
};

static Args parse(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto integer = [&]() { if (++i >= argc) throw std::runtime_error("missing value for " + s); return std::stoi(argv[i]); };
    auto text = [&]() { if (++i >= argc) throw std::runtime_error("missing value for " + s); return std::string(argv[i]); };
    if (s == "--kernel") a.kernel = integer();
    else if (s == "--M") a.M = integer();
    else if (s == "--N") a.N = integer();
    else if (s == "--K") a.K = integer();
    else if (s == "--seed") a.seed = integer();
    else if (s == "--device") a.device = integer();
    else if (s == "--warmup") a.warmup = integer();
    else if (s == "--iters") a.iters = integer();
    else if (s == "--mode") a.mode = text();
    else throw std::runtime_error("unknown argument " + s);
  }
  if (a.kernel < 0 || a.kernel > 19 || a.M <= 0 || a.N <= 0 || a.K <= 0 ||
      a.warmup < 0 || a.iters <= 0 || (a.mode != "benchmark" && a.mode != "profile"))
    throw std::runtime_error("invalid argument value");
  return a;
}

static const char *kernel_name(int id) {
  static const char *names[] = {
    "cublas_strict_fp32", "naive", "coalesced", "shared_tiled",
    "register_1d", "register_2d", "vectorized", "bank_reordered",
    "bank_padded", "autotuned", "warp_tiled", "double_buffered",
    "async_double_buffered", "b_smem_permuted", "cutlass_like_tiling",
    "cutlass_like_pipeline", "register_prefetch_swizzle",
    "register_prefetch_2cta", "register_prefetch_fast_epilogue",
    "cutlass_register_pipeline"};
  return names[id];
}

static bool native_safe(const Args &a) {
  if (a.kernel <= 2) return true;
  if (a.kernel >= 3 && a.kernel <= 8) return a.M % 64 == 0 && a.N % 64 == 0 && a.K % 8 == 0;
  if (a.kernel == 9 || a.kernel == 10 || a.kernel == 12 || a.kernel == 13)
    return a.M % 128 == 0 && a.N % 128 == 0 && a.K % 16 == 0;
  if (a.kernel == 11) return a.M % 128 == 0 && a.N % 256 == 0 && a.K % 16 == 0;
  if (a.kernel == 14) return a.M % 128 == 0 && a.N % 64 == 0 && a.K % 8 == 0;
  if (a.kernel == 15) return a.M % 128 == 0 && a.N % 64 == 0 && a.K % 16 == 0;
  if (a.kernel == 16) return a.M % 128 == 0 && a.N % 128 == 0 && a.K % 16 == 0;
  if (a.kernel == 17) return a.M % 128 == 0 && a.N % 128 == 0 && a.K % 16 == 0;
  if (a.kernel == 18) return a.M % 128 == 0 && a.N % 128 == 0 && a.K % 16 == 0;
  if (a.kernel == 19) return a.M % 128 == 0 && a.N % 64 == 0 && a.K % 8 == 0;
  return false;
}

static cublasStatus_t strict_sgemm(cublasHandle_t h, int M, int N, int K,
                                  const float *A, const float *B, float *C) {
  const float alpha = 1.0f, beta = 0.0f;
  // Row-major AB is column-major B^T A^T. PEDANTIC excludes TF32 and
  // reduced-precision shortcuts from the strict baseline.
  return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                      &alpha, B, CUDA_R_32F, N, A, CUDA_R_32F, K,
                      &beta, C, CUDA_R_32F, N,
                      CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
}

int main(int argc, char **argv) {
  try {
    Args a = parse(argc, argv);
    if (a.mode == "profile" && a.kernel == 0)
      throw std::runtime_error("profile mode is custom-kernel-only; kernel 0 is cuBLAS");
    if (a.kernel > 0 && !native_safe(a))
      throw std::runtime_error("shape is not natively safe for this historical kernel; no fallback is allowed");

    CUDA_CHECK(cudaSetDevice(a.device));
    const size_t ae = size_t(a.M) * a.K, be = size_t(a.K) * a.N, ce = size_t(a.M) * a.N;
    std::mt19937 gen(a.seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> hA(ae), hB(be), hOut(ce), hRef;
    for (float &x : hA) x = dist(gen);
    for (float &x : hB) x = dist(gen);

    float *A = nullptr, *B = nullptr, *C = nullptr, *R = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&A), ae * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&B), be * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&C), ce * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), ae * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), be * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(C, 0, ce * sizeof(float)));

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    const float alpha = 1.0f, beta = 0.0f;
    auto launch = [&]() {
      if (a.kernel == 0) {
        if (strict_sgemm(handle, a.M, a.N, a.K, A, B, C) != CUBLAS_STATUS_SUCCESS)
          throw std::runtime_error("strict cuBLAS launch failed");
      } else {
        run_kernel(a.kernel, a.M, a.N, a.K, alpha, A, B, beta, C, handle);
      }
    };

    if (a.mode == "profile") {
      // Exactly one target launch. Initialization above uses runtime memcpy,
      // never cuBLAS or another compute kernel.
      launch();
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      std::cout << "profile_kernel=" << a.kernel << ",name=" << kernel_name(a.kernel)
                << ",M=" << a.M << ",N=" << a.N << ",K=" << a.K << '\n';
      cublasDestroy(handle); cudaFree(A); cudaFree(B); cudaFree(C);
      return 0;
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&R), ce * sizeof(float)));
    CUDA_CHECK(cudaMemset(R, 0, ce * sizeof(float)));
    CUBLAS_CHECK(strict_sgemm(handle, a.M, a.N, a.K, A, B, R));
    CUDA_CHECK(cudaDeviceSynchronize());
    hRef.resize(ce);
    CUDA_CHECK(cudaMemcpy(hRef.data(), R, ce * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < a.warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> samples; samples.reserve(a.iters);
    for (int i = 0; i < a.iters; ++i) {
      CUDA_CHECK(cudaEventRecord(start)); launch(); CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); samples.push_back(ms);
    }
    CUDA_CHECK(cudaMemcpy(hOut.data(), C, ce * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs = 0, error2 = 0, ref2 = 0; long long nonfinite = 0;
    for (size_t i = 0; i < ce; ++i) {
      double x = hOut[i], y = hRef[i];
      if (!std::isfinite(x) || !std::isfinite(y)) ++nonfinite;
      double e = std::abs(x - y); max_abs = std::max(max_abs, e);
      error2 += e * e; ref2 += y * y;
    }
    const double norm_l2 = std::sqrt(error2 / (ref2 + 1e-30));
    const bool pass = max_abs < 1e-2 && norm_l2 < 1e-4 && nonfinite == 0;
    std::sort(samples.begin(), samples.end());
    double mean = 0; for (float x : samples) mean += x; mean /= samples.size();
    const double median = samples[samples.size() / 2];
    const double p95 = samples[std::min(samples.size() - 1,
      size_t(std::ceil(0.95 * samples.size())) - 1)];
    const double tflops = 2.0 * double(a.M) * a.N * a.K / (median * 1e-3) / 1e12;
    std::cout << "kernel=" << a.kernel << ",name=" << kernel_name(a.kernel)
              << ",M=" << a.M << ",N=" << a.N << ",K=" << a.K
              << ",warmup=" << a.warmup << ",iters=" << a.iters
              << ",strict_fp32=1,pass=" << int(pass) << ",max_abs=" << max_abs
              << ",norm_l2=" << norm_l2 << ",median_ms=" << median
              << ",mean_ms=" << mean << ",p95_ms=" << p95
              << ",tflops=" << tflops << '\n';

    cudaEventDestroy(start); cudaEventDestroy(stop); cublasDestroy(handle);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(R);
    return pass ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
  }
}
