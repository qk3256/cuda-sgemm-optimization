# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

## Overview

Running the kernels on a NVIDIA A6000 (Ampere):

![](benchmark_results.png)

GFLOPs at matrix size 4096x4096:
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to cuBLAS |
|:------------------------------------|----------:|:-------------------------------|
| 1: Naive                            |   `309.0` | 1.3%                           |
| 2: GMEM Coalescing                  |  `1986.5` | 8.5%                           |
| 3: SMEM Caching                     |  `2980.3` | 12.8%                          |
| 4: 1D Blocktiling                   |  `8474.7` | 36.5%                          |
| 5: 2D Blocktiling                   | `15971.7` | 68.7%                          |
| 7: Avoid Bank Conflicts (Linearize) | `16213.4` | 69.7%                          |
| 8: Avoid Bank Conflicts (Offset)    | `16459.2` | 70.8%                          |
| 11: Double Buffering                | `17278.3` | 74.3%                          |
| 6: Vectorized Mem Access            | `18237.3` | 78.4%                          |
| 9: Autotuning                       | `19721.0` | 84.8%                          |
| 10: Warptiling                      | `21779.3` | 93.7%                          |
| 0: cuBLAS                           | `23249.6` | 100.0%                         |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit 12, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt` and change:
    ```cmake
    set(CUDA_COMPUTE_CAPABILITY 80)
    ```
1. Build: `mkdir build && cd build && cmake .. && cmake --build .`
1. Run one of the kernels: `DEVICE=<device_id> ./sgemm <kernel number>`
1. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`

Credit goes to [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the benchmarking setup.

## A100 SM80 reproducible experiment

The experiment branch targets `sm_80` by default. On an A100 host:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/home/why/.local/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cmake -S . -B build-a100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DCUDA_COMPUTE_CAPABILITY=80
cmake --build build-a100 --parallel
DEVICE=0 SEED=1234 CORRECTNESS_WARMUP=3 CORRECTNESS_ITERS=5 \
  SGEMM_EXE=$PWD/build-a100/sgemm_experiment scripts/check_correctness.sh
DEVICE=0 WARMUP=10 ITERS=30 SGEMM_EXE=$PWD/build-a100/sgemm_experiment \
  scripts/run_benchmarks.sh
DEVICE=0 SGEMM_EXE=$PWD/build-a100/sgemm_experiment scripts/run_autotune.sh
KERNEL=10 M=4096 N=4096 K=4096 DEVICE=0 \
  SGEMM_EXE=$PWD/build-a100/sgemm_experiment scripts/run_ncu.sh
```

`build/sgemm_experiment` accepts arbitrary `--M --N --K`, `--kernel`, `--seed`, and `--device`. For custom kernels whose original implementation lacks boundary guards, non-tile-aligned shapes are executed by an explicitly reported safe naive fallback (`fallback_naive=1`); this prevents out-of-bounds accesses while keeping the requested kernel visible in the result row. CUDA Event timing uses preallocated buffers, warmup, asynchronous event pairs, one post-sequence synchronization, and median/mean/P95/std statistics.

Results are written to `results/correctness.csv`, `results/a100_benchmark.csv`, `results/a100_benchmark.json`, and `results/autotune_*.csv`. Nsight Compute artifacts belong under `profiles/`; if counters require privilege, review and run `scripts/run_ncu_root.sh` manually.
