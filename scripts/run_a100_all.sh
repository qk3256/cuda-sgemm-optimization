#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
export PATH=/home/why/.local/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

DEVICE=${DEVICE:-0}
SEED=${SEED:-1234}
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-5}
CORRECTNESS_WARMUP=${CORRECTNESS_WARMUP:-1}
CORRECTNESS_ITERS=${CORRECTNESS_ITERS:-1}

echo '[1/5] GPU/toolchain check'
nvidia-smi -L
nvcc --version | tail -n 1
ncu -v | head -n 4

echo '[2/5] SM80 build'
cmake -S . -B build-a100 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCUDA_COMPUTE_CAPABILITY=80
cmake --build build-a100 --parallel

echo '[3/5] correctness'
DEVICE="$DEVICE" SEED="$SEED" \
CORRECTNESS_WARMUP="$CORRECTNESS_WARMUP" \
CORRECTNESS_ITERS="$CORRECTNESS_ITERS" \
SGEMM_EXE="$ROOT/build-a100/sgemm_experiment" \
scripts/check_correctness.sh

echo '[4/5] benchmark'
DEVICE="$DEVICE" SEED="$SEED" WARMUP="$WARMUP" ITERS="$ITERS" \
SGEMM_EXE="$ROOT/build-a100/sgemm_experiment" \
scripts/run_benchmarks.sh

echo '[5/5] bounded kernel-10 autotune'
DEVICE="$DEVICE" WARMUP="$WARMUP" ITERS="$ITERS" \
SHAPE_CLASSES="${SHAPE_CLASSES:-large,medium,small,tiny}" \
scripts/run_autotune_real.sh

if [ -f profiles/raw/kernel_10_4096x4096x4096.ncu-rep ]; then
  ncu --import profiles/raw/kernel_10_4096x4096x4096.ncu-rep \
    --csv --page raw > profiles/csv/kernel_10_4096x4096x4096.csv
fi
if [ -f profiles/raw/kernel_10_128x4096x4096.ncu-rep ]; then
  ncu --import profiles/raw/kernel_10_128x4096x4096.ncu-rep \
    --csv --page raw > profiles/csv/kernel_10_128x4096x4096.csv
fi

echo 'Completed. Results:'
printf '  %s\n' results/correctness.csv results/a100_benchmark.csv \
  results/a100_benchmark.json results/autotune_candidates.csv \
  results/autotune_best.csv
