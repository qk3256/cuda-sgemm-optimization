#!/usr/bin/env bash
set -euo pipefail

WT="/home/why/workspace/lab/cuda-sgemm/SGEMM_NEW_OPT_20261001"
NCU="/usr/local/cuda-12.4/bin/ncu"
RUNNER="$WT/build-newopt/new_opt_runner"
OUT="$WT/new_opt/profiles"
SECTIONS=(--section SpeedOfLight --section LaunchStats --section Occupancy
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis
  --section WarpStateStats --section SchedulerStats --section InstructionStats
  --section SourceCounters)

collect() {
  local name="$1" kernel="$2"
  sudo "$NCU" "${SECTIONS[@]}" --force-overwrite -o "$OUT/$name" \
    "$RUNNER" --mode profile --kernel "$kernel" --M 4096 --N 4096 --K 4096
  sudo chown "$(id -u):$(id -g)" "$OUT/$name.ncu-rep"
}

collect final_k26_handwritten 26
collect final_k22_cutlass_exact 22

echo "Generated:"
echo "  $OUT/final_k26_handwritten.ncu-rep"
echo "  $OUT/final_k22_cutlass_exact.ncu-rep"
