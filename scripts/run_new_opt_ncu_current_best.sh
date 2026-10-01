#!/usr/bin/env bash
set -euo pipefail

WT="/home/why/workspace/lab/cuda-sgemm/SGEMM_NEW_OPT_20261001"
NCU="/usr/local/cuda-12.4/bin/ncu"
RUNNER="$WT/build-newopt/new_opt_runner"
SECTIONS=(--section SpeedOfLight --section LaunchStats --section Occupancy
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis
  --section WarpStateStats --section SchedulerStats --section InstructionStats
  --section SourceCounters)

collect() {
  local name="$1" kernel="$2" m="$3" n="$4" k="$5"
  sudo "$NCU" "${SECTIONS[@]}" --force-overwrite \
    -o "$WT/new_opt/profiles/$name" "$RUNNER" --mode profile \
    --kernel "$kernel" --M "$m" --N "$n" --K "$k"
  sudo chown "$(id -u):$(id -g)" "$WT/new_opt/profiles/$name.ncu-rep"
}

collect best_k17_4096 17 4096 4096 4096
collect best_k19_smallm 19 128 4096 4096
