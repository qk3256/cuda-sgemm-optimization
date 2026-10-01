#!/usr/bin/env bash
set -euo pipefail

WORKTREE="/home/why/workspace/lab/cuda-sgemm/SGEMM_NEW_OPT_20261001"
NCU="/usr/local/cuda-12.4/bin/ncu"
RUNNER="$WORKTREE/build-newopt/new_opt_runner"
OUT="$WORKTREE/new_opt/profiles"

mkdir -p "$OUT"

sections=(
  --section SpeedOfLight
  --section LaunchStats
  --section Occupancy
  --section ComputeWorkloadAnalysis
  --section MemoryWorkloadAnalysis
  --section WarpStateStats
  --section SchedulerStats
  --section InstructionStats
  --section SourceCounters
)

profile_one() {
  local report_name="$1"
  local kernel="$2"
  local m="$3"
  local n="$4"
  local k="$5"

  echo "Profiling custom kernel $kernel at M=$m N=$n K=$k -> $report_name.ncu-rep"
  sudo "$NCU" "${sections[@]}" --force-overwrite \
    -o "$OUT/$report_name" \
    "$RUNNER" --mode profile --kernel "$kernel" --M "$m" --N "$n" --K "$k"
  sudo chown "$(id -u):$(id -g)" "$OUT/$report_name.ncu-rep"
}

# Baseline winners established by new_opt/NEW_OPT_RESULTS.csv.
# Profile mode launches exactly one custom kernel and never invokes cuBLAS.
profile_one baseline_k9_4096 9 4096 4096 4096
profile_one baseline_k4_smallm 4 128 4096 4096

echo "Reports written under $OUT"
