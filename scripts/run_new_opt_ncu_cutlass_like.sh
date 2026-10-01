#!/usr/bin/env bash
set -euo pipefail

WORKTREE="/home/why/workspace/lab/cuda-sgemm/SGEMM_NEW_OPT_20261001"
NCU="/usr/local/cuda-12.4/bin/ncu"
REPORT="$WORKTREE/new_opt/profiles/cutlass_like_k14_4096"

sudo "$NCU" \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
  --section WarpStateStats --section SchedulerStats \
  --section InstructionStats --section SourceCounters \
  --force-overwrite -o "$REPORT" \
  "$WORKTREE/build-newopt/new_opt_runner" \
  --mode profile --kernel 14 --M 4096 --N 4096 --K 4096

sudo chown "$(id -u):$(id -g)" "$REPORT.ncu-rep"
echo "Wrote $REPORT.ncu-rep"
