#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
echo "sudo ncu --section SpeedOfLight --section LaunchStats --section Occupancy --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section WarpStateStats --section SchedulerStats --section InstructionStats --csv -o $ROOT/profiles/raw/kernel_10_4096.ncu-rep -- $ROOT/build/sgemm_experiment --kernel 10 --M 4096 --N 4096 --K 4096 --device \${DEVICE:-0}"
