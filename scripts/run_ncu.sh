#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); mkdir -p "$ROOT/profiles/raw" "$ROOT/profiles/csv" "$ROOT/profiles/summaries"
EXE=${SGEMM_EXE:-$ROOT/build/sgemm_experiment}; NCU=${NCU:-ncu}; DEVICE=${DEVICE:-0}; KERNEL=${KERNEL:-10}; M=${M:-4096}; N=${N:-4096}; K=${K:-4096}
LOG="$ROOT/profiles/raw/kernel_${KERNEL}_${M}x${N}x${K}.log"; CSV="$ROOT/profiles/csv/kernel_${KERNEL}_${M}x${N}x${K}.csv"; REP="$ROOT/profiles/raw/kernel_${KERNEL}_${M}x${N}x${K}.ncu-rep"
cmd=("$NCU" --section SpeedOfLight --section LaunchStats --section Occupancy --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section WarpStateStats --section SchedulerStats --section InstructionStats --csv -o "$REP" -- "$EXE" --kernel "$KERNEL" --M "$M" --N "$N" --K "$K" --device "$DEVICE")
printf 'command:' > "$LOG"; printf ' %q' "${cmd[@]}" >> "$LOG"; printf '\n' >> "$LOG"
if ! command -v "$NCU" >/dev/null 2>&1 || [[ ! -x "$EXE" ]]; then echo 'NCU unavailable or experiment executable missing' | tee -a "$LOG"; : > "$CSV"; exit 0; fi
"${cmd[@]}" 2>&1 | tee -a "$LOG" | tee "$CSV" >/dev/null || true
