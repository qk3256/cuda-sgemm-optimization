#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC="$ROOT/src/runner.cu"; BUILD="$ROOT/build-a100"; EXE="$BUILD/sgemm_experiment"
DEVICE=${DEVICE:-0}; WARMUP=${WARMUP:-2}; ITERS=${ITERS:-5}
OUT="$ROOT/results/autotune_candidates.csv"; BEST="$ROOT/results/autotune_best.csv"
echo 'shape_class,M,N,K,kernel,config,compile_status,correctness,median_ms,tflops,relative_cublas,registers,shared_bytes,spill_stores,spill_loads,rejection_reason' > "$OUT"
echo 'shape_class,M,N,K,kernel,config,median_ms,tflops,relative_cublas' > "$BEST"
tmp=$(mktemp -d /tmp/sgemm-autotune.XXXXXX); cp "$SRC" "$tmp/runner.cu.orig"
restore() { cp "$tmp/runner.cu.orig" "$SRC"; }
trap restore EXIT INT TERM
configs=('BM=64 BN=128 BK=16 WM=32 WN=64 WNITER=1 TM=4 TN=4 NUM_THREADS=128' 'BM=128 BN=128 BK=16 WM=64 WN=64 WNITER=4 TM=8 TN=4 NUM_THREADS=128' 'BM=128 BN=128 BK=16 WM=64 WN=64 WNITER=2 TM=4 TN=4 NUM_THREADS=256')
shapes=('large 4096 4096 4096' 'medium 512 4096 4096' 'small 128 4096 4096' 'tiny 16 4096 4096')
shape_filter=${SHAPE_CLASSES:-large,medium,small,tiny}
filtered=(); for q in "${shapes[@]}"; do cls=${q%% *}; case ",$shape_filter," in *,$cls,*) filtered+=("$q");; esac; done
shapes=("${filtered[@]}")
for cfg in "${configs[@]}"; do
  for kv in $cfg; do key=${kv%%=*}; val=${kv#*=}; sed -Ei "s/const uint K10_${key} = [0-9]+;/const uint K10_${key} = ${val};/" "$SRC"; done
  log=$(mktemp /tmp/sgemm-autotune-build.XXXXXX)
  if ! cmake --build "$BUILD" --parallel >"$log" 2>&1; then
    for q in "${shapes[@]}"; do read -r cls M N K <<< "$q"; echo "$cls,$M,$N,$K,10,\"$cfg\",build_failed,not_run,,,,,,,,,$(tr '\n' ' ' < "$log")" >> "$OUT"; done
    continue
  fi
  resources=$(cuobjdump --dump-resource-usage "$BUILD/sgemm" 2>/dev/null)
  kernel_resources=$(echo "$resources" | awk '/sgemmWarptiling/{seen=1} seen && /REG:/{print; exit}')
  regs=$(echo "$kernel_resources" | sed -n 's/.*REG:\s*\([0-9][0-9]*\).*/\1/p'); shared=$(echo "$kernel_resources" | sed -n 's/.*SHARED:\s*\([0-9][0-9]*\).*/\1/p')
  for q in "${shapes[@]}"; do
    read -r cls M N K <<< "$q"; line=$($EXE --kernel 10 --M "$M" --N "$N" --K "$K" --seed 1234 --device "$DEVICE" --warmup "$WARMUP" --iters "$ITERS" 2>&1) || true
    if [[ "$line" == kernel=* ]]; then pass=$(echo "$line" | sed -n 's/.*pass=\([^,]*\).*/\1/p'); med=$(echo "$line" | sed -n 's/.*median_ms=\([^,]*\).*/\1/p'); tf=$(echo "$line" | sed -n 's/.*tflops=\([^,]*\).*/\1/p'); echo "$cls,$M,$N,$K,10,\"$cfg\",built,$pass,$med,$tf,,${regs:-},${shared:-},,," >> "$OUT"; else echo "$cls,$M,$N,$K,10,\"$cfg\",runtime_failed,not_run,,,,,,,,,$line" >> "$OUT"; fi
  done
done
for cls in large medium small tiny; do awk -F, -v c="$cls" 'NR>1 && $1==c && $7=="built" && $8=="1"' "$OUT" | sort -t, -k10,10nr | head -n 1 | awk -F, '{print $1","$2","$3","$4","$5","$6","$9","$10",,"}' >> "$BEST"; done
echo "Wrote $OUT and $BEST"
