#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); mkdir -p "$ROOT/results"
OUT="$ROOT/results/autotune_candidates.csv"; BEST="$ROOT/results/autotune_best.csv"; EXE=${SGEMM_EXE:-$ROOT/build/sgemm_experiment}; DEVICE=${DEVICE:-0}
echo 'shape_class,M,N,K,kernel,config,compile_status,correctness,median_ms,tflops,relative_cublas,registers,shared_bytes,stack_bytes,spill_stores,spill_loads,rejection_reason' > "$OUT"
echo 'shape_class,M,N,K,kernel,config,median_ms,tflops,relative_cublas' > "$BEST"
configs=('k9:BM=64,BN=64,BK=16,TM=4,TN=4' 'k9:BM=128,BN=128,BK=16,TM=8,TN=8' 'k10:BM=64,BN=128,BK=16,WM=32,WN=64,WNITER=1,TM=4,TN=4' 'k10:BM=128,BN=128,BK=16,WM=64,WN=64,WNITER=4,TM=8,TN=4')
shapes=('large 4096 4096 4096' 'medium 512 4096 4096' 'small 128 4096 4096' 'tiny 16 4096 4096')
if [[ ! -x "$EXE" ]]; then for q in "${shapes[@]}"; do read -r cls M N K <<< "$q"; for c in "${configs[@]}"; do IFS=: read -r kid cfg <<< "$c"; echo "$cls,$M,$N,$K,${kid#k},$cfg,not_compiled,not_run,,,,,,,,,,CUDA_TOOLCHAIN_UNAVAILABLE" >> "$OUT"; done; done; exit 0; fi
for q in "${shapes[@]}"; do read -r cls M N K <<< "$q"; for c in "${configs[@]}"; do IFS=: read -r kid cfg <<< "$c"; k=${kid#k}; line=$($EXE --kernel "$k" --M "$M" --N "$N" --K "$K" --device "$DEVICE" --warmup 10 --iters 20 2>/dev/null) || true; if [[ "$line" == kernel=* ]]; then vals=$(echo "$line"|awk -F, '{for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]=="median_ms"||a[1]=="tflops")printf "%s,",a[2]}}'); echo "$cls,$M,$N,$K,$k,$cfg,compiled,measured,$vals,,,,,,," >> "$OUT"; else echo "$cls,$M,$N,$K,$k,$cfg,compile_or_runtime_error,not_run,,,,,,,,,,${line//,/ }" >> "$OUT"; fi; done; done
