#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); mkdir -p "$ROOT/results"
CSV="$ROOT/results/a100_benchmark.csv"; JSON="$ROOT/results/a100_benchmark.json"; EXE=${SGEMM_EXE:-$ROOT/build/sgemm_experiment}; SEED=${SEED:-1234}; DEVICE=${DEVICE:-0}; WARMUP=${WARMUP:-10}; ITERS=${ITERS:-30}
echo 'kernel,M,N,K,seed,device,fallback_naive,pass,max_abs,max_rel,norm_l2,nan_inf,median_ms,mean_ms,p95_ms,std_ms,tflops,status' > "$CSV"; echo '[' > "$JSON"; first=1
shapes=('4096 4096 4096' '2048 4096 4096' '512 4096 4096' '128 4096 4096' '16 4096 4096' '1000 1536 768' '127 4097 1025')
if [[ ! -x "$EXE" ]]; then echo ']' > "$JSON"; for k in 0 1 2 3 4 5 6 7 8 9 10 11 12; do for s in "${shapes[@]}"; do read -r M N K <<< "$s"; echo "$k,$M,$N,$K,$SEED,$DEVICE,,,,,,,,,,,,BLOCKED_NO_CUDA_EXECUTABLE" >> "$CSV"; done; done; echo "Wrote blocked results to $CSV"; exit 0; fi
for k in 0 1 2 3 4 5 6 7 8 9 10 11 12; do for s in "${shapes[@]}"; do read -r M N K <<< "$s"; line=$($EXE --kernel "$k" --M "$M" --N "$N" --K "$K" --seed "$SEED" --device "$DEVICE" --warmup "$WARMUP" --iters "$ITERS" 2>&1) || true; if [[ "$line" == kernel=* ]]; then echo "$line" | awk -F, '{for(i=1;i<=NF;i++){split($i,a,"="); printf "%s%s",(i>1?",":""),a[2]} print ",measured"}' >> "$CSV"; obj=$(echo "$line" | awk -F, '{printf "{\"%s\":\"%s\"", $1, $2; for(i=3;i<=NF;i++){split($i,a,"="); printf ",\"%s\":\"%s\"",a[1],a[2]} print "}"}'); [[ $first -eq 0 ]] && echo ',' >> "$JSON"; echo "$obj" >> "$JSON"; first=0; else echo "$k,$M,$N,$K,$SEED,$DEVICE,,,,,,,,,,,,RUNTIME_ERROR" >> "$CSV"; fi; done; done
echo ']' >> "$JSON"; echo "Wrote $CSV and $JSON"
