#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT="$ROOT/results/correctness.csv"; mkdir -p "$ROOT/results"
echo 'kernel,M,N,K,seed,device,fallback_naive,pass,max_abs,max_rel,norm_l2,nan_inf,median_ms,mean_ms,p95_ms,std_ms,tflops,status' > "$OUT"
EXE=${SGEMM_EXE:-$ROOT/build/sgemm_experiment}; SEED=${SEED:-1234}; DEVICE=${DEVICE:-0}
WARMUP=${CORRECTNESS_WARMUP:-3}; ITERS=${CORRECTNESS_ITERS:-5}
shapes=('4096 4096 4096' '2048 4096 4096' '512 4096 4096' '128 4096 4096' '16 4096 4096' '1000 1536 768' '127 4097 1025')
if [[ ! -x "$EXE" ]]; then
  for k in $(seq 0 12); do for s in "${shapes[@]}"; do read -r M N K <<< "$s"; echo "$k,$M,$N,$K,$SEED,$DEVICE,,,,,,,,,,,,BLOCKED_NO_CUDA_EXECUTABLE" >> "$OUT"; done; done
  echo "Wrote $OUT (blocked: $EXE not found)"; exit 0
fi
  for k in $(seq 0 12); do for s in "${shapes[@]}"; do read -r M N K <<< "$s"; line=$($EXE --kernel "$k" --M "$M" --N "$N" --K "$K" --seed "$SEED" --device "$DEVICE" --warmup "$WARMUP" --iters "$ITERS" --mode correctness 2>&1) || true; if [[ "$line" == kernel=* ]]; then echo "$line" | awk -F, '{for(i=1;i<=NF;i++){split($i,a,"="); printf "%s%s", (i>1?",":""), a[2]} print ",measured"}' >> "$OUT"; else echo "$k,$M,$N,$K,$SEED,$DEVICE,,,,,,,,,,,,RUNTIME_ERROR" >> "$OUT"; fi; done; done
echo "Wrote $OUT"
