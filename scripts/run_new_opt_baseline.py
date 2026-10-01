#!/usr/bin/env python3
"""Benchmark every historical kernel under one explicit protocol."""
import argparse, csv, subprocess
from pathlib import Path

FIELDS = ["experiment", "kernel", "name", "M", "N", "K", "warmup", "iters",
          "strict_fp32", "pass", "max_abs", "norm_l2", "median_ms", "mean_ms",
          "p95_ms", "tflops", "status"]

def parse_line(text):
    line = next((x for x in text.splitlines() if x.startswith("kernel=")), "")
    return {part.split("=", 1)[0]: part.split("=", 1)[1]
            for part in line.split(",") if "=" in part}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    shapes = [(4096, 4096, 4096), (128, 4096, 4096)]
    rows = []
    for M, N, K in shapes:
        for kernel in range(13):
            cmd = [args.exe, "--kernel", str(kernel), "--M", str(M), "--N", str(N),
                   "--K", str(K), "--warmup", str(args.warmup), "--iters", str(args.iters)]
            p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            print(p.stdout.strip(), flush=True)
            d = parse_line(p.stdout)
            row = {k: d.get(k, "") for k in FIELDS}
            row["experiment"] = "baseline_20261001"
            row["status"] = "measured" if p.returncode == 0 and d else "error"
            rows.append(row)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

if __name__ == "__main__": main()
