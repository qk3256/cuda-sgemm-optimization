#!/usr/bin/env python3
"""Run the bounded, reproducible final A100 SGEMM retest.

The script intentionally runs only the representative stages and both existing
double-buffer implementations (11 and 12), so the better one can be selected
from this round instead of inferred from an unverified historical claim.
"""
import argparse, csv, os, re, subprocess, sys
from pathlib import Path

KERNEL_NAMES = {0: "cublas_fp32", 1: "naive", 3: "shared_memory_tiling",
                5: "register_tiling_2d", 10: "warp_tiling",
                11: "double_buffering", 12: "async_barrier_double_buffering"}
PERF_SHAPES = [(4096, 4096, 4096, "large"), (128, 4096, 4096, "medium"),
               (1000, 1536, 768, "llm_like")]
BOUNDARY = (127, 4097, 1025, "nondiv")
FIELDS = ["git_commit", "GPU", "CUDA_version", "dtype", "M", "N", "K",
          "kernel_id", "kernel_name", "warm-up", "measurements", "round",
          "median_ms", "p95_ms", "mean_ms", "std_ms", "TFLOP/s",
          "speedup_vs_naive", "ratio_vs_cublas", "fallback", "correctness", "status", "source_log"]
CFIELDS = ["git_commit", "GPU", "CUDA_version", "dtype", "M", "N", "K",
           "kernel_id", "kernel_name", "round", "max_abs", "max_rel",
           "norm_l2", "nan_inf", "fallback", "pass", "status", "source_log"]

def kv(line):
    return {p.split("=", 1)[0]: p.split("=", 1)[1] for p in line.strip().split(",") if "=" in p}

def cmd1(cmd):
    try: return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception: return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="build-a100/sgemm_experiment")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="cuda_gemm_audit/final_retest")
    args = ap.parse_args()
    exe, out = Path(args.exe), Path(args.out)
    if not exe.exists(): raise SystemExit(f"missing executable: {exe}; build with cmake --build build-a100 --parallel")
    raw = out / "raw_logs"; raw.mkdir(parents=True, exist_ok=True)
    commit = cmd1(["git", "rev-parse", "HEAD"])
    gpu = cmd1(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", str(args.device)])
    cuda = cmd1(["bash", "-lc", "nvcc --version | sed -n 's/.*release \\([^,]*\\).*/\\1/p'"])
    records, correctness = [], []
    def run(k, shape, tag, rnd, warm=10, iters=50):
        M,N,K,_ = shape; log = raw / f"{tag}_k{k}_r{rnd}.log"
        c = [str(exe), "--kernel", str(k), "--M", str(M), "--N", str(N), "--K", str(K),
             "--seed", "1234", "--device", str(args.device), "--warmup", str(warm), "--iters", str(iters)]
        p = subprocess.run(c, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write_text(p.stdout)
        line = next((x for x in p.stdout.splitlines() if x.startswith("kernel=")), None)
        d = kv(line) if line else {}
        status = "measured" if line and p.returncode == 0 else "RUNTIME_ERROR"
        fallback = d.get("fallback_naive", "unknown")
        records.append({"shape": (M,N,K), "kernel": k, "round": rnd, "d": d,
                        "tag": tag, "status": status, "log": str(log)})
        correctness.append({"shape": (M,N,K), "kernel": k, "round": rnd, "d": d,
                            "tag": tag, "status": status, "log": str(log)})
    for M,N,K,tag in PERF_SHAPES:
        for k in (0,1,3,5,10,11,12):
            for rnd in (1,2,3): run(k, (M,N,K,tag), tag, rnd)
    for k in (0,1,3,5,10,11,12): run(k, BOUNDARY, "nondiv", 1, 10, 1)

    # Derive relative metrics from the median of the three rounds per pair.
    med = {}
    for r in records:
        if r["status"] == "measured" and "median_ms" in r["d"]:
            med.setdefault((r["shape"], r["kernel"]), []).append(float(r["d"]["median_ms"]))
    base = {(s, k): sum(v)/len(v) for (s,k),v in med.items()}
    with (out/"results.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in records:
            d=r["d"]; s=r["shape"]; naive=base.get((s,1)); blas=base.get((s,0)); tm=float(d["median_ms"]) if "median_ms" in d else None
            row={"git_commit":commit,"GPU":gpu,"CUDA_version":cuda,"dtype":"FP32","M":s[0],"N":s[1],"K":s[2],"kernel_id":r["kernel"],"kernel_name":KERNEL_NAMES[r["kernel"]],"warm-up":10,"measurements":50 if r["tag"]!="nondiv" else 1,"round":r["round"],"median_ms":d.get("median_ms",""),"p95_ms":d.get("p95_ms",""),"mean_ms":d.get("mean_ms",""),"std_ms":d.get("std_ms",""),"TFLOP/s":d.get("tflops",""),"speedup_vs_naive":(naive/tm if naive and tm else ""),"ratio_vs_cublas":(blas/tm if blas and tm else ""),"fallback":d.get("fallback_naive","unknown"),"correctness":d.get("pass",""),"status":r["status"],"source_log":r["log"]}; w.writerow(row)
    with (out/"correctness.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=CFIELDS); w.writeheader()
        for r in correctness:
            d=r["d"]; s=r["shape"]; w.writerow({"git_commit":commit,"GPU":gpu,"CUDA_version":cuda,"dtype":"FP32","M":s[0],"N":s[1],"K":s[2],"kernel_id":r["kernel"],"kernel_name":KERNEL_NAMES[r["kernel"]],"round":r["round"],"max_abs":d.get("max_abs",""),"max_rel":d.get("max_rel",""),"norm_l2":d.get("norm_l2",""),"nan_inf":d.get("nan_inf",""),"fallback":d.get("fallback_naive","unknown"),"pass":d.get("pass",""),"status":r["status"],"source_log":r["log"]})
    (out/"environment.md").write_text(f"# Final retest environment\n\n- Git commit: `{commit}`\n- GPU {args.device}: {gpu}\n- CUDA Toolkit: {cuda}\n- Protocol: seed 1234; warm-up 10; CUDA Event measurements 50 (3 rounds); boundary case 1 measurement.\n- Executable: `{exe}`\n")
    print(f"wrote {out/'results.csv'} and {out/'correctness.csv'}; measured records: {sum(r['status']=='measured' for r in records)}/{len(records)}")

if __name__ == "__main__": main()
