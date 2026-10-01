# New Optimization Profiling Notes

## Symbol-safe collection

`new_opt_runner --mode profile` performs allocation and deterministic initialization, launches exactly one requested custom kernel, synchronizes, and exits. It does not create a cuBLAS handle or run a reference GEMM. This prevents Nsight Compute from accidentally profiling cuBLAS.

Collection script:

```bash
cd ~/workspace/lab/cuda-sgemm/SGEMM_NEW_OPT_20261001
./scripts/run_new_opt_ncu.sh
```

Reports:

- `new_opt/profiles/baseline_k9_4096.ncu-rep`
- `new_opt/profiles/baseline_k4_smallm.ncu-rep`

Selected sections cover:

- Speed of Light and compute workload
- memory workload and L2/DRAM behavior
- launch resources and theoretical/achieved occupancy
- scheduler and warp-state statistics
- instruction statistics and source counters

## Baseline report status

The first unprivileged attempt correctly connected to `new_opt_runner` and printed only `profile_kernel=9,name=autotuned`, but the NVIDIA driver returned `ERR_NVGPUCTRPERM`. No report was accepted from that attempt. The sudo collection script is the authoritative path.

## Metrics to extract

For every retained experiment, record at least:

- SM throughput and FP32 pipe utilization
- achieved occupancy, registers/thread, shared memory/block
- DRAM throughput, L2 behavior, global-load sectors/useful bytes
- shared-memory bank-conflict or wavefront excess indicators
- active and eligible warps per scheduler, issue active
- long scoreboard, short scoreboard, MIO throttle, LG throttle, math-pipe throttle

## Interpretation discipline

- A large stall category is a symptom counted in warp-state sampling, not a direct prediction that eliminating it yields the same percentage speedup.
- High occupancy is not an objective by itself; it is useful only when additional eligible warps hide the observed dependency or memory latency.
- High cache hit rate does not prove low cache pressure; throughput and sector counts must be considered together.
- Static register/shared-memory usage constrains theoretical residency, while achieved occupancy and scheduler metrics show what occurred dynamically.

## NCU baseline findings

Pending privileged collection.

