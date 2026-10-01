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

### Square, kernel 9

- custom symbol verified: `sgemmAutotuned<128,128,16,8,8>`
- FMA pipe: 87.44% of peak sustained active
- SM throughput: 82.87%; DRAM throughput: 4.85%
- issue active: 49.18%
- active / eligible warps per scheduler: 3.80 / 1.53
- registers: 102/thread; shared memory: 16 KiB/block
- theoretical / achieved occupancy: 25.0% / 23.73%, register-limited to two CTAs/SM
- L2 hit rate: 75.38%
- global sectors: 142,606,336 actual versus 138,412,032 ideal (3% excess)
- shared wavefronts: 754,974,720 actual versus 436,207,616 ideal (42% excess)
- principal warp-state values per issued instruction: math-pipe throttle 2.04, long scoreboard 0.62, short scoreboard 0.37, MIO throttle 0.29 cycles; `not selected` 2.12 is evidence of available eligible work, not itself a defect.

The square kernel is compute-heavy rather than DRAM-bound. Its main actionable inefficiency is shared-memory wavefront amplification, while low register-limited occupancy reduces latency-hiding margin.

### Small-M, kernel 4

- custom symbol verified: `sgemm1DBlocktiling<64,64,8,8>`
- grid: 128 CTAs, only 0.59 waves/SM
- FMA pipe: 37.61%; SM throughput: 29.64%; issue active: 30.97%
- active / eligible warps per scheduler: 4.71 / 0.77
- registers: 46/thread; shared memory: 4 KiB/block
- theoretical / achieved occupancy: 50.0% / 29.35%
- global sectors and shared wavefronts exactly match their ideal counts
- principal warp-state values per issued instruction: long scoreboard 5.13, MIO throttle 2.40, barrier 1.49, wait 1.29, short scoreboard 1.26 cycles

The small-M kernel is dominated first by insufficient grid waves and then by poor latency hiding in the few resident CTAs. It does not have the bank-conflict or global-coalescing problem observed in the square kernel.

## Experiment 1 profile

The permutation reduced excess shared wavefronts from 318.8M to 67.1M, raised issue active from 49.18% to 54.93%, and nearly removed short-scoreboard stalls. It did not improve latency because total executed instructions increased 12.8% and register use rose. This distinguishes a successfully changed micro-metric from a successful kernel optimization.

## CUTLASS-like synchronous hierarchy, kernel 14

- FMA pipe 88.63%; SM throughput 88.16%; DRAM throughput 8.52%
- issue active 50.86%; active/eligible warps per scheduler 3.71/1.35
- 102 registers/thread, 6 KiB shared/block, 25% theoretical and 23.21% achieved occupancy
- global sectors 209.7M actual / 205.5M ideal
- shared wavefronts 486.5M actual / 453.0M ideal
- main stalls per issued instruction: math pipe 1.58, long scoreboard 1.12, wait 0.96, barrier 0.48, short scoreboard 0.34

The variant is compute-heavy with mostly efficient global/shared accesses. Its residual gap is associated with limited eligible work, global-load dependency, barriers, and instruction scheduling rather than DRAM bandwidth.

## Kernel 17 square profile

- FMA pipe 96.23%; SM throughput 91.23%; DRAM throughput 4.95%
- issue active 54.15%; active/eligible warps per scheduler 3.79/1.83
- theoretical/achieved occupancy 25.0%/23.69%, 16 resident warps/SM
- 128 registers/thread, 32 KiB shared/block
- global sectors exactly ideal: 136,314,880
- shared wavefronts exactly ideal: 436,207,616
- long scoreboard 0.043, short scoreboard 0.080, MIO throttle 0.024, barrier 0.176 cycles/issued instruction
- dominant residual categories: math-pipe throttle 2.349, not-selected 2.386, wait 0.895

This established that kernel 17 was no longer memory-bound; the remaining opportunity was instruction organization under the 128-register cap.

## Kernel 19 small-M profile

- only 64 CTAs, 0.15 grid waves/SM
- achieved occupancy 6.26%, active/eligible warps per scheduler 1.00/0.31
- long scoreboard 1.087 cycles/issued instruction
- global transactions ideal, but shared wavefront excess about 18%

Grid underfill is the primary small-M limitation. The shared excess was still actionable and led to kernel 23's K-aware A layout.

## Final profile collection

`scripts/run_new_opt_ncu_final_compare.sh` collects the final hand-written kernel 26 and exact CUTLASS kernel 22. The report files are:

- `new_opt/profiles/final_k26_handwritten.ncu-rep`
- `new_opt/profiles/final_k22_cutlass_exact.ncu-rep`

### Final square comparison

| Metric | hand-written k26 | exact CUTLASS k22 |
|---|---:|---:|
| NCU duration | 9.24 ms | 9.37 ms |
| compute/SM throughput | 98.28% | 97.22% |
| FP32 FMA pipe, active-cycle basis | 98.67% | 97.57% |
| issue active | 54.63% | 63.00% |
| active / eligible warps per scheduler | 3.84 / 1.48 | 3.83 / 1.62 |
| achieved occupancy | 23.98% | 23.93% |
| registers/thread | 128 | 128 |
| shared memory/block (kernel payload) | 12.29 KiB static | 12.54 KiB dynamic |
| global L2 sectors actual / ideal | 203,423,744 / 203,423,744 | 203,423,744 / 203,423,744 |
| shared wavefronts actual / ideal | 486,539,264 / 452,984,832 | 320,503,808 / 319,979,520 |
| L2 hit rate | 86.15% | 88.98% |
| executed warp instructions | 2.378B | 2.780B |

Both kernels have perfectly coalesced global traffic and the same register-limited 25% theoretical occupancy. CUTLASS has the superior shared-memory layout: only 0.52M excess wavefronts versus 33.55M for the hand-written kernel. It also has more eligible warps and higher issue activity. The hand-written kernel nevertheless executes **14.5% fewer warp instructions** and sustains a higher FMA-pipe utilization. Its simpler iterator/address machinery more than compensates for the residual shared-wavefront amplification under this shape.

Relevant stalls per issued instruction tell the same story. Hand-written/CUTLASS values are: long scoreboard 0.096/0.002, short scoreboard 0.048/0.003, MIO throttle 0.017/0.001, barrier 1.483/1.365, math-pipe throttle 1.723/1.484, and wait 0.911/0.564. The hand-written kernel does not win by having lower per-instruction stalls; it wins by requiring substantially fewer instructions while keeping the FMA pipe almost continuously active.

