# New Optimization Experiment Log

All experiments began on 2026-10-01 in branch `codex/new-opt-20261001`. One principal variable should change per experiment. A result is kept only after correctness and both target shapes are considered.

## Experiment 0 — trustworthy baseline and inventory

- Current bottleneck evidence: historical labels and reports were not trustworthy for the new run; an old NCU report had captured a cuBLAS `ampere_sgemm` symbol instead of the custom kernel.
- Hypothesis: a profile-only executable plus strict comparison protocol will identify the real starting point.
- Modification: added `new_opt_runner`, fixed-seed correctness, pedantic FP32 cuBLAS, CUDA Event samples, median/mean/p95, native-shape safety checks, and a profile mode that launches exactly one custom kernel and never invokes cuBLAS.
- Expected metric: not an optimization; establishes valid timing and symbol attribution.
- Correctness: all 26 `(13 kernels) x (2 shapes)` runs passed.
- Performance:
  - square winner kernel 9: 15.55 TFLOP/s stable confirmation, 87.2% of strict cuBLAS
  - small-M winner kernel 4: 4.47 TFLOP/s, 34.6% of strict cuBLAS
- Profiler change: N/A; baseline reports awaiting counter permission.
- Conclusion: **keep**. This is the experimental foundation.

## Experiment 1 — permute the shared-B layout

- Current bottleneck evidence:
  - square baseline shared accesses generated 754,974,720 wavefronts versus 436,207,616 ideal, an excess of 318,767,104 (42%).
  - global accesses generated only 3% excessive L2 sectors.
  - FMA-pipe utilization was already 87.44%, while DRAM throughput was only 4.85% of peak.
  - for fixed `dotIdx` and register element `i`, adjacent logical B thread tiles are eight floats apart and therefore alias onto four banks.
- Hypothesis: permuting shared B from `[column_tile][element_in_TN]` to `[element_in_TN][column_tile]` will turn the compute-phase warp access into adjacent banks and reduce excess shared wavefronts.
- Modification: retained CTA `128x128x16`, thread tile `8x8`, and 256 threads. Global B loads remain `float4`; only their shared-memory placement and the corresponding shared read index changed.
- Expected profiler metric: substantially lower `memory_l1_wavefronts_shared - memory_l1_wavefronts_shared_ideal`, followed by higher issue-active/FMA utilization if the saved shared cycles exceed the layout overhead.
- Correctness: pass on both target shapes.
- Performance:
  - square: 8.869 ms, 15.497 TFLOP/s, approximately 87.0% of stable strict cuBLAS and 0.3% below the initial custom best.
  - small-M: 1.299 ms, 3.305 TFLOP/s, below the 4.467 TFLOP/s shape baseline.
- Static resources: 106 registers/thread and 16 KiB shared/block, versus 102 registers/thread for kernel 9.
- Profiler change: pending `exp1_k13_b_permuted_4096.ncu-rep`.
- Profiler change: shared excess fell from 318.8M to 67.1M wavefronts; issue active rose 49.18% to 54.93% and short scoreboard fell 0.373 to 0.037 cycles/issued instruction. However, executed instructions grew from 2.431B to 2.741B and registers rose from 102 to 106.
- Conclusion: **revert/not selected**. The layout fixed the intended conflict, but scalar scatter stores and address permutation cost more instructions than they saved.

## Direct CUTLASS/article replication track

After Experiment 1 established the bottleneck mechanics, development switched from single-variable micro-edits to explicit high-performance design replications. Each variant remains separately selectable.

### Kernel 14 — CUTLASS tile hierarchy on the existing mainloop

- Design: CTA `128x64x8`, warp `64x32`, four warps, thread `8x8`, synchronous single-stage mainloop.
- Result: 16.148 TFLOP/s square, 4.438 TFLOP/s small-M; pass.
- Change from kernel 9: +3.9% square. Registers 102/thread; shared memory 6 KiB/block.
- NCU: FMA 88.63%, SM throughput 88.16%, 2% global-sector excess, 6.9% shared-wavefront excess. Long scoreboard 1.12 and barrier 0.48 cycles/issued instruction remain.
- Conclusion: **keep as structural baseline**.

### Kernel 15 — old producer/consumer double buffer on CUTLASS tiles

- Design: same tile hierarchy, but different warps perform load/compute work around block barriers.
- Result: 14.893 TFLOP/s square, 3.887 TFLOP/s small-M; pass.
- Conclusion: **revert**. More synchronization without same-thread register prefetch does not reproduce CUTLASS's pipeline.

### Kernels 16–18 — article-style register prefetch, A swizzle, coalesced C

- Design: CTA `128x128x16`, 256 threads, thread `8x8`; next A/B tile first enters per-thread float4 registers, overlaps current shared-tile compute, then is committed to the alternate shared buffer. A is transposed/swizzled; each thread's C columns are two separated float4 groups.
- Kernel 16: 16.380 TFLOP/s, but 130+ registers/thread prevented two resident CTAs.
- Kernel 17: `__launch_bounds__(256,2)` constrained allocation to 128 registers/thread and raised performance to 17.131 TFLOP/s, **96.16% of a same-session 17.82 TFLOP/s strict cuBLAS**.
- Kernel 18: alpha=1/beta=0 specialized epilogue gave no material change (17.133 TFLOP/s).
- Conclusion: **kernel 17 is current square best**; resource tuning, not the epilogue, produced the major gain.

### Kernel 19 — register-prefetch pipeline with CUTLASS `128x64x8` hierarchy

- Design: 128 threads, four `64x32` warp tiles, two stages, per-thread two A float4 plus one B float4 prefetch, bank-aware A layout and coalesced split-column C stores.
- Result: 16.438 TFLOP/s square and **4.718 TFLOP/s small-M**; pass.
- Resources: 128 registers/thread, 12 KiB shared/block, 32-byte compiler stack frame.
- Conclusion: **keep for small-M, not for square**. The smaller N tile doubles grid concurrency and improves small-M, but BK=8/control/address overhead prevents it matching kernel 17 on the square case.

