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

### Kernel 17 loop-unroll sweep

- Evidence: kernel 17 had essentially ideal global/shared transactions and 96.23% FMA-pipe utilization, leaving loop and scheduling overhead as the plausible square bottleneck.
- Change: only the outer BK loop unroll factor changed.
- Result: unroll 4 produced 17.196 TFLOP/s; unroll 8 increased the compiler stack from 16 to 32 bytes and regressed to 16.847 TFLOP/s.
- Conclusion: **keep unroll 4; reject unroll 8**. More static instruction exposure is not free under a hard 128-register residency cap.

### Exact CUTLASS integration — kernels 21 and 22

- Purpose: place CUTLASS and custom kernels under the identical correctness and timing harness instead of comparing against the standalone profiler's loop.
- Kernel 21: `device::Gemm` with the selected `128x64x8`, warp `64x32x8`, two-stage SIMT configuration; 17.431 TFLOP/s square and 8.144 TFLOP/s small-M.
- Kernel 22: the exact generated `DefaultGemmUniversal + GemmUniversalAdapter` type; 17.506 TFLOP/s square and 8.176 TFLOP/s small-M.
- Conclusion: **keep as a clearly labelled reference, not as custom work**. The standalone profiler's 18.80 TFLOP/s square result is not used in the final ratio because its timing loop differs.

### Kernel 23 — K-aware shared swizzle and two-way unroll

- Current bottleneck evidence: kernel 19 showed about 18% shared wavefront excess. Source-level bank analysis found that cooperative A stores used different K addresses but the same bank because the row permutation did not include K.
- Hypothesis: XORing the float4 K-group bit into the A shared-row permutation will separate store banks while preserving conflict-free compute loads, where K is warp-uniform.
- Modification: retain CTA `128x64x8`, warp `64x32`, thread `8x8`, 128 threads, and two stages. Change only A shared placement/read addressing; then separately apply `#pragma unroll 2` to the BK loop.
- Correctness: pass on square and small-M; square matched strict cuBLAS bit-for-bit in the fixed test.
- Before outer-loop unroll: 17.139 TFLOP/s square, 5.497 TFLOP/s small-M.
- After outer-loop unroll: **17.566 TFLOP/s square and 6.668 TFLOP/s small-M**. Static resources are 128 registers/thread, 12 KiB shared/block, and zero compiler stack bytes.
- Stable square ratio: **98.56–98.59% of strict FP32 cuBLAS** in a 200-iteration cuBLAS/custom/CUTLASS/custom/cuBLAS sandwich.
- Conclusion: **keep as the pre-CTA-swizzle base**. It also slightly exceeds the exact CUTLASS Universal reference (17.506 TFLOP/s) under the same runner.

### Kernels 24–29 — CTA scheduling swizzle

- Evidence: per-CTA memory transactions and compute were already efficient, leaving cross-CTA cache locality and scheduling order as a low-risk remaining variable.
- Modification: reproduce CUTLASS identity-swizzle grouping while preserving the kernel 23 CTA, warp, thread, and shared layouts. Sweep N-group sizes 2, 4, 8, 16, 32, and 64.
- Results: 17.520, 17.517, **17.723**, 17.719, 17.700, and 17.649 TFLOP/s respectively.
- Conclusion: **keep group 8 as kernel 26**. It reaches 99.45–99.47% of the stable 17.817–17.822 TFLOP/s strict-cuBLAS baseline and exceeds the exact CUTLASS reference under the same harness.

### Resource and specialization checks — kernels 30–33

- Residency sweep: 3 CTA/SM used 168 registers and reached 17.524 TFLOP/s; 2 CTA/SM used 234 registers and fell to 16.378; 5 CTA/SM forced 96 registers plus a 40-byte stack and fell to 16.669. Four CTA/SM and 128 registers is the optimum.
- Compile-time `N=K=4096` specialization regressed to 17.451 TFLOP/s; the compiler already optimized the dynamic-stride loop effectively.
- Conclusion: **reject all**; retain the general kernel 26.

### Multi-size square sweep

- The same kernel 26 was tested at 1024, 2048, 3072, 3584, 4096, 5120, 6144, and 8192 without retuning.
- It is poor at 1024 because the grid is too small, ties cuBLAS at 2048 in the initial sweep, reaches 99.4% at 4096, and reaches 96.9% at 8192.
- Short 50-iteration samples appeared to exceed cuBLAS at 3072 and 6144. Longer 200-iteration sandwich repeats were clock-state sensitive and did not show a consistent win.
- Conclusion: do not claim a general cuBLAS victory. The defensible headline is **99.4% of strict FP32 cuBLAS at 4096 cubed**.

