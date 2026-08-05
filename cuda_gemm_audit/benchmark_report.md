# Benchmark audit report

## Evidence used

The primary reusable timing evidence is `results/representative_benchmark.csv`
and the corresponding application logs under `profiles/raw/`. It contains
CUDA Event measurements with fixed seed 1234, GPU 0, preallocated buffers,
warmup, and median/mean/P95/std output from `experiment_runner.cu`.

The table in `benchmark_results.csv` is intentionally an audit subset, not a
replacement for the repository's result files. The original full scripts write
one process per kernel/shape and their current CSVs are partial at audit time.

## Most credible measured rows

| Shape | Kernel | Median | TFLOP/s | Relative cuBLAS | Source |
|---|---|---:|---:|---:|---|
| 4096³ | cuBLAS FP32 | 9.28563 ms | 14.8012 | 1.000000 | `results/representative_benchmark.csv` |
| 4096³ | warp tiling 10 | 10.4038 ms | 13.2104 | 0.892515 | `results/representative_benchmark.csv` |
| 128×4096×4096 | cuBLAS FP32 | 0.329728 ms | 13.0258 | 1.000000 | `results/representative_benchmark.csv` |
| 128×4096×4096 | warp tiling 10 | 1.38138 ms | 3.10919 | 0.238697 | `results/representative_benchmark.csv` |

The 4096³ rows used 3 warmups and 5 measured iterations. The 128×4096×4096
warp row used 2 warmups and 3 measured iterations. These are not the required
50-iteration, three-round audit re-test; that stronger evidence is marked
missing rather than inferred.

## Correctness evidence

The harness uses cuBLAS strict FP32 output as reference and reports max absolute
error, stabilized max relative error, normalized L2, and NaN/Inf count. Existing
measured rows have `pass=1`; smoke logs include aligned and non-divisible cases.
The full requested 13×7 correctness matrix was not independently confirmed in
this audit because the long process-per-row run was interrupted and the current
default Codex namespace does not expose GPU devices.

## NCU evidence

The two root-collected reports are stronger profiler evidence than timing alone:

- 4096³ kernel 10: 9.310176 ms NCU duration, 97.804318% SM throughput,
  9.031275% DRAM throughput, 122 registers/thread, 13.568 KB shared/block.
- 128×4096×4096 kernel 10: 323.392 us NCU duration, 88.764492% SM throughput,
  16.842719% DRAM throughput, 122 registers/thread, 13.568 KB shared/block.

Both profiles support a compute-throughput-dominant classification for the
profiled kernel/shape pairs. They do not prove the same classification for all
kernel IDs or shapes.

