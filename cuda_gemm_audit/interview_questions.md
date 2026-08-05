# Interview evidence pack

1. **What does the coalescing stage change?** Kernel 2 changes the thread/data
   mapping so consecutive threads load consecutive global-memory elements; the
   implementation is `src/kernels/2_kernel_global_mem_coalesce.cuh`.

2. **Why use shared-memory tiling?** Kernel 3 loads A/B tiles into shared memory,
   synchronizes, and reuses each value across the block instead of rereading
   global memory for every multiply.

3. **What is register tiling here?** Kernels 4-10 keep per-thread accumulators
   and short operand vectors in registers; kernel 10 additionally assigns each
   warp a sub-tile and reuses `regM`/`regN` across its output tile.

4. **How does arithmetic intensity explain the profile?** GEMM performs
   `2*M*N*K` FLOPs while reusing tile data; the NCU evidence shows much higher
   SM than DRAM throughput for the two profiled kernel-10 shapes, consistent
   with compute throughput dominating those measurements.

5. **What are the bank-conflict variants?** Kernel 7 reorganizes B's shared
   layout; kernel 8 adds extra columns (`BN + extraCols`) to change shared-memory
   addressing. These mechanisms are present in code; a complete per-stage NCU
   comparison is not part of the audit evidence.

6. **What is register pressure evidence?** NCU reported 122 registers/thread
   for the profiled `ampere_sgemm_128x64_nn` kernel. That is measured resource
   evidence; it is not by itself proof of an occupancy bottleneck.

7. **How is occupancy treated?** NCU collection includes LaunchStats and
   Occupancy sections, but the audit only claims the directly extracted
   registers/shared-memory/SOL values. It does not infer a numeric occupancy
   conclusion without a verified metric row.

8. **Why use CUDA Events?** GPU work is asynchronous; CUDA Events measure GPU
   elapsed time around the launch sequence without inserting a host
   synchronization inside every iteration. The harness synchronizes once after
   recording the measured events.

9. **How is TFLOP/s computed?** The harness uses `2*M*N*K` floating-point
   operations divided by median latency in seconds, then divides by `1e12`.

10. **Why does the custom kernel trail cuBLAS?** The measured 4096³ warp-tile
    row is 13.2104 TFLOP/s versus 14.8012 TFLOP/s cuBLAS (0.892515×). The audit
    can state that gap as measured; attributing it to a single cause would need
    additional instruction/occupancy/algorithm evidence.

