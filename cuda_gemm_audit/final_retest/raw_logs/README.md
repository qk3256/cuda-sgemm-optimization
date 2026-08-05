# Final retest raw logs

The planned 10-warmup/50-event/3-round invocations were not launched. The
current Codex execution namespace has no `/dev/nvidia*` GPU device nodes and
the current tool policy does not permit an elevated GPU invocation. Placeholder
rows in the sibling CSV files therefore use `NOT_RUN_GPU_UNAVAILABLE`; no
latency, TFLOP/s, or error value is asserted for this retest.

Earlier raw NCU logs remain in `profiles/raw/` and were deliberately not
reused as final-retest measurements.

