# Final retest environment

- Project: `/home/why/workspace/infra/cuda-sgemm/SGEMM_CUDA`
- Branch before retest: `codex/a100-sgemm-experiment`
- Backup commit before retest: `af6d1ba` (`Back up A100 SGEMM audit and experiment artifacts`)
- Intended GPU: GPU 0, NVIDIA A100-SXM4-40GB, compute capability 8.0
- Intended CUDA: 12.4.131 at `/usr/local/cuda-12.4`
- Intended dtype: FP32
- Intended seed: 1234

## Runtime availability

The current non-elevated Codex execution namespace does not expose
`/dev/nvidia*`; `nvidia-smi -L` does not return a usable device from this
namespace. The user's node161 shell previously exposed both A100 devices and
successfully produced the existing NCU reports, but this retest could not be
run from the current tool namespace. No performance number in this directory
is fabricated or copied from an earlier protocol.

## Build state

The previously configured `build-a100` uses CUDA 12.4 and SM80. The backup
commit contains code, scripts, CSV, and Markdown; large build and NCU artifacts
remain outside that commit as noted in the final audit.

