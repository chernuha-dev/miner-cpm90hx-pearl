# Pearl miner for two CMP 90HX

Target rig: HiveOS, Kryptex pool, two NVIDIA CMP 90HX. Current Forge Miner baseline: 62.5 + 63.9 = **126.4 TH/s** at **360 W** total (0.351 TH/s/W). These are the numbers to beat using **pool-accepted shares**, not only the miner's local speed display.

## First step: capture the rig environment

Run `bash scripts/rig_probe.sh` on the rig and share its output. The script reads hardware and toolchain details; it does not read wallet addresses, pool passwords, or miner config files. In particular, we need the CUDA compute capability, driver, power limit, and whether `nvcc` is already available.

```bash
bash scripts/rig_probe.sh
```

## Implementation path

1. Confirm the exact PearlHash work and share format used by Kryptex against the current Pearl protocol, including any pool-specific encoding.
2. Implement a small host loop for Kryptex jobs and submissions, with one worker and CUDA context per GPU. Keep both cards busy while handling job changes and stale shares.
3. Implement the PearlHash CUDA path for the rig's architecture. Validate proof generation against a reference before optimizing.
4. Benchmark kernel time, accepted shares, wall power, rejected and stale shares. Tune launch shape, batching, data movement and power limit from measurements.

The official Pearl miner currently documents its CUDA tests for sm90 GPUs. We will verify the CMP 90HX capabilities from the rig before choosing the kernel path.

References: [Pearl source](https://github.com/pearl-research-labs/pearl), [NVIDIA CMP specifications](https://www.nvidia.com/en-us/cmp/), [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/).
