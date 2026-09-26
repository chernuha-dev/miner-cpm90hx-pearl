# Pearl miner for two CMP 90HX

Work in progress. The CUDA miner has produced a **verified share accepted by Kryptex** with both CMP 90HX cards enabled. Forge Miner remains the production miner on the rig while we work toward a higher accepted-share rate per watt.

The implementation starts from [CPPminer](https://github.com/1640675651/CPPminer) at commit `6785ad349a33f7bebb3ffe1dc1ad876b56d66ef8` under its MIT license. See [LICENSE](LICENSE). We are adapting its protocol, proof encoding, and CUDA path for current Pearl and Kryptex.

Target rig: HiveOS, Kryptex pool, two NVIDIA CMP 90HX. Current Forge Miner baseline: 62.5 + 63.9 = **126.4 TH/s** at **360 W** total (0.351 TH/s/W). These are the numbers to beat using **pool-accepted shares**, not only the miner's local speed display.

The miner charges **no developer fee**. The legacy upstream fee interface now always authorizes with the configured user wallet and has no developer wallet or fee scheduler.

## Current rig result

The independent sm_86 tensor-core kernel uses two GPUs on disjoint search ranges. Its production-size full scan measured **50.26 TMAC/s** in an offline hard-target test (1.400 s scan, 0.017 s preparation, row batch 16, column batch 1024). A live Kryptex run with the current four-warp kernel found a GPU1 share, passed the local certificate-version-3 verifier, and received `{"id":2,"result":true,"error":null}` from the pool. The live scan held about 50 TMAC/s. During the preceding live run, the GPUs drew 179.47 and 179.55 W; power draw for the current kernel was not sampled separately.

TMAC/s is this miner's count of int8 matrix multiply-accumulate work. Forge reports TH/s, so a sustained accepted-share comparison is still needed before claiming a direct speed or profit ratio. The current scan and power readings do not establish an advantage over Forge.

See [the rig benchmark notes](docs/rig-benchmarks.md) for tested launch shapes and validation results.

## Rig environment

The rig runs HiveOS / Ubuntu 22.04, driver 610.43.03, CUDA 12.8 and two sm_86 CMP 90HX cards with 10 GiB each. Run `bash scripts/rig_probe.sh` to refresh the report. The script does not read wallet addresses, pool passwords, or miner config files.

```bash
bash scripts/rig_probe.sh
```

## Implementation path

1. Confirm the exact PearlHash work and share format used by Kryptex against the current Pearl protocol, including any pool-specific encoding.
2. Implement a small host loop for Kryptex jobs and submissions, with one worker and CUDA context per GPU. Keep both cards busy while handling job changes and stale shares.
3. Implement the PearlHash CUDA path for the rig's architecture. Validate proof generation against a reference before optimizing.
4. Benchmark kernel time, accepted shares, wall power, rejected and stale shares. Tune launch shape, batching, data movement and power limit from measurements.

The first cuBLAS INT8 probe measured 35.7 TOPS per card at 2048³, and 41.6 / 43.2 TOPS at 4096³. Those figures characterize hardware throughput; they are not PearlHash rates.

An [Ampere kernel survey and rig benchmark](docs/open-pearl-miner-evaluation.md) found a faster isolated tensor-core path in `minerjed/open-pearl-miner`, but its hash-tile geometry and license prevent direct inclusion here.

Current Kryptex Stratum observation: `mining.authorize` sends an object containing `agent`, `type: "v2"`, `wallet`, and `worker`. `mining.notify` sends `header`, `height`, `job_id`, `target`, `cert_version`. `mining.submit` sends `job_id` and a gzip-compressed, base64-encoded `plain_proof`. A real Forge share was accepted with this format. Capture files are private and are not committed.

References: [Pearl source](https://github.com/pearl-research-labs/pearl), [NVIDIA CMP specifications](https://www.nvidia.com/en-us/cmp/), [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/).
