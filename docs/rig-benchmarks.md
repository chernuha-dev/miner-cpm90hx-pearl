# CMP 90HX rig benchmark, 2026-09-27

Rig: two sm_86 CMP 90HX cards, HiveOS / Ubuntu 22.04, CUDA 12.8. Tests stopped `forgeminer-prl.service` and restored it afterward. The full-scan rows below use production dimensions M=131072, N=262144, K=2048 and a target too hard to produce a share, so every run covers the full search space. `TMAC/s` counts matrix multiply-accumulate operations for visited hash tiles divided by scan wall time.

| Kernel and batch (row:column periods) | Full scan | Local rate, two GPUs |
| --- | ---: | ---: |
| BM128, BN128, 8 warps; 16:64 | 1.766 s | 39.86 TMAC/s |
| BM128, BN128, 8 warps; 16:1024 | 1.554 s | 45.28 TMAC/s |
| BM64, BN128, 8 warps; 16:1024 | 1.499 s | 46.96 TMAC/s |
| BM64, BN128, 4 warps; 16:1024 | 1.400 s | 50.26 TMAC/s |

The earlier multi-GPU code launched the same range on both cards. Column-period partitioning removed this duplicated work. Each GPU now generates the same A and B locally from shared seeds; the generated A keys are checked for equality before scanning.

An isolated production-size batch benchmark measured BM64/BN128/4-warps at 31.89 TMAC/s on one GPU. Increasing BN to 256 yielded 30.17 TMAC/s with four warps, so the shipped kernel retains BN128. `tests/bench_ampere_scan.py` reproduces the batch comparison. `tests/test_ampere_scan.py` compares transcript words with independent NumPy dot products for all tested shapes.

Both GPU0 and GPU1 found offline shares that passed the Rust certificate-version-3 verifier. A live Kryptex share from GPU1 with the four-warp kernel passed the same verifier and was accepted by the pool (`result: true`, `error: null`). A prior live run sampled GPU power at 179.47 W and 179.55 W. The legacy fee scheduler has been replaced by a wallet-only pass-through; no developer wallet is present.

Forge's observed baseline from the rig owner is 126.4 TH/s at 360 W. Its reported TH/s and our local TMAC/s are different counters. A sustained accepted-share and wall-power comparison is required to establish profitability; the current implementation has not demonstrated an advantage over Forge.

## Five-minute pool comparison

Both runs used the same rig, Kryptex PRL pool, wallet and displayed share difficulty `2097152`. Counts are pool-accepted shares from the miner logs, with no rejected shares in either run. Times are UTC on 2026-09-26.

| Miner | Window | Accepted / rejected | Sampled GPU power |
| --- | --- | ---: | ---: |
| This miner | 21:19:39–21:24:39 | 1 / 0 | 358–359 W total |
| ForgeMiner v1.8.1 | 21:25:05–21:30:05 | 7 / 0 | 359–360 W total |

Our miner completed 208 attempts and its scan-weighted rate was 49.63 TMAC/s. Including job changes and proof work, the five-minute window averaged 47.53 TMAC/s by its own work counter. Forge displayed about 126–128 TH/s during its window. The direct pool result favors Forge on this rig at present. Five-minute share counts are noisy; 7:1 is the observed count ratio, not a precise long-run income multiplier. Forge remains the running service.

## Kernel bottleneck after the comparison

The rig already uses a persistent `cmpunlocker-90hx-stockflow` NVIDIA kernel module. Our and Forge's runs both draw about 180 W per GPU, so enabling the existing unlock is not a missing step.

On one CMP 90HX, a production-size `--profile-scan=3` at row batch 16 and column batch 512 measured 25.16 TMAC/s per batch. The transcript GEMM used 21.443 ms (98.1%); CUDA BLAKE3 jackpot used 0.323 ms (1.5%); synchronization used 0.088 ms (0.4%). The full one-GPU sweep took 2.828 s (24.88 TMAC/s). Thus the main limit is the matrix/transcript kernel, not proof submission, pool protocol, or host synchronization. The profiler's inherited `C_hist` and `16x GemmEx` labels refer to an older path and do not describe the Ampere implementation.

Additional production-batch measurements on one GPU were 30.87 TMAC/s for the shipped BM64/BN128/BK128, four-warp, one-stage shape; 28.11 for two stages; 30.84 for three stages; 30.55 for BK64; and 27.82 for BK32. These variations did not close the gap to Forge. The default binary remains BM64/BN128/BK128, four warps, one stage. A substantially different sm_86 compute kernel is the next performance task.
