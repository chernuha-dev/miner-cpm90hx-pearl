# open-pearl-miner evaluation

Evaluated [minerjed/open-pearl-miner](https://github.com/minerjed/open-pearl-miner) at commit `41ef42158b110a088662f3aac1f7fd9c064bbe1b` on 2026-09-26. The reference checkout and its build remain separate from this repository at `/opt/minerjed-reference` on the rig. No code from that project is included here.

## Compatibility

The reference has an Ampere tensor-core kernel with fused INT8 GEMM and transcript folding. Its [pool configuration](https://github.com/minerjed/open-pearl-miner/blob/main/python/pool_common.py) specifies M=N=131072 and K=4096, and its [proof code](https://github.com/minerjed/open-pearl-miner/blob/main/python/pearl_host.py) works on 16×16 hash tiles. Its README describes LuckyPool support. Our captured Kryptex jobs and locally verified Forge proofs use M=131072, N=262144, K=2048, rank=128, 8×16 hash tiles and certificate version 3. The kernel and proof pipeline therefore cannot be substituted directly into this miner.

The project's [license](https://github.com/minerjed/open-pearl-miner/blob/main/LICENSE) requires retaining a 2% developer fee in distributed derivatives; its README instead says 1%. Do not copy or adapt its implementation into this public repository without resolving the licensing terms. The hardware and algorithmic idea can be evaluated separately while an independently implemented sm_86 kernel is developed from permissively licensed sources.

## Rig benchmark

Built its unmodified CUDA library with CUDA 12.8, CUTLASS 3.9.2 headers and `-gencode arch=compute_86,code=sm_86`. Stopped Forge for the GPU measurements and restarted it immediately afterward. Each measurement used one GPU, zero-filled 4096×4096 input matrices, K=2048, R=128, a zero target, three warmups and 20 timed `p40_pearl_pow_split` calls with synchronization. Rate is `M×N×K / elapsed`, expressed in TMAC/s; the reference benchmark's hard-coded `2^20` MACs per 16×16 tile would overstate the rate at K=2048.

| CMP 90HX | Mean per call | GEMM-equivalent rate |
| --- | ---: | ---: |
| GPU 0 | 0.923 ms | 37.22 TMAC/s |
| GPU 1 | 0.921 ms | 37.32 TMAC/s |

These are isolated kernel-path measurements for the reference's 16×16 format. They are not Kryptex hashrate, accepted shares, or end-to-end throughput. Our current CUDA fallback measured about 4.96 TMAC/s in a production-size *full scan* at row/column batch 16/64, including work excluded from this reference benchmark. The two figures should not be read as a direct speedup ratio. Forge's observed baseline is 62.5 and 63.9 TH/s as reported by Forge under the current pool job.

## Next implementation step

Build an independently authored sm_86 tensor-core path for the current 8×16 transcript and K=2048, compare its transcript and proof with the existing Rust verifier, then require accepted Kryptex shares before replacing Forge. Measure full-scan throughput and wall power on both cards.
