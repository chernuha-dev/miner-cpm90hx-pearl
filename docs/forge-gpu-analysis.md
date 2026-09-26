# Forge GPU analysis on the CMP 90HX rig

This is an observation log for designing an independent Pearl kernel. Captured
third-party binaries and disassemblies stay on the rig and are not part of this
repository. Instruction counts are **static** disassembly counts, not runtime
instruction totals or throughput measurements.

## Runtime observations

Forge Miner 1.8.1 loads its sm_86 GPU code through `cuModuleLoadData`, so
running `cuobjdump` on the host executable alone does not reveal the search
kernel. The temporary `tools/cuda_module_probe.c` interceptor observed two
module loads per GPU. The selected Pearl entry point is named
`search_stratum_cutlass`. With the pool's M=131072, N=262144 job, it launched
grid 1024×1024×1 and block 256×1×1. This gives one 128×256 output tile per
thread block. The symbol metadata identifies INT8 `m16n8k32` Tensor Core MMA
and a 128×256×64 thread-block tile.

The selected kernel reports 251 registers per thread, zero static shared
memory, and zero dynamic shared memory. Its sm_86 disassembly contains 4096
`IMMA`, 2450 `LOP3`, 1048 `LDG`, and no `SHFL`, `BAR`, `LDSM`, `LDS`, or
`STS` instructions. It has four static `STG` instructions. This indicates a
register-heavy, fused search path rather than a shared-memory GEMM followed
by a separate transcript scan.

For comparison, our production BM64/BN128 Triton kernel reports 138 registers
per thread and 24 KiB shared memory. Its static disassembly contains 64
`IMMA`, 192 `SHFL`, 12 `BAR`, 24 `LDSM`, 28 `STS`, and 9 `LDS` instructions.
It writes transcript words for a later BLAKE3/target kernel. Its MMA loop is
rolled, so static instruction counts cannot be compared as dynamic work.

The Forge block covers 16×16 Pearl hash tiles of size 8×16, exactly 256 tiles
for 256 threads. Combined with the absence of cross-thread shuffles/barriers,
this **suggests** one thread owns all 128 accumulator cells for one hash tile
and reduces its transcript in registers. The instruction and launch evidence
does not by itself prove the lane-to-tile mapping. Forge's embedded symbol
mentions asynchronous copy layout types, while the selected compiled kernel
has no shared-memory instructions; the generated instructions are the stronger
evidence for this build and GPU.

## Independent implementation direction

Build an sm_86 INT8 Tensor Core kernel using public CUTLASS/CuTe primitives:
128×256 output tiles, K=64 pipeline, 256 threads, and an accumulator layout
that maps each 8×16 Pearl hash tile to one thread's registers. Fold all 16
milestones and test the target inside that kernel. Validate every transcript
word against the local reference verifier before any pool comparison. Measure
full accepted shares and GPU power at the same clock and power limits as Forge.

SRBMiner-MULTI 3.6.9 is installed on the rig and exposes a `--pearl-k2` option,
but Forge was sufficient to identify a specific kernel-layout bottleneck.
