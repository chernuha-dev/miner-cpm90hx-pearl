"""Build the sm_86 Pearl 8x16 transcript scan kernel (requires Triton and CuPy).

The emitted cubin is loaded by the C++ miner through the CUDA driver API. The
kernel consumes the miner's step-major noisy matrices and writes 16 cumulative
XOR words per 8x16 hash tile. BLAKE3 and target comparison run in CUDA C++.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import triton
import triton.language as tl


@triton.jit
def ampere_scan(
    A, B, T, M, N, RPI0, CPI0,
    BM: tl.constexpr = 128,
    BN: tl.constexpr = 128,
    BK: tl.constexpr = 128,
):
    row_split = 128 // BM
    col_split = 256 // BN
    pr_slice = tl.program_id(0)
    pr = pr_slice // row_split
    pc_slice = tl.program_id(1)
    rr = tl.arange(0, BM)
    cc = tl.arange(0, BN)
    kk = tl.arange(0, BK)

    row_tile = (pr_slice % row_split) * (BM // 8) + rr // 8
    row_u = rr % 8
    row_base = tl.where(row_tile < 8, row_tile, 16 + row_tile - 8)
    row = (RPI0 + pr) * 128 + row_base + (row_u // 2) * 32 + (row_u % 2) * 8

    col_tile = (pc_slice % col_split) * (BN // 16) + cc // 16
    col_v = cc % 16
    col_base = tl.where(col_tile < 8, col_tile * 2, 16 + (col_tile - 8) * 2)
    col = (CPI0 + pc_slice // col_split) * 256 + col_base + (col_v // 2) * 32 + col_v % 2

    acc = tl.full((BM, BN), 0, tl.int32)
    for step in range(16):
        a = tl.load(A + (step * M + row[:, None]) * 128 + kk[None, :])
        b = tl.load(B + (step * N + col[None, :]) * 128 + kk[:, None])
        acc = tl.dot(a, b, acc)

        tile_xor = tl.reshape(acc.to(tl.uint32), (BM // 8, 8, BN // 16, 16))
        tile_xor = tl.xor_sum(tile_xor, 3)
        tile_xor = tl.xor_sum(tile_xor, 1)
        tile_xor = tl.reshape(tile_xor, (BM * BN // 128,))
        tile = tl.arange(0, BM * BN // 128)
        local_col = tile % (BN // 16)
        row_tile_out = tile // (BN // 16)
        col_tile_out = (pc_slice % col_split) * (BN // 16) + local_col
        half = col_tile_out // 8
        tile_in_half = row_tile_out * 8 + col_tile_out % 8
        batch_col_periods = tl.num_programs(1) // col_split
        local_cta = (pr * batch_col_periods + pc_slice // col_split) * 2 + half
        tl.store(T + (local_cta * 16 + step) * 128
                 + (pr_slice % row_split) * BM + tile_in_half, tile_xor)


class _Ptr:
    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def data_ptr(self):
        return int(self.array.data.ptr)


def build_cubin(output: Path, bm: int = 64, warps: int = 4) -> None:
    import cupy as cp

    cp.cuda.Device(0).use()
    a = cp.zeros((16, 128, 128), dtype=cp.int8)
    b = cp.zeros((16, 256, 128), dtype=cp.int8)
    t = cp.empty((2, 16, 128), dtype=cp.uint32)
    kernel = ampere_scan[(128 // bm, 2)](
        _Ptr(a, tl.int8), _Ptr(b, tl.int8), _Ptr(t, tl.uint32),
        128, 256, 0, 0, BM=bm, num_warps=warps, num_stages=1,
    )
    cp.cuda.runtime.deviceSynchronize()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(kernel.asm["cubin"])
    print(
        f"{output}: symbol={kernel.name} regs={kernel.n_regs} "
        f"shared={kernel.metadata.shared} bytes={output.stat().st_size}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("kernels/ampere_sm86.cubin"))
    parser.add_argument("--bm", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--warps", type=int, choices=(4, 8, 16), default=4)
    args = parser.parse_args()
    build_cubin(args.output, args.bm, args.warps)
