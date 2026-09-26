"""Compare the sm_86 transcript kernel with independent NumPy dot products."""

from __future__ import annotations

import sys
from pathlib import Path

import cupy as cp
import numpy as np
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))
from ampere_scan import _Ptr, ampere_scan  # noqa: E402


def row_base(tile: int) -> int:
    return tile if tile < 8 else 16 + tile - 8


def col_base(tile: int) -> int:
    return tile * 2 if tile < 8 else 16 + (tile - 8) * 2


def main() -> None:
    m, n = 256, 512
    row_period0, col_period0 = 1, 1
    rng = np.random.default_rng(20260926)
    a_host = rng.integers(-64, 64, size=(16, m, 128), dtype=np.int8)
    b_host = rng.integers(-64, 64, size=(16, n, 128), dtype=np.int8)
    a, b = cp.asarray(a_host), cp.asarray(b_host)
    transcript = cp.empty((2, 16, 128), dtype=cp.uint32)
    ampere_scan[(1, 2)](
        _Ptr(a, tl.int8), _Ptr(b, tl.int8), _Ptr(transcript, tl.uint32),
        m, n, row_period0, col_period0, num_warps=8, num_stages=1,
    )
    got = cp.asnumpy(transcript)

    row_pattern = [(u // 2) * 32 + (u % 2) * 8 for u in range(8)]
    col_pattern = [(v // 2) * 32 + (v % 2) for v in range(16)]
    for row_tile, col_tile in ((0, 0), (1, 1), (7, 8), (15, 15)):
        rows = [row_period0 * 128 + row_base(row_tile) + x for x in row_pattern]
        cols = [col_period0 * 256 + col_base(col_tile) + x for x in col_pattern]
        accum = np.zeros((8, 16), dtype=np.int32)
        cta = col_tile // 8
        tile = row_tile * 8 + col_tile % 8
        for step in range(16):
            aa = a_host[step, rows, :].astype(np.int32)
            bb = b_host[step, cols, :].astype(np.int32)
            accum += aa @ bb.T
            want = np.bitwise_xor.reduce(accum.view(np.uint32).reshape(-1))
            have = got[cta, step, tile]
            if want != have:
                raise AssertionError((row_tile, col_tile, step, int(want), int(have)))
    print("PASS: 64 randomized transcript words match NumPy")


if __name__ == "__main__":
    main()
