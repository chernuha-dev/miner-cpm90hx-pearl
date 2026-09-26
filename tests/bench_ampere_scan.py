"""Compare Ampere transcript block shapes on a production-size batch."""

from __future__ import annotations

import sys
from pathlib import Path

import cupy as cp
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))
from ampere_scan import _Ptr, ampere_scan  # noqa: E402


def main() -> None:
    m, n = 131072, 262144
    row_batch, col_batch = 16, 512
    a = cp.ones((16, m, 128), dtype=cp.int8)
    b = cp.ones((16, n, 128), dtype=cp.int8)
    t = cp.empty((row_batch * col_batch * 2, 16, 128), dtype=cp.uint32)
    for bm in (128, 64):
        grid = (row_batch * 128 // bm, col_batch * 2)

        def launch() -> None:
            ampere_scan[grid](
                _Ptr(a, tl.int8), _Ptr(b, tl.int8), _Ptr(t, tl.uint32),
                m, n, 0, 0, BM=bm, num_warps=8, num_stages=1,
            )

        for _ in range(3):
            launch()
        cp.cuda.runtime.deviceSynchronize()
        times = []
        for _ in range(10):
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            launch()
            stop.record()
            stop.synchronize()
            times.append(cp.cuda.get_elapsed_time(start, stop))
        times.sort()
        elapsed_ms = times[len(times) // 2]
        mac = row_batch * 128 * col_batch * 256 * 2048
        print(f"BM={bm} median={elapsed_ms:.3f} ms "
              f"rate={mac / (elapsed_ms * 1e-3) / 1e12:.2f} TMAC/s")


if __name__ == "__main__":
    main()
