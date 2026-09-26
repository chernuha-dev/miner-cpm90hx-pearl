"""Compare Ampere transcript block shapes on a production-size batch."""

from __future__ import annotations

import sys
from pathlib import Path

import cupy as cp
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))
from ampere_scan import _Ptr, ampere_scan, ampere_dot_probe  # noqa: E402


def main() -> None:
    m, n = 131072, 262144
    row_batch, col_batch = 16, 512
    a = cp.ones((16, m, 128), dtype=cp.int8)
    b = cp.ones((16, n, 128), dtype=cp.int8)
    t = cp.empty((row_batch * col_batch * 2, 16, 128), dtype=cp.uint32)
    o = cp.empty((row_batch * 128 // 64, col_batch * 256 // 128, 128), dtype=cp.int32)
    variants = (("dot-only", ampere_dot_probe, o, 1, False),
                ("dot-reuse", ampere_dot_probe, o, 1, True),
                ("dot-unroll4", ampere_dot_probe, o, 4, False),
                ("dot-unroll16", ampere_dot_probe, o, 16, False),
                ("transcript", ampere_scan, t, 1, False))
    for name, fn, output, unroll, reuse in variants:
        grid = (row_batch * 128 // 64, col_batch * 256 // 128)

        def probe_launch():
            kwargs = {"UNROLL": unroll, "REUSE": reuse} if name != "transcript" else {}
            return fn[grid](_Ptr(a, tl.int8), _Ptr(b, tl.int8),
                            _Ptr(output, tl.uint32 if name == "transcript" else tl.int32),
                            m, n, 0, 0, BM=64, BN=128, num_warps=4, num_stages=1,
                            **kwargs)

        kernel = None
        for _ in range(3):
            kernel = probe_launch()
        cp.cuda.runtime.deviceSynchronize()
        samples = []
        for _ in range(10):
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            probe_launch()
            stop.record()
            stop.synchronize()
            samples.append(cp.cuda.get_elapsed_time(start, stop))
        samples.sort()
        elapsed_ms = samples[len(samples) // 2]
        mac = row_batch * 128 * col_batch * 256 * 2048
        print(f"{name} regs={kernel.n_regs} shared={kernel.metadata.shared} "
              f"median={elapsed_ms:.3f} ms rate={mac / (elapsed_ms * 1e-3) / 1e12:.2f} TMAC/s", flush=True)

    for bm, bn, bk, warps, stages in ((64, 128, 128, 4, 1),
                                       (64, 128, 128, 4, 2),
                                       (64, 128, 128, 4, 3),
                                       (64, 128, 64, 4, 1),
                                       (64, 128, 64, 4, 2),
                                       (64, 128, 32, 4, 1)):
        grid = (row_batch * 128 // bm, col_batch * 256 // bn)

        def launch():
            return ampere_scan[grid](
                _Ptr(a, tl.int8), _Ptr(b, tl.int8), _Ptr(t, tl.uint32),
                m, n, 0, 0, BM=bm, BN=bn, BK=bk,
                num_warps=warps, num_stages=stages,
            )

        kernel = None
        for _ in range(3):
            kernel = launch()
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
        print(f"BM={bm} BN={bn} BK={bk} warps={warps} stages={stages} regs={kernel.n_regs} "
              f"shared={kernel.metadata.shared} median={elapsed_ms:.3f} ms "
              f"rate={mac / (elapsed_ms * 1e-3) / 1e12:.2f} TMAC/s", flush=True)


if __name__ == "__main__":
    main()
