"""Measure the factored scan alone on a production-size GPU batch."""

import sys
from pathlib import Path

import cupy as cp
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))
from ampere_scan import _Ptr  # noqa: E402
from factored_scan import factored_scan, factored_scan_i8  # noqa: E402


def main():
    m, n = 131072, 262144
    row_batch, col_batch = 16, 512
    w = cp.ones((16, m, 128), dtype=cp.float16)
    e = cp.ones((n, 128), dtype=cp.float16)
    wlo = cp.ones((16, m, 128), dtype=cp.int8)
    whi = cp.zeros((16, m, 128), dtype=cp.int8)
    ei8 = cp.ones((n, 128), dtype=cp.int8)
    t = cp.empty((row_batch * col_batch * 2, 16, 128), dtype=cp.uint32)
    mac = row_batch * 128 * col_batch * 256 * 2048
    for bm, bn, warps in ((64, 128, 4), (128, 128, 4), (128, 256, 8),
                          (128, 256, 16)):
        grid = (row_batch * 128 // bm, col_batch * 256 // bn)

        def launch():
            return factored_scan[grid](
                _Ptr(w, tl.float16), _Ptr(e, tl.float16), _Ptr(t, tl.uint32),
                m, n, 0, 0, BM=bm, BN=bn, num_warps=warps, num_stages=1)

        try:
            kernel = launch()
            launch()
            cp.cuda.runtime.deviceSynchronize()
            samples = []
            for _ in range(7):
                start, stop = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                launch()
                stop.record()
                stop.synchronize()
                samples.append(cp.cuda.get_elapsed_time(start, stop))
            samples.sort()
            elapsed_ms = samples[len(samples) // 2]
            print(f"BM={bm} BN={bn} warps={warps} regs={kernel.n_regs} "
                  f"shared={kernel.metadata.shared} median={elapsed_ms:.3f} ms "
                  f"rate={mac / (elapsed_ms * 1e-3) / 1e12:.2f} equivalent TMAC/s",
                  flush=True)
        except Exception as exc:
            print(f"BM={bm} BN={bn} warps={warps}: {exc}", flush=True)

    for bm, bn, warps in ((64, 128, 4), (128, 128, 4),
                          (128, 256, 8), (128, 256, 16)):
        grid = (row_batch * 128 // bm, col_batch * 256 // bn)

        def launch_i8():
            return factored_scan_i8[grid](
                _Ptr(wlo, tl.int8), _Ptr(whi, tl.int8), _Ptr(ei8, tl.int8),
                _Ptr(t, tl.uint32), m, n, 0, 0, BM=bm, BN=bn,
                num_warps=warps, num_stages=1)

        try:
            kernel = launch_i8()
            launch_i8()
            cp.cuda.runtime.deviceSynchronize()
            samples = []
            for _ in range(7):
                start, stop = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                launch_i8()
                stop.record()
                stop.synchronize()
                samples.append(cp.cuda.get_elapsed_time(start, stop))
            samples.sort()
            elapsed_ms = samples[len(samples) // 2]
            print(f"i8 BM={bm} BN={bn} warps={warps} regs={kernel.n_regs} "
                  f"shared={kernel.metadata.shared} median={elapsed_ms:.3f} ms "
                  f"rate={mac / (elapsed_ms * 1e-3) / 1e12:.2f} equivalent TMAC/s",
                  flush=True)
        except Exception as exc:
            print(f"i8 BM={bm} BN={bn} warps={warps}: {exc}", flush=True)


if __name__ == "__main__":
    main()
