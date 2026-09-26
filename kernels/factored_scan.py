"""Experimental zero-B rank-factorized Pearl scan for sm_86.

W[step,row,rank] is the cumulative signed A projection through the B
permutation pair table. E[col,rank] is the dense B noise basis. This path is
valid only when the pool's B signal is exactly zero and W/accumulator bounds
make FP16 inputs and FP32 accumulation exact for integers.
"""

import triton
import triton.language as tl


@triton.jit
def factored_scan(W, E, T, M, N, RPI0, CPI0,
                  BM: tl.constexpr = 64, BN: tl.constexpr = 128):
    row_split = 128 // BM
    col_split = 256 // BN
    pr_slice = tl.program_id(0)
    pr = pr_slice // row_split
    pc_slice = tl.program_id(1)
    rr = tl.arange(0, BM)
    cc = tl.arange(0, BN)
    kk = tl.arange(0, 128)

    row_tile = (pr_slice % row_split) * (BM // 8) + rr // 8
    row_u = rr % 8
    row_base = tl.where(row_tile < 8, row_tile, 16 + row_tile - 8)
    row = (RPI0 + pr) * 128 + row_base + (row_u // 2) * 32 + (row_u % 2) * 8

    col_tile = (pc_slice % col_split) * (BN // 16) + cc // 16
    col_v = cc % 16
    col_base = tl.where(col_tile < 8, col_tile * 2, 16 + (col_tile - 8) * 2)
    col = (CPI0 + pc_slice // col_split) * 256 + col_base + (col_v // 2) * 32 + col_v % 2

    basis = tl.load(E + col[None, :] * 128 + kk[:, None])
    for step in range(16):
        projected = tl.load(W + (step * M + row[:, None]) * 128 + kk[None, :])
        acc = tl.dot(projected, basis)
        tile_xor = tl.reshape(acc.to(tl.int32).to(tl.uint32),
                              (BM // 8, 8, BN // 16, 16))
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
