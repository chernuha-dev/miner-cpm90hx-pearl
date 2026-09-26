#pragma once

// Independent sm86 transcript prototype using public CUTLASS v2.11 primitives.
// A and B are step-major, with rows/columns permuted so every thread owns one
// complete 8x16 Pearl hash tile in its accumulator registers.
#include <cstdint>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/default_mma.h"

using PearlTensorMma = typename cutlass::gemm::threadblock::DefaultMma<
    int8_t, cutlass::layout::RowMajor, 16,
    int8_t, cutlass::layout::ColumnMajor, 16,
    int32_t, cutlass::layout::RowMajor,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    3, cutlass::arch::OpMultiplyAdd,
    false, cutlass::gemm::SharedMemoryClearOption::kNone>::ThreadblockMma;

__host__ __device__ inline int pearl_perm_row(int physical) {
    const int within = physical % 64;
    const int packed = (physical / 64) * 64 + (within % 8) * 8 + within / 8;
    const int tile = packed / 8;
    const int u = packed % 8;
    const int row_base = tile < 8 ? tile : tile + 8;
    return row_base + (u / 2) * 32 + (u % 2) * 8;
}

__host__ __device__ inline int pearl_perm_col(int physical) {
    const int within = physical % 64;
    const int packed = (physical / 64) * 64 + ((within % 8) / 2) * 16
                       + (within / 8) * 2 + (within % 2);
    const int tile = packed / 16;
    const int v = packed % 16;
    return tile * 2 + (v / 2) * 32 + (v % 2);
}

__global__ void pearl_perm_a(const int8_t *input, int8_t *output, int m) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = 16 * m * 128;
    if (idx >= total) return;
    const int k = idx % 128;
    const int row = (idx / 128) % m;
    const int step = idx / (m * 128);
    const int logical = (row / 128) * 128 + pearl_perm_row(row % 128);
    output[idx] = input[(step * m + logical) * 128 + k];
}

__global__ void pearl_perm_b(const int8_t *input, int8_t *output, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = 16 * n * 128;
    if (idx >= total) return;
    const int k = idx % 128;
    const int col = (idx / 128) % n;
    const int step = idx / (n * 128);
    const int logical = (col / 256) * 256 + pearl_perm_col(col % 256);
    output[idx] = input[(step * n + logical) * 128 + k];
}

__global__ void pearl_cutlass_transcript(int8_t *a, int8_t *b, uint32_t *out,
                                          int m, int n, int rpi0, int cpi0) {
    extern __shared__ __align__(16) unsigned char buffer[];
    auto &storage = *reinterpret_cast<PearlTensorMma::SharedStorage *>(buffer);
    const int tid = threadIdx.x;
    const int row_period = rpi0 + blockIdx.x;
    const int col_period = cpi0 + blockIdx.y;
    PearlTensorMma mma(storage, tid, tid / 32, tid % 32);
    typename PearlTensorMma::IteratorA::Params ap(
        cutlass::layout::RowMajor::packed({m, 128}));
    typename PearlTensorMma::IteratorB::Params bp(
        cutlass::layout::ColumnMajor::packed({128, n}));
    PearlTensorMma::FragmentC accum;
    accum.clear();
    for (int step = 0; step < 16; ++step) {
        int8_t *sa = a + size_t(step) * m * 128;
        int8_t *sb = b + size_t(step) * n * 128;
        typename PearlTensorMma::IteratorA ia(
            ap, sa, {m, 128}, tid, {row_period * 128, 0});
        typename PearlTensorMma::IteratorB ib(
            bp, sb, {128, n}, tid, {0, col_period * 256});
        mma(2, accum, ia, ib, accum);
        uint32_t xor_word = 0;
        #pragma unroll
        for (int i = 0; i < PearlTensorMma::FragmentC::kElements; ++i)
            xor_word ^= uint32_t(accum[i]);
        const size_t cta = size_t(blockIdx.x) * gridDim.y + blockIdx.y;
        out[(cta * 16 + step) * 256 + tid] = xor_word;
        mma.wind_down();
    }
}
