#pragma once

// Experimental register-only sm86 path. The PTX m16n8k32 fragment coordinates
// are used to load packed A/B directly from global memory without shared memory.
#include "cutlass/arch/mma_sm80.h"
#include "cutlass/gemm/gemm.h"

using PearlDirectMma = cutlass::arch::Mma<
    cutlass::gemm::GemmShape<16, 8, 32>, 32,
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    cutlass::arch::OpMultiplyAdd>;

__global__ void pearl_direct_transcript(const int8_t *a, const int8_t *b,
                                         uint32_t *out, int m, int n,
                                         int rpi0, int cpi0) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int group = (tid % 32) / 4;
    const int four = tid % 4;
    const int row0 = (rpi0 + blockIdx.x) * 128 + (warp % 2) * 64;
    const int col0 = (cpi0 + blockIdx.y) * 256 + (warp / 2) * 64;
    PearlDirectMma op;
    PearlDirectMma::FragmentC accum[4][8];
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 8; ++ni)
            accum[mi][ni].clear();

    #pragma unroll
    for (int step = 0; step < 16; ++step) {
        const int8_t *as = a + size_t(step) * m * 128;
        const int8_t *bs = b + size_t(step) * n * 128;
        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            PearlDirectMma::FragmentA fa[4];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int ar0 = row0 + mi * 16 + group;
                const int ar1 = ar0 + 8;
                const int ak = chunk * 32 + four * 4;
                auto *d = reinterpret_cast<uint32_t *>(&fa[mi]);
                d[0] = *reinterpret_cast<const uint32_t *>(as + size_t(ar0) * 128 + ak);
                d[1] = *reinterpret_cast<const uint32_t *>(as + size_t(ar1) * 128 + ak);
                d[2] = *reinterpret_cast<const uint32_t *>(as + size_t(ar0) * 128 + ak + 16);
                d[3] = *reinterpret_cast<const uint32_t *>(as + size_t(ar1) * 128 + ak + 16);
            }
            #pragma unroll
            for (int ni = 0; ni < 8; ++ni) {
                PearlDirectMma::FragmentB fb;
                const int bc = col0 + ni * 8 + group;
                const int bk = chunk * 32 + four * 4;
                auto *d = reinterpret_cast<uint32_t *>(&fb);
                d[0] = *reinterpret_cast<const uint32_t *>(bs + size_t(bc) * 128 + bk);
                d[1] = *reinterpret_cast<const uint32_t *>(bs + size_t(bc) * 128 + bk + 16);
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi)
                    op(accum[mi][ni], fa[mi], fb, accum[mi][ni]);
            }
        }
        uint32_t word = 0;
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi)
            #pragma unroll
            for (int ni = 0; ni < 8; ++ni)
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    word ^= uint32_t(accum[mi][ni][i]);
        const size_t cta = size_t(blockIdx.x) * gridDim.y + blockIdx.y;
        out[(cta * 16 + step) * 256 + tid] = word;
    }
}
