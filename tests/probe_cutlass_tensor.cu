// Inspect public CUTLASS sm80 INT8 accumulator partition on sm86.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/default_mma.h"

using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
    int8_t, cutlass::layout::RowMajor, 16,
    int8_t, cutlass::layout::ColumnMajor, 16,
    int32_t, cutlass::layout::RowMajor,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    2, cutlass::arch::OpMultiplyAdd,
    false, cutlass::gemm::SharedMemoryClearOption::kNone>::ThreadblockMma;

__global__ void inspect_mma(int8_t *a, int8_t *b, int32_t *out) {
    extern __shared__ __align__(16) unsigned char buffer[];
    auto &storage = *reinterpret_cast<DefaultMma::SharedStorage *>(buffer);
    const int tid = threadIdx.x;
    DefaultMma mma(storage, tid, tid / 32, tid % 32);
    typename DefaultMma::IteratorA::Params ap(
        cutlass::layout::RowMajor::packed({128, 128}));
    typename DefaultMma::IteratorB::Params bp(
        cutlass::layout::ColumnMajor::packed({128, 256}));
    typename DefaultMma::IteratorA ia(ap, a, {128, 128}, tid, {0, 0});
    typename DefaultMma::IteratorB ib(bp, b, {128, 256}, tid, {0, 0});
    DefaultMma::FragmentC acc;
    acc.clear();
    mma(2, acc, ia, ib, acc);
    #pragma unroll
    for (int i = 0; i < DefaultMma::FragmentC::kElements; ++i)
        out[tid * DefaultMma::FragmentC::kElements + i] = acc[i];
}

int main() {
    printf("fragment_cells=%zu shared_bytes=%zu\n",
           size_t(DefaultMma::FragmentC::kElements), sizeof(DefaultMma::SharedStorage));
    std::vector<int8_t> a(128 * 128, 0), b(256 * 128, 0);
    for (int row = 0; row < 128; ++row) {
        int within = row % 64;
        int logical_row = (row / 64) * 64 + (within % 8) * 8 + within / 8;
        for (int k = 0; k < 4; ++k) a[row * 128 + k] = int8_t(logical_row);
        a[row * 128 + 4] = 1;
        a[row * 128 + 5] = 64;
    }
    for (int col = 0; col < 256; ++col) {
        int within = col % 64;
        int logical_col = (col / 64) * 64 + ((within % 8) / 2) * 16
                          + (within / 8) * 2 + (within % 2);
        for (int k = 0; k < 4; ++k) b[col * 128 + k] = 64;
        b[col * 128 + 4] = int8_t(logical_col % 128);
        b[col * 128 + 5] = int8_t(2 * (logical_col / 128));
    }
    int8_t *da, *db;
    int32_t *dout;
    cudaMalloc(&da, a.size());
    cudaMalloc(&db, b.size());
    cudaMalloc(&dout, 128 * 256 * sizeof(int32_t));
    cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice);
    inspect_mma<<<1, 256, sizeof(DefaultMma::SharedStorage)>>>(da, db, dout);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    std::vector<int32_t> out(128 * 256);
    cudaMemcpy(out.data(), dout, out.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
    std::vector<int> seen(128 * 256, 0);
    int single_tile_threads = 0;
    bool tile_seen[256] = {};
    for (int tid = 0; tid < 256; ++tid) {
        int tile = -1;
        bool same_tile = true;
        bool row_seen[128] = {}, col_seen[256] = {};
        for (int i = 0; i < 128; ++i) {
            int value = out[tid * 128 + i];
            if (value < 0 || value >= 128 * 256) {
                fprintf(stderr, "invalid value thread=%d cell=%d value=%d\n", tid, i, value);
                return 1;
            }
            ++seen[value];
            int row = value / 256, col = value % 256;
            row_seen[row] = true;
            col_seen[col] = true;
            int t = (row / 8) * 16 + col / 16;
            if (tile == -1) tile = t;
            if (tile != t) same_tile = false;
        }
        single_tile_threads += same_tile;
        if (same_tile) {
            if (tile_seen[tile]) {
                fprintf(stderr, "duplicate logical tile %d\n", tile);
                return 1;
            }
            tile_seen[tile] = true;
        }
        if (tid < 4) {
            int nr = 0, nc = 0;
            for (bool v : row_seen) nr += v;
            for (bool v : col_seen) nc += v;
            printf("thread %d first=%d last=%d tile=%d single=%d rows=%d cols=%d\n",
                   tid, out[tid * 128], out[tid * 128 + 127], tile, same_tile, nr, nc);
            if (tid == 0) {
                printf("thread 0 rows:");
                for (int row = 0; row < 128; ++row) if (row_seen[row]) printf(" %d", row);
                printf("\nthread 0 cols:");
                for (int col = 0; col < 256; ++col) if (col_seen[col]) printf(" %d", col);
                printf("\n");
            }
        }
    }
    int unique = 0, missing = 0, duplicated = 0;
    for (int count : seen) {
        unique += count == 1;
        missing += count == 0;
        duplicated += count > 1;
    }
    printf("single_tile_threads=%d/256 unique=%d missing=%d duplicated=%d\n",
           single_tile_threads, unique, missing, duplicated);
    cudaFree(da); cudaFree(db); cudaFree(dout);
    return unique == 128 * 256 && single_tile_threads == 256 ? 0 : 1;
}
