#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "cutlass_sm86_transcript.cuh"

int main() {
    constexpr int m = 128, n = 256;
    std::mt19937 rng(27);
    std::uniform_int_distribution<int> dist(-7, 7);
    std::vector<int8_t> a(16 * m * 128), b(16 * n * 128);
    for (auto &v : a) v = int8_t(dist(rng));
    for (auto &v : b) v = int8_t(dist(rng));
    std::vector<int8_t> ap(a.size()), bp(b.size());
    for (int step = 0; step < 16; ++step) {
        for (int physical = 0; physical < m; ++physical) {
            const int logical = pearl_perm_row(physical);
            for (int k = 0; k < 128; ++k)
                ap[(step * m + physical) * 128 + k] =
                    a[(step * m + logical) * 128 + k];
        }
        for (int physical = 0; physical < n; ++physical) {
            const int logical = pearl_perm_col(physical);
            for (int k = 0; k < 128; ++k)
                bp[(step * n + physical) * 128 + k] =
                    b[(step * n + logical) * 128 + k];
        }
    }
    int8_t *da = nullptr, *db = nullptr;
    uint32_t *dout = nullptr;
    cudaMalloc(&da, ap.size());
    cudaMalloc(&db, bp.size());
    cudaMalloc(&dout, 16 * 256 * sizeof(uint32_t));
    cudaMemcpy(da, ap.data(), ap.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(db, bp.data(), bp.size(), cudaMemcpyHostToDevice);
    pearl_cutlass_transcript<<<dim3(1, 1), 256,
                               sizeof(PearlTensorMma::SharedStorage)>>>(
        da, db, dout, m, n, 0, 0);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    std::vector<uint32_t> got(16 * 256), expected(16 * 256, 0);
    cudaMemcpy(got.data(), dout, got.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            int32_t acc = 0;
            const int tile_row = (row % 8) + 8 * ((row % 32) / 16);
            const int tile_col = (col % 32) / 2;
            const int tile = tile_row * 16 + tile_col;
            for (int step = 0; step < 16; ++step) {
                for (int k = 0; k < 128; ++k)
                    acc += int(a[(step * m + row) * 128 + k]) *
                           int(b[(step * n + col) * 128 + k]);
                expected[step * 256 + tile] ^= uint32_t(acc);
            }
        }
    }
    for (int step = 0; step < 16; ++step) {
        for (int row_tile = 0; row_tile < 16; ++row_tile) {
            for (int col_tile = 0; col_tile < 16; ++col_tile) {
                const int warp = (col_tile / 4) * 2 + row_tile / 8;
                const int lane = (row_tile % 8) * 4 + col_tile % 4;
                const int thread = warp * 32 + lane;
                const int tile = row_tile * 16 + col_tile;
                if (got[step * 256 + thread] != expected[step * 256 + tile]) {
                    fprintf(stderr, "step %d tile (%d,%d) thread %d: got %08x expected %08x\n",
                            step, row_tile, col_tile, thread,
                            got[step * 256 + thread], expected[step * 256 + tile]);
                    return 1;
                }
            }
        }
    }
    cudaFuncAttributes attr{};
    cudaFuncGetAttributes(&attr, pearl_cutlass_transcript);
    printf("16 milestones x 256 hash tiles: exact transcript words; "
           "regs=%d static_smem=%zu dynamic_smem=%zu\n",
           attr.numRegs, attr.sharedSizeBytes, sizeof(PearlTensorMma::SharedStorage));
    cudaFree(da); cudaFree(db); cudaFree(dout);
    return 0;
}
