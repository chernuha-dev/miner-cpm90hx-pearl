#include <cstdio>
#include <cuda_runtime.h>
#include "direct_mma_sm86.cuh"

int main() {
    constexpr int m = 131072, n = 262144;
    constexpr int row_batch = 16, col_batch = 512;
    int8_t *a = nullptr, *b = nullptr;
    uint32_t *out = nullptr;
    cudaMalloc(&a, size_t(m) * 2048);
    cudaMalloc(&b, size_t(n) * 2048);
    cudaMalloc(&out, size_t(row_batch) * col_batch * 16 * 256 * 4);
    cudaMemset(a, 0, size_t(m) * 2048);
    cudaMemset(b, 0, size_t(n) * 2048);
    cudaEvent_t start, end;
    cudaEventCreate(&start); cudaEventCreate(&end);
    for (int i = 0; i < 2; ++i)
        pearl_direct_transcript<<<dim3(row_batch, col_batch), 256>>>(a,b,out,m,n,0,0);
    cudaDeviceSynchronize();
    float ms_sum = 0;
    for (int i = 0; i < 5; ++i) {
        cudaEventRecord(start);
        pearl_direct_transcript<<<dim3(row_batch, col_batch), 256>>>(a,b,out,m,n,0,0);
        cudaEventRecord(end);
        cudaError_t err = cudaEventSynchronize(end);
        if (err != cudaSuccess) {
            fprintf(stderr, "kernel: %s\n", cudaGetErrorString(err));
            return 1;
        }
        float ms = 0; cudaEventElapsedTime(&ms, start, end);
        ms_sum += ms;
        printf("run %d: %.3f ms\n", i, ms);
    }
    printf("direct MMA mean %.3f ms/batch %.2f TMAC/s scan-only\n",
           ms_sum / 5, 0.549755813888 * 5000 / ms_sum);
    cudaFree(a); cudaFree(b); cudaFree(out);
    return 0;
}
