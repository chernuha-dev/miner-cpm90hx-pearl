#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "cutlass_sm86_transcript.cuh"

static void check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    constexpr int m = 131072, n = 262144;
    constexpr int row_batch = 16, col_batch = 512;
    const size_t bytes_a = size_t(16) * m * 128;
    const size_t bytes_b = size_t(16) * n * 128;
    const size_t words_out = size_t(row_batch) * col_batch * 16 * 256;
    int8_t *a, *b, *pa, *pb;
    uint32_t *out;
    check(cudaMalloc(&a, bytes_a), "alloc A");
    check(cudaMalloc(&b, bytes_b), "alloc B");
    check(cudaMalloc(&pa, bytes_a), "alloc perm A");
    check(cudaMalloc(&pb, bytes_b), "alloc perm B");
    check(cudaMalloc(&out, words_out * sizeof(uint32_t)), "alloc out");
    check(cudaMemset(a, 1, bytes_a), "init A");
    check(cudaMemset(b, 1, bytes_b), "init B");
    auto prep = [&]() {
        pearl_perm_a<<<(bytes_a + 255) / 256, 256>>>(a, pa, m);
        pearl_perm_b<<<(bytes_b + 255) / 256, 256>>>(b, pb, n);
    };
    auto scan = [&]() {
        pearl_cutlass_transcript<<<dim3(row_batch, col_batch), 256,
                                   sizeof(PearlTensorMma::SharedStorage)>>>(
            pa, pb, out, m, n, 0, 0);
    };
    prep();
    scan();
    check(cudaDeviceSynchronize(), "warmup");
    auto measure = [&](const char *name, bool include_prep) {
        float times[5];
        for (int i = 0; i < 5; ++i) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            if (include_prep) prep();
            scan();
            cudaEventRecord(stop);
            check(cudaEventSynchronize(stop), "benchmark");
            check(cudaEventElapsedTime(&times[i], start, stop), "elapsed");
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        float ms = 0;
        for (float t : times) ms += t / 5;
        double mac = double(row_batch) * 128 * col_batch * 256 * 2048;
        printf("%s mean=%.3f ms equivalent=%.2f TMAC/s\n",
               name, ms, mac / (ms * 1e-3) / 1e12);
    };
    measure("scan", false);
    measure("prep+scan", true);
    cudaFree(a); cudaFree(b); cudaFree(pa); cudaFree(pb); cudaFree(out);
}
