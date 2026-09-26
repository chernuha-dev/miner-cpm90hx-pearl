#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call) do { \
  cudaError_t status = (call); \
  if (status != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
    std::exit(1); \
  } \
} while (0)

#define CHECK_CUBLAS(call) do { \
  cublasStatus_t status = (call); \
  if (status != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "%s: cuBLAS status %d\n", #call, int(status)); \
    std::exit(1); \
  } \
} while (0)

static void benchmark(int device, int size) {
  CHECK_CUDA(cudaSetDevice(device));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::printf("GPU %d: %s, sm_%d%d, GEMM %d x %d x %d\n",
              device, prop.name, prop.major, prop.minor, size, size, size);

  const size_t matrix_bytes = size_t(size) * size;
  int8_t *a = nullptr, *b = nullptr;
  int32_t *c = nullptr;
  CHECK_CUDA(cudaMalloc(&a, matrix_bytes));
  CHECK_CUDA(cudaMalloc(&b, matrix_bytes));
  CHECK_CUDA(cudaMalloc(&c, matrix_bytes * sizeof(int32_t)));
  CHECK_CUDA(cudaMemset(a, 1, matrix_bytes));
  CHECK_CUDA(cudaMemset(b, 1, matrix_bytes));
  CHECK_CUDA(cudaMemset(c, 0, matrix_bytes * sizeof(int32_t)));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  const int32_t alpha = 1, beta = 0;
  auto gemm = [&] {
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      size, size, size, &alpha, a, CUDA_R_8I, size, b, CUDA_R_8I, size,
      &beta, c, CUDA_R_32I, size, CUBLAS_COMPUTE_32I,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };

  for (int i = 0; i < 5; ++i) gemm();
  CHECK_CUDA(cudaDeviceSynchronize());
  int32_t first = 0;
  CHECK_CUDA(cudaMemcpy(&first, c, sizeof(first), cudaMemcpyDeviceToHost));
  if (first != size) {
    std::fprintf(stderr, "Incorrect GEMM result: got %d, expected %d\n", first, size);
    std::exit(1);
  }

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  constexpr int iterations = 40;
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) gemm();
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  const double tops = 2.0 * size * size * size * iterations / (ms * 1e9);
  std::printf("  %.3f ms/GEMM, %.3f INT8 TOPS (cuBLAS), result=%d\n",
              ms / iterations, tops, first);

  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(c));
  CHECK_CUDA(cudaFree(b));
  CHECK_CUDA(cudaFree(a));
}

int main(int argc, char **argv) {
  int count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&count));
  const int size = argc > 1 ? std::atoi(argv[1]) : 2048;
  if (size < 256 || size % 32 != 0) {
    std::fprintf(stderr, "Size must be >=256 and divisible by 32\n");
    return 2;
  }
  for (int device = 0; device < count; ++device) benchmark(device, size);
  return 0;
}
