// Inspect public CUTLASS sm80 INT8 accumulator partition on sm86.
#include <cstdio>
#include <cstdint>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>;
using DefaultKernel = typename cutlass::gemm::kernel::DefaultGemm<
    int8_t, cutlass::layout::RowMajor, 16,
    int8_t, cutlass::layout::ColumnMajor, 16,
    int32_t, cutlass::layout::RowMajor, int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2, false, cutlass::arch::OpMultiplyAdd,
    cutlass::gemm::SharedMemoryClearOption::kNone>::GemmKernel;

int main() {
    using Mma = DefaultKernel::Mma;
    printf("threads=%d fragment_cells=%d shared_bytes=%zu\n",
           DefaultKernel::kThreadCount, Mma::FragmentC::kElements,
           sizeof(Mma::SharedStorage));
}
