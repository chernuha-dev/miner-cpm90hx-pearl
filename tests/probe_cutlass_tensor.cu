// Inspect public CUTLASS sm80 INT8 accumulator partition on sm86.
#include <cstdio>
#include <cstdint>
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

int main() {
    printf("fragment_cells=%d shared_bytes=%zu\n",
           DefaultMma::FragmentC::kElements, sizeof(DefaultMma::SharedStorage));
}
