# INT8 GEMM probe

Measures cuBLAS INT8 tensor operations on each visible CUDA GPU. This is a **hardware probe**, not a PearlHash speed estimate: PearlHash also includes noise, transcripts, hashing, and proof generation.

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_86 bench/int8_gemm.cu -lcublas -o bench/int8_gemm
./bench/int8_gemm 2048
./bench/int8_gemm 4096
```

The binary is a local build artifact and should not be committed.
