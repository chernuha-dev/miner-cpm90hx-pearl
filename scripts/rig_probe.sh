#!/usr/bin/env bash
set -u

printf '=== system ===\n'
uname -a
if [ -r /etc/os-release ]; then
  sed -n '/^NAME=/p; /^VERSION=/p; /^ID=/p' /etc/os-release
fi

printf '\n=== NVIDIA driver and GPUs ===\n'
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi is unavailable\n'
else
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,compute_cap,driver_version,memory.total,power.limit,power.draw,temperature.gpu,clocks.current.graphics,clocks.current.memory --format=csv,noheader || nvidia-smi -L
  printf '\n=== CUDA processes ===\n'
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true
fi

printf '\n=== CUDA toolchain ===\n'
NVCC_BIN="$(command -v nvcc || true)"
if [ -z "$NVCC_BIN" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
  NVCC_BIN=/usr/local/cuda/bin/nvcc
fi
if [ -n "$NVCC_BIN" ]; then
  "$NVCC_BIN" --version | tail -4
else
  printf 'nvcc is unavailable\n'
fi
if command -v cmake >/dev/null 2>&1; then
  cmake --version | head -1
else
  printf 'cmake is unavailable\n'
fi
if command -v g++ >/dev/null 2>&1; then
  g++ --version | head -1
else
  printf 'g++ is unavailable\n'
fi
