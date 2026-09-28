// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file clock_holder.cu
 * @brief Keeps the GPU clocks that Nsight Compute locked for the whole CUDA performance A/B
 *        (parity/perf_ab.py run --lock-clocks, ADR 0018).
 *
 * `ncu --clock-control boost|base clock_holder` profiles this program's single kernel and locks
 * the GPU's SM and memory clocks for as long as the profiled process lives; the lock applies to
 * every process on the GPU, so both timed programs (the unpatched original and dynG, neither of
 * them profiled) run at the same fixed clocks. No root is needed (profiling must be allowed to
 * non-admin users, RmProfilingAdminOnly=0). The program launches one kernel, prints
 * `clock_holder ready pid=<pid>`, then idles (no GPU work) until its standard input reaches end
 * of file, and exits normally so that ncu restores the clocks. perf_ab.py also runs
 * `ncu --clock-control reset` afterwards, because a killed ncu session leaves the clocks locked.
 */
#include <cuda_runtime.h>

#include <unistd.h>

#include <cstdio>

namespace {

__global__ void clock_holder_kernel(int* flag) {
  if (threadIdx.x == 0) {
    *flag = 1;
  }
}

}  // namespace

int main() {
  int* flag = nullptr;
  if (cudaMalloc(&flag, sizeof(int)) != cudaSuccess) {
    std::fprintf(stderr, "clock_holder: cudaMalloc failed\n");
    return 1;
  }
  clock_holder_kernel<<<1, 32>>>(flag);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "clock_holder: the kernel failed\n");
    return 1;
  }
  std::printf("clock_holder ready pid=%d\n", static_cast<int>(getpid()));
  std::fflush(stdout);
  char buffer[256];
  while (read(0, buffer, sizeof buffer) > 0) {
  }
  cudaFree(flag);
  return 0;
}
