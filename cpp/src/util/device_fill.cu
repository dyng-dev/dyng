// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file device_fill.cu
 * @brief The fill kernel and its explicit instantiations.
 */
#include "util/cuda_check.hpp"
#include "util/device_fill.hpp"
#include "util/kernel_registry.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace dyng::detail {

namespace {

constexpr int fill_block_size = 256;
// Grid-stride loop: a bounded grid (a few waves on any current GPU) covers any count.
constexpr std::size_t fill_max_blocks = 4096;

template <typename value_t>
__global__ void fill_kernel(value_t* data, std::size_t count, value_t value) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    data[i] = value;
  }
}

}  // namespace

template <typename value_t>
void fill_async(stream_ref stream, value_t* data, std::size_t count, value_t value) {
  if (count == 0) {
    return;
  }
  const std::size_t blocks =
      std::min(fill_max_blocks, (count + fill_block_size - 1) / fill_block_size);
  const auto native = static_cast<cudaStream_t>(stream.get());
  fill_kernel<value_t>
      <<<static_cast<unsigned int>(blocks), fill_block_size, 0, native>>>(data, count, value);
  DYNG_CHECK_KERNEL(native);
}

#define DYNG_INSTANTIATE_FILL(type)                                     \
  template void fill_async<type>(stream_ref, type*, std::size_t, type); \
  DYNG_REGISTER_KERNEL(fill_kernel<type>)

DYNG_INSTANTIATE_FILL(std::uint8_t);
DYNG_INSTANTIATE_FILL(std::int32_t);
DYNG_INSTANTIATE_FILL(std::uint32_t);
DYNG_INSTANTIATE_FILL(std::int64_t);
DYNG_INSTANTIATE_FILL(std::uint64_t);
DYNG_INSTANTIATE_FILL(float);
DYNG_INSTANTIATE_FILL(double);

#undef DYNG_INSTANTIATE_FILL

}  // namespace dyng::detail
