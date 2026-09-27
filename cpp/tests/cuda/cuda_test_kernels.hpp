// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_test_kernels.hpp
 * @brief Host launchers of the test kernels (cuda_test_kernels.cu), callable from .cpp tests.
 */
#pragma once

#include <dyng/core/memory.hpp>
#include <dyng/core/stream.hpp>

#include <cstdint>

namespace dyng::test {

/// Raise `bits` in a device error word from `threads` threads (atomicOr).
void launch_raise_device_errors(stream_ref stream, std::uint32_t* word, std::uint32_t bits,
                                int threads);

/// Launch a kernel with an invalid configuration (2048 threads per block) and check it with
/// DYNG_CHECK_KERNEL, which must throw cuda_error.
void launch_invalid_configuration(stream_ref stream);

/// Write value + i to data[i] for i < count on the device (a kernel, not a memset).
void launch_write_sequence(stream_ref stream, std::int64_t* data, std::int64_t count,
                           std::int64_t value);

/// The CCCL 3.x memory-resource concept accepts dynG's resources through the adapter of
/// util/cccl_memory_resource.cuh, and a CCCL-shaped resource adapts back into a
/// memory_resource_ref (PLAN Section 4.7.2): allocates `bytes` through both directions on
/// `stream`, writes and reads the memory, and returns true if every step worked.
bool cccl_adapters_round_trip(memory_resource_ref device_memory, stream_ref stream,
                              std::size_t bytes);

/// Whether the toolkit's CCCL has the 3.x memory-resource concept (CUDA 13; CUDA 12 ships 2.x).
bool cccl3_memory_resource_available();

}  // namespace dyng::test
