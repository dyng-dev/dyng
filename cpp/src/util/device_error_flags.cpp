// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file device_error_flags.cpp
 * @brief The sticky device error word (host part). The class needs the CUDA runtime; the
 *        description and the exception mapping are compiled in every build.
 */
#include "util/device_error_flags.hpp"

#include "core/resources_access.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>

#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

#if DYNG_HAS_CUDA
#include "util/cuda_check.hpp"

#include <cuda_runtime_api.h>
#endif

namespace dyng::detail {

std::string describe_device_errors(std::uint32_t bits) {
  if (bits == 0) {
    return "none";
  }
  struct named {
    device_error kind;
    const char* name;
  };
  constexpr named kinds[] = {
      {device_error::capacity, "capacity"},
      {device_error::invalid_input, "invalid_input"},
      {device_error::parent_cycle, "parent_cycle"},
      {device_error::internal, "internal"},
  };
  std::ostringstream out;
  const char* separator = "";
  std::uint32_t known = 0;
  for (const named& k : kinds) {
    known |= bits_of(k.kind);
    if ((bits & bits_of(k.kind)) != 0) {
      out << separator << k.name;
      separator = ", ";
    }
  }
  if ((bits & ~known) != 0) {
    out << separator << "unknown bits 0x" << std::hex << (bits & ~known);
  }
  return out.str();
}

void throw_device_errors(std::uint32_t bits, std::string_view context, std::string_view detail) {
  std::string message =
      concat_message("dyng: ", context, ": device error (", describe_device_errors(bits), ")");
  if (!detail.empty()) {
    message += concat_message(": ", detail);
  }
  const std::uint32_t known = bits_of(device_error::capacity) |
                              bits_of(device_error::invalid_input) |
                              bits_of(device_error::parent_cycle);
  if ((bits & bits_of(device_error::internal)) != 0 || (bits & ~known) != 0) {
    throw internal_error(message + " (a library bug; please report it)");
  }
  if ((bits & bits_of(device_error::capacity)) != 0) {
    throw capacity_error(message);
  }
  throw invalid_argument_error(message);
}

#if DYNG_HAS_CUDA

device_error_flags::device_error_flags(const resources& res)
    : word_(1, res.stream(), res.memory(), res.device()),
      mirror_(1, res.stream(), resources_access::staging_memory(res), -1) {
  mirror_[0] = 0;
  const scoped_device guard(res.device());
  reset(res.stream());
}

void device_error_flags::reset(stream_ref stream) {
  DYNG_CUDA_TRY(cudaMemsetAsync(word_.data(), 0, sizeof(std::uint32_t),
                                static_cast<cudaStream_t>(stream.get())));
}

void device_error_flags::enqueue_readback(stream_ref stream) {
  DYNG_CUDA_TRY(cudaMemcpyAsync(mirror_.data(), word_.data(), sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream.get())));
}

std::uint32_t device_error_flags::mirrored() const noexcept {
  return *static_cast<const volatile std::uint32_t*>(mirror_.data());
}

std::uint32_t device_error_flags::read(stream_ref stream) {
  enqueue_readback(stream);
  DYNG_CUDA_TRY(cudaStreamSynchronize(static_cast<cudaStream_t>(stream.get())));
  return mirrored();
}

void device_error_flags::throw_if_raised(std::string_view context, std::string_view detail) const {
  const std::uint32_t bits = mirrored();
  if (bits != 0) {
    throw_device_errors(bits, context, detail);
  }
}

#else  // !DYNG_HAS_CUDA

device_error_flags::device_error_flags(const resources& /*res*/) {
  throw not_supported_error("dyng: device error flags need the cuda backend, which is not built");
}

void device_error_flags::reset(stream_ref /*stream*/) {}

void device_error_flags::enqueue_readback(stream_ref /*stream*/) {}

std::uint32_t device_error_flags::mirrored() const noexcept {
  return 0;
}

std::uint32_t device_error_flags::read(stream_ref /*stream*/) {
  return 0;
}

void device_error_flags::throw_if_raised(std::string_view /*context*/,
                                         std::string_view /*detail*/) const {}

#endif  // DYNG_HAS_CUDA

}  // namespace dyng::detail
