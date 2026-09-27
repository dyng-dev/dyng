// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernel_registry.cpp
 * @brief The registry of the library's CUDA kernels.
 */
#include "util/kernel_registry.hpp"

#include <mutex>
#include <vector>

namespace dyng::detail {

namespace {

struct registry_state {
  std::mutex mutex;
  std::vector<kernel_entry> kernels;
};

// Constructed on first use: registrars of other translation units run during static
// initialisation, in an unspecified order.
registry_state& registry() noexcept {
  static registry_state state;
  return state;
}

}  // namespace

void register_kernel(const void* function, const char* name) noexcept {
  registry_state& state = registry();
  const std::lock_guard<std::mutex> lock(state.mutex);
  try {
    state.kernels.push_back(kernel_entry{function, name});
  } catch (...) {
    // Out of memory while the library loads: the kernel is then loaded lazily at its first
    // launch instead of by warm_up(); nothing else depends on the entry.
  }
}

std::vector<kernel_entry> registered_kernels() {
  registry_state& state = registry();
  const std::lock_guard<std::mutex> lock(state.mutex);
  return state.kernels;
}

}  // namespace dyng::detail
