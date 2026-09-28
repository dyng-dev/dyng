// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernel_registry.hpp
 * @brief The registry of the library's CUDA kernels, read by resources::warm_up() (PLAN Section
 *        4.7.1: one of the three pieces of process-global state, with the log level and sink).
 *
 * Every .cu file of the library registers its kernels (every explicit instantiation of a kernel
 * template) at load time:
 *
 *     DYNG_REGISTER_KERNEL(fill_kernel<std::int32_t>);
 *
 * warm_up() then asks the runtime for each kernel's attributes, which loads its module under lazy
 * loading (the CUDA 12.2+ default), so no kernel is loaded inside a timed call.
 *
 * Host-only: the registry stores the host-side handles of the kernels as `const void*`.
 */
#pragma once

#include <cstddef>
#include <vector>

namespace dyng::detail {

/**
 * @brief One registered kernel.
 */
struct kernel_entry {
  const void* function = nullptr;  ///< the kernel's host handle (what cudaFuncGetAttributes takes)
  const char* name = nullptr;      ///< its source name, for messages
};

/**
 * @brief Add a kernel to the registry (called by kernel_registrar at load time).
 * @param[in] function The kernel's host handle.
 * @param[in] name     Its source name (a string literal).
 */
void register_kernel(const void* function, const char* name) noexcept;

/**
 * @brief A snapshot of the registry.
 * @return Every registered kernel, in registration order.
 */
[[nodiscard]] std::vector<kernel_entry> registered_kernels();

/**
 * @brief Registers one kernel when constructed (a namespace-scope static per kernel).
 */
struct kernel_registrar {
  /**
   * @brief Register a kernel.
   * @param[in] function The kernel's host handle.
   * @param[in] name     Its source name.
   */
  kernel_registrar(const void* function, const char* name) noexcept {
    register_kernel(function, name);
  }
};

}  // namespace dyng::detail

#define DYNG_KERNEL_REGISTRY_CONCAT_INNER(a, b) a##b
#define DYNG_KERNEL_REGISTRY_CONCAT(a, b) DYNG_KERNEL_REGISTRY_CONCAT_INNER(a, b)

/// Register a kernel (or one instantiation of a kernel template) for resources::warm_up().
#define DYNG_REGISTER_KERNEL(...)                                            \
  static const ::dyng::detail::kernel_registrar DYNG_KERNEL_REGISTRY_CONCAT( \
      dyng_kernel_registrar_, __LINE__)(reinterpret_cast<const void*>(&__VA_ARGS__), #__VA_ARGS__)
