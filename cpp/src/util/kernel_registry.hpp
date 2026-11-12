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

/// The kernel group of the translation unit: cmake/dyng_kernel_groups.cmake defines
/// DYNG_KERNEL_GROUP (a string literal) for the CUDA sources of a group.
#ifdef DYNG_KERNEL_GROUP
#define DYNG_KERNEL_REGISTRY_GROUP DYNG_KERNEL_GROUP
#else
#define DYNG_KERNEL_REGISTRY_GROUP nullptr
#endif

namespace dyng::detail {

/**
 * @brief One registered kernel.
 */
struct kernel_entry {
  const void* function = nullptr;  ///< the kernel's host handle (what cudaFuncGetAttributes takes)
  const char* name = nullptr;      ///< its source name, for messages
  /// its kernel group (core/kernel_groups.hpp), or nullptr for a kernel outside every group
  const char* group = nullptr;
};

/**
 * @brief Add a kernel to the registry (called by kernel_registrar at load time).
 * @param[in] function The kernel's host handle.
 * @param[in] name     Its source name (a string literal).
 * @param[in] group    Its kernel group (a string literal), or nullptr; the default is the
 *                     calling translation unit's group (DYNG_KERNEL_REGISTRY_GROUP).
 */
void register_kernel(const void* function, const char* name,
                     const char* group = DYNG_KERNEL_REGISTRY_GROUP) noexcept;

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
   * @param[in] group    Its kernel group, or nullptr; the default is the constructing translation
   *                     unit's group (DYNG_KERNEL_REGISTRY_GROUP).
   */
  kernel_registrar(const void* function, const char* name,
                   const char* group = DYNG_KERNEL_REGISTRY_GROUP) noexcept {
    register_kernel(function, name, group);
  }
};

}  // namespace dyng::detail

#define DYNG_KERNEL_REGISTRY_CONCAT_INNER(a, b) a##b
#define DYNG_KERNEL_REGISTRY_CONCAT(a, b) DYNG_KERNEL_REGISTRY_CONCAT_INNER(a, b)

/// Register a kernel (or one instantiation of a kernel template) for resources::warm_up().
#define DYNG_REGISTER_KERNEL(...)                                            \
  static const ::dyng::detail::kernel_registrar DYNG_KERNEL_REGISTRY_CONCAT( \
      dyng_kernel_registrar_, __LINE__)(reinterpret_cast<const void*>(&__VA_ARGS__), #__VA_ARGS__)
