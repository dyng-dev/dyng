// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file resources_access.hpp
 * @brief Internal access to the parts of a resources handle that are not public: the workspace
 *        pool (ADR 0015), the CUDA device record (PLAN Section 4.6 rule 7) and the host staging
 *        resource (PLAN Section 4.7.1).
 */
#pragma once

#include "core/cuda_runtime.hpp"

#include <dyng/core/memory.hpp>

namespace dyng {
class resources;
}  // namespace dyng

namespace dyng::detail {

class workspace_pool;

/**
 * @brief Internal access to the parts of a resources handle that are not public.
 */
struct resources_access {
  /**
   * @brief The workspace pool of a handle (shared by all its copies).
   * @param[in] res The handle.
   * @return Its pool.
   */
  static workspace_pool& workspaces(const resources& res) noexcept;

  /**
   * @brief What the CUDA backend recorded about its device when the handle was created.
   * @param[in] res A handle of the CUDA backend.
   * @return The device record (shared by every copy of the handle).
   * @throws internal_error if `res` is not a CUDA handle.
   */
  static const cuda_device_properties& device_properties(const resources& res);

  /**
   * @brief Override the recorded cooperative-launch capability (tests of the engine selection:
   *        engine::automatic must throw not_supported_error on a device without it).
   * @param[in] res       A handle of the CUDA backend (affects every copy).
   * @param[in] available The capability to report from now on.
   * @throws internal_error if `res` is not a CUDA handle.
   */
  static void force_cooperative_launch(const resources& res, bool available);

  /**
   * @brief The resource for host staging buffers (batch uploads, control-block mirrors).
   * @param[in] res The handle.
   * @return The pinned host resource for the CUDA backend, the host resource otherwise.
   */
  static memory_resource_ref staging_memory(const resources& res) noexcept;

  /**
   * @brief Host threads for the host-side work of a call (graph builds and applies, tree imports
   *        and checks): the thread count of an OpenMP handle, the OpenMP default for a CUDA handle
   *        (its host work runs next to the device; 1 without OpenMP), 1 for the sequential
   *        backend.
   * @param[in] res The handle.
   * @return The thread count (>= 1).
   */
  static int host_threads(const resources& res) noexcept;
};

}  // namespace dyng::detail
