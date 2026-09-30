// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file context.hpp
 * @brief framework::context: what every hook of one run receives (PLAN Section 4.5.2): the
 *        resources of the call, the algorithm's name, the workspace pool and the device error
 *        word of the run.
 *
 * A context lives for one compute() or one update() of one result (in dyng::update(res, g,
 * batch, r1, r2, ...) each result has its own). It owns no memory: scratch space is leased from
 * the pool of the resources handle (ADR 0015), frontiers are owned by the enactor.
 *
 * Device errors. A hook that reads back a device error word (for example with the control block
 * of a persistent kernel, merged into a synchronization it makes anyway) records it with
 * raise_device_error() instead of throwing; the enactor checks the recorded bits after the last
 * hook of the phase and throws the precise exception (throw_device_errors: capacity_error,
 * invalid_argument_error or internal_error) at the API boundary. A hook may also throw directly;
 * the enactor's check is the uniform path.
 */
#pragma once

#include "core/resources_access.hpp"
#include "framework/workspace.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace dyng::detail::framework {

/**
 * @brief The context of one run of one problem.
 */
class context {
 public:
  /**
   * @brief A context for a run through `res`.
   * @param[in] res       The resources of the call (must outlive the context).
   * @param[in] algorithm The algorithm's name (`problem_t::name`, e.g. "sssp"); must outlive the
   *                      context (a string literal).
   */
  context(const resources& res, std::string_view algorithm) noexcept
      : res_(&res), algorithm_(algorithm) {}

  /**
   * @brief The resources of the call.
   * @return The handle.
   */
  [[nodiscard]] const resources& res() const noexcept {
    return *res_;
  }

  /**
   * @brief The algorithm's name.
   * @return E.g. "sssp" (the prefix of its profiler stages and messages).
   */
  [[nodiscard]] std::string_view algorithm() const noexcept {
    return algorithm_;
  }

  /**
   * @brief The backend of the call.
   * @return res().get_backend().
   */
  [[nodiscard]] backend get_backend() const noexcept {
    return res_->get_backend();
  }

  /**
   * @brief Whether the call runs on the CUDA backend.
   * @return true for resources::cuda().
   */
  [[nodiscard]] bool on_cuda() const noexcept {
    return res_->get_backend() == backend::cuda;
  }

  /**
   * @brief Host threads of the call's host-side work (the thread count of an OpenMP or CUDA
   *        handle, 1 for the sequential backend).
   * @return The thread count.
   */
  [[nodiscard]] int host_threads() const noexcept {
    return resources_access::host_threads(*res_);
  }

  /**
   * @brief The workspace pool of the resources handle.
   * @return The pool (shared by every copy of the handle).
   */
  [[nodiscard]] workspace_pool& workspaces() const noexcept {
    return resources_access::workspaces(*res_);
  }

  /**
   * @brief Lease a workspace of the run (ordered on the stream of a CUDA handle; ADR 0015).
   * @tparam workspace_t The workspace type (derived from pooled_workspace).
   * @return The lease; the caller sizes the workspace.
   * @throws std::bad_alloc if a new workspace cannot be created.
   * @throws cuda_error     if the stream cannot be made to wait for the workspace's last use.
   */
  template <typename workspace_t>
  [[nodiscard]] workspace_pool::lease<workspace_t> lease_workspace() const {
    return workspaces().acquire<workspace_t>(*res_);
  }

  /**
   * @brief The engine the enactor chose for this run (before the commit of an update, before
   *        reset() of a compute()); hooks that differ between the engines read it (for example
   *        the subtraction on G_t of an aggregate_delta problem whose fused engine has its own).
   * @return engine::fused or engine::operators; engine::automatic before the enactor chose.
   */
  [[nodiscard]] engine chosen_engine() const noexcept {
    return engine_;
  }

  /**
   * @brief Record the enactor's choice of engine (the enactors call this).
   * @param[in] e engine::fused or engine::operators.
   */
  void set_chosen_engine(engine e) noexcept {
    engine_ = e;
  }

  /**
   * @brief Record device error bits (device_error in util/device_error_flags.hpp); the enactor
   *        throws after the phase's last hook.
   * @param[in] bits   The error word read back from the device (0 records nothing).
   * @param[in] detail Optional precise description (kept for the first non-zero word).
   */
  void raise_device_error(std::uint32_t bits, std::string_view detail = {}) {
    if (bits == 0) {
      return;
    }
    if (errors_ == 0) {
      error_detail_.assign(detail.data(), detail.size());
    }
    errors_ |= bits;
  }

  /**
   * @brief The recorded device error bits.
   * @return The union of every raise_device_error() since the last clear.
   */
  [[nodiscard]] std::uint32_t device_errors() const noexcept {
    return errors_;
  }

  /**
   * @brief The description of the first recorded device error.
   * @return The detail given to raise_device_error() (may be empty).
   */
  [[nodiscard]] const std::string& device_error_detail() const noexcept {
    return error_detail_;
  }

  /**
   * @brief Forget the recorded device errors (the enactor does this before throwing).
   */
  void clear_device_errors() noexcept {
    errors_ = 0;
    error_detail_.clear();
  }

 private:
  const resources* res_;
  std::string_view algorithm_;
  engine engine_ = engine::automatic;
  std::uint32_t errors_ = 0;
  std::string error_detail_;
};

}  // namespace dyng::detail::framework
