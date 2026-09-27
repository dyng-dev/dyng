// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file resources.cpp
 * @brief The resources handle.
 */
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "framework/workspace.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>

#include <string>

#if DYNG_HAS_OPENMP
#include <omp.h>
#endif

namespace dyng {

namespace detail {

/// Shared state of a resources handle and its copies.
struct resources_state {
  backend kind = backend::sequential;  ///< the backend
  int device = -1;                     ///< CUDA device, -1 for host backends
  stream_ref stream{};                 ///< stream all work is ordered on
  int num_threads = 1;                 ///< host threads of the backend
  memory_resource_ref memory{default_host_memory_resource()};   ///< allocations
  memory_resource_ref staging{default_host_memory_resource()};  ///< host staging buffers
  copy_policy copies = copy_policy::allow;                      ///< implicit-copy policy
  profiler* attached_profiler = nullptr;                        ///< not owned
  cuda_device_properties device_props;  ///< recorded once by resources::cuda()
  // Last member: destroyed first, while the memory resources it allocated from are still valid.
  workspace_pool workspaces;  ///< engine scratch shared by every result run through the handle
};

workspace_pool& resources_access::workspaces(const resources& res) noexcept {
  return res.state_->workspaces;
}

const cuda_device_properties& resources_access::device_properties(const resources& res) {
  if (res.state_->kind != backend::cuda) {
    DYNG_FAIL("device_properties() of a ", to_string(res.state_->kind), " handle");
  }
  return res.state_->device_props;
}

void resources_access::force_cooperative_launch(const resources& res, bool available) {
  if (res.state_->kind != backend::cuda) {
    DYNG_FAIL("force_cooperative_launch() on a ", to_string(res.state_->kind), " handle");
  }
  res.state_->device_props.cooperative_launch = available;
}

memory_resource_ref resources_access::staging_memory(const resources& res) noexcept {
  return res.state_->staging;
}

int resources_access::host_threads(const resources& res) noexcept {
  switch (res.state_->kind) {
    case backend::openmp:
      return res.state_->num_threads > 0 ? res.state_->num_threads : 1;
    case backend::cuda:
#if DYNG_HAS_OPENMP
      return omp_get_max_threads() > 0 ? omp_get_max_threads() : 1;
#else
      return 1;
#endif
    case backend::sequential:
      break;
  }
  return 1;
}

}  // namespace detail

namespace {

resources make_default() {
  switch (default_backend()) {
    case backend::cuda:
      return resources::cuda();
    case backend::openmp:
      return resources::openmp();
    case backend::sequential:
      break;
  }
  return resources::sequential();
}

}  // namespace

resources::resources() : resources(make_default()) {}

resources::resources(std::shared_ptr<detail::resources_state> state) noexcept
    : state_(std::move(state)) {}

// A move copies the shared_ptr instead of stealing it: a moved-from handle must stay usable,
// since every accessor is noexcept and has no precondition.
resources::resources(const resources& other) noexcept = default;

resources::resources(resources&& other) noexcept : state_(other.state_) {}

resources& resources::operator=(const resources& other) noexcept = default;

resources& resources::operator=(resources&& other) noexcept {
  state_ = other.state_;
  return *this;
}

resources::~resources() = default;

resources resources::sequential() {
  auto state = std::make_shared<detail::resources_state>();
  state->kind = backend::sequential;
  state->num_threads = 1;
  return resources(std::move(state));
}

resources resources::openmp(int num_threads) {
#if DYNG_HAS_OPENMP
  DYNG_EXPECTS(num_threads >= 0, "resources::openmp(): num_threads must be >= 0, got ",
               num_threads);
  auto state = std::make_shared<detail::resources_state>();
  state->kind = backend::openmp;
  state->num_threads = num_threads > 0 ? num_threads : omp_get_max_threads();
  return resources(std::move(state));
#else
  (void)num_threads;
  throw not_supported_error(
      "dyng: the openmp backend is not built (configure with DYNG_ENABLE_OPENMP=ON); "
      "available: sequential");
#endif
}

resources resources::cuda(int device, stream_ref stream) {
#if DYNG_HAS_CUDA
  auto state = std::make_shared<detail::resources_state>();
  state->kind = backend::cuda;
  state->device = device;
  state->stream = stream;
  state->num_threads = 1;
  // Validates the device (invalid_argument_error; not_supported_error without a visible device)
  // and records the capabilities engine::automatic reads (cooperative launch, SM count).
  state->device_props = detail::query_cuda_device(device);
  state->memory = default_device_memory_resource(device);
  state->staging = default_pinned_host_memory_resource();
  return resources(std::move(state));
#else
  (void)device;
  (void)stream;
  throw not_supported_error(
      std::string("dyng: the cuda backend is not built (configure with DYNG_ENABLE_CUDA=ON); "
                  "available: sequential") +
      (DYNG_HAS_OPENMP ? ", openmp" : ""));
#endif
}

backend resources::get_backend() const noexcept {
  return state_->kind;
}

int resources::device() const noexcept {
  return state_->device;
}

stream_ref resources::stream() const noexcept {
  return state_->stream;
}

int resources::num_threads() const noexcept {
  return state_->num_threads;
}

memory_space resources::default_space() const noexcept {
  return state_->kind == backend::cuda ? memory_space::device : memory_space::host;
}

memory_resource_ref resources::memory() const noexcept {
  return state_->memory;
}

void resources::set_memory_resource(memory_resource_ref mr) {
  if (state_->kind != backend::cuda) {
    DYNG_EXPECTS(is_host_accessible(mr.space()),
                 "a host backend needs a host-accessible memory resource");
  } else {
    DYNG_EXPECTS(mr.space() == memory_space::device || mr.space() == memory_space::managed,
                 "the cuda backend needs a device or managed memory resource");
  }
  state_->workspaces.release_idle();
  state_->memory = mr;
}

copy_policy resources::get_copy_policy() const noexcept {
  return state_->copies;
}

void resources::set_copy_policy(copy_policy policy) noexcept {
  state_->copies = policy;
}

void resources::attach_profiler(profiler* p) noexcept {
  state_->attached_profiler = p;
}

profiler* resources::get_profiler() const noexcept {
  return state_->attached_profiler;
}

void resources::release_workspaces() const noexcept {
  state_->workspaces.release_idle();
}

std::size_t resources::workspace_bytes() const {
  return state_->workspaces.statistics().idle_bytes;
}

void resources::warm_up() const {
  if (state_->kind == backend::cuda) {
    (void)detail::cuda_warm_up(state_->device, state_->stream, state_->memory);
  }
}

void resources::synchronize() const {
  if (state_->kind == backend::cuda) {
    detail::cuda_synchronize(state_->device, state_->stream);
  }
}

}  // namespace dyng
