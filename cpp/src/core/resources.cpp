// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file resources.cpp
 * @brief The resources handle.
 */
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
  memory_resource_ref memory{default_host_memory_resource()};  ///< allocations
  copy_policy copies = copy_policy::allow;                     ///< implicit-copy policy
  profiler* attached_profiler = nullptr;                       ///< not owned
};

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
  (void)device;
  (void)stream;
  throw not_supported_error(
      std::string("dyng: the cuda backend is not built; available: sequential") +
      (DYNG_HAS_OPENMP ? ", openmp" : ""));
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
  }
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

void resources::warm_up() const {}

void resources::synchronize() const {
  if (state_->kind == backend::cuda) {
    state_->stream.synchronize();
  }
}

}  // namespace dyng
