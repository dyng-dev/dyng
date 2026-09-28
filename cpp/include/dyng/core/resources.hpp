// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file resources.hpp
 * @brief resources: the execution handle passed to every compute(), update() and apply().
 * @ingroup core
 */
#pragma once

#include <dyng/core/backend.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/stream.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace dyng {

class profiler;

namespace detail {
struct resources_state;
struct resources_access;
}  // namespace detail

/**
 * @brief What to do when an input array lives in a memory space that does not match the backend.
 *
 * The library then copies the array once. The default `allow` logs the copy at debug level.
 * @ingroup core
 */
enum class copy_policy : std::uint8_t {
  allow,  ///< copy; log at debug level
  warn,   ///< copy; log at warn level (the default inside timed benchmark runs)
  error,  ///< throw invalid_argument_error instead of copying
};

/**
 * @brief Execution resources: backend, device, stream, memory resource, thread count, profiler.
 *
 * Cheap to copy: copies share one handle, so set_memory_resource(), set_copy_policy() and
 * attach_profiler() affect every copy. Call them during setup, not while another thread uses the
 * handle; for independent settings create a new resources object (PLAN Section 4.7.4).
 * The library never creates streams, sets environment variables, or changes the global OpenMP
 * thread count on its own.
 * A moved-from handle stays valid: moving shares the handle like a copy (the source keeps it), so
 * no accessor ever meets an empty handle.
 *
 * **Scratch memory.** The handle also caches the engines' workspaces (scratch memory such as the
 * frontier lists of sssp): a compute() or update() leases one from the handle, sizes it once and
 * returns it, so results that run one after another through the same handle (for example the K
 * objectives updated by dyng::update_each()) share one workspace, and a steady-state update
 * allocates no scratch memory (ADR 0015). Calls that run concurrently on copies of one handle get
 * distinct workspaces. The cache lives as long as the last copy of the handle, or until
 * release_workspaces().
 * @ingroup core
 */
class resources {
 public:
  /**
   * @brief Resources for default_backend() with its default settings.
   * @throws not_supported_error if the default backend cannot be initialised.
   */
  resources();

  /**
   * @brief The sequential (reference) backend: one host thread, host memory.
   * @return New resources for backend::sequential.
   */
  [[nodiscard]] static resources sequential();

  /**
   * @brief The OpenMP backend.
   * @param[in] num_threads Threads used by OpenMP parallel regions of the library; 0 = the
   *                        OpenMP default at the time of the call (omp_get_max_threads(), which
   *                        honours OMP_NUM_THREADS).
   * @return New resources for backend::openmp.
   * @throws not_supported_error    if the library was built without OpenMP.
   * @throws invalid_argument_error if `num_threads` is negative.
   */
  [[nodiscard]] static resources openmp(int num_threads = 0);

  /**
   * @brief The CUDA backend on one device and stream.
   *
   * Records the device's capabilities once (cooperative launch, multiprocessor count), which the
   * engines read instead of querying the device on every call. Memory comes from the device's
   * default resource (default_device_memory_resource(): a stream-ordered cudaMallocAsync pool);
   * host staging buffers from default_pinned_host_memory_resource(). The library never creates
   * streams: all work is ordered on `stream`, which must outlive every copy of the handle and
   * every buffer allocated through it. Every call that enqueues work makes `device` current for
   * its duration and restores the caller's current device.
   * @param[in] device CUDA device ordinal.
   * @param[in] stream Stream for all work, created on `device`; the default is the per-thread
   *                   default stream (`cudaStreamPerThread`), never the legacy stream.
   * @return New resources for backend::cuda.
   * @throws not_supported_error    if the library was built without CUDA, or no device is visible.
   * @throws invalid_argument_error if `device` is not a visible device.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync Queries the device once; enqueues no work on `stream`.
   */
  [[nodiscard]] static resources cuda(int device = 0, stream_ref stream = {});

  /**
   * @brief Share the handle of `other` (a cheap copy; settings stay shared).
   * @param[in] other The handle to share.
   */
  resources(const resources& other) noexcept;

  /**
   * @brief Share the handle of `other`; `other` keeps it and stays valid (a move is a copy).
   * @param[in] other The handle to share.
   */
  resources(resources&& other) noexcept;

  /**
   * @brief Share the handle of `other` (a cheap copy; settings stay shared).
   * @param[in] other The handle to share.
   * @return `*this`.
   */
  resources& operator=(const resources& other) noexcept;

  /**
   * @brief Share the handle of `other`; `other` keeps it and stays valid (a move is a copy).
   * @param[in] other The handle to share.
   * @return `*this`.
   */
  resources& operator=(resources&& other) noexcept;

  /// @brief Release this reference to the shared handle.
  ~resources();

  /**
   * @brief The backend.
   * @return The backend chosen at construction.
   */
  [[nodiscard]] backend get_backend() const noexcept;

  /**
   * @brief The CUDA device.
   * @return The device ordinal for the CUDA backend, -1 for host backends.
   */
  [[nodiscard]] int device() const noexcept;

  /**
   * @brief The stream all work is ordered on.
   * @return The stream (the per-thread default stream unless one was given).
   */
  [[nodiscard]] stream_ref stream() const noexcept;

  /**
   * @brief The number of host threads the backend uses.
   * @return 1 for sequential and cuda, the OpenMP team size for openmp.
   */
  [[nodiscard]] int num_threads() const noexcept;

  /**
   * @brief The memory space results and containers are placed in by default.
   * @return memory_space::host for host backends, memory_space::device for cuda.
   */
  [[nodiscard]] memory_space default_space() const noexcept;

  /**
   * @brief The memory resource used for library allocations.
   * @return A reference to the current resource (the default host resource for host backends,
   *         default_device_memory_resource(device()) for cuda).
   */
  [[nodiscard]] memory_resource_ref memory() const noexcept;

  /**
   * @brief Replace the memory resource (affects every copy of this handle).
   *
   * The cached workspaces are released first (they were allocated from the previous resource).
   * @param[in] mr The new resource; must outlive every allocation made through it.
   * @throws invalid_argument_error if the resource's space does not suit the backend (host
   *                                backends need a host-accessible space, cuda a device or
   *                                managed space).
   * @async The idle workspaces are returned to the previous resource in stream order (after the
   *        work already enqueued on the handle's stream); calls made after this one returns use
   *        the new resource.
   */
  void set_memory_resource(memory_resource_ref mr);

  /**
   * @brief The policy for implicit host/device copies of inputs.
   * @return The current policy (default copy_policy::allow).
   */
  [[nodiscard]] copy_policy get_copy_policy() const noexcept;

  /**
   * @brief Set the policy for implicit host/device copies of inputs (affects every copy).
   * @param[in] policy The new policy.
   */
  void set_copy_policy(copy_policy policy) noexcept;

  /**
   * @brief Attach a profiler that records the library's stages and counters.
   * @param[in] p The profiler (not owned; must outlive its attachment); nullptr detaches.
   */
  void attach_profiler(profiler* p) noexcept;

  /**
   * @brief The attached profiler.
   * @return The profiler, or nullptr if none is attached.
   */
  [[nodiscard]] profiler* get_profiler() const noexcept;

  /**
   * @brief Free the cached workspaces that are not in use (affects every copy).
   *
   * The next compute() or update() through the handle sizes a new workspace (allocates). Safe to
   * call while another thread uses the handle: a workspace in use returns to the cache afterwards.
   * @async Device workspaces are released in stream order (the memory is reused only after the
   *        work already enqueued on the handle's stream); host workspaces are freed at once.
   */
  void release_workspaces() const noexcept;

  /**
   * @brief The scratch memory the handle caches (shared by every copy).
   * @return Bytes held by the cached workspaces that are not in use right now.
   */
  [[nodiscard]] std::size_t workspace_bytes() const;

  /**
   * @brief Initialise the backend ahead of timed work.
   *
   * CUDA: creates the device's context, loads every kernel of the library (under lazy module
   * loading, the CUDA default, each kernel would otherwise be loaded inside its first call), and
   * primes the stream and the memory resource. This replaces setting CUDA_MODULE_LOADING=EAGER in
   * the environment, which a library must not do. A no-op for host backends.
   * @throws cuda_error if the CUDA runtime reports an error.
   * @sync
   */
  void warm_up() const;

  /**
   * @brief Wait for all work enqueued through these resources.
   *
   * A no-op for host backends, whose calls complete before returning.
   * @throws cuda_error if the CUDA runtime reports an error.
   * @sync
   */
  void synchronize() const;

 private:
  friend struct detail::resources_access;
  explicit resources(std::shared_ptr<detail::resources_state> state) noexcept;
  std::shared_ptr<detail::resources_state> state_;
};

}  // namespace dyng
