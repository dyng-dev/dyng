// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file workspace.hpp
 * @brief The workspace pool: scratch memory of the engines, owned by a resources handle and shared
 *        by every result that is computed or updated through it (ADR 0015).
 *
 * An engine's workspace is scratch: its contents on entry to a run are irrelevant, apart from
 * invariants the workspace type keeps itself (for sssp: the generation stamps, whose counter lives
 * in the same object). Results therefore do not own one. A run leases a workspace of its type
 * from the pool of the resources handle it runs with, sizes it (a no-op once it is large enough),
 * uses it and returns it. Results that run one after another (the K objectives of a multi-weight
 * graph updated by dyng::update_each(), as MOSP's shared SospWorkspace) share one workspace;
 * results that run concurrently on copies of one handle lease distinct workspaces, so the pool
 * holds as many workspaces of a type as were ever in use at the same time.
 *
 * Steady state (invariant I9): once a workspace of the type has been sized for the largest graph
 * of a stable workload, leasing it allocates nothing.
 *
 * A lease that ends while an exception propagates discards its workspace (a failed run may leave
 * flags set that the next run relies on being clear); the next lease creates a fresh one.
 *
 * Device workspaces (CUDA backend; their arrays are detail::scratch_buffer, see
 * framework/scratch_buffer.hpp) follow the same pattern: they allocate from the handle's memory
 * resource on its stream when sized and free there when destroyed, so the pool must be emptied
 * before its memory resource goes away (resources::set_memory_resource() does that; the default
 * resources outlive every handle).
 */
#pragma once

#include "core/resources_access.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

namespace dyng::detail {

/**
 * @brief Base of every pooled workspace type (type erasure for the pool).
 */
class pooled_workspace {
 public:
  pooled_workspace() = default;                                   ///< default
  pooled_workspace(const pooled_workspace&) = delete;             ///< not copyable
  pooled_workspace& operator=(const pooled_workspace&) = delete;  ///< not copyable
  pooled_workspace(pooled_workspace&&) = delete;                  ///< not movable
  pooled_workspace& operator=(pooled_workspace&&) = delete;       ///< not movable
  virtual ~pooled_workspace() = default;                          ///< releases the memory

  /**
   * @brief The memory the workspace holds.
   * @return Bytes of its arrays (capacity, not size).
   */
  [[nodiscard]] virtual std::size_t bytes() const noexcept = 0;
};

/**
 * @brief Counters of a workspace pool (for tests and memory reports).
 */
struct workspace_pool_statistics {
  std::size_t idle = 0;         ///< workspaces in the pool, not leased
  std::size_t leased = 0;       ///< workspaces leased right now
  std::size_t idle_bytes = 0;   ///< bytes held by the idle workspaces
  std::uint64_t created = 0;    ///< workspaces created since the pool was made
  std::uint64_t discarded = 0;  ///< workspaces dropped after a failed run
  std::uint64_t leases = 0;     ///< leases handed out
};

/**
 * @brief A pool of engine workspaces, keyed by workspace type; thread-safe.
 */
class workspace_pool {
 public:
  /**
   * @brief Exclusive use of one workspace; returns it to the pool when it ends.
   * @tparam workspace_t The workspace type (derived from pooled_workspace, default-constructible).
   */
  template <typename workspace_t>
  class lease {
   public:
    lease(const lease&) = delete;             ///< not copyable
    lease& operator=(const lease&) = delete;  ///< not copyable

    /**
     * @brief Take over another lease.
     * @param[in,out] other The lease to take over; it holds nothing afterwards.
     */
    lease(lease&& other) noexcept
        : pool_(std::exchange(other.pool_, nullptr)),
          workspace_(std::move(other.workspace_)),
          exceptions_(other.exceptions_) {}

    lease& operator=(lease&&) = delete;  ///< not assignable

    /// @brief Return the workspace (or discard it if an exception is propagating).
    ~lease() {
      if (pool_ != nullptr && workspace_ != nullptr) {
        pool_->give_back(key(), std::move(workspace_), std::uncaught_exceptions() > exceptions_);
      }
    }

    /**
     * @brief The workspace.
     * @return A reference valid for the lifetime of the lease.
     */
    [[nodiscard]] workspace_t& get() const noexcept {
      return static_cast<workspace_t&>(*workspace_);
    }

    /**
     * @brief The workspace.
     * @return A reference valid for the lifetime of the lease.
     */
    workspace_t& operator*() const noexcept {
      return get();
    }

    /**
     * @brief The workspace.
     * @return A pointer valid for the lifetime of the lease.
     */
    workspace_t* operator->() const noexcept {
      return &get();
    }

   private:
    friend class workspace_pool;
    lease(workspace_pool* pool, std::unique_ptr<pooled_workspace> workspace) noexcept
        : pool_(pool), workspace_(std::move(workspace)), exceptions_(std::uncaught_exceptions()) {}

    static std::type_index key() noexcept {
      return std::type_index(typeid(workspace_t));
    }

    workspace_pool* pool_;
    std::unique_ptr<pooled_workspace> workspace_;
    int exceptions_;
  };

  workspace_pool() = default;                                 ///< an empty pool
  workspace_pool(const workspace_pool&) = delete;             ///< not copyable
  workspace_pool& operator=(const workspace_pool&) = delete;  ///< not copyable
  workspace_pool(workspace_pool&&) = delete;                  ///< not movable
  workspace_pool& operator=(workspace_pool&&) = delete;       ///< not movable
  ~workspace_pool() = default;                                ///< frees every idle workspace

  /**
   * @brief Lease a workspace of type `workspace_t`: an idle one (the most recently returned) or a
   *        new, empty one.
   * @tparam workspace_t The workspace type (derived from pooled_workspace, default-constructible).
   * @return The lease; the caller sizes the workspace.
   * @throws std::bad_alloc if a new workspace cannot be created.
   */
  template <typename workspace_t>
  [[nodiscard]] lease<workspace_t> acquire() {
    static_assert(std::is_base_of_v<pooled_workspace, workspace_t>,
                  "workspace_pool: a workspace type must derive from pooled_workspace");
    std::unique_ptr<pooled_workspace> taken = take(lease<workspace_t>::key());
    if (taken == nullptr) {
      try {
        taken = std::make_unique<workspace_t>();
      } catch (...) {
        give_back(lease<workspace_t>::key(), nullptr, true);  // ends the counted lease
        throw;
      }
      note_created();
    }
    return lease<workspace_t>(this, std::move(taken));
  }

  /**
   * @brief Free every idle workspace (leased ones return to the pool later and stay).
   */
  void release_idle() noexcept;

  /**
   * @brief The pool's counters.
   * @return A snapshot.
   */
  [[nodiscard]] workspace_pool_statistics statistics() const;

 private:
  struct entry {
    std::type_index key;
    std::unique_ptr<pooled_workspace> workspace;
  };

  /// An idle workspace of this type (the most recently returned), or nullptr; counts a lease.
  std::unique_ptr<pooled_workspace> take(std::type_index key);
  /// Return a leased workspace; `discard` drops it instead (nullptr: a lease that never got one).
  void give_back(std::type_index key, std::unique_ptr<pooled_workspace> workspace,
                 bool discard) noexcept;
  void note_created() noexcept;

  mutable std::mutex mutex_;
  std::vector<entry> idle_;
  std::size_t leased_ = 0;
  std::uint64_t created_ = 0;
  std::uint64_t discarded_ = 0;
  std::uint64_t leases_ = 0;
};

}  // namespace dyng::detail
