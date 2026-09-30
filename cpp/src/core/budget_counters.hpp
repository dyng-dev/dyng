// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file budget_counters.hpp
 * @brief The allocation and host-synchronization counters behind invariant I9 (PLAN Section
 *        4.5.5): counted only in builds with DYNG_DEBUG_BUDGETS=ON.
 *
 * The library's own memory resources (host, pinned host, CUDA stream-ordered pool) call
 * note_allocation() for every allocation, and the library's stream synchronization
 * (detail::cuda_synchronize(), which resources::synchronize() uses) calls note_host_sync(). A
 * reader takes a snapshot before and after a phase and subtracts (framework/budgets.hpp does that
 * for the algorithm work of an update). The functions are always called; they count only in a
 * build with DYNG_DEBUG_BUDGETS=ON (budget_counters.cpp is then compiled with the define) and are
 * empty calls otherwise.
 *
 * Attribution. The counters are kept per calling thread: a count made on a thread goes to that
 * thread's counters, and a snapshot reads the calling thread's counters, so an update measured on
 * one thread is not charged with the allocations or synchronizations of library calls on other
 * user threads (PLAN 4.7.4 allows concurrent calls on distinct containers). The exception are the
 * worker threads of an OpenMP parallel region (a thread inside a parallel region that is not the
 * region's primary thread): the thread that opened the region cannot be named from inside it, so
 * their counts go to one shared worker tally that every snapshot includes. Counts of OpenMP
 * workers are therefore exact when one thread at a time runs OpenMP regions of the library, and
 * may be charged to every thread that measures while another thread's OpenMP region counts
 * (library code makes no allocation and no synchronization in a parallel region; the workers note
 * only the growth of their per-thread lists and, in the conformance executables, host heap
 * allocations).
 *
 * Reservations: code that grows a reusable array on purpose (a workspace created or enlarged, a
 * scratch buffer grown, a result's arrays grown for new vertices) calls note_reservation(). An
 * algorithm phase that reserved is a *reserving* run: its allocations are reported but not held
 * against the steady-state budget ("once reserved", PLAN 4.5.5); a run that reserved nothing must
 * stay within it. framework/scratch_buffer.hpp and the workspace pool note their growth
 * themselves.
 *
 * Container work: the graph's own materializations inside an update (its device copy uploaded on
 * first use after a host commit, its device in-edges, its host copy downloaded) run inside a
 * container_scope. Their allocations and synchronizations are still counted, and also recorded as
 * container work, which a budget does not hold against the algorithm (container growth is
 * reported, never failed: PLAN 4.5.5, I9).
 *
 * Instrumentation: the synchronizations the profiler makes for profiler_options::sync_stages (and
 * anything else measurement code does) run inside an instrumentation_scope. They are counted and
 * recorded as instrumentation, which a budget does not hold against the algorithm either.
 *
 * What is not counted: allocations of a memory resource the user installed
 * (resources::set_memory_resource), std::vector growth of the host engines, and synchronizations
 * an engine makes with the CUDA runtime directly. A test binary can count host allocations too by
 * calling note_allocation() from a replacement of the global operator new (the conformance kit's
 * check C8 does that); engines that synchronize directly call note_host_sync() next to the call.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace dyng::detail {

/**
 * @brief A snapshot of the process-wide counters (differences of two snapshots are the counts of
 *        a phase).
 */
struct budget_counters {
  std::int64_t allocations = 0;      ///< allocations through the library's memory resources
  std::int64_t allocated_bytes = 0;  ///< bytes of those allocations
  std::int64_t host_syncs = 0;       ///< host synchronizations with a stream
  std::int64_t reservations = 0;     ///< deliberate growths of reusable arrays (note_reservation)
  std::int64_t container_allocations = 0;  ///< the part of `allocations` that was container work
  std::int64_t container_host_syncs = 0;   ///< the part of `host_syncs` that was container work
  /// The part of `allocations` that was instrumentation (instrumentation_scope).
  std::int64_t instrumentation_allocations = 0;
  /// The part of `host_syncs` that was instrumentation (profiler_options::sync_stages).
  std::int64_t instrumentation_host_syncs = 0;

  /**
   * @brief The allocations that were neither container work nor instrumentation.
   * @return allocations - container_allocations - instrumentation_allocations.
   */
  [[nodiscard]] std::int64_t own_allocations() const noexcept {
    return allocations - container_allocations - instrumentation_allocations;
  }

  /**
   * @brief The host synchronizations that were neither container work nor instrumentation.
   * @return host_syncs - container_host_syncs - instrumentation_host_syncs.
   */
  [[nodiscard]] std::int64_t own_host_syncs() const noexcept {
    return host_syncs - container_host_syncs - instrumentation_host_syncs;
  }

  /**
   * @brief The sum of two stretches of counts (e.g. the two halves of an update).
   * @param[in] other The other counts.
   * @return This plus `other`, field by field.
   */
  [[nodiscard]] budget_counters plus(const budget_counters& other) const noexcept {
    budget_counters d;
    d.allocations = allocations + other.allocations;
    d.allocated_bytes = allocated_bytes + other.allocated_bytes;
    d.host_syncs = host_syncs + other.host_syncs;
    d.reservations = reservations + other.reservations;
    d.container_allocations = container_allocations + other.container_allocations;
    d.container_host_syncs = container_host_syncs + other.container_host_syncs;
    d.instrumentation_allocations = instrumentation_allocations + other.instrumentation_allocations;
    d.instrumentation_host_syncs = instrumentation_host_syncs + other.instrumentation_host_syncs;
    return d;
  }

  /**
   * @brief The counts between an earlier snapshot and this one.
   * @param[in] earlier The snapshot taken first.
   * @return This minus `earlier`, field by field.
   */
  [[nodiscard]] budget_counters since(const budget_counters& earlier) const noexcept {
    budget_counters d;
    d.allocations = allocations - earlier.allocations;
    d.allocated_bytes = allocated_bytes - earlier.allocated_bytes;
    d.host_syncs = host_syncs - earlier.host_syncs;
    d.reservations = reservations - earlier.reservations;
    d.container_allocations = container_allocations - earlier.container_allocations;
    d.container_host_syncs = container_host_syncs - earlier.container_host_syncs;
    d.instrumentation_allocations =
        instrumentation_allocations - earlier.instrumentation_allocations;
    d.instrumentation_host_syncs = instrumentation_host_syncs - earlier.instrumentation_host_syncs;
    return d;
  }
};

/**
 * @brief Whether libdyng was built with DYNG_DEBUG_BUDGETS=ON (the counters count).
 * @return true in a budgets build.
 */
[[nodiscard]] bool budgets_enabled() noexcept;

/**
 * @brief The current values of the calling thread's counters, plus the shared tally of OpenMP
 *        workers (see the file comment).
 * @return A snapshot (all zero when budgets_enabled() is false).
 */
[[nodiscard]] budget_counters budget_snapshot() noexcept;

/**
 * @brief Count one allocation (a no-op unless budgets_enabled()).
 * @param[in] bytes Its size.
 */
void note_allocation(std::size_t bytes) noexcept;

/**
 * @brief Count one host synchronization with a stream (a no-op unless budgets_enabled()).
 */
void note_host_sync() noexcept;

/**
 * @brief Count one deliberate growth of a reusable array (a workspace, a scratch buffer, a
 *        result's arrays); a no-op unless budgets_enabled(). See the file comment.
 */
void note_reservation() noexcept;

/**
 * @brief Record the counts of a stretch of container work (container_scope does it).
 * @param[in] allocations Allocations of the work that no inner scope recorded.
 * @param[in] host_syncs  Host synchronizations of the work that no inner scope recorded.
 */
void note_container_work(std::int64_t allocations, std::int64_t host_syncs) noexcept;

/**
 * @brief Record the counts of a stretch of instrumentation (instrumentation_scope does it).
 * @param[in] allocations Allocations of the stretch that no inner scope recorded.
 * @param[in] host_syncs  Host synchronizations of the stretch that no inner scope recorded.
 */
void note_instrumentation_work(std::int64_t allocations, std::int64_t host_syncs) noexcept;

/**
 * @brief Marks the counts between its construction and destruction as container work (see the
 *        file comment). Nested scopes count once.
 */
class container_scope {
 public:
  /// @brief Start the scope.
  container_scope() noexcept : start_(budget_snapshot()) {}
  container_scope(const container_scope&) = delete;             ///< not copyable
  container_scope& operator=(const container_scope&) = delete;  ///< not copyable
  container_scope(container_scope&&) = delete;                  ///< not movable
  container_scope& operator=(container_scope&&) = delete;       ///< not movable

  /// @brief Record the scope's counts as container work.
  ~container_scope() {
    const budget_counters d = budget_snapshot().since(start_);
    note_container_work(d.own_allocations(), d.own_host_syncs());
  }

 private:
  budget_counters start_;
};

/**
 * @brief Marks the counts between its construction and destruction as instrumentation (the
 *        profiler's synchronizations of profiler_options::sync_stages; see the file comment).
 *        Nested scopes (of either kind) count once.
 */
class instrumentation_scope {
 public:
  /// @brief Start the scope.
  instrumentation_scope() noexcept : start_(budget_snapshot()) {}
  instrumentation_scope(const instrumentation_scope&) = delete;             ///< not copyable
  instrumentation_scope& operator=(const instrumentation_scope&) = delete;  ///< not copyable
  instrumentation_scope(instrumentation_scope&&) = delete;                  ///< not movable
  instrumentation_scope& operator=(instrumentation_scope&&) = delete;       ///< not movable

  /// @brief Record the scope's counts as instrumentation.
  ~instrumentation_scope() {
    const budget_counters d = budget_snapshot().since(start_);
    note_instrumentation_work(d.own_allocations(), d.own_host_syncs());
  }

 private:
  budget_counters start_;
};

}  // namespace dyng::detail
