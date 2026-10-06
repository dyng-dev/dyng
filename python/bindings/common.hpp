// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file common.hpp
 * @brief Shared helpers of the nanobind bindings (dyng._core): holders, locking around native
 *        calls, array conversion, and the bind functions of every file.
 *
 * The extension module is private: the typed layer python/dyng (the .py files) owns the public names, does
 * the dtype dispatch and passes arrays of exactly the element type a function takes (the array
 * arguments are `noconvert`, so nothing is converted, and nothing is narrowed, here).
 *
 * Locking (ADR 0011, item 5). The GIL is released around every native call that runs algorithms or I/O
 * (PLAN Section 5.4 rule 3). Two locks then keep concurrent Python threads memory-safe:
 *   - every container and result holder has a reader/writer lock: calls that only read it share
 *     it, calls that change it (apply, update) take it exclusively;
 *   - every native call holds a process-wide lock (call_gate, writer-preferring) in shared mode;
 *     reading a profiler's records, attaching a profiler and changing a handle's copy policy take
 *     it exclusively (while_idle), so the records are never read while a call may write them
 *     (ADR 0023, note 5) and the shared handle's fields never change under a running call.
 * Locks are always taken after the GIL is released and released before it is taken back, so a
 * thread waiting for a lock never holds the GIL.
 *
 * Result arrays (ADR 0011, item 6). A result's state lives in reference-counted storage
 * (result_holder). Every exported array (dyng.Array, and through it NumPy, DLPack and the array
 * interface) holds a reference to the state it views, and an update copies the state first
 * (copy-on-write) while such a reference exists, so an exported array never sees its memory
 * change or go away: it keeps showing the state it was read from.
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dyng::python {

namespace nb = nanobind;

// ---------------------------------------------------------------------------------------------
// Type codes of the instantiations (the suffixes of the native names: GraphI32I64U, ..._i32_u)
// ---------------------------------------------------------------------------------------------

/**
 * @brief The code of an id or weight type in native names ("I32", "I64", "U").
 * @tparam value_t std::int32_t, std::int64_t or dyng::unweighted.
 */
template <typename value_t>
struct type_code;

/// @brief int32: "I32".
template <>
struct type_code<std::int32_t> {
  static constexpr const char* upper = "I32";  ///< in class names
  static constexpr const char* lower = "i32";  ///< in function names
};

/// @brief int64: "I64".
template <>
struct type_code<std::int64_t> {
  static constexpr const char* upper = "I64";  ///< in class names
  static constexpr const char* lower = "i64";  ///< in function names
};

/// @brief unweighted: "U".
template <>
struct type_code<unweighted> {
  static constexpr const char* upper = "U";  ///< in class names
  static constexpr const char* lower = "u";  ///< in function names
};

/**
 * @brief A native class or function name with type suffixes.
 * @param[in] base  The base name.
 * @param[in] codes The type codes, appended in order.
 * @param[in] sep   The separator before each code ("" for classes, "_" for functions).
 * @return The name.
 */
std::string suffixed(const char* base, std::initializer_list<const char*> codes, const char* sep);

// ---------------------------------------------------------------------------------------------
// Holders and locking
// ---------------------------------------------------------------------------------------------

/**
 * @brief A container or result owned by a Python object, with its reader/writer lock.
 * @tparam value_t The C++ object (dyng::graph<V,E,W>, sssp::result<V>, cycle_count::result).
 */
template <typename value_t>
struct holder {
  /**
   * @brief Take ownership of `v`.
   * @param[in] v The object.
   */
  explicit holder(value_t&& v) : value(std::move(v)) {}
  holder(const holder&) = delete;             ///< not copyable
  holder& operator=(const holder&) = delete;  ///< not copyable
  holder(holder&&) = delete;                  ///< not movable (the lock)
  holder& operator=(holder&&) = delete;       ///< not movable (the lock)
  ~holder() = default;                        ///< destroys the object

  value_t value;                    ///< the object
  mutable std::shared_mutex mutex;  ///< shared: read-only calls; exclusive: apply / update
};

/**
 * @brief A result owned by a Python object: its state in reference-counted storage, a generation
 *        counter and a reader/writer lock.
 *
 * Arrays exported from the state (owned_view()) share its ownership, so the memory they view
 * lives as long as they do. for_update() gives an update a state no export references: the
 * current one when nothing else holds it, otherwise a clone (copy-on-write), so an export keeps
 * showing the state it was taken from, unchanged. The generation counts the updates; dyng.Array
 * compares it with the generation it was read at to raise StaleResultError. The holder also
 * remembers the resources of the call that last wrote the state (writer()): the stream on which
 * a reader of device memory must order itself (dyng.Array's host copies, `__cuda_array_interface__`
 * and DLPack stream ordering, PLAN Section 5.4 rule 4).
 * @tparam value_t sssp::result<V> or cycle_count::result.
 */
template <typename value_t>
class result_holder {
 public:
  /**
   * @brief Take ownership of `v`.
   * @param[in] v The result.
   */
  explicit result_holder(value_t&& v) : state_(std::make_shared<value_t>(std::move(v))) {}

  /**
   * @brief Take ownership of `v`, written by a call with the resources `writer`.
   * @param[in] v      The result.
   * @param[in] writer The resources of the call that produced it.
   */
  result_holder(value_t&& v, const resources& writer)
      : state_(std::make_shared<value_t>(std::move(v))), writer_(writer) {}
  result_holder(const result_holder&) = delete;             ///< not copyable
  result_holder& operator=(const result_holder&) = delete;  ///< not copyable
  result_holder(result_holder&&) = delete;                  ///< not movable (the lock)
  result_holder& operator=(result_holder&&) = delete;       ///< not movable (the lock)
  ~result_holder() = default;                               ///< drops this reference to the state

  /**
   * @brief The current state (call with `mutex` held, shared or exclusive).
   * @return The state.
   */
  [[nodiscard]] const value_t& get() const noexcept {
    return *state_;
  }

  /**
   * @brief The current state, shared with the caller (call with `mutex` held; for exports).
   * @return A reference that keeps the state alive.
   */
  [[nodiscard]] std::shared_ptr<const value_t> share() const {
    return state_;
  }

  /**
   * @brief The state to change in place without affecting its arrays (set_options(); call with
   *        `mutex` held exclusively).
   * @return The state.
   */
  [[nodiscard]] value_t& for_options() noexcept {
    return *state_;
  }

  /**
   * @brief The state an update may change (call with `mutex` held exclusively): a clone for
   *        `res` when an export still references the current state; advances the generation.
   * @param[in] res The resources of the update (those of the clone).
   * @return The state to update.
   */
  [[nodiscard]] value_t& for_update(const resources& res) {
    // Exports are only created under the shared lock, so while the exclusive lock is held the
    // count can only fall. The acquire fence orders this thread's writes after the reads of an
    // export released on another thread.
    if (state_.use_count() > 1) {
      state_ = std::make_shared<value_t>(state_->clone(res));
    } else {
      std::atomic_thread_fence(std::memory_order_acquire);
    }
    generation_.fetch_add(1, std::memory_order_acq_rel);
    writer_ = res;
    return *state_;
  }

  /**
   * @brief The resources of the call that last wrote the state (call with `mutex` held, shared
   *        or exclusive): their stream orders the state's device memory.
   * @return The resources, or none for a holder built without them.
   */
  [[nodiscard]] const std::optional<resources>& writer() const noexcept {
    return writer_;
  }

  /**
   * @brief The number of updates so far (starts at 0).
   * @return The generation.
   */
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_.load(std::memory_order_acquire);
  }

  mutable std::shared_mutex mutex;  ///< shared: read-only calls; exclusive: update, set_options

 private:
  std::shared_ptr<value_t> state_;
  std::atomic<std::uint64_t> generation_{0};
  std::optional<resources> writer_;
};

/**
 * @brief A reader/writer lock that prefers writers: once a thread waits for exclusive ownership,
 *        new shared owners wait behind it.
 *
 * The native-call lock needs this: every native call holds it shared, and with a
 * reader-preferring lock (glibc's std::shared_mutex) a thread waiting to read a profiler's
 * records or to attach one would wait forever while other threads keep calling. Shared owners
 * never take it again while they hold it (native calls do not nest), so the preference cannot
 * deadlock. Meets the SharedMutex requirements used by std::shared_lock and std::unique_lock.
 */
class call_gate {
 public:
  call_gate() = default;                            ///< unlocked
  call_gate(const call_gate&) = delete;             ///< not copyable
  call_gate& operator=(const call_gate&) = delete;  ///< not copyable
  call_gate(call_gate&&) = delete;                  ///< not movable
  call_gate& operator=(call_gate&&) = delete;       ///< not movable
  ~call_gate() = default;                           ///< must be unlocked

  /// @brief Take shared ownership (waits while a writer holds or waits for the lock).
  void lock_shared() {
    std::unique_lock<std::mutex> l(mutex_);
    changed_.wait(l, [this] { return !writer_ && waiting_writers_ == 0; });
    ++readers_;
  }

  /// @brief Release shared ownership.
  void unlock_shared() {
    std::lock_guard<std::mutex> l(mutex_);
    if (--readers_ == 0) {
      changed_.notify_all();
    }
  }

  /// @brief Take exclusive ownership (waits for the shared owners to finish).
  void lock() {
    std::unique_lock<std::mutex> l(mutex_);
    ++waiting_writers_;
    changed_.wait(l, [this] { return !writer_ && readers_ == 0; });
    --waiting_writers_;
    writer_ = true;
  }

  /// @brief Release exclusive ownership.
  void unlock() {
    std::lock_guard<std::mutex> l(mutex_);
    writer_ = false;
    changed_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  std::size_t readers_ = 0;
  std::size_t waiting_writers_ = 0;
  bool writer_ = false;
};

/**
 * @brief The process-wide lock of native calls (shared by every call; exclusive for reading a
 *        profiler's records, attaching a profiler and changing a handle's copy policy).
 * @return The lock.
 */
call_gate& native_call_mutex();

/**
 * @brief Run `f` without the GIL, holding the native-call lock in shared mode.
 *
 * Every lock `f` takes must be taken inside `f` (after the GIL is released).
 * @tparam function_t A callable without arguments.
 * @param[in] f The native work.
 * @return What `f` returns.
 */
template <typename function_t>
decltype(auto) without_gil(function_t&& f) {
  nb::gil_scoped_release release;
  std::shared_lock<call_gate> call_lock(native_call_mutex());
  return std::forward<function_t>(f)();
}

/**
 * @brief Run `f` while no native call runs: without the GIL, holding the native-call lock
 *        exclusively (reading a profiler's records, attaching a profiler, changing the copy
 *        policy of a shared resources handle).
 * @tparam function_t A callable without arguments.
 * @param[in] f The work.
 * @return What `f` returns.
 */
template <typename function_t>
decltype(auto) while_idle(function_t&& f) {
  nb::gil_scoped_release release;
  std::unique_lock<call_gate> lock(native_call_mutex());
  return std::forward<function_t>(f)();
}

/**
 * @brief Take a list of reader/writer locks in address order (a fixed order, so two calls that
 *        lock the same objects can never deadlock); an object listed more than once is locked
 *        once, in the strongest of its modes.
 */
class lock_set {
 public:
  /**
   * @brief Add a lock.
   * @param[in] m         The lock.
   * @param[in] exclusive Exclusive (a write) or shared (a read).
   */
  void add(std::shared_mutex& m, bool exclusive) {
    entries_.push_back({&m, exclusive});
  }

  /**
   * @brief Take every lock (call without the GIL).
   */
  void lock();

  lock_set() = default;                           ///< an empty set
  lock_set(const lock_set&) = delete;             ///< not copyable
  lock_set& operator=(const lock_set&) = delete;  ///< not copyable
  lock_set(lock_set&&) = delete;                  ///< not movable
  lock_set& operator=(lock_set&&) = delete;       ///< not movable
  ~lock_set();                                    ///< releases the locks taken

 private:
  struct entry {
    std::shared_mutex* mutex;
    bool exclusive;
  };
  std::vector<entry> entries_;
  std::vector<entry> taken_;
};

// ---------------------------------------------------------------------------------------------
// Arrays
// ---------------------------------------------------------------------------------------------

/**
 * @brief A read-only 1-D C-contiguous host array argument of element type `value_t`.
 * @tparam value_t The element type.
 */
template <typename value_t>
using in_array = nb::ndarray<const value_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/**
 * @brief The element type of a weight array of graphs with weight type `weight_t` (int32 for
 *        unweighted graphs, whose weight arrays are always empty or absent).
 * @tparam weight_t The graph's weight type.
 */
template <typename weight_t>
using weight_scalar_t = std::conditional_t<is_unweighted_v<weight_t>, std::int32_t, weight_t>;

/**
 * @brief The array_view of an array argument (host memory; no copy).
 * @tparam value_t The element type.
 * @param[in] a The array.
 * @return A view of its elements.
 */
template <typename value_t>
array_view<const value_t> view_of(const in_array<value_t>& a) {
  return host_view(a.data(), a.shape(0));
}

/**
 * @brief Move a vector into a NumPy array that owns it (no copy of the elements).
 * @tparam value_t The element type.
 * @param[in] v The vector.
 * @return A 1-D NumPy array of v.size() elements.
 */
template <typename value_t>
nb::ndarray<nb::numpy, value_t, nb::ndim<1>> to_numpy(std::vector<value_t>&& v) {
  auto* heap = new std::vector<value_t>(std::move(v));
  nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<value_t>*>(p); });
  const std::size_t shape[1] = {heap->size()};
  return nb::ndarray<nb::numpy, value_t, nb::ndim<1>>(heap->data(), 1, shape, owner);
}

/**
 * @brief Move a vector into a 2-D C-order NumPy array of shape (rows, cols) that owns it.
 * @tparam value_t The element type.
 * @param[in] v    The vector (rows * cols elements).
 * @param[in] rows The number of rows.
 * @param[in] cols The number of columns.
 * @return The array.
 */
template <typename value_t>
nb::ndarray<nb::numpy, value_t, nb::ndim<2>> to_numpy_2d(std::vector<value_t>&& v, std::size_t rows,
                                                         std::size_t cols) {
  auto* heap = new std::vector<value_t>(std::move(v));
  nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<value_t>*>(p); });
  const std::size_t shape[2] = {rows, cols};
  return nb::ndarray<nb::numpy, value_t, nb::ndim<2>>(heap->data(), 2, shape, owner);
}

/**
 * @brief A read-only, framework-neutral array (nanobind's array-API object: the buffer protocol,
 *        __dlpack__ and __dlpack_device__) viewing memory that `keep` keeps alive (no copy).
 * @tparam value_t The element type.
 * @param[in] v    The elements.
 * @param[in] keep The owner of the memory (a result state, shared with its holder).
 * @return The array; it holds a reference to `keep` until the last view of it is gone.
 */
template <typename value_t>
nb::ndarray<nb::array_api, const value_t, nb::ndim<1>> owned_view(
    array_view<const value_t> v, std::shared_ptr<const void> keep) {
  auto* heap = new std::shared_ptr<const void>(std::move(keep));
  nb::capsule owner(heap,
                    [](void* p) noexcept { delete static_cast<std::shared_ptr<const void>*>(p); });
  const std::size_t shape[1] = {v.size()};
  const bool device = !is_host_accessible(v.space()) || v.space() == memory_space::managed;
  return nb::ndarray<nb::array_api, const value_t, nb::ndim<1>>(
      v.data(), 1, shape, owner, nullptr, nb::dtype<value_t>(),
      device ? nb::device::cuda::value : nb::device::cpu::value, device ? v.device() : 0);
}

/**
 * @brief A read-only 2-D C-order array (shape (rows, cols)) viewing memory that `keep` keeps
 *        alive (no copy); as owned_view().
 * @tparam value_t The element type.
 * @param[in] v    The elements (rows * cols, row-major).
 * @param[in] cols The number of columns (> 0).
 * @param[in] keep The owner of the memory.
 * @return The array.
 */
template <typename value_t>
nb::ndarray<nb::array_api, const value_t, nb::ndim<2>> owned_view_2d(
    array_view<const value_t> v, std::size_t cols, std::shared_ptr<const void> keep) {
  auto* heap = new std::shared_ptr<const void>(std::move(keep));
  nb::capsule owner(heap,
                    [](void* p) noexcept { delete static_cast<std::shared_ptr<const void>*>(p); });
  const std::size_t shape[2] = {cols == 0 ? 0 : v.size() / cols, cols};
  const bool device = !is_host_accessible(v.space()) || v.space() == memory_space::managed;
  return nb::ndarray<nb::array_api, const value_t, nb::ndim<2>>(
      v.data(), 2, shape, owner, nullptr, nb::dtype<value_t>(),
      device ? nb::device::cuda::value : nb::device::cpu::value, device ? v.device() : 0);
}

/**
 * @brief Export one array of a result's current state (the shared lock is taken without the GIL).
 * @tparam value_t    The result type.
 * @tparam function_t `array_view<const T>(const value_t&)`: which array.
 * @param[in] h     The holder.
 * @param[in] which Picks the array.
 * @return The array (owned_view()).
 */
template <typename value_t, typename function_t>
auto export_array(const result_holder<value_t>& h, function_t&& which) {
  std::shared_ptr<const value_t> state;
  const auto v = without_gil([&] {
    std::shared_lock<std::shared_mutex> lock(h.mutex);
    state = h.share();
    return which(*state);
  });
  return owned_view(v, std::shared_ptr<const void>(std::move(state)));
}

/**
 * @brief The resources of the call that last wrote a result's state (result_holder::writer()),
 *        read under its shared lock (without the GIL).
 * @tparam value_t The result type.
 * @param[in] h The holder.
 * @return The resources, or none.
 */
template <typename value_t>
std::optional<resources> writer_of(const result_holder<value_t>& h) {
  return without_gil([&] {
    std::shared_lock<std::shared_mutex> lock(h.mutex);
    return h.writer();
  });
}

/**
 * @brief Read something of a result's current state under its shared lock (without the GIL).
 * @tparam value_t    The result type.
 * @tparam function_t `T(const value_t&)`.
 * @param[in] h The holder.
 * @param[in] f The read.
 * @return What `f` returns.
 */
template <typename value_t, typename function_t>
decltype(auto) read_result(const result_holder<value_t>& h, function_t&& f) {
  return without_gil([&] {
    std::shared_lock<std::shared_mutex> lock(h.mutex);
    return f(h.get());
  });
}

// ---------------------------------------------------------------------------------------------
// The bind functions (one per file)
// ---------------------------------------------------------------------------------------------

void bind_errors(nb::module_& m);       ///< errors.cpp
void bind_resources(nb::module_& m);    ///< resources.cpp
void bind_graph(nb::module_& m);        ///< graph.cpp
void bind_batch(nb::module_& m);        ///< batch.cpp
void bind_sssp(nb::module_& m);         ///< sssp.cpp
void bind_cycle_count(nb::module_& m);  ///< cycle_count.cpp
void bind_mosp(nb::module_& m);         ///< mosp.cpp
void bind_update(nb::module_& m);       ///< update.cpp
void bind_io(nb::module_& m);           ///< io.cpp
void bind_generators(nb::module_& m);   ///< generators.cpp
void bind_profiler(nb::module_& m);     ///< profiler.cpp
void bind_registry(nb::module_& m);     ///< registry.cpp
void bind_testing(nb::module_& m);      ///< testing.cpp
void bind_arrays(nb::module_& m);       ///< arrays.cpp

}  // namespace dyng::python
