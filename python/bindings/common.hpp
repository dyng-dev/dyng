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
 * Locking (ADR 0025). The GIL is released around every native call that runs algorithms or I/O
 * (PLAN Section 5.4 rule 3). Two locks then keep concurrent Python threads memory-safe:
 *   - every container and result holder has a reader/writer lock: calls that only read it share
 *     it, calls that change it (apply, update) take it exclusively;
 *   - every native call holds a process-wide lock in shared mode; reading a profiler's records
 *     takes it exclusively, so the records are never read while a call may write them (ADR 0023,
 *     note 5).
 * Locks are always taken after the GIL is released and released before it is taken back, so a
 * thread waiting for a lock never holds the GIL.
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/types.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
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
 * @brief The process-wide lock of native calls (shared by every call, exclusive for reading a
 *        profiler's records).
 * @return The lock.
 */
std::shared_mutex& native_call_mutex();

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
  std::shared_lock<std::shared_mutex> call_lock(native_call_mutex());
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
 *        __dlpack__ and __dlpack_device__) viewing memory that `owner` keeps alive (no copy).
 * @tparam value_t The element type.
 * @param[in] v     The elements (host memory).
 * @param[in] owner The Python object that owns the memory (kept alive by the array).
 * @return The array.
 */
template <typename value_t>
nb::ndarray<nb::array_api, const value_t, nb::ndim<1>> owned_view(array_view<const value_t> v,
                                                                  nb::handle owner) {
  const std::size_t shape[1] = {v.size()};
  const bool device = !is_host_accessible(v.space()) || v.space() == memory_space::managed;
  return nb::ndarray<nb::array_api, const value_t, nb::ndim<1>>(
      v.data(), 1, shape, owner, nullptr, nb::dtype<value_t>(),
      device ? nb::device::cuda::value : nb::device::cpu::value, device ? v.device() : 0);
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
void bind_update(nb::module_& m);       ///< update.cpp
void bind_io(nb::module_& m);           ///< io.cpp
void bind_generators(nb::module_& m);   ///< generators.cpp
void bind_profiler(nb::module_& m);     ///< profiler.cpp
void bind_registry(nb::module_& m);     ///< registry.cpp
void bind_testing(nb::module_& m);      ///< testing.cpp

}  // namespace dyng::python
