// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/wedge.cuh (forEachWedge's and TriangleBody's frontier
// filters, the wedges of a source as loops) and include/frontier.hpp (the exactly-once rule)
/**
 * @file local_patterns_engine.hpp
 * @brief EXPERIMENTAL PUBLIC (detail); may change in any minor release. The host half of the
 *        local-pattern engine that <dyng/experimental/local_patterns.cuh> hands to the library:
 *        the type-erased Op (the kernels table), the work of one source on the host, the
 *        exactly-once rule, the Op concept's checks, and the driver's entry points in libdyng.
 *
 * One engine, one driver (ADR 0058): libdyng's driver (Step 0, the views of G_t and G_{t+1},
 * the frontiers, the workspace, the commit, the delta) runs every pattern, the four of
 * triad_count (built into the library) and the Ops of users; what depends on the Op is a table of
 * two functions (local_patterns::kernels): `launch` (the Op's counting kernel on cuda,
 * local_patterns_kernels.cuh) and `source` (the work of one source on the host backends, here).
 * Host-compilable: no CUDA header. Nothing here is meant to be named by users.
 * @ingroup experimental
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/experimental/local_patterns.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

namespace dyng::detail::local_patterns {

/// The version of the hand-off between the header and libdyng (checked by the driver).
inline constexpr std::uint32_t abi_version = 1;

// ---- The exactly-once rule (ownership::min_member, invariant I2) on keys -----------------------

/// Whether the wedge (f, b) of a frontier phase belongs to a smaller frontier member: b is in the
/// frontier and b < f (ESCHER-GPU's forEachWedge). No flags (a full count): never.
DYNG_HD_INLINE bool min_member_skips_wedge(const unsigned char* in_frontier, int b, int f) {
#if defined(DYNG_MUTATION_TRIAD_SKIP_FRONTIER_FILTER_B)
  // MUTATION (M9a criterion 11, triad_skip_frontier_filter_b; ESCHER-GPU's wedge.cuh:101).
  (void)in_frontier;
  (void)b;
  (void)f;
  return false;
#else
  return in_frontier != nullptr && in_frontier[b] && b < f;
#endif
}

/// Whether the triangle (f, b, k) closed by k belongs to a smaller frontier member: k is in the
/// frontier and k < f (ESCHER-GPU's TriangleBody). No flags (a full count): never.
DYNG_HD_INLINE bool min_member_skips_closing(const unsigned char* in_frontier, int k, int f) {
#if defined(DYNG_MUTATION_TRIAD_SKIP_FRONTIER_FILTER_K)
  // MUTATION (M9a criterion 11, triad_skip_frontier_filter_k; ESCHER-GPU's wedge.cuh:129).
  (void)in_frontier;
  (void)k;
  (void)f;
  return false;
#else
  return in_frontier != nullptr && in_frontier[k] && k < f;
#endif
}

// ---- The bins handle of the host ---------------------------------------------------------------

/// host_bins: an Op named a bin outside [0, num_bins) (it was not written).
inline constexpr std::uint32_t host_invalid_bin = 1U;
/// host_bins: a bin, or a count of the Op's own (overflow()), passed 2^64 - 1.
inline constexpr std::uint32_t host_overflow = 2U;

/**
 * @brief The bins handle an Op gets on the host: one worker's private bins and its status word
 *        (host_invalid_bin, host_overflow), added up by the driver after the pass.
 * @tparam num_bins The Op's bins.
 */
template <int num_bins>
struct host_bins {
  std::uint64_t* bins;    ///< num_bins counters
  std::uint32_t* status;  ///< the error bits of this worker

  /**
   * @brief Add `value` to bin `bin`; a bin outside [0, num_bins) sets host_invalid_bin instead.
   * @param[in] bin   The bin.
   * @param[in] value The count.
   */
  DYNG_HD void add(int bin, std::uint64_t value) const {
    if (bin < 0 || bin >= num_bins) {
      *status |= host_invalid_bin;
      return;
    }
#if defined(__CUDA_ARCH__)
    bins[bin] += value;
#else
    if (__builtin_add_overflow(bins[bin], value, &bins[bin])) {
      *status |= host_overflow;
    }
#endif
  }

  /// Report a count past 2^64 - 1 that the Op found itself (a sum or product of its own).
  DYNG_HD void overflow() const {
    *status |= host_overflow;
  }
};

/**
 * @brief Whether an Op is one of the library's built-in patterns (triad_count's four), which the
 *        library's own headers mark: its device bins handle tests the sign of a bin only (the
 *        built-in Ops never name a bin at or above num_bins) and adds without a carry test when
 *        the Op adds 1 per call (`unit_counts`), as M9a's engine did; the Ops of users get a
 *        handle that tests both bounds and every carry.
 * @tparam op_t The Op.
 */
template <class op_t>
struct builtin_op : std::false_type {};

// ---- The Op concept ----------------------------------------------------------------------------

/// A bins handle with the interface of the engine's (the checks of the Op concept call with it).
struct probe_bins {
  /// add(bin, value)
  DYNG_HD void add(int, unsigned long long) const {}
  /// overflow()
  DYNG_HD void overflow() const {}
};

/// Whether op_t declares `static constexpr int num_bins` (a constant expression).
template <class op_t, class = void>
struct has_num_bins : std::false_type {};
/// (detected)
template <class op_t>
struct has_num_bins<op_t, std::void_t<std::integral_constant<int, op_t::num_bins>>>
    : std::is_same<std::remove_cv_t<decltype(op_t::num_bins)>, int> {};

/// Whether op_t declares `static constexpr dyng::experimental::shape shape`.
template <class op_t, class = void>
struct has_shape : std::false_type {};
/// (detected)
template <class op_t>
struct has_shape<op_t, std::void_t<std::integral_constant<experimental::shape, op_t::shape>>>
    : std::is_same<std::remove_cv_t<decltype(op_t::shape)>, experimental::shape> {};

/// Whether op_t has begin_wedge(const hyperedge_rows&, int, int).
template <class op_t, class = void>
struct has_begin_wedge : std::false_type {};
/// (detected)
template <class op_t>
struct has_begin_wedge<op_t, std::void_t<decltype(std::declval<op_t&>().begin_wedge(
                                 std::declval<const experimental::hyperedge_rows&>(), 0, 0))>>
    : std::true_type {};

/// Whether op_t has triangle(const hyperedge_rows&, int, bins&).
template <class op_t, class = void>
struct has_triangle : std::false_type {};
/// (detected)
template <class op_t>
struct has_triangle<op_t, std::void_t<decltype(std::declval<op_t&>().triangle(
                              std::declval<const experimental::hyperedge_rows&>(), 0,
                              std::declval<probe_bins&>()))>> : std::true_type {};

/// Whether op_t has pair(const hyperedge_rows&, int, int, bins&).
template <class op_t, class = void>
struct has_pair : std::false_type {};
/// (detected)
template <class op_t>
struct has_pair<op_t, std::void_t<decltype(std::declval<op_t&>().pair(
                          std::declval<const experimental::hyperedge_rows&>(), 0, 0,
                          std::declval<probe_bins&>()))>> : std::true_type {};

/// Whether op_t has key(const hyperedge_rows&, int, bins&).
template <class op_t, class = void>
struct has_key : std::false_type {};
/// (detected)
template <class op_t>
struct has_key<op_t, std::void_t<decltype(std::declval<op_t&>().key(
                         std::declval<const experimental::hyperedge_rows&>(), 0,
                         std::declval<probe_bins&>()))>> : std::true_type {};

/// Whether op_t has flush(bins&).
template <class op_t, class = void>
struct has_flush : std::false_type {};
/// (detected)
template <class op_t>
struct has_flush<op_t,
                 std::void_t<decltype(std::declval<op_t&>().flush(std::declval<probe_bins&>()))>>
    : std::true_type {};

/// The shape of op_t (triangles when it declares none: the checks report that).
template <class op_t>
constexpr experimental::shape shape_of() noexcept {
  if constexpr (has_shape<op_t>::value) {
    return op_t::shape;
  } else {
    return experimental::shape::triangles;
  }
}

/// The num_bins of op_t (0 when it declares none: the checks report that).
template <class op_t>
constexpr int bins_of() noexcept {
  if constexpr (has_num_bins<op_t>::value) {
    return op_t::num_bins;
  } else {
    return 0;
  }
}

/**
 * @brief The Op concept of count_local_patterns: one static_assert, with its own message, per
 *        missing or ill-formed member (cpp/tests/compile_fail/local_patterns_op.cpp).
 * @tparam op_t The Op.
 * @return Whether op_t is an Op.
 */
template <class op_t>
constexpr bool check_op() {
  constexpr bool num_bins_ok = has_num_bins<op_t>::value;
  static_assert(num_bins_ok,
                "dyng::experimental::count_local_patterns: the Op must declare `static constexpr "
                "int num_bins = <N>;`, the number of its 64-bit bins");
  constexpr bool bins_in_range =
      !num_bins_ok || (bins_of<op_t>() >= 1 && bins_of<op_t>() <= experimental::max_pattern_bins);
  static_assert(bins_in_range,
                "dyng::experimental::count_local_patterns: the Op's num_bins must be in [1, "
                "dyng::experimental::max_pattern_bins] (1024: the bins live in a GPU block's "
                "shared memory)");
  constexpr bool shape_ok = has_shape<op_t>::value;
  static_assert(shape_ok,
                "dyng::experimental::count_local_patterns: the Op must declare `static constexpr "
                "dyng::experimental::shape shape = dyng::experimental::shape::triangles;` (or "
                "pairs, or keys)");
  constexpr bool copyable = std::is_trivially_copyable_v<op_t>;
  static_assert(copyable,
                "dyng::experimental::count_local_patterns: the Op must be trivially copyable (the "
                "engine copies it into every GPU thread and host worker): no virtual function, no "
                "user-provided copy, move or destructor, no member that owns memory");
  constexpr bool constructible = std::is_default_constructible_v<op_t>;
  static_assert(constructible,
                "dyng::experimental::count_local_patterns: the Op must be default-constructible "
                "(compute() without an Op value-initializes one: op_t{})");
  constexpr experimental::shape s = shape_of<op_t>();
  constexpr bool triangles = s == experimental::shape::triangles;
  constexpr bool begin_ok = !shape_ok || !triangles || has_begin_wedge<op_t>::value;
  static_assert(begin_ok,
                "dyng::experimental::count_local_patterns: an Op of shape triangles must have "
                "`DYNG_HD void begin_wedge(const dyng::experimental::hyperedge_rows& rows, int a, "
                "int b)`");
  constexpr bool triangle_ok = !shape_ok || !triangles || has_triangle<op_t>::value;
  static_assert(triangle_ok,
                "dyng::experimental::count_local_patterns: an Op of shape triangles must have "
                "`template <class bins_t> DYNG_HD void triangle(const "
                "dyng::experimental::hyperedge_rows& rows, int c, bins_t& bins)`");
  constexpr bool pair_ok = !shape_ok || s != experimental::shape::pairs || has_pair<op_t>::value;
  static_assert(pair_ok,
                "dyng::experimental::count_local_patterns: an Op of shape pairs must have "
                "`template <class bins_t> DYNG_HD void pair(const "
                "dyng::experimental::hyperedge_rows& rows, int a, int b, bins_t& bins)`");
  constexpr bool key_ok = !shape_ok || s != experimental::shape::keys || has_key<op_t>::value;
  static_assert(key_ok,
                "dyng::experimental::count_local_patterns: an Op of shape keys must have "
                "`template <class bins_t> DYNG_HD void key(const "
                "dyng::experimental::hyperedge_rows& rows, int a, bins_t& bins)`");
  constexpr bool flush_ok = has_flush<op_t>::value;
  static_assert(flush_ok,
                "dyng::experimental::count_local_patterns: the Op must have `template <class "
                "bins_t> DYNG_HD void flush(bins_t& bins)` (it adds the Op's register-held partial "
                "sums to the bins; an Op that adds in its callbacks has an empty flush)");
  return num_bins_ok && bins_in_range && shape_ok && copyable && constructible && begin_ok &&
         triangle_ok && pair_ok && key_ok && flush_ok;
}

// ---- The hand-off ------------------------------------------------------------------------------

/// What a counting kernel of an Op reads (the driver fills it; local_patterns_kernels.cuh).
struct device_pass {
  line_rows adj;                               ///< the line graph's rows (device memory)
  experimental::hyperedge_rows rows;           ///< the h2v rows (device memory)
  const int* sources = nullptr;                ///< the frontier's keys, or nullptr: the keys 1..n
  int num_sources = 0;                         ///< the sources
  const long long* offsets = nullptr;          ///< the sources' wedge offsets (triangles, pairs)
  long long num_wedges = 0;                    ///< the sources' wedges (triangles, pairs)
  bool all_neighbours = false;                 ///< a frontier phase: every neighbour of a source
  const unsigned char* in_frontier = nullptr;  ///< the frontier's flags (nullptr: a full count)
  unsigned long long* out = nullptr;           ///< the phase's bins (num_bins words)
  unsigned long long* error = nullptr;         ///< the call's error word
  void* stream = nullptr;                      ///< the cudaStream_t of the call
};

/// What the work of one source reads on the host.
struct host_pass {
  line_rows adj;                               ///< the line graph's rows (host memory)
  experimental::hyperedge_rows rows;           ///< the h2v rows (host memory)
  const int* keys = nullptr;                   ///< the frontier's keys, or nullptr: keys 1..n
  const unsigned char* in_frontier = nullptr;  ///< the frontier's flags (nullptr: a full count)
};

/// The cuda half of an Op: launch its counting kernel for a pass; returns cudaGetLastError().
using launch_fn = int (*)(const void* op, const device_pass& pass);
/// The host half of an Op: the work of source `s` into one worker's bins and status word.
using source_fn = void (*)(const void* op, const host_pass& pass, std::int64_t s,
                           std::uint64_t* bins, std::uint32_t* status);

/// The type-erased Op the driver runs (one table per Op type; the Op's value travels beside it).
struct kernels {
  std::uint32_t abi = abi_version;  ///< the hand-off version the table was built with
  int num_bins = 0;                 ///< the Op's bins
  experimental::shape pattern_shape = experimental::shape::triangles;  ///< the Op's shape
  bool builtin = false;        ///< one of triad_count's patterns (its messages and limits)
  bool merge_limit = true;     ///< refuse a hypergraph whose member codes could reach INT_MAX
  std::size_t op_size = 0;     ///< sizeof the Op
  launch_fn launch = nullptr;  ///< cuda (nullptr: the library has no cuda backend)
  source_fn source = nullptr;  ///< host
};

/// The key of source s on the host: the frontier's keys[s], or s + 1 in a full count.
inline int host_source_key(const int* keys, std::int64_t s) noexcept {
  return keys != nullptr ? keys[s] : static_cast<int>(s) + 1;
}

/// The first wedge of source a on the host: every neighbour in a frontier phase, the
/// out-neighbours (b > a) in a full count.
inline int host_wedge_begin(const line_rows& adj, int a, bool all_neighbours) noexcept {
  return all_neighbours ? adj.loc[a] : adj.out[a];
}

/**
 * @brief The work of one source on the host (ESCHER-GPU's wedge engine as loops; M9a's host
 *        engine, ADR 0054): the source's wedges (a, b) that pass the frontier filter, closed into
 *        their triangles (a, b, k), k > b, that pass the closing filter (triangles), or one pair
 *        per wedge (pairs), or the source itself (keys); one copy of the Op, flushed at the end.
 * @tparam op_t The Op.
 * @param[in]  op_bytes The Op's value (its bytes).
 * @param[in]  p        The pass.
 * @param[in]  s        The source.
 * @param[out] bins     The worker's bins.
 * @param[out] status   The worker's status word.
 */
template <class op_t>
void host_source(const void* op_bytes, const host_pass& p, std::int64_t s, std::uint64_t* bins,
                 std::uint32_t* status) {
  host_bins<op_t::num_bins> handle{bins, status};
  op_t op;
  std::memcpy(static_cast<void*>(&op), op_bytes, sizeof(op_t));
  const int a = host_source_key(p.keys, s);
  if constexpr (op_t::shape == experimental::shape::keys) {
    if constexpr (!builtin_op<op_t>::value) {
      if (p.rows.size(a - 1) == 0) {
        return;  // a dead id below the bound (a full count): no hyperedge
      }
    }
    op.key(p.rows, a - 1, handle);
  } else {
    const line_rows& adj = p.adj;
    const bool all_neighbours = p.keys != nullptr;
    const int end_a = adj.loc[a] + adj.len[a];
    for (int pos = host_wedge_begin(adj, a, all_neighbours); pos < end_a; ++pos) {
      const int b = adj.flat[pos];
      if (min_member_skips_wedge(p.in_frontier, b, a)) {
        continue;
      }
      if constexpr (op_t::shape == experimental::shape::pairs) {
        op.pair(p.rows, a - 1, b - 1, handle);
      } else {
        op.begin_wedge(p.rows, a - 1, b - 1);
        int x = pos + 1;
        int y = adj.out[b];
        const int end_b = adj.loc[b] + adj.len[b];
        while (x < end_a && y < end_b) {
          const int ka = adj.flat[x];
          const int kb = adj.flat[y];
          if (ka == kb) {
            if (!min_member_skips_closing(p.in_frontier, ka, a)) {
              op.triangle(p.rows, ka - 1, handle);
            }
            ++x;
            ++y;
          } else if (ka < kb) {
            ++x;
          } else {
            ++y;
          }
        }
      }
    }
  }
  op.flush(handle);
}

}  // namespace dyng::detail::local_patterns

namespace dyng::detail {

/// The state behind an experimental result (defined in libdyng).
struct local_patterns_state;

/// Deletes a local_patterns_state (in libdyng, where it is complete).
struct local_patterns_state_deleter {
  /// Delete `st`.
  void operator()(local_patterns_state* st) const noexcept;
};

/// The owner of an experimental result's state.
using local_patterns_state_ptr =
    std::unique_ptr<local_patterns_state, local_patterns_state_deleter>;

/**
 * @brief The driver of count_local_patterns::compute() (libdyng; instantiated for the supported
 *        types): the full count of the Op `k` with the value `op`.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in] res Execution resources.
 * @param[in] h   The hypergraph.
 * @param[in] k   The Op's table.
 * @param[in] op  The Op's value (k.op_size bytes).
 * @return The state of the new result.
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] local_patterns_state_ptr local_patterns_compute(
    const resources& res, const hypergraph<vertex_t, weight_t>& h, const local_patterns::kernels& k,
    const void* op);

/**
 * @brief The driver of count_local_patterns::update() (libdyng): the update of one result.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in]     res    Execution resources.
 * @param[in,out] h      The hypergraph.
 * @param[in]     batch  The batch.
 * @param[in,out] st     The result's state.
 * @param[in]     target The result (its address: dyng::update()'s duplicate check).
 * @return The stats.
 */
template <typename vertex_t, typename weight_t>
experimental::local_patterns_stats local_patterns_update(
    const resources& res, hypergraph<vertex_t, weight_t>& h,
    const hyperedge_batch_view<vertex_t, weight_t>& batch, local_patterns_state_ptr& st,
    const void* target);

/**
 * @brief The update participant of an experimental result (dyng::update()).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in,out] st     The result's state.
 * @param[in]     target The result.
 * @param[out]    out    Receives the stats.
 * @return The participant.
 */
template <typename vertex_t, typename weight_t>
std::unique_ptr<hyper_update_participant<vertex_t, weight_t>> make_local_patterns_participant(
    local_patterns_state_ptr& st, const void* target, experimental::local_patterns_stats& out);

/// The counts of a result (host memory; invalid_argument_error for a moved-from result,
/// stale_result_error for a poisoned one).
[[nodiscard]] array_view<const std::uint64_t> local_patterns_counts(const local_patterns_state* st);
/// The count of one bin (invalid_argument_error outside [0, num_bins)).
[[nodiscard]] std::uint64_t local_patterns_count(const local_patterns_state* st, int bin);
/// The sum of the bins (capacity_error past 2^64 - 1).
[[nodiscard]] std::uint64_t local_patterns_total(const local_patterns_state* st);
/// The number of bins.
[[nodiscard]] int local_patterns_num_bins(const local_patterns_state* st);
/// The hypergraph version matched (0 for a moved-from result).
[[nodiscard]] std::uint64_t local_patterns_graph_version(const local_patterns_state* st) noexcept;
/// A deep copy (not_supported_error for an unavailable backend).
[[nodiscard]] local_patterns_state_ptr local_patterns_clone(const resources& res,
                                                            const local_patterns_state* st);
/// The Op's value into `out` (`size` bytes; invalid_argument_error if the sizes differ).
void local_patterns_op(const local_patterns_state* st, void* out, std::size_t size);

}  // namespace dyng::detail
