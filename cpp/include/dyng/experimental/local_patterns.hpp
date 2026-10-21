// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/wedge.cuh (intersectSize, venn3, intersects3)
/**
 * @file local_patterns.hpp
 * @brief EXPERIMENTAL PUBLIC; may change in any minor release. The host-compilable part of
 *        experimental::count_local_patterns: the shapes, the rows an Op reads (hyperedge_rows,
 *        member_span) and the set helpers on them, the row accessor, and the stats of an update.
 *
 * Everything here compiles with a plain host compiler and includes no CUDA header, so a host
 * program can name these types (and read a hypergraph's rows with access_rows()); counting a
 * pattern needs the extension header <dyng/experimental/local_patterns.cuh> and nvcc (PLAN
 * Section 4.1: the one public header that needs nvcc by nature).
 * @ingroup experimental
 */
#pragma once

#include <dyng/core/memory.hpp>
#include <dyng/core/stats.hpp>
#include <dyng/core/types.hpp>
#include <dyng/hypergraph/hyper_apply_summary.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <climits>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <vector>

/**
 * @defgroup experimental experimental
 * @brief EXPERIMENTAL PUBLIC; may change in any minor release: count_local_patterns<op_t>, the
 *        extension point that counts a new hypergraph pattern with triad_count's wedge engine.
 *
 * A **local pattern** is something counted per small group of hyperedges: per triangle of the
 * line graph (three hyperedges that pairwise share a vertex), per adjacent pair (two hyperedges
 * that share a vertex), or per hyperedge. An **Op** says what to count: a small trivially
 * copyable struct with `num_bins` 64-bit bins, a `shape` (shape::triangles, shape::pairs or
 * shape::keys) and the callbacks of its shape, which the engine calls with 0-based hyperedge ids
 * and the rows handle (hyperedge_rows: the members of a hyperedge as a member_span of opaque,
 * ascending codes; hyperedge_rows::decode() gives the vertex id of a code). The engine
 * (ESCHER-GPU's wedge engine, the one triad_count runs) owns everything else: the full count of
 * compute(), the exact delta of update() (the triads through the deleted hyperedges subtracted
 * on the hypergraph before the batch, those through the inserted ones added after it, each
 * counted once, at its smallest changed member), the frontier, the sign and the bins. The four
 * patterns of triad_count are four such Ops, built into the library.
 *
 * The page "Count a new hypergraph pattern (an Op)" of the documentation is the tutorial; the
 * Op concept is documented with count_local_patterns (local_patterns.cuh).
 *
 * Status: experimental (PLAN Section 5.9): any minor release may change it, with a CHANGELOG
 * entry; the API check tracks and reports its changes without failing on them.
 */

#if defined(__CUDACC__)
/// Marks a callback of an Op (and any function it calls) that runs on the host and the device.
#define DYNG_HD __host__ __device__
/// A small function of the engine and its set helpers, inlined on the host and the device.
#define DYNG_HD_INLINE __host__ __device__ __forceinline__
#else
/// Marks a callback of an Op (and any function it calls) that runs on the host and the device.
#define DYNG_HD
/// A small function of the engine and its set helpers, inlined on the host and the device.
#define DYNG_HD_INLINE inline
#endif

namespace dyng::detail::local_patterns {
struct rows_factory;
}  // namespace dyng::detail::local_patterns

namespace dyng::experimental {

/**
 * @brief What an Op counts over: the triangles of the line graph, its edges (adjacent pairs of
 *        hyperedges), or the hyperedges themselves (ESCHER-GPU's countTriangles, countPairs and
 *        countKeys).
 * @ingroup experimental
 */
enum class shape : std::uint8_t {
  triangles,  ///< begin_wedge(rows, a, b), then triangle(rows, c, bins) per triangle {a, b, c}
  pairs,      ///< pair(rows, a, b, bins) per pair of hyperedges that share a vertex
  keys,       ///< key(rows, a, bins) per hyperedge
};

/**
 * @brief The lower-case name of a shape ("triangles", "pairs", "keys").
 * @param[in] s The shape.
 * @return A static string naming `s`.
 * @ingroup experimental
 */
[[nodiscard]] std::string_view to_string(shape s) noexcept;

/**
 * @brief The largest num_bins of an Op: its bins live in the block's shared memory on the GPU
 *        (8 bytes each: 8 KiB at most), and on the stack of every host thread.
 * @ingroup experimental
 */
inline constexpr int max_pattern_bins = 1024;

/**
 * @brief An opaque member code: the codes of a hyperedge's members are distinct and ascending in
 *        the vertex ids, so two rows can be merged and compared code by code, and
 *        hyperedge_rows::decode() gives the vertex id of a code. Nothing else is meant by its
 *        value.
 * @ingroup experimental
 */
using member_code = std::int32_t;

/**
 * @brief The members of one hyperedge: its member codes, ascending (a contiguous, read-only
 *        span in the memory the engine runs in; never stored beyond the callback).
 * @ingroup experimental
 */
class member_span {
 public:
  /// An empty span.
  constexpr member_span() noexcept = default;

  /**
   * @brief The span of `count` codes from `first`.
   * @param[in] first The first code.
   * @param[in] count The number of codes.
   */
  DYNG_HD_INLINE constexpr member_span(const member_code* first, int count) noexcept
      : first_(first), count_(count) {}

  /**
   * @brief The first code.
   * @return A pointer to it.
   */
  [[nodiscard]] DYNG_HD_INLINE constexpr const member_code* begin() const noexcept {
    return first_;
  }

  /**
   * @brief Past the last code.
   * @return begin() + size().
   */
  [[nodiscard]] DYNG_HD_INLINE constexpr const member_code* end() const noexcept {
    return first_ + count_;
  }

  /**
   * @brief The number of members.
   * @return The span's length (the hyperedge's size).
   */
  [[nodiscard]] DYNG_HD_INLINE constexpr int size() const noexcept {
    return count_;
  }

  /**
   * @brief Whether the span is empty (a dead hyperedge id).
   * @return size() == 0.
   */
  [[nodiscard]] DYNG_HD_INLINE constexpr bool empty() const noexcept {
    return count_ == 0;
  }

  /**
   * @brief The code of member `i`.
   * @param[in] i The position, in [0, size()).
   * @return The code.
   */
  [[nodiscard]] DYNG_HD_INLINE constexpr member_code operator[](int i) const noexcept {
    return first_[i];
  }

 private:
  const member_code* first_ = nullptr;
  int count_ = 0;
};

/**
 * @brief The rows handle the engine passes to an Op's callbacks: the members of every hyperedge
 *        by its 0-based id, as member spans (no decode pass: the codes are the store's own).
 *
 * Trivially copyable and cheap to pass by value; it points into the memory of the engine's
 * backend (device memory on cuda), so it is valid only inside a callback, or for the life of the
 * row_access it came from.
 * @ingroup experimental
 */
class hyperedge_rows {
 public:
  /// A handle of no rows (row() must not be called on it).
  constexpr hyperedge_rows() noexcept = default;

  /**
   * @brief The members of hyperedge `id`.
   * @param[in] id A hyperedge id the engine passed (a callback's argument).
   * @return Its member codes, ascending.
   */
  [[nodiscard]] DYNG_HD_INLINE member_span row(int id) const {
    return {flat_ + loc_[id + 1], len_[id + 1]};
  }

  /**
   * @brief The number of members of hyperedge `id`.
   * @param[in] id A hyperedge id the engine passed.
   * @return row(id).size().
   */
  [[nodiscard]] DYNG_HD_INLINE int size(int id) const {
    return len_[id + 1];
  }

  /**
   * @brief The vertex id of a member code.
   * @param[in] code A code of a member_span of these rows.
   * @return The member's vertex id, in [0, num_vertices()).
   */
  [[nodiscard]] DYNG_HD_INLINE static constexpr std::int32_t decode(member_code code) noexcept {
    return code - 1;
  }

 private:
  friend struct detail::local_patterns::rows_factory;
  const member_code* flat_ = nullptr;  // the store's payload
  const int* loc_ = nullptr;           // by key: the offset of the row
  const int* len_ = nullptr;           // by key: the length of the row
};

/**
 * @brief The sizes of the regions of three member sets that the classification of a triad
 *        reads (venn3()).
 * @ingroup experimental
 */
struct venn_counts {
  int ab = 0;   ///< |A ∩ B|
  int bc = 0;   ///< |B ∩ C|
  int ca = 0;   ///< |C ∩ A|
  int abc = 0;  ///< |A ∩ B ∩ C|
};

}  // namespace dyng::experimental

namespace dyng::detail::local_patterns {

/**
 * @brief The line graph's rows by key (ESCHER-GPU's RowView of H2H): the layout of the
 *        hypergraph's internal row view (cpp/src/hypergraph/row_views.hpp), engine-only.
 */
struct line_rows {
  const int* flat = nullptr;  ///< the payload: the neighbour keys of every key, sorted
  const int* loc = nullptr;   ///< key -> the offset of its row in flat
  const int* len = nullptr;   ///< key -> the length of its row (0 for an absent key)
  const int* out = nullptr;   ///< key -> the offset of its first entry greater than the key
  int max_key = 0;            ///< the arrays cover the keys 0..max_key
};

/// min(x, y) (the device's integer min on the device).
DYNG_HD_INLINE int min_of(int x, int y) {
#if defined(__CUDA_ARCH__)
  return min(x, y);
#else
  return x < y ? x : y;
#endif
}

/// max(x, y) (the device's integer max on the device).
DYNG_HD_INLINE int max_of(int x, int y) {
#if defined(__CUDA_ARCH__)
  return max(x, y);
#else
  return x < y ? y : x;
#endif
}

// ---- Set primitives on sorted, contiguous rows (ESCHER-GPU's kernel/wedge.cuh) ----------------

/// |A ∩ B| of two sorted rows.
DYNG_HD_INLINE int intersect_size(const int* a, int na, const int* b, int nb) {
  int x = 0, y = 0, c = 0;
  while (x < na && y < nb) {
    int va = a[x], vb = b[y];
    c += (va == vb);
    x += (va <= vb);
    y += (vb <= va);
  }
  return c;
}

/// One merge pass over sorted rows a, b, c: |B ∩ C|, |C ∩ A| and |A ∩ B ∩ C|. The rows end at
/// INT_MAX, so no row may hold INT_MAX (count_local_patterns and triad_count's compute() refuse a
/// hypergraph whose largest member code could be INT_MAX).
DYNG_HD_INLINE void venn3(const int* a, int na, const int* b, int nb, const int* c, int nc,
                          int& cbc, int& cca, int& g) {
  int x = 0, y = 0, z = 0;
  cbc = cca = g = 0;
  while (true) {
    int va = x < na ? a[x] : INT_MAX;
    int vb = y < nb ? b[y] : INT_MAX;
    int vc = z < nc ? c[z] : INT_MAX;
    int m = min_of(va, min_of(vb, vc));
    if (m == INT_MAX) break;
    int ia = va == m, ib = vb == m, ic = vc == m;
    cbc += ib & ic;
    cca += ic & ia;
    g += ia & ib & ic;
    x += ia;
    y += ib;
    z += ic;
  }
}

/// Whether sorted rows a, b and c share an element.
DYNG_HD_INLINE bool intersects3(const int* a, int na, const int* b, int nb, const int* c, int nc) {
  int x = 0, y = 0, z = 0;
  while (x < na && y < nb && z < nc) {
    int m = max_of(a[x], max_of(b[y], c[z]));
    if (a[x] == m && b[y] == m && c[z] == m) return true;
    x += a[x] < m;
    y += b[y] < m;
    z += c[z] < m;
  }
  return false;
}

}  // namespace dyng::detail::local_patterns

namespace dyng::experimental {

/**
 * @brief |A ∩ B|: the number of members two hyperedges share (one merge of their codes).
 * @param[in] a The members of one hyperedge.
 * @param[in] b The members of another.
 * @return The size of the intersection.
 * @ingroup experimental
 */
[[nodiscard]] DYNG_HD_INLINE int intersect_size(member_span a, member_span b) noexcept {
  return detail::local_patterns::intersect_size(a.begin(), a.size(), b.begin(), b.size());
}

/**
 * @brief The pairwise and the triple intersections of three hyperedges in one merge of their
 *        codes (the regions of their Venn diagram follow: only a = |A| - ab - ca + abc, ...).
 * @param[in] a The members of the first hyperedge.
 * @param[in] b The members of the second.
 * @param[in] c The members of the third.
 * @return |A ∩ B|, |B ∩ C|, |C ∩ A| and |A ∩ B ∩ C|.
 * @ingroup experimental
 */
[[nodiscard]] DYNG_HD_INLINE venn_counts venn3(member_span a, member_span b,
                                               member_span c) noexcept {
  venn_counts v;
  int x = 0, y = 0, z = 0;
  const int na = a.size(), nb = b.size(), nc = c.size();
  while (true) {
    const int va = x < na ? a[x] : INT_MAX;
    const int vb = y < nb ? b[y] : INT_MAX;
    const int vc = z < nc ? c[z] : INT_MAX;
    const int m = detail::local_patterns::min_of(va, detail::local_patterns::min_of(vb, vc));
    if (m == INT_MAX) break;
    const int ia = va == m, ib = vb == m, ic = vc == m;
    v.ab += ia & ib;
    v.bc += ib & ic;
    v.ca += ic & ia;
    v.abc += ia & ib & ic;
    x += ia;
    y += ib;
    z += ic;
  }
  return v;
}

/**
 * @brief Whether three hyperedges share a member.
 * @param[in] a The members of the first hyperedge.
 * @param[in] b The members of the second.
 * @param[in] c The members of the third.
 * @return true if some vertex is in all three.
 * @ingroup experimental
 */
[[nodiscard]] DYNG_HD_INLINE bool intersects3(member_span a, member_span b,
                                              member_span c) noexcept {
  return detail::local_patterns::intersects3(a.begin(), a.size(), b.begin(), b.size(), c.begin(),
                                             c.size());
}

/**
 * @brief The rows the engine reads of one hypergraph state, as access_rows() returns them: the
 *        hyperedges' members by id (the handle the callbacks get) and the state's id bound.
 *
 * It points into the hypergraph's storage (device memory on cuda: space() and device() say
 * where) and is valid until the next apply() or update() of the hypergraph; reading it does not
 * copy anything.
 * @ingroup experimental
 */
class row_access {
 public:
  /// No rows.
  constexpr row_access() noexcept = default;

  /**
   * @brief The members of every hyperedge by id: hyperedges().row(id) for every id in
   *        [0, id_bound()); a dead id has an empty row.
   * @return The rows handle (as an Op's callbacks receive it).
   */
  [[nodiscard]] hyperedge_rows hyperedges() const noexcept {
    return hyperedges_;
  }

  /**
   * @brief The hyperedge id bound of the state (hypergraph::hyperedge_id_bound()).
   * @return One more than the largest id that may be live.
   */
  [[nodiscard]] std::int64_t id_bound() const noexcept {
    return id_bound_;
  }

  /**
   * @brief Whether the state keeps its line graph (what the triangles and pairs shapes read).
   * @return true if the line graph is maintained.
   */
  [[nodiscard]] bool has_line_graph() const noexcept {
    return has_line_graph_;
  }

  /**
   * @brief Where the rows are.
   * @return memory_space::device for a hypergraph of the cuda backend, memory_space::host
   *         otherwise.
   */
  [[nodiscard]] memory_space space() const noexcept {
    return space_;
  }

  /**
   * @brief The CUDA device of device rows.
   * @return The device ordinal, or -1 for host rows.
   */
  [[nodiscard]] int device() const noexcept {
    return device_;
  }

 private:
  friend struct detail::local_patterns::rows_factory;
  hyperedge_rows hyperedges_;
  detail::local_patterns::line_rows line_;
  int h2v_keys_ = 0;
  std::int64_t id_bound_ = 0;
  memory_space space_ = memory_space::host;
  int device_ = -1;
  bool has_line_graph_ = false;
};

/**
 * @brief Counters of one count_local_patterns update (fields are only ever appended).
 *
 * Every counter is deterministic (identical on every backend, thread count and run). The
 * inherited fields mean what they mean for triad_count::stats: `affected` is the number of bins
 * whose count changed, `iterations` is 0, `frontier_visits` is deleted_frontier +
 * inserted_frontier, `fallback_used` is false, `converged` is true, and `engine_used` is
 * engine::fused on cuda and engine::operators on the host backends.
 * @ingroup experimental
 */
struct local_patterns_stats : update_stats {
  /// What applying the batch did to the hypergraph (the inserted ids among it).
  hyper_apply_summary batch;
  /// The hyperedges of the subtract phase: the live ids the batch deleted (distinct).
  std::int64_t deleted_frontier = 0;
  /// The hyperedges of the add phase: the hyperedges the batch inserted.
  std::int64_t inserted_frontier = 0;
  /// The subtract phase's count, summed over the bins (what the deleted hyperedges took away).
  std::uint64_t removed = 0;
  /// The add phase's count, summed over the bins (what the inserted hyperedges brought).
  std::uint64_t added = 0;
  /// The change of every bin: the counts after the update minus the counts before it (num_bins
  /// values).
  std::vector<std::int64_t> delta;
};

}  // namespace dyng::experimental

namespace dyng::detail::local_patterns {

/// The engine's constructors of the rows handles (the encoding stays inside the library).
struct rows_factory {
  /// The handle of the h2v rows (by key = id + 1; the members + 1, sorted).
  DYNG_HD_INLINE static experimental::hyperedge_rows make(const int* flat, const int* loc,
                                                          const int* len) noexcept {
    experimental::hyperedge_rows r;
    r.flat_ = flat;
    r.loc_ = loc;
    r.len_ = len;
    return r;
  }

  /// The row access of a state: its h2v handle, its line graph and their bounds.
  static experimental::row_access make_access(experimental::hyperedge_rows hyperedges,
                                              const line_rows& line, int h2v_keys,
                                              std::int64_t id_bound, memory_space space, int device,
                                              bool has_line_graph) noexcept {
    experimental::row_access a;
    a.hyperedges_ = hyperedges;
    a.line_ = line;
    a.h2v_keys_ = h2v_keys;
    a.id_bound_ = id_bound;
    a.space_ = space;
    a.device_ = device;
    a.has_line_graph_ = has_line_graph;
    return a;
  }

  /// The line graph's rows of a state (keys = ids + 1).
  static const line_rows& line(const experimental::row_access& a) noexcept {
    return a.line_;
  }

  /// The h2v keys of a state (1..h2v_keys: the ids below the bound, + 1).
  static int h2v_keys(const experimental::row_access& a) noexcept {
    return a.h2v_keys_;
  }
};

}  // namespace dyng::detail::local_patterns

namespace dyng::detail {

/**
 * @brief Whether count_local_patterns and access_rows() are compiled for
 *        hypergraph<vertex_t, weight_t>: int32_t vertex ids, unweighted or int32_t weights (the
 *        types of triad_count; the weights are ignored).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 */
template <typename vertex_t, typename weight_t>
inline constexpr bool local_patterns_supported_v =
    std::is_same_v<vertex_t, std::int32_t> &&
    (std::is_same_v<weight_t, unweighted> || std::is_same_v<weight_t, std::int32_t>);

/**
 * @brief The implementation of experimental::access_rows() (instantiated for the supported
 *        types).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in] h The hypergraph.
 * @return Its rows.
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] experimental::row_access local_patterns_access_rows(
    const hypergraph<vertex_t, weight_t>& h);

}  // namespace dyng::detail

/**
 * @brief The message of count_local_patterns' and access_rows()'s static_assert for an
 *        unsupported hypergraph type.
 * @ingroup experimental
 */
#define DYNG_LOCAL_PATTERNS_TYPES_MESSAGE                                                         \
  "dyng::experimental::count_local_patterns supports hypergraph<int32_t, unweighted or int32_t> " \
  "only (the types of triad_count; the weights are ignored)"

namespace dyng::experimental {

/**
 * @brief The rows the engine reads of `h`'s current state (the hyperedges' members by id).
 *
 * Nothing is copied or built (under line_graph_policy::automatic the line graph exists only once
 * an algorithm asked for it: has_line_graph() says).
 * @tparam vertex_t Vertex id type (int32_t).
 * @tparam weight_t Hyperedge weight type (unweighted or int32_t).
 * @param[in] h The hypergraph.
 * @return Its rows, valid until the next apply() or update() of `h`.
 * @throws invalid_argument_error if `h` was poisoned by a failed apply().
 * @ingroup experimental
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] row_access access_rows(const hypergraph<vertex_t, weight_t>& h) {
  static_assert(detail::local_patterns_supported_v<vertex_t, weight_t>,
                DYNG_LOCAL_PATTERNS_TYPES_MESSAGE);
  return detail::local_patterns_access_rows(h);
}

}  // namespace dyng::experimental
