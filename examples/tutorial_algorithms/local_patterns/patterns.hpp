// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file patterns.hpp
 * @brief The Ops of the tutorial "Count a new hypergraph pattern (an Op)"
 *        (docs/tutorials/count_a_new_hypergraph_pattern.md): four patterns counted by
 *        experimental::count_local_patterns, none of them an enumerator of triad_count::pattern.
 *
 * Host-compilable: the Ops need only <dyng/experimental/local_patterns.hpp> (DYNG_HD is empty
 * without nvcc), so the host-only test can run their callbacks; counting them with the engine
 * needs <dyng/experimental/local_patterns.cuh> and nvcc (tutorial_test.cu). The tutorial quotes
 * this file between its `// [tutorial: ...]` markers.
 */
#pragma once

#include <dyng/experimental/local_patterns.hpp>

#include <cstdint>

namespace tutorial {

namespace dx = dyng::experimental;

// [tutorial: line_graph_triangles]
/// The triangles of the line graph: the triples of hyperedges that pairwise share a vertex (one
/// bin). triad_count classifies the same triples into its 30 h-motifs, so this count equals
/// triad_count's h_motif total().
struct line_graph_triangles {
  static constexpr int num_bins = 1;
  static constexpr dx::shape shape = dx::shape::triangles;

  std::uint64_t count = 0;  // this worker's partial sum, added to the bin by flush()

  DYNG_HD void begin_wedge(const dx::hyperedge_rows& /*rows*/, int /*a*/, int /*b*/) {}

  template <class bins_t>
  DYNG_HD void triangle(const dx::hyperedge_rows& /*rows*/, int /*c*/, bins_t& /*bins*/) {
    ++count;
  }

  template <class bins_t>
  DYNG_HD void flush(bins_t& bins) {
    if (count != 0) bins.add(0, count);
  }
};
// [tutorial: line_graph_triangles end]

// [tutorial: strong_pairs]
/// The pairs of hyperedges that share at least `k` vertices (one bin). `k` is a parameter of the
/// Op's value: count_local_patterns<strong_pairs>::compute(res, h, strong_pairs{3}).
struct strong_pairs {
  static constexpr int num_bins = 1;
  static constexpr dx::shape shape = dx::shape::pairs;

  int k = 2;                // the threshold
  std::uint64_t count = 0;  // this worker's partial sum

  template <class bins_t>
  DYNG_HD void pair(const dx::hyperedge_rows& rows, int a, int b, bins_t& /*bins*/) {
    if (dx::intersect_size(rows.row(a), rows.row(b)) >= k) ++count;
  }

  template <class bins_t>
  DYNG_HD void flush(bins_t& bins) {
    if (count != 0) bins.add(0, count);
  }
};
// [tutorial: strong_pairs end]

// [tutorial: core_triads]
/// The triangles of the line graph by the size of their core, the vertices all three hyperedges
/// share: bin 0 none, bin 1 one, bin 2 two or more (three bins, added to directly).
struct core_triads {
  static constexpr int num_bins = 3;
  static constexpr dx::shape shape = dx::shape::triangles;

  int a = 0;  // the wedge (a, b) that the next triangles close
  int b = 0;

  DYNG_HD void begin_wedge(const dx::hyperedge_rows& /*rows*/, int wedge_a, int wedge_b) {
    a = wedge_a;
    b = wedge_b;
  }

  template <class bins_t>
  DYNG_HD void triangle(const dx::hyperedge_rows& rows, int c, bins_t& bins) {
    const dx::venn_counts v = dx::venn3(rows.row(a), rows.row(b), rows.row(c));
    bins.add(v.abc < 2 ? v.abc : 2, 1);
  }

  template <class bins_t>
  DYNG_HD void flush(bins_t& /*bins*/) {}
};
// [tutorial: core_triads end]

// [tutorial: hub_pairs]
/// The pairs of hyperedges that both contain the vertex `hub` (one bin): the members are opaque
/// codes, and rows.decode() gives a code's vertex id.
struct hub_pairs {
  static constexpr int num_bins = 1;
  static constexpr dx::shape shape = dx::shape::pairs;

  std::int32_t hub = 0;  // the vertex
  std::uint64_t count = 0;

  DYNG_HD bool has_hub(const dx::hyperedge_rows& rows, int id) const {
    for (const dx::member_code code : rows.row(id)) {
      if (rows.decode(code) == hub) return true;
    }
    return false;
  }

  template <class bins_t>
  DYNG_HD void pair(const dx::hyperedge_rows& rows, int a, int b, bins_t& /*bins*/) {
    if (has_hub(rows, a) && has_hub(rows, b)) ++count;
  }

  template <class bins_t>
  DYNG_HD void flush(bins_t& bins) {
    if (count != 0) bins.add(0, count);
  }
};
// [tutorial: hub_pairs end]

}  // namespace tutorial
