// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_kit_traits.hpp
 * @brief test_traits of the kit's hypergraph fakes (framework/fake_hyper_problems.hpp), the
 *        test-only algorithms that run the conformance kit's hypergraph axis (ADR 0043) on every
 *        backend until triad_count (M9) does: test_hyper_pairs (the overlapping pairs of live
 *        hyperedges; its suite is hypergraph_kit_test.cpp) and test_hyper_squares (the sum of
 *        |e|^2; the partner of C10).
 *
 * They are not in the library's registry, so the traits give their registry entry themselves
 * (registry_entry(); a registered hypergraph algorithm is found through its manifest like any
 * other) and list each other as test_partners for C10.
 */
#pragma once

#include "conformance/hypergraph_generators.hpp"
#include "conformance/test_traits.hpp"
#include "conformance/type_list.hpp"
#include "framework/fake_hyper_problems.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/registry.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/hypergraph/hypergraph_properties.hpp>
#include <dyng/testing/line_graph_oracle.hpp>

#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

namespace dyng::conformance {

namespace tags {
/// The tag of the test-only hypergraph fake test_hyper_pairs.
struct test_hyper_pairs;
/// The tag of the test-only hypergraph fake test_hyper_squares.
struct test_hyper_squares;
}  // namespace tags

namespace hyper_kit_detail {

namespace fakes = ::dyng::test::framework_fakes;

/// The hypergraph types of the fakes: every instantiated combination (int32, unweighted) and
/// (int32, int32).
using hypergraph_types =
    type_list<hypergraph<std::int32_t, unweighted>, hypergraph<std::int32_t, std::int32_t>>;

/// The registry entry of a fake (as a manifest would declare it).
inline algorithm_info entry_of(std::string_view name, std::string_view title) {
  algorithm_info info;
  info.name = name;
  info.title = title;
  info.family = algorithm_family::aggregate_delta;
  info.container = container_kind::hypergraph;
  info.maturity = maturity_level::experimental;
  info.determinism_level = determinism::bitwise;
  info.oracle = oracle_kind::compute;
  info.backends = {backend::sequential, backend::openmp, backend::cuda};
  return info;
}

/// The sizes of the kit's hypergraphs (the medium one has more than 1,024 vertices, so the
/// skewed mix inserts a hyperedge of more than 1,024 members).
inline hypergraph_shape shape_of(size_class size) {
  switch (size) {
    case size_class::tiny:
      return {6, 4, 3, 9};
    case size_class::small:
      return {40, 30, 5, 9};
    case size_class::medium:
      return {1100, 300, 8, 9};
  }
  return {};
}

/// The members of every id of `h` on the host (h2v()).
template <typename container_t>
void h2v_on_host(const resources& res, const container_t& h, std::vector<std::int64_t>& offsets,
                 std::vector<std::int32_t>& members) {
  const auto view = h.h2v();
  offsets = to_vector(res, view.row_ptr);
  const std::vector<typename container_t::vertex_type> cols = to_vector(res, view.col_ind);
  members.assign(cols.begin(), cols.end());
  if (offsets.empty()) offsets.assign(1, 0);
}

}  // namespace hyper_kit_detail

/// The overlapping pairs of live hyperedges (the edges of the line graph).
template <>
struct test_traits<tags::test_hyper_pairs> {
  static constexpr std::string_view name = "test_hyper_pairs";  ///< the fake's name
  static constexpr oracle_kind oracle = oracle_kind::compute;   ///< update chain == compute
  static constexpr determinism level = determinism::bitwise;    ///< exact counts
  static constexpr bool history_independent = true;             ///< C5 applies
  using container_types = hyper_kit_detail::hypergraph_types;   ///< every instantiation
  template <typename container_t>
  using result = hyper_kit_detail::fakes::hyper_pairs_result;  ///< the result
  using stats = hyper_kit_detail::fakes::hyper_count_stats;    ///< the stats
  using snapshot = std::int64_t;                               ///< the count
  using test_partners = type_list<tags::test_hyper_squares>;   ///< C10's partner

  /// Not in the library's registry: its entry, as a manifest would declare it.
  static const algorithm_info* registry_entry() {
    static const algorithm_info info =
        hyper_kit_detail::entry_of(name, "Overlapping pairs of hyperedges (kit fake)");
    return &info;
  }
  /// It reads the line graph, kept under every preset (automatic builds it on first use).
  static void require(hypergraph_properties& /*p*/) {}
  /// One extra preset: the line graph maintained from construction.
  static std::vector<hypergraph_properties> extra_properties() {
    hypergraph_properties p;
    p.line_graph = line_graph_policy::sorted_merge;
    return {p};
  }
  /// The kit's hypergraphs.
  static hypergraph_shape shape(size_class size) {
    return hyper_kit_detail::shape_of(size);
  }
  /// compute() (one engine; the choice is ignored).
  template <typename container_t>
  static result<container_t> compute(const resources& res, const container_t& h, engine /*e*/) {
    return hyper_kit_detail::fakes::compute_pairs(res, h);
  }
  /// update() through run_update().
  template <typename container_t>
  static stats update(const resources& res, container_t& h,
                      const typename container_t::batch_view_type& batch, result<container_t>& r) {
    return hyper_kit_detail::fakes::update_pairs(res, h, batch, r);
  }
  /// The count.
  template <typename container_t>
  static snapshot take(const resources& /*res*/, const result<container_t>& r) {
    return r.value;
  }
  /// The independent oracle: the line graph rebuilt from h2v (MOSP_ESCHER's algorithm).
  template <typename container_t>
  static snapshot oracle_of(const resources& res, const container_t& h) {
    std::vector<std::int64_t> offsets;
    std::vector<std::int32_t> members;
    hyper_kit_detail::h2v_on_host(res, h, offsets, members);
    const testing::line_graph_rows rows =
        testing::rebuild_line_graph(offsets, members, static_cast<std::int32_t>(h.num_vertices()));
    return static_cast<std::int64_t>(rows.neighbours.size()) / 2;
  }
  /// The counts of the two phases.
  static std::vector<std::int64_t> deterministic(const stats& s) {
    return {s.removed, s.added};
  }
  /// No options.
  template <typename container_t>
  static std::vector<std::function<void()>> invalid_options(const resources& /*res*/,
                                                            const container_t& /*h*/) {
    return {};
  }
  /// One read-back per count on CUDA.
  static std::int64_t host_sync_budget(backend b) {
    return b == backend::cuda ? 2 : 0;
  }
};

/// The sum of |e|^2 over the live hyperedges (C10's partner of test_hyper_pairs).
template <>
struct test_traits<tags::test_hyper_squares> {
  static constexpr std::string_view name = "test_hyper_squares";  ///< the fake's name
  static constexpr oracle_kind oracle = oracle_kind::compute;     ///< update chain == compute
  static constexpr determinism level = determinism::bitwise;      ///< exact sums
  static constexpr bool history_independent = true;               ///< C5 applies
  using container_types = hyper_kit_detail::hypergraph_types;     ///< every instantiation
  template <typename container_t>
  using result = hyper_kit_detail::fakes::hyper_squares_result;  ///< the result
  using stats = hyper_kit_detail::fakes::hyper_count_stats;      ///< the stats
  using snapshot = std::int64_t;                                 ///< the sum
  using test_partners = type_list<tags::test_hyper_pairs>;       ///< C10's partner

  /// Not in the library's registry: its entry, as a manifest would declare it.
  static const algorithm_info* registry_entry() {
    static const algorithm_info info =
        hyper_kit_detail::entry_of(name, "Sum of squared hyperedge sizes (kit fake)");
    return &info;
  }
  /// No requirement.
  static void require(hypergraph_properties& /*p*/) {}
  /// The kit's hypergraphs.
  static hypergraph_shape shape(size_class size) {
    return hyper_kit_detail::shape_of(size);
  }
  /// compute().
  template <typename container_t>
  static result<container_t> compute(const resources& res, const container_t& h, engine /*e*/) {
    return hyper_kit_detail::fakes::compute_squares(res, h);
  }
  /// update() through run_update().
  template <typename container_t>
  static stats update(const resources& res, container_t& h,
                      const typename container_t::batch_view_type& batch, result<container_t>& r) {
    return hyper_kit_detail::fakes::update_squares(res, h, batch, r);
  }
  /// The sum.
  template <typename container_t>
  static snapshot take(const resources& /*res*/, const result<container_t>& r) {
    return r.value;
  }
  /// The counts of the two phases.
  static std::vector<std::int64_t> deterministic(const stats& s) {
    return {s.removed, s.added};
  }
  /// No options.
  template <typename container_t>
  static std::vector<std::function<void()>> invalid_options(const resources& /*res*/,
                                                            const container_t& /*h*/) {
    return {};
  }
  /// One read-back per count on CUDA.
  static std::int64_t host_sync_budget(backend b) {
    return b == backend::cuda ? 2 : 0;
  }
};

}  // namespace dyng::conformance
