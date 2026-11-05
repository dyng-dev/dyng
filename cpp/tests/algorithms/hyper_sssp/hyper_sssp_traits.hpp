// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hyper_sssp_traits.hpp
 * @brief The conformance-kit traits of hyper_sssp, a hypergraph algorithm (PLAN Section 8.2; the
 *        kit's hypergraph axis, ADR 0043; M11a criterion 15).
 *
 * The kit compares the distances and parents by id (bitwise), and C2 also against the independent
 * oracle (dyng::testing::hyper_sssp_reference). hyper_sssp reads the slack_csr line graph only
 * (HD9): the kit's escher presets are not its (accepts()), and every slack_csr preset gets the
 * vertex-to-hyperedge rows and the slack_csr line graph (require()). C4 compares the two CUDA
 * engines (M11b: engine::fused, MOSP_ESCHER's kernels, and engine::operators, the shared sssp
 * engine of O12; the host backend has one engine), and the test-only tag hyper_sssp_shared_engine
 * runs C0-C12 again with engine::operators as the kit's engine (M11b criterion 3: the kit on both
 * engines). C10's partners: a second hyper_sssp
 * result of another source on the same hypergraph (the test-only tag hyper_sssp_other_source, the
 * manifest's entry) and the kit's fake test_hyper_pairs. C0 checks the openmp refusal (HD14): the
 * manifest lists sequential and cuda.
 */
#pragma once

#include "conformance/hypergraph_generators.hpp"
#include "conformance/hypergraph_kit_traits.hpp"
#include "conformance/test_traits.hpp"
#include "conformance/type_list.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/registry.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/hyper_sssp.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/hypergraph/hypergraph_properties.hpp>
#include <dyng/testing/hyper_sssp_reference.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <string_view>
#include <vector>

namespace dyng::conformance {

namespace tags {
/// The test-only tag of hyper_sssp from another source vertex (C10's partner).
struct hyper_sssp_other_source;
/// The test-only tag of hyper_sssp on the shared engine (the kit's engine::operators on cuda).
struct hyper_sssp_shared_engine;
}  // namespace tags

namespace hyper_sssp_kit_detail {

/// The traits of hyper_sssp from the source vertex `source_vertex` (the kit's hypergraphs have at
/// least two vertices), with `kit_engine` as the CUDA engine of the kit's engine::automatic.
template <std::int32_t source_vertex, engine kit_engine = engine::automatic>
struct traits_of {
  static constexpr oracle_kind oracle = oracle_kind::compute;  ///< update chain == compute
  static constexpr determinism level = determinism::bitwise;   ///< identical bytes
  /// By content (content_snapshot): the inverse of a batch re-inserts the deleted hyperedges
  /// under ids of the allocator's choice, so the result by id may differ, by content it may not.
  static constexpr bool history_independent = true;
  /// The instantiated hypergraph type (weighted hyperedges only).
  using container_types = type_list<hypergraph<std::int32_t, std::int32_t>>;
  template <typename container_t>
  using result = hyper_sssp::result<std::int32_t>;  ///< the result
  using stats = hyper_sssp::stats;                  ///< update()'s stats
  using snapshot = std::vector<std::int64_t>;       ///< the distances, then the parents, by id

  /// slack_csr storage with its line graph and the vertex-to-hyperedge rows (HD9).
  static void require(hypergraph_properties& props) {
    props.store_v2h = true;
    if (props.storage == hypergraph_storage::slack_csr &&
        (props.line_graph == line_graph_policy::none ||
         props.line_graph == line_graph_policy::sorted_merge)) {
      props.line_graph = line_graph_policy::slack_csr;
    }
  }
  /// hyper_sssp reads the slack_csr line graph only: the escher presets are not its.
  static bool accepts(const hypergraph_properties& props) {
    return props.storage == hypergraph_storage::slack_csr;
  }
  /// The generated hypergraphs: weights in [1, 2^28] (the medium shape the heaviest, so its long
  /// paths pass the CUDA engine's 32-bit width and fall back).
  static hypergraph_shape shape(size_class size) {
    switch (size) {
      case size_class::tiny:
        return {6, 4, 3, 9};
      case size_class::small:
        return {40, 30, 5, 1000};
      case size_class::medium:
        return {1100, 300, 8, 1 << 28};
    }
    return {};
  }
  /// compute() from the source vertex with the engine of the kit (cuda: automatic and fused run
  /// MOSP_ESCHER's kernels, operators the shared sssp engine; automatic is `kit_engine`).
  template <typename container_t>
  static result<container_t> compute(const resources& res, const container_t& h, engine e) {
    hyper_sssp::options opt;
    opt.cuda_engine = e == engine::automatic ? kit_engine : e;
    return hyper_sssp::compute(res, h, source_vertex, opt);
  }
  /// update().
  template <typename container_t>
  static stats update(const resources& res, container_t& h,
                      const typename container_t::batch_view_type& batch, result<container_t>& r) {
    return hyper_sssp::update(res, h, batch, r);
  }
  /// The distances, then the parents, on the host.
  template <typename container_t>
  static snapshot take(const resources& res, const result<container_t>& r) {
    snapshot out = to_vector(res, r.hyperedge_distances());
    const std::vector<std::int32_t> p = to_vector(res, r.hyperedge_parents());
    out.insert(out.end(), p.begin(), p.end());
    return out;
  }
  /// C5's snapshot by content: per live hyperedge (weight, distance, its members, the members of
  /// its parent; none for -1), sorted. Ids the allocator chooses do not enter it.
  template <typename container_t>
  static std::vector<std::vector<std::int64_t>> content_snapshot(const resources& res,
                                                                 const result<container_t>& r,
                                                                 const container_t& h) {
    const auto v = h.h2v();
    const std::vector<std::int64_t> off = to_vector(res, v.row_ptr);
    const std::vector<std::int32_t> mem = to_vector(res, v.col_ind);
    const auto w = h.weights();
    const std::vector<std::int64_t> d = to_vector(res, r.hyperedge_distances());
    const std::vector<std::int32_t> p = to_vector(res, r.hyperedge_parents());
    std::vector<std::vector<std::int64_t>> out;
    for (std::size_t i = 0; i + 1 < off.size(); ++i) {
      if (off[i] == off[i + 1]) continue;  // dead
      std::vector<std::int64_t> row{w[i], i < d.size() ? d[i] : 0, off[i + 1] - off[i]};
      row.insert(row.end(), mem.begin() + off[i], mem.begin() + off[i + 1]);
      const std::int32_t q = i < p.size() ? p[i] : -1;
      if (q >= 0)
        row.insert(row.end(), mem.begin() + off[static_cast<std::size_t>(q)],
                   mem.begin() + off[static_cast<std::size_t>(q) + 1]);
      out.push_back(std::move(row));
    }
    std::sort(out.begin(), out.end());
    return out;
  }
  /// The independent oracle (dyng::testing::hyper_sssp_reference) on the hypergraph's rows.
  template <typename container_t>
  static snapshot oracle_of(const resources& res, const container_t& h) {
    const auto v = h.h2v();
    const std::vector<std::int64_t> offsets = to_vector(res, v.row_ptr);
    const std::vector<std::int32_t> members = to_vector(res, v.col_ind);
    const auto w = h.weights();
    const std::vector<std::int64_t> weights(w.begin(), w.end());
    auto ref = testing::hyper_sssp_reference(offsets, members, h.num_vertices(), weights,
                                             source_vertex, true);
    snapshot out = std::move(ref.first);
    out.resize(static_cast<std::size_t>(h.hyperedge_id_bound()),
               std::numeric_limits<std::int64_t>::max() / 4);
    std::vector<std::int32_t> p = std::move(ref.second);
    p.resize(static_cast<std::size_t>(h.hyperedge_id_bound()), -1);
    out.insert(out.end(), p.begin(), p.end());
    return out;
  }
  /// The counters equal across runs and backends (fallback_used, relaxations and iterations are
  /// schedule-dependent on cuda: HD7).
  static std::vector<std::int64_t> deterministic(const stats& s) {
    return {s.line_graph_insertions, s.line_graph_deletions, s.invalidated};
  }
  /// Options compute() and set_options() reject.
  template <typename container_t>
  static std::vector<std::function<void()>> invalid_options(const resources& res,
                                                            const container_t& h) {
    return {
        [&res, &h] {
          hyper_sssp::options o;
          o.work_budget = -1;
          (void)hyper_sssp::compute(res, h, source_vertex, o);
        },
        [&res, &h] {
          hyper_sssp::options o;
          o.max_iterations = 0;
          (void)hyper_sssp::compute(res, h, source_vertex, o);
        },
        [&res, &h] {
          hyper_sssp::options o;
          o.on_budget = static_cast<hyper_sssp::budget_action>(9);
          (void)hyper_sssp::compute(res, h, source_vertex, o);
        },
        [&res, &h] {
          hyper_sssp::options o;
          o.cuda_engine = static_cast<engine>(17);
          (void)hyper_sssp::compute(res, h, source_vertex, o);
        },
        [&res, &h] { (void)hyper_sssp::compute(res, h, h.num_vertices()); },
        [&res, &h] {
          hyper_sssp::result<std::int32_t> r = hyper_sssp::compute(res, h, source_vertex);
          hyper_sssp::options o;
          o.work_budget = std::numeric_limits<double>::quiet_NaN();
          r.set_options(o);
        },
    };
  }
  /// C8: the host synchronizations of an update: none on the host backend; on cuda the engine's
  /// own run-dependent bound (one per pointer-jumping and push round, ADR 0091).
  static std::int64_t host_sync_budget(backend b) {
    return b == backend::cuda ? run_dependent_budget : 0;
  }
};

/// The registry entry of the test-only tag: hyper_sssp's (the manifest's).
inline const algorithm_info* hyper_sssp_entry() {
  return find_algorithm("hyper_sssp");
}

}  // namespace hyper_sssp_kit_detail

/// hyper_sssp from vertex 0.
template <>
struct test_traits<tags::hyper_sssp> : hyper_sssp_kit_detail::traits_of<0> {
  static constexpr std::string_view name = "hyper_sssp";  ///< as in the manifest
  /// C10's test-only partners: hyper_sssp from another source, and the kit's fake.
  using test_partners = type_list<tags::hyper_sssp_other_source, tags::test_hyper_pairs>;
};

/// hyper_sssp from vertex 1 (test-only tag, the manifest's entry; C10's partner).
template <>
struct test_traits<tags::hyper_sssp_other_source> : hyper_sssp_kit_detail::traits_of<1> {
  static constexpr std::string_view name = "hyper_sssp";  ///< the manifest's
  /// hyper_sssp's registry entry.
  static const algorithm_info* registry_entry() {
    return hyper_sssp_kit_detail::hyper_sssp_entry();
  }
  /// C10's test-only partners.
  using test_partners = type_list<tags::hyper_sssp, tags::test_hyper_pairs>;
};

/// hyper_sssp from vertex 0 on the shared engine (test-only tag, the manifest's entry; M11b).
template <>
struct test_traits<tags::hyper_sssp_shared_engine>
    : hyper_sssp_kit_detail::traits_of<0, engine::operators> {
  static constexpr std::string_view name = "hyper_sssp";  ///< the manifest's
  /// hyper_sssp's registry entry.
  static const algorithm_info* registry_entry() {
    return hyper_sssp_kit_detail::hyper_sssp_entry();
  }
  /// C10's test-only partners: the as-is engine from another source, and the kit's fake.
  using test_partners = type_list<tags::hyper_sssp_other_source, tags::test_hyper_pairs>;
};

}  // namespace dyng::conformance
