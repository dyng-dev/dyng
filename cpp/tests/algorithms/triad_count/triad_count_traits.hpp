// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_count_traits.hpp
 * @brief The conformance-kit traits of triad_count, a hypergraph algorithm (PLAN Section 8.2; the
 *        kit's hypergraph axis, ADR 0043): one suite per pattern.
 *
 * tags::triad_count (the registered algorithm) runs pattern::h_motif; the test-only tags
 * triad_count_type1, triad_count_type2 and triad_count_type3 run StatHyper's types with the same
 * registry entry (the manifest's), so the kit (C0-C12, both hypergraph instantiations, every
 * preset, mix and shape) covers all four patterns. C2 compares every state with the independent
 * oracle (dyng::testing::triad_reference). The partners of C10 are another pattern of
 * triad_count and the kit's fake test_hyper_pairs.
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
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/hypergraph/hypergraph_properties.hpp>
#include <dyng/testing/triad_reference.hpp>
#include <dyng/triad_count.hpp>

#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

namespace dyng::conformance {

namespace tags {
/// The test-only tag of triad_count with pattern::stathyper_type1.
struct triad_count_type1;
/// The test-only tag of triad_count with pattern::stathyper_type2.
struct triad_count_type2;
/// The test-only tag of triad_count with pattern::stathyper_type3.
struct triad_count_type3;
}  // namespace tags

namespace triad_kit_detail {

/// The oracle's pattern of a triad_count pattern.
constexpr testing::triad_pattern oracle_pattern(triad_count::pattern p) {
  switch (p) {
    case triad_count::pattern::h_motif:
      return testing::triad_pattern::h_motif;
    case triad_count::pattern::stathyper_type1:
      return testing::triad_pattern::stathyper_type1;
    case triad_count::pattern::stathyper_type2:
      return testing::triad_pattern::stathyper_type2;
    case triad_count::pattern::stathyper_type3:
      return testing::triad_pattern::stathyper_type3;
  }
  return testing::triad_pattern::h_motif;
}

/// The traits of one pattern (the kit compares the counts exactly).
template <triad_count::pattern kind>
struct traits_of {
  static constexpr oracle_kind oracle = oracle_kind::compute;     ///< update chain == compute
  static constexpr determinism level = determinism::exact_value;  ///< identical values
  static constexpr bool history_independent = true;               ///< counts of the hypergraph
  static constexpr int kit_openmp_threads = 8;                    ///< openmp(8) (M9a 10)
  /// The instantiated hypergraph types (DYNG_FOR_EACH_HYPERGRAPH_TYPE).
  using container_types =
      type_list<hypergraph<std::int32_t, unweighted>, hypergraph<std::int32_t, std::int32_t>>;
  template <typename container_t>
  using result = triad_count::result;           ///< the result
  using stats = triad_count::stats;             ///< update()'s stats
  using snapshot = std::vector<std::uint64_t>;  ///< the counts

  /// The hypergraph requirements: h_motif and types 2-3 read the line graph and v2h (a preset
  /// without a line graph gets automatic); type 1 needs neither.
  static void require(hypergraph_properties& props) {
    if (kind == triad_count::pattern::stathyper_type1) {
      return;
    }
    props.store_v2h = true;
    if (props.line_graph == line_graph_policy::none) {
      props.line_graph = line_graph_policy::automatic;
    }
  }
  /// One extra preset: the line graph maintained from construction (the gates' preset).
  static std::vector<hypergraph_properties> extra_properties() {
    hypergraph_properties p;
    p.line_graph = line_graph_policy::sorted_merge;
    return {p};
  }
  /// The generated hypergraphs (the medium one has more than 1,024 vertices, so the kit's skewed
  /// mix inserts a hyperedge of more than 1,024 members).
  static hypergraph_shape shape(size_class size) {
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
  /// compute() with the engine of the kit (cuda: automatic and fused run the wedge engine,
  /// operators is not_supported_error).
  template <typename container_t>
  static result<container_t> compute(const resources& res, const container_t& h, engine e) {
    triad_count::options opt;
    opt.kind = kind;
    opt.cuda_engine = e;
    return triad_count::compute(res, h, opt);
  }
  /// update().
  template <typename container_t>
  static stats update(const resources& res, container_t& h,
                      const typename container_t::batch_view_type& batch, result<container_t>& r) {
    return triad_count::update(res, h, batch, r);
  }
  /// The counts on the host.
  template <typename container_t>
  static snapshot take(const resources& res, const result<container_t>& r) {
    (void)res;
    const auto c = r.counts();
    return snapshot(c.data(), c.data() + c.size());
  }
  /// The independent oracle (dyng::testing::triad_reference) on the live hyperedges.
  template <typename container_t>
  static snapshot oracle_of(const resources& res, const container_t& h) {
    const auto list = h.to_hyperedge_list(res);
    return testing::triad_reference(list.offsets, list.members, list.num_vertices,
                                    oracle_pattern(kind));
  }
  /// The counters equal across runs and backends: the frontier sizes, the triads each phase
  /// counted (every backend is incremental from M9a step 4) and the per-bin delta.
  static std::vector<std::int64_t> deterministic(const stats& s) {
    std::vector<std::int64_t> out{s.deleted_frontier, s.inserted_frontier,
                                  static_cast<std::int64_t>(s.triads_removed),
                                  static_cast<std::int64_t>(s.triads_added)};
    out.insert(out.end(), s.delta.begin(), s.delta.end());
    return out;
  }
  /// Options compute() and set_options() reject: an invalid pattern, an invalid engine, a change
  /// of the pattern.
  template <typename container_t>
  static std::vector<std::function<void()>> invalid_options(const resources& res,
                                                            const container_t& h) {
    return {
        [&res, &h] {
          triad_count::options o;
          o.kind = static_cast<triad_count::pattern>(9);
          (void)triad_count::compute(res, h, o);
        },
        [&res, &h] {
          triad_count::options o;
          o.kind = kind;
          o.cuda_engine = static_cast<engine>(17);
          (void)triad_count::compute(res, h, o);
        },
        [&res, &h] {
          triad_count::options o;
          o.kind = kind;
          triad_count::result r = triad_count::compute(res, h, o);
          o.kind = kind == triad_count::pattern::h_motif ? triad_count::pattern::stathyper_type1
                                                         : triad_count::pattern::h_motif;
          r.set_options(o);
        },
    };
  }
  /// C8: the host synchronizations of an update: on cuda the wedge count of each phase and the
  /// read of the bins (type 1 launches no wedges: the bins only); none on the host backends.
  static std::int64_t host_sync_budget(backend b) {
    if (b != backend::cuda) {
      return 0;
    }
    return kind == triad_count::pattern::stathyper_type1 ? 1 : 3;
  }
};

/// The registry entry of the test-only tags: triad_count's (the manifest's).
inline const algorithm_info* triad_count_entry() {
  return find_algorithm("triad_count");
}

}  // namespace triad_kit_detail

/// triad_count, pattern h_motif.
template <>
struct test_traits<tags::triad_count> : triad_kit_detail::traits_of<triad_count::pattern::h_motif> {
  static constexpr std::string_view name = "triad_count";  ///< as in the manifest
  /// C10's test-only partners: another pattern of triad_count and the kit's fake.
  using test_partners = type_list<tags::triad_count_type1, tags::test_hyper_pairs>;
};

/// triad_count, pattern stathyper_type1 (test-only tag, the manifest's entry).
template <>
struct test_traits<tags::triad_count_type1>
    : triad_kit_detail::traits_of<triad_count::pattern::stathyper_type1> {
  static constexpr std::string_view name = "triad_count";  ///< the manifest's
  /// triad_count's registry entry.
  static const algorithm_info* registry_entry() {
    return triad_kit_detail::triad_count_entry();
  }
  /// C10's test-only partners.
  using test_partners = type_list<tags::triad_count_type2, tags::test_hyper_pairs>;
};

/// triad_count, pattern stathyper_type2 (test-only tag).
template <>
struct test_traits<tags::triad_count_type2>
    : triad_kit_detail::traits_of<triad_count::pattern::stathyper_type2> {
  static constexpr std::string_view name = "triad_count";  ///< the manifest's
  /// triad_count's registry entry.
  static const algorithm_info* registry_entry() {
    return triad_kit_detail::triad_count_entry();
  }
  /// C10's test-only partners.
  using test_partners = type_list<tags::triad_count_type3, tags::test_hyper_pairs>;
};

/// triad_count, pattern stathyper_type3 (test-only tag).
template <>
struct test_traits<tags::triad_count_type3>
    : triad_kit_detail::traits_of<triad_count::pattern::stathyper_type3> {
  static constexpr std::string_view name = "triad_count";  ///< the manifest's
  /// triad_count's registry entry.
  static const algorithm_info* registry_entry() {
    return triad_kit_detail::triad_count_entry();
  }
  /// C10's test-only partners.
  using test_partners = type_list<tags::triad_count, tags::test_hyper_pairs>;
};

}  // namespace dyng::conformance
