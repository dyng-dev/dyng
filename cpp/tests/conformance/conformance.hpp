// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file conformance.hpp
 * @brief The conformance kit (PLAN Section 8.2): checks C0-C12 for every registered algorithm,
 *        every backend of the build and every instantiated graph type, from the algorithm's
 *        test_traits (conformance/test_traits.hpp) and one line
 *
 *            DYNG_CONFORMANCE_SUITE(<name>);
 *
 *        in cpp/tests/algorithms/<name>/<name>_conformance_test.cpp.
 *
 * | Check | Property |
 * |---|---|
 * | C0  | the traits agree with the registry (the manifest): oracle kind, determinism level, a sequential backend first (invariant I8); a backend of this binary that the manifest does not list rejects compute() with not_supported_error (so the manifest cannot drop a backend the library runs) |
 * | C1  | update(empty batch) leaves the result unchanged, affected == 0 |
 * | C2  | the oracle over random graphs x sizes x batch mixes x 3 batches: oracle = compute: after every batch the result equals compute(G_i) (and an independent oracle, if the traits have one); oracle = reference: both are near the converged reference |
 * | C3  | the backends agree (sequential == OpenMP == CUDA) at the declared level, deterministic counters included |
 * | C4  | fused engine == operators engine where a backend has both |
 * | C5  | a batch followed by its inverse returns the original result (history-independent algorithms) |
 * | C6  | two runs give identical output and deterministic counters |
 * | C7  | invalid input is rejected with the documented exception (nothing changes) or counted as skipped |
 * | C8  | budgets (DYNG_DEBUG_BUDGETS builds): once reserved, the algorithm phase allocates nothing (library memory resources and, where the counting operator new is linked, the host heap) and synchronizes at most the problem's budget; the commit is reported separately |
 * | C9  | stats sanity: 0 <= affected <= n; applied + skipped == requested; the vertex count |
 * | C10 | dyng::update(res, g, b, r1, r2) equals each update alone on a copy of the graph, for every other registered algorithm on the same graph type |
 * | C11 | stale results are detected (a separate g.apply(), a result of another graph) |
 * | C12 | stream ordering on non-default streams (CUDA) |
 *
 * The suite source is compiled twice, like the algorithm suites (support/gtest_helpers.hpp): into
 * the host executable (label cpu: sequential and OpenMP) and with DYNG_TEST_CUDA=1 into the CUDA
 * executable (label gpu: cuda; the cross-backend check compares it with the host backends).
 * Every randomized check prints its seed; DYNG_TEST_SEED=<seed> replays one.
 */
#pragma once

#include "conformance/allocation_counter.hpp"
#include "conformance/generators.hpp"
#include "conformance/registry.hpp"
#include "conformance/test_traits.hpp"
#include "conformance/type_list.hpp"
#include "core/budget_counters.hpp"
#include "framework/budgets.hpp"
#include "support/gtest_helpers.hpp"
#include "support/test_seeds.hpp"

#include <dyng/config.hpp>
#include <dyng/core/array_view.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/registry.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stream.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#if defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace dyng::conformance {

/// The registered algorithms built into this test binary (their test_traits are defined).
using registered_algorithms = typename filter<manifest_algorithms, has_test_traits>::type;

/// One typed case of the suite: an algorithm on one of its graph types.
template <typename tag_t, typename graph_t>
struct suite_case {
  using tag = tag_t;                  ///< the algorithm's tag
  using traits = test_traits<tag_t>;  ///< its traits
  using graph_type = graph_t;         ///< the graph type
};

/// The cases of an algorithm: one per graph type of its traits.
template <typename tag_t>
struct cases_of {
  template <typename graph_t>
  using make = suite_case<tag_t, graph_t>;  ///< the case of one graph type
  /// ::testing::Types of the cases
  using type = typename as_gtest_types<
      typename transform<typename test_traits<tag_t>::graph_types, make>::type>::type;
};

/// The name of a type in a case name.
template <typename type_t>
std::string type_name() {
  if constexpr (is_unweighted_v<type_t>) {
    return "unweighted";
  } else {
    return (std::is_signed_v<type_t> ? "i" : "u") + std::to_string(8 * sizeof(type_t));
  }
}

/// GoogleTest names of the cases: "v<V>_e<E>_w<W>", e.g. v32_e64_unweighted.
struct case_name {
  template <typename case_t>
  static std::string GetName(int /*index*/) {  // NOLINT(readability-identifier-naming): gtest API
    using graph_t = typename case_t::graph_type;
    return "v" + type_name<typename graph_t::vertex_type>().substr(1) + "_e" +
           type_name<typename graph_t::edge_type>().substr(1) + "_w" +
           (is_unweighted_v<typename graph_t::weight_type>
                ? std::string("unweighted")
                : type_name<typename graph_t::weight_type>().substr(1));
  }
};

// ------------------------------------------------------------------------------------------------
// Registration checks (invariant I8): what DYNG_CONFORMANCE_SUITE requires at compile time
// ------------------------------------------------------------------------------------------------

namespace kit_detail {

template <typename traits_t, typename graph_t, typename = void>
struct has_compute : std::false_type {};
template <typename traits_t, typename graph_t>
struct has_compute<
    traits_t, graph_t,
    std::void_t<decltype(traits_t::template compute<graph_t>(
        std::declval<const resources&>(), std::declval<const graph_t&>(), engine::automatic))>>
    : std::true_type {};

template <typename traits_t, typename graph_t, typename = void>
struct has_update : std::false_type {};
template <typename traits_t, typename graph_t>
struct has_update<traits_t, graph_t,
                  std::void_t<decltype(traits_t::template update<graph_t>(
                      std::declval<const resources&>(), std::declval<graph_t&>(),
                      std::declval<const edge_batch_view<typename graph_t::vertex_type,
                                                         typename graph_t::weight_type>&>(),
                      std::declval<typename traits_t::template result<graph_t>&>()))>>
    : std::true_type {};

template <typename traits_t, typename = void>
struct has_compare : std::false_type {};
template <typename traits_t>
struct has_compare<traits_t, std::void_t<decltype(traits_t::compare(
                                 std::declval<const typename traits_t::snapshot&>(),
                                 std::declval<const typename traits_t::snapshot&>()))>>
    : std::true_type {};

template <typename traits_t, typename graph_t, typename = void>
struct has_oracle_of : std::false_type {};
template <typename traits_t, typename graph_t>
struct has_oracle_of<traits_t, graph_t,
                     std::void_t<decltype(traits_t::template oracle_of<graph_t>(
                         std::declval<const resources&>(), std::declval<const graph_t&>()))>>
    : std::true_type {};

template <typename traits_t, typename graph_t, typename = void>
struct has_near_reference : std::false_type {};
template <typename traits_t, typename graph_t>
struct has_near_reference<traits_t, graph_t,
                          std::void_t<decltype(traits_t::template near_reference<graph_t>(
                              std::declval<const resources&>(), std::declval<const graph_t&>(),
                              std::declval<const typename traits_t::snapshot&>()))>>
    : std::true_type {};

template <typename traits_t, typename = void>
struct has_extra_properties : std::false_type {};
template <typename traits_t>
struct has_extra_properties<traits_t, std::void_t<decltype(traits_t::extra_properties())>>
    : std::true_type {};

template <typename traits_t, typename = void>
struct has_num_weights : std::false_type {};
template <typename traits_t>
struct has_num_weights<traits_t, std::void_t<decltype(traits_t::num_weights)>> : std::true_type {};

/// The weight columns of the kit's weighted graphs: test_traits::num_weights, default 1.
template <typename traits_t>
constexpr int weight_columns() {
  if constexpr (has_num_weights<traits_t>::value) {
    return traits_t::num_weights;
  } else {
    return 1;
  }
}

template <typename traits_t, typename = void>
struct has_run_budget : std::false_type {};
template <typename traits_t>
struct has_run_budget<traits_t,
                      std::void_t<decltype(traits_t::host_sync_budget(
                          backend::cuda, std::declval<const typename traits_t::stats&>()))>>
    : std::true_type {};

/// C8's expected host synchronizations of one update: test_traits::host_sync_budget(backend,
/// stats) when the traits have it (a budget that depends on the engine that ran and its counters),
/// else host_sync_budget(backend). run_dependent_budget: C8 checks the problem's own budget.
template <typename traits_t>
std::int64_t expected_host_syncs(backend b, const typename traits_t::stats& s) {
  if constexpr (has_run_budget<traits_t>::value) {
    return traits_t::host_sync_budget(b, s);
  } else {
    (void)s;
    return traits_t::host_sync_budget(b);
  }
}

template <typename traits_t, typename = void>
struct has_oracle_kind : std::false_type {};
template <typename traits_t>
struct has_oracle_kind<traits_t, std::void_t<decltype(traits_t::oracle)>>
    : std::is_same<std::decay_t<decltype(traits_t::oracle)>, oracle_kind> {};

template <typename traits_t, typename = void>
struct has_level : std::false_type {};
template <typename traits_t>
struct has_level<traits_t, std::void_t<decltype(traits_t::level)>>
    : std::is_same<std::decay_t<decltype(traits_t::level)>, determinism> {};

template <typename stats_t, typename = void>
struct has_batch : std::false_type {};
template <typename stats_t>
struct has_batch<stats_t, std::void_t<decltype(std::declval<stats_t&>().batch)>>
    : std::is_same<std::decay_t<decltype(std::declval<stats_t&>().batch)>, apply_summary> {};

}  // namespace kit_detail

/**
 * @brief The compile-time registration rules of one graph type of an algorithm.
 * @tparam traits_t The algorithm's traits.
 * @tparam graph_t  One of its graph types.
 * @return true.
 */
template <typename traits_t, typename graph_t>
constexpr bool registration_check_graph() {
  static_assert(kit_detail::has_compute<traits_t, graph_t>::value,
                "dyng conformance kit: an algorithm cannot be registered without compute() "
                "(invariant I8): test_traits::compute<graph_t>(res, g, engine) must exist for "
                "every graph type of graph_types");
  static_assert(kit_detail::has_update<traits_t, graph_t>::value,
                "dyng conformance kit: test_traits::update<graph_t>(res, g, batch, result) must "
                "exist for every graph type of graph_types");
  static_assert(!kit_detail::has_oracle_kind<traits_t>::value ||
                    traits_t::oracle != oracle_kind::reference ||
                    kit_detail::has_near_reference<traits_t, graph_t>::value,
                "dyng conformance kit: oracle_kind::reference needs "
                "test_traits::near_reference<graph_t>(res, g, snapshot) (the converged reference "
                "and its tolerance)");
  return true;
}

/// @copydoc registration_check_graph (every graph type of a list).
template <typename traits_t, typename... graphs_t>
constexpr bool registration_check_graphs(type_list<graphs_t...> /*list*/) {
  return (registration_check_graph<traits_t, graphs_t>() && ...);
}

/**
 * @brief The compile-time registration rules of an algorithm (invariant I8, PLAN Section 8.2):
 *        DYNG_CONFORMANCE_SUITE fails to compile with these messages.
 * @tparam tag_t The algorithm's tag.
 * @return true.
 */
template <typename tag_t>
constexpr bool registration_check() {
  static_assert(has_traits<tag_t>::value,
                "dyng conformance kit: the algorithm has no test_traits specialisation; write "
                "cpp/tests/algorithms/<name>/<name>_traits.hpp (docs/developer/conformance.md)");
  using traits_t = test_traits<tag_t>;
  static_assert(traits_t::graph_types::size > 0,
                "dyng conformance kit: test_traits::graph_types lists no graph type");
  static_assert(kit_detail::has_oracle_kind<traits_t>::value,
                "dyng conformance kit: test_traits must declare its oracle kind, `static constexpr "
                "oracle_kind oracle = oracle_kind::compute;` (or reference); invariant I8");
  static_assert(kit_detail::has_level<traits_t>::value,
                "dyng conformance kit: test_traits must declare its determinism level, `static "
                "constexpr determinism level = ...;`");
  static_assert(kit_detail::has_batch<typename traits_t::stats>::value,
                "dyng conformance kit: the algorithm's stats must derive from update_stats and "
                "contain `apply_summary batch` (ADR 0006)");
  static_assert(!kit_detail::has_level<traits_t>::value ||
                    traits_t::level != determinism::tolerance ||
                    kit_detail::has_compare<traits_t>::value,
                "dyng conformance kit: an algorithm with determinism::tolerance must provide "
                "test_traits::compare(expected, actual) with its tolerance");
  return registration_check_graphs<traits_t>(typename traits_t::graph_types{});
}

// ------------------------------------------------------------------------------------------------
// Helpers of the checks
// ------------------------------------------------------------------------------------------------

namespace kit_detail {

/// The backends of the algorithm that this test binary runs (suite_backends() of the binary,
/// restricted to the manifest's backends).
template <typename traits_t>
std::vector<backend> backends() {
  std::vector<backend> out;
  const algorithm_info* info = find_algorithm(traits_t::name);
  if (info == nullptr) {
    return out;
  }
  for (const backend b : test::suite_backends()) {
    if (std::find(info->backends.begin(), info->backends.end(), b) != info->backends.end()) {
      out.push_back(b);
    }
  }
  return out;
}

/// The backends a cross-backend comparison runs on (sequential first).
template <typename traits_t>
std::vector<backend> comparison_backends() {
  std::vector<backend> out;
  const algorithm_info* info = find_algorithm(traits_t::name);
  if (info == nullptr) {
    return out;
  }
  for (const backend b : test::comparison_backends()) {
    if (std::find(info->backends.begin(), info->backends.end(), b) != info->backends.end()) {
      out.push_back(b);
    }
  }
  return out;
}

/// Resources for a backend (OpenMP with four threads, so the parallel paths run).
inline resources resources_for(backend b) {
  return test::make_resources(b, b == backend::openmp ? 4 : 0);
}

/// Two snapshots at the algorithm's level (traits::compare, else ==).
template <typename traits_t>
::testing::AssertionResult same(const typename traits_t::snapshot& expected,
                                const typename traits_t::snapshot& actual) {
  if constexpr (has_compare<traits_t>::value) {
    return traits_t::compare(expected, actual);
  } else {
    if (expected == actual) {
      return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << "expected " << ::testing::PrintToString(expected)
                                         << "\n  actual " << ::testing::PrintToString(actual);
  }
}

/// A property preset of the checks.
struct preset {
  std::string label;       ///< for traces
  graph_properties props;  ///< the graph's properties (the algorithm's requirements applied)
};

/// The presets: the defaults, set() semantics, and the traits' extras; each with require().
template <typename traits_t>
std::vector<preset> presets() {
  std::vector<preset> out;
  graph_properties plain;
  traits_t::require(plain);
  out.push_back({"upsert_last_wins()", plain});
  graph_properties sets = plain;
  sets.semantics = batch_semantics::set();
  out.push_back({"set()", sets});
  if constexpr (has_extra_properties<traits_t>::value) {
    int i = 0;
    for (graph_properties p : traits_t::extra_properties()) {
      traits_t::require(p);
      out.push_back({"extra properties " + std::to_string(i++), p});
    }
  }
  return out;
}

/// Whether a mix applies to a graph type and preset (reweight: weighted graphs, upsert only).
template <typename graph_t>
bool mix_applies(batch_mix mix, const graph_properties& props) {
  if (mix == batch_mix::reweight) {
    return !is_unweighted_v<typename graph_t::weight_type> && !props.semantics.as_sets &&
           props.semantics.on_existing_insert == batch_semantics::existing_insert::upsert;
  }
  return true;
}

/// One algorithm result kept up to date on one graph (the unit of every check).
template <typename case_t>
struct chain {
  using traits = typename case_t::traits;                            ///< the traits
  using graph_type = typename case_t::graph_type;                    ///< the graph type
  using result_type = typename traits::template result<graph_type>;  ///< the result type
  using stats_type = typename traits::stats;                         ///< the stats type
  using batch_type = edge_batch<typename graph_type::vertex_type,
                                typename graph_type::weight_type>;  ///< the batch type

  resources res;                 ///< the backend
  std::optional<graph_type> g;   ///< the graph
  std::optional<result_type> r;  ///< the result

  /// compute() on the model's graph.
  chain(resources on, const graph_model<graph_type>& model, const graph_properties& props,
        engine e = engine::automatic)
      : res(std::move(on)) {
    g.emplace(model.build(res, props));
    r.emplace(traits::template compute<graph_type>(res, *g, e));
  }

  /// update() with a batch.
  stats_type step(const batch_type& b) {
    return traits::template update<graph_type>(res, *g, b.view(), *r);
  }

  /// A host snapshot of the result.
  [[nodiscard]] typename traits::snapshot take() const {
    return traits::template take<graph_type>(res, *r);
  }

  /// compute() on the current graph, as a snapshot.
  [[nodiscard]] typename traits::snapshot recompute() const {
    return traits::template take<graph_type>(
        res, traits::template compute<graph_type>(res, *g, engine::automatic));
  }
};

/// C2's oracle check of one chain after a batch (at the declared oracle kind).
template <typename case_t>
void expect_oracle(const chain<case_t>& c) {
  using traits = typename case_t::traits;
  using graph_t = typename case_t::graph_type;
  const auto updated = c.take();
  if constexpr (traits::oracle == oracle_kind::compute) {
    const auto computed = c.recompute();
    EXPECT_TRUE(same<traits>(computed, updated)) << "update chain != compute(G_i)";
    if constexpr (has_oracle_of<traits, graph_t>::value) {
      EXPECT_TRUE(same<traits>(traits::template oracle_of<graph_t>(c.res, *c.g), computed))
          << "compute(G_i) != the independent oracle";
    }
  } else {
    EXPECT_TRUE(traits::template near_reference<graph_t>(c.res, *c.g, updated))
        << "the update chain is not near the converged reference";
    EXPECT_TRUE(traits::template near_reference<graph_t>(c.res, *c.g, c.recompute()))
        << "compute(G_i) is not near the converged reference";
  }
}

/// The deterministic counters of a stats object, with the batch summary's.
template <typename traits_t>
std::vector<std::int64_t> counters(const typename traits_t::stats& s) {
  std::vector<std::int64_t> out = traits_t::deterministic(s);
  const apply_summary& b = s.batch;
  out.insert(out.end(), {s.affected, b.inserted_edges, b.updated_edges, b.deleted_edges,
                         b.ignored_deletions, b.dropped_self_loops, b.cancelled_pairs,
                         b.inserted_vertices, b.num_vertices_after, b.ignored_insertions});
  return out;
}

/// A random model of a size class.
template <typename case_t>
graph_model<typename case_t::graph_type> model_of(size_class size, std::mt19937_64& rng) {
  graph_model<typename case_t::graph_type> m =
      random_model<typename case_t::graph_type>(case_t::traits::shape(size), rng);
  m.num_weights = weight_columns<typename case_t::traits>();
  return m;
}

/**
 * C4 on `backends`: where a backend runs both a fused and an operators engine (compute() with
 * engine::fused and engine::operators, and the updates report different stats::engine_used), the
 * two results agree at the traits' level and so do the deterministic counters, after compute()
 * and after each of three consecutive batches, for every preset, the small and medium sizes and
 * every applicable batch mix. Returns whether some backend compared (C4 is skipped otherwise);
 * stops at the first disagreement. Also used by the kit's own test with a fake two-engine
 * algorithm (engines_agree_test.cpp).
 */
template <typename case_t>
bool engines_agree(const std::vector<backend>& backends) {
  using traits = typename case_t::traits;
  using graph_t = typename case_t::graph_type;
  bool compared = false;
  for (const backend b : backends) {
    SCOPED_TRACE(std::string(to_string(b)));
    for (const preset& p : presets<traits>()) {
      SCOPED_TRACE(p.label);
      for (const size_class size : {size_class::small, size_class::medium}) {
        for (const batch_mix mix : all_mixes()) {
          if (!mix_applies<graph_t>(mix, p.props)) {
            continue;
          }
          SCOPED_TRACE("size " + std::to_string(static_cast<int>(size)) + ", mix " +
                       std::string(to_string(mix)));
          std::mt19937_64 rng(4000 + static_cast<std::uint64_t>(size) * 17 +
                              static_cast<std::uint64_t>(mix));
          graph_model<graph_t> model = model_of<case_t>(size, rng);
          std::optional<chain<case_t>> fused;
          std::optional<chain<case_t>> operators;
          try {
            fused.emplace(resources_for(b), model, p.props, engine::fused);
            operators.emplace(resources_for(b), model, p.props, engine::operators);
          } catch (const not_supported_error&) {
            break;  // this backend has one engine
          }
          if (!same<traits>(fused->take(), operators->take())) {
            ADD_FAILURE() << "compute(): the fused and the operators engine disagree";
            return true;
          }
          for (int step = 0; step < 3; ++step) {
            SCOPED_TRACE("batch " + std::to_string(step));
            const generated_batch<graph_t> gen = random_batch(model, mix, rng);
            typename traits::stats sf;
            typename traits::stats so;
            try {
              sf = fused->step(gen.batch);
              so = operators->step(gen.batch);
            } catch (const not_supported_error&) {
              break;
            }
            if (sf.engine_used == so.engine_used) {
              break;  // the backend ignores the choice (one engine)
            }
            compared = true;
            const bool agree = same<traits>(fused->take(), operators->take());
            EXPECT_TRUE(same<traits>(fused->take(), operators->take()));
            EXPECT_EQ(traits::deterministic(sf), traits::deterministic(so));
            if (!agree || traits::deterministic(sf) != traits::deterministic(so)) {
              return true;  // the first disagreement is enough (its trace names the case)
            }
          }
        }
      }
    }
  }
  return compared;
}

}  // namespace kit_detail

// ------------------------------------------------------------------------------------------------
// The suite
// ------------------------------------------------------------------------------------------------

/// The typed suite of the kit; DYNG_CONFORMANCE_SUITE instantiates it for one algorithm.
template <typename case_t>
class conformance : public ::testing::Test {};

TYPED_TEST_SUITE_P(conformance);

// C0: the traits agree with the registry (the manifest), which lists a sequential backend first.
TYPED_TEST_P(conformance, C0_TheTraitsAgreeWithTheRegistry) {
  using traits = typename TypeParam::traits;
  const algorithm_info* info = find_algorithm(traits::name);
  ASSERT_NE(info, nullptr) << traits::name << " is not in dyng::algorithms() (its manifest, "
                           << "scripts/regen.py, or DYNG_ALGORITHMS)";
  EXPECT_EQ(info->oracle, traits::oracle) << "the manifest's oracle differs from the traits'";
  EXPECT_EQ(info->determinism_level, traits::level) << "the manifest's determinism differs";
  ASSERT_FALSE(info->backends.empty());
  EXPECT_EQ(info->backends.front(), backend::sequential)
      << "invariant I8: every algorithm has the sequential reference backend";
  EXPECT_EQ(info->container, container_kind::graph);
  // The kit runs the manifest's backends only: a backend the library runs but the manifest omits
  // would lose its conformance coverage silently (and the registry would misreport it).
  using graph_t = typename TypeParam::graph_type;
  for (const backend b : test::suite_backends()) {
    if (std::find(info->backends.begin(), info->backends.end(), b) != info->backends.end()) {
      continue;
    }
    SCOPED_TRACE(std::string(to_string(b)) + " (not in the manifest)");
    std::mt19937_64 rng(10);
    const resources res = kit_detail::resources_for(b);
    const graph_t g = kit_detail::model_of<TypeParam>(size_class::tiny, rng)
                          .build(res, kit_detail::presets<traits>().front().props);
    EXPECT_THROW((void)traits::template compute<graph_t>(res, g, engine::automatic),
                 not_supported_error)
        << traits::name << " runs on " << to_string(b)
        << ", which its manifest does not list in `backends`";
  }
  if (kit_detail::backends<traits>().empty()) {
    GTEST_SKIP() << "no backend of " << traits::name << " in this test binary";
  }
}

// C1: an empty batch changes nothing.
TYPED_TEST_P(conformance, C1_AnEmptyBatchChangesNothing) {
  using traits = typename TypeParam::traits;
  using graph_t = typename TypeParam::graph_type;
  for (const backend b : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(b)));
    for (const kit_detail::preset& p : kit_detail::presets<traits>()) {
      SCOPED_TRACE(p.label);
      for (const size_class size : {size_class::tiny, size_class::small}) {
        std::mt19937_64 rng(11);
        kit_detail::chain<TypeParam> c(kit_detail::resources_for(b),
                                       kit_detail::model_of<TypeParam>(size, rng), p.props);
        const auto before = c.take();
        const std::uint64_t version = c.g->version();
        const typename kit_detail::chain<TypeParam>::batch_type empty;
        const auto s = c.step(empty);
        EXPECT_EQ(s.affected, 0);
        EXPECT_TRUE(kit_detail::same<traits>(before, c.take()));
        EXPECT_EQ(s.batch.inserted_edges + s.batch.deleted_edges + s.batch.updated_edges, 0);
        EXPECT_EQ(c.r->graph_version(), c.g->version());
        EXPECT_GE(c.g->version(), version);
        (void)sizeof(graph_t);
      }
    }
  }
}

// C2: the oracle over random graphs x sizes x batch mixes x three consecutive batches.
TYPED_TEST_P(conformance, C2_UpdateChainsEqualTheOracle) {
  using traits = typename TypeParam::traits;
  using graph_t = typename TypeParam::graph_type;
  for (const backend b : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(b)));
    for (const kit_detail::preset& p : kit_detail::presets<traits>()) {
      SCOPED_TRACE(p.label);
      for (const std::uint64_t seed : test::test_seeds(2000, 2)) {
        SCOPED_TRACE(test::seed_trace(seed));
        for (const size_class size : {size_class::tiny, size_class::small, size_class::medium}) {
          for (const batch_mix mix : all_mixes()) {
            if (!kit_detail::mix_applies<graph_t>(mix, p.props)) {
              continue;
            }
            SCOPED_TRACE("size " + std::to_string(static_cast<int>(size)) + ", mix " +
                         std::string(to_string(mix)));
            std::mt19937_64 rng(seed * 131 + static_cast<std::uint64_t>(size) * 17 +
                                static_cast<std::uint64_t>(mix));
            graph_model<graph_t> model = kit_detail::model_of<TypeParam>(size, rng);
            kit_detail::chain<TypeParam> c(kit_detail::resources_for(b), model, p.props);
            kit_detail::expect_oracle(c);
            for (int step = 0; step < 3; ++step) {
              SCOPED_TRACE("batch " + std::to_string(step));
              const generated_batch<graph_t> gen = random_batch(model, mix, rng);
              (void)c.step(gen.batch);
              ASSERT_EQ(static_cast<std::int64_t>(c.g->num_vertices()),
                        static_cast<std::int64_t>(model.num_vertices));
              ASSERT_EQ(static_cast<std::int64_t>(c.g->num_edges()),
                        static_cast<std::int64_t>(model.weights.size()));
              kit_detail::expect_oracle(c);
              if (::testing::Test::HasFailure()) {
                return;  // the first failing chain is enough (its seed is in the trace)
              }
            }
          }
        }
      }
    }
  }
}

// C3: the backends agree at the declared level (sequential first, the reference).
TYPED_TEST_P(conformance, C3_BackendsAgree) {
  using traits = typename TypeParam::traits;
  using graph_t = typename TypeParam::graph_type;
  const std::vector<backend> backends = kit_detail::comparison_backends<traits>();
  if (backends.size() < 2) {
    GTEST_SKIP() << "only one backend of " << traits::name << " in this build";
  }
  for (const kit_detail::preset& p : kit_detail::presets<traits>()) {
    SCOPED_TRACE(p.label);
    for (const std::uint64_t seed : test::test_seeds(3000, 2)) {
      SCOPED_TRACE(test::seed_trace(seed));
      for (const batch_mix mix :
           {batch_mix::mixed, batch_mix::heavy, batch_mix::reweight, batch_mix::grow}) {
        if (!kit_detail::mix_applies<graph_t>(mix, p.props)) {
          continue;
        }
        SCOPED_TRACE("mix " + std::string(to_string(mix)));
        std::mt19937_64 rng(seed * 7 + static_cast<std::uint64_t>(mix));
        graph_model<graph_t> model = kit_detail::model_of<TypeParam>(size_class::small, rng);
        std::vector<kit_detail::chain<TypeParam>> chains;
        chains.reserve(backends.size());
        for (const backend b : backends) {
          chains.emplace_back(kit_detail::resources_for(b), model, p.props);
        }
        for (std::size_t i = 1; i < chains.size(); ++i) {
          EXPECT_TRUE(kit_detail::same<traits>(chains[0].take(), chains[i].take()))
              << "compute() on " << to_string(backends[i]) << " differs from sequential";
        }
        for (int step = 0; step < 3; ++step) {
          SCOPED_TRACE("batch " + std::to_string(step));
          const generated_batch<graph_t> gen = random_batch(model, mix, rng);
          const auto reference = kit_detail::counters<traits>(chains[0].step(gen.batch));
          const auto expected = chains[0].take();
          for (std::size_t i = 1; i < chains.size(); ++i) {
            SCOPED_TRACE(std::string(to_string(backends[i])));
            EXPECT_EQ(kit_detail::counters<traits>(chains[i].step(gen.batch)), reference);
            EXPECT_TRUE(kit_detail::same<traits>(expected, chains[i].take()));
          }
        }
      }
    }
  }
}

// C4: the fused engine equals the operators engine where a backend has both.
TYPED_TEST_P(conformance, C4_EnginesAgree) {
  using traits = typename TypeParam::traits;
  if (!kit_detail::engines_agree<TypeParam>(kit_detail::backends<traits>())) {
    GTEST_SKIP() << "no backend of " << traits::name
                 << " in this build has both a fused and an operators engine";
  }
}

// C5: a batch followed by its inverse returns the original result.
TYPED_TEST_P(conformance, C5_ABatchAndItsInverseCancel) {
  using traits = typename TypeParam::traits;
  using graph_t = typename TypeParam::graph_type;
  if (!traits::history_independent) {
    GTEST_SKIP() << traits::name << " is not history-independent";
  }
  for (const backend b : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(b)));
    for (const kit_detail::preset& p : kit_detail::presets<traits>()) {
      SCOPED_TRACE(p.label);
      for (const std::uint64_t seed : test::test_seeds(5000, 2)) {
        SCOPED_TRACE(test::seed_trace(seed));
        for (const batch_mix mix :
             {batch_mix::insert_only, batch_mix::delete_only, batch_mix::mixed, batch_mix::local,
              batch_mix::heavy, batch_mix::reweight}) {
          if (!kit_detail::mix_applies<graph_t>(mix, p.props)) {
            continue;
          }
          SCOPED_TRACE("mix " + std::string(to_string(mix)));
          std::mt19937_64 rng(seed * 3 + static_cast<std::uint64_t>(mix));
          graph_model<graph_t> model = kit_detail::model_of<TypeParam>(size_class::small, rng);
          kit_detail::chain<TypeParam> c(kit_detail::resources_for(b), model, p.props);
          const auto before = c.take();
          const generated_batch<graph_t> gen = random_batch(model, mix, rng);
          ASSERT_TRUE(gen.invertible);
          (void)c.step(gen.batch);
          (void)c.step(gen.inverse);
          EXPECT_TRUE(kit_detail::same<traits>(before, c.take()));
        }
      }
    }
  }
}

// C6: two runs give identical output (and deterministic counters).
TYPED_TEST_P(conformance, C6_RunsAreDeterministic) {
  using traits = typename TypeParam::traits;
  using graph_t = typename TypeParam::graph_type;
  for (const backend b : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(b)));
    for (const kit_detail::preset& p : kit_detail::presets<traits>()) {
      SCOPED_TRACE(p.label);
      for (const size_class size : {size_class::small, size_class::medium}) {
        for (const batch_mix mix : {batch_mix::mixed, batch_mix::heavy, batch_mix::grow}) {
          SCOPED_TRACE("mix " + std::string(to_string(mix)));
          std::mt19937_64 rng(6000 + static_cast<std::uint64_t>(mix));
          graph_model<graph_t> model = kit_detail::model_of<TypeParam>(size, rng);
          kit_detail::chain<TypeParam> first(kit_detail::resources_for(b), model, p.props);
          kit_detail::chain<TypeParam> second(kit_detail::resources_for(b), model, p.props);
          EXPECT_TRUE(kit_detail::same<traits>(first.take(), second.take()));
          for (int step = 0; step < 3; ++step) {
            const generated_batch<graph_t> gen = random_batch(model, mix, rng);
            EXPECT_EQ(kit_detail::counters<traits>(first.step(gen.batch)),
                      kit_detail::counters<traits>(second.step(gen.batch)));
            EXPECT_TRUE(kit_detail::same<traits>(first.take(), second.take()));
          }
        }
      }
    }
  }
}

// C7: invalid input is rejected with the documented exception (and changes nothing), or counted
// as skipped per the batch semantics.
TYPED_TEST_P(conformance, C7_InvalidInputIsRejectedOrCounted) {
  using traits = typename TypeParam::traits;
  using graph_t = typename TypeParam::graph_type;
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;
  using batch_t = edge_batch<vertex_t, weight_t>;
  constexpr bool weighted = !is_unweighted_v<weight_t>;
  // Batches with the kit graph's weight columns (test_traits::num_weights), all weights 1.
  constexpr int columns = kit_detail::weight_columns<traits>();
  const auto make_batch = []() { return weighted ? batch_t(columns) : batch_t(); };
  const auto insert = [](batch_t& b, vertex_t u, vertex_t v) {
    if constexpr (weighted) {
      const std::vector<weight_t> w(static_cast<std::size_t>(columns), weight_t{1});
      b.insert_edge(u, v, host_view(w));
    } else {
      b.insert_edge(u, v);
    }
  };
  for (const backend be : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(be)));
    for (const kit_detail::preset& p : kit_detail::presets<traits>()) {
      SCOPED_TRACE(p.label);
      std::mt19937_64 rng(7000);
      graph_model<graph_t> model = kit_detail::model_of<TypeParam>(size_class::small, rng);
      kit_detail::chain<TypeParam> c(kit_detail::resources_for(be), model, p.props);
      const auto before = c.take();
      // Rejected: nothing changes, and the result stays current and usable.
      const auto expect_rejected = [&](const batch_t& b, const char* what) {
        SCOPED_TRACE(what);
        const std::uint64_t version = c.g->version();
        EXPECT_THROW((void)c.step(b), invalid_argument_error);
        EXPECT_EQ(c.g->version(), version);
        EXPECT_EQ(c.r->graph_version(), c.g->version());
        EXPECT_TRUE(kit_detail::same<traits>(before, c.take()));
      };
      {
        batch_t b = make_batch();
        insert(b, vertex_t{-1}, vertex_t{0});
        expect_rejected(b, "an insertion with a negative id");
      }
      {
        batch_t b = make_batch();
        b.delete_edge(vertex_t{0}, vertex_t{-2});
        expect_rejected(b, "a deletion with a negative id");
      }
      if constexpr (weighted) {
        // One weight column more than the graph has (K + 1 for a K-column graph).
        batch_t b(columns + 1);
        const std::vector<weight_t> w(static_cast<std::size_t>(columns) + 1, weight_t{1});
        b.insert_edge(vertex_t{0}, vertex_t{1}, host_view(w));
        expect_rejected(b, "a batch with one weight column more than the graph has");
      }
      // Bad options: each must throw invalid_argument_error.
      for (const std::function<void()>& bad : traits::invalid_options(c.res, *c.g)) {
        EXPECT_THROW(bad(), invalid_argument_error) << "an invalid option was accepted";
      }
      // The result still updates.
      EXPECT_NO_THROW((void)c.step(batch_t{}));
      // Counted as skipped: a deletion of a missing edge, and (set semantics) an insertion of an
      // existing edge and a self-loop.
      vertex_t u = 0;
      vertex_t v = 0;
      for (vertex_t x = 0; x < model.num_vertices && u == v; ++x) {
        for (vertex_t y = 0; y < model.num_vertices; ++y) {
          if (x != y && model.weights.count({x, y}) == 0) {
            u = x;
            v = y;
            break;
          }
        }
      }
      ASSERT_NE(u, v) << "the generated graph is complete";
      {
        batch_t b = make_batch();
        b.delete_edge(u, v);
        const auto s = c.step(b);
        EXPECT_EQ(s.batch.ignored_deletions, 1) << "a deletion of a missing edge is counted";
        EXPECT_EQ(s.affected, 0);
        EXPECT_TRUE(kit_detail::same<traits>(before, c.take()));
      }
      if (p.props.semantics.as_sets && !model.weights.empty()) {
        const auto existing = model.weights.begin()->first;
        batch_t b = make_batch();
        insert(b, existing.first, existing.second);
        insert(b, vertex_t{0}, vertex_t{0});
        const auto s = c.step(b);
        EXPECT_EQ(s.batch.ignored_insertions, 1) << "an insertion of an existing edge is counted";
        EXPECT_EQ(s.batch.dropped_self_loops, 1) << "a self-loop is dropped and counted";
        EXPECT_EQ(s.affected, 0);
        EXPECT_TRUE(kit_detail::same<traits>(before, c.take()));
      }
      // Duplicates in one batch: the result still equals compute().
      {
        batch_t b = make_batch();
        insert(b, u, v);
        insert(b, u, v);
        (void)c.step(b);
        kit_detail::expect_oracle(c);
      }
      // A deletion of a missing edge under missing_delete::error is rejected.
      {
        graph_properties strict = p.props;
        strict.semantics.on_missing_delete = batch_semantics::missing_delete::error;
        kit_detail::chain<TypeParam> s(kit_detail::resources_for(be), model, strict);
        const auto strict_before = s.take();
        batch_t b = make_batch();
        b.delete_edge(u, v);
        const std::uint64_t version = s.g->version();
        EXPECT_THROW((void)s.step(b), invalid_argument_error);
        EXPECT_EQ(s.g->version(), version);
        EXPECT_TRUE(kit_detail::same<traits>(strict_before, s.take()));
      }
    }
  }
}

// C8: budgets of the algorithm phase once reserved (DYNG_DEBUG_BUDGETS builds; invariant I9),
// with engine::automatic and, where a backend has a second engine, engine::operators (a device
// without cooperative launch runs that engine under engine::automatic: M7).
TYPED_TEST_P(conformance, C8_TheAlgorithmPhaseStaysWithinItsBudget) {
  using traits = typename TypeParam::traits;
  using graph_t = typename TypeParam::graph_type;
  if (!detail::budgets_enabled()) {
    GTEST_SKIP() << "not a DYNG_DEBUG_BUDGETS build (the dev presets are)";
  }
  namespace fw = ::dyng::detail::framework;
  // Strict budgets: an excess throws internal_error here (a Debug build of the library only logs
  // it, framework/budgets.hpp).
  const fw::strict_budgets_scope strict;
  for (const backend b : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(b)));
    for (const kit_detail::preset& p : kit_detail::presets<traits>()) {
      SCOPED_TRACE(p.label);
      for (const size_class size : {size_class::small, size_class::medium}) {
        for (const batch_mix mix : {batch_mix::mixed, batch_mix::local, batch_mix::reweight}) {
          if (!kit_detail::mix_applies<graph_t>(mix, p.props)) {
            continue;
          }
          SCOPED_TRACE("mix " + std::string(to_string(mix)));
          std::optional<engine> automatic_ran;  // the engine engine::automatic chose
          for (const engine e : {engine::automatic, engine::operators}) {
            SCOPED_TRACE(e == engine::automatic ? "engine automatic" : "engine operators");
            if (e == engine::operators && automatic_ran == engine::operators) {
              break;  // engine::automatic already measured the operators engine
            }
            std::mt19937_64 rng(8000 + static_cast<std::uint64_t>(mix));
            graph_model<graph_t> model = kit_detail::model_of<TypeParam>(size, rng);
            const graph_model<graph_t> base = model;
            const generated_batch<graph_t> gen = random_batch(model, mix, rng);
            const resources res = kit_detail::resources_for(b);
            // The steady state ("once reserved"): a twin result on the same graph takes the same
            // batch first, so the handle's pooled workspaces have the batch's shapes; the measured
            // update then reserves nothing. The OpenMP engines' per-thread lists grow with the
            // largest share a thread has taken so far (the dynamic schedule), so a measured run
            // that still reserved is repeated after another warm-up (at most five times).
            std::optional<kit_detail::chain<TypeParam>> measured;
            fw::budget_report report;
            typename traits::stats stats{};
            bool counted = false;
            bool single_engine = false;
            for (int attempt = 0; attempt < 5; ++attempt) {
              std::optional<kit_detail::chain<TypeParam>> warm;
              try {
                warm.emplace(res, base, p.props, e);
              } catch (const not_supported_error&) {
                single_engine = true;  // the backend does not run this engine
                break;
              }
              (void)warm->step(gen.batch);
              measured.emplace(res, base, p.props, e);
              host_allocation_counter armed;
              counted = armed.counting();
              try {
                stats = measured->step(gen.batch);
              } catch (const internal_error& err) {
                ADD_FAILURE() << "the budget check failed: " << err.what();
                break;
              }
              armed.stop();
              report = fw::last_budget_report();
              if (!report.reserving()) {
                break;
              }
            }
            if (::testing::Test::HasFailure()) {
              return;
            }
            if (single_engine || (e == engine::operators && automatic_ran == stats.engine_used)) {
              break;  // the backend has one engine (or ignores the choice)
            }
            if (e == engine::automatic) {
              automatic_ran = stats.engine_used;
            }
            EXPECT_TRUE(report.measured);
            EXPECT_FALSE(report.reserving())
                << "the steady-state update still reserved (" << report.used.reservations
                << " times) after five warm-ups";
            EXPECT_EQ(report.used.own_allocations(), 0)
                << report.used.allocated_bytes << " bytes"
                << (counted ? " (library memory and host heap)" : " (library memory)");
            EXPECT_EQ(report.limit.allocations, 0) << "the problem declares no steady-state budget";
            const std::int64_t expected = kit_detail::expected_host_syncs<traits>(b, stats);
            if (expected == run_dependent_budget) {
              // The budget depends on counters the stats do not carry: the problem's own bound.
              EXPECT_NE(report.limit.host_syncs, fw::budget::unlimited)
                  << "the problem declares no steady-state budget";
              EXPECT_LE(report.used.own_host_syncs(), report.limit.host_syncs);
            } else {
              EXPECT_LE(report.used.own_host_syncs(), expected);
              EXPECT_EQ(report.limit.host_syncs, expected)
                  << "the problem's budget differs from the traits'";
            }
            const detail::budget_counters commit = fw::last_commit_counts();
            ::testing::Test::RecordProperty(
                "commit_allocations_" + std::string(to_string(b)),
                static_cast<int>(commit.allocations));  // container growth: reported only
            EXPECT_TRUE(kit_detail::same<traits>(measured->recompute(), measured->take()));
          }
        }
      }
    }
  }
}

// C9: stats sanity.
TYPED_TEST_P(conformance, C9_StatsAreSane) {
  using traits = typename TypeParam::traits;
  using graph_t = typename TypeParam::graph_type;
  for (const backend b : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(b)));
    for (const kit_detail::preset& p : kit_detail::presets<traits>()) {
      SCOPED_TRACE(p.label);
      for (const batch_mix mix : all_mixes()) {
        if (!kit_detail::mix_applies<graph_t>(mix, p.props)) {
          continue;
        }
        SCOPED_TRACE("mix " + std::string(to_string(mix)));
        std::mt19937_64 rng(9000 + static_cast<std::uint64_t>(mix));
        graph_model<graph_t> model = kit_detail::model_of<TypeParam>(size_class::small, rng);
        kit_detail::chain<TypeParam> c(kit_detail::resources_for(b), model, p.props);
        for (int step = 0; step < 3; ++step) {
          const generated_batch<graph_t> gen = random_batch(model, mix, rng);
          const auto s = c.step(gen.batch);
          const apply_summary& a = s.batch;
          EXPECT_GE(s.affected, 0);
          EXPECT_LE(s.affected, std::max<std::int64_t>(a.num_vertices_after, 1));
          EXPECT_EQ(a.num_vertices_after, static_cast<std::int64_t>(c.g->num_vertices()));
          const auto requested =
              static_cast<std::int64_t>(gen.batch.num_insertions() + gen.batch.num_deletions());
          EXPECT_EQ(a.inserted_edges + a.updated_edges + a.ignored_insertions + a.deleted_edges +
                        a.ignored_deletions + a.dropped_self_loops,
                    requested)
              << "applied + skipped != requested";
          EXPECT_NE(s.engine_used, engine::automatic) << "the stats name the engine that ran";
          EXPECT_TRUE(s.converged);
        }
      }
    }
  }
}

// C10: one dyng::update of several results equals each update alone on a copy of the graph, for
// every other registered algorithm of this build on the same graph type and backend.
TYPED_TEST_P(conformance, C10_OneUpdateOfSeveralResultsEqualsSeparateUpdates) {
  using traits = typename TypeParam::traits;
  using graph_t = typename TypeParam::graph_type;
  int partners = 0;
  for_each_type(registered_algorithms{}, [&](auto* partner_tag) {
    using partner_tag_t = std::remove_pointer_t<decltype(partner_tag)>;
    using partner = test_traits<partner_tag_t>;
    if constexpr (!std::is_same_v<partner_tag_t, typename TypeParam::tag> &&
                  contains_v<typename partner::graph_types, graph_t>) {
      using partner_case = suite_case<partner_tag_t, graph_t>;
      for (const backend b : kit_detail::backends<traits>()) {
        const std::vector<backend> theirs = kit_detail::backends<partner>();
        if (std::find(theirs.begin(), theirs.end(), b) == theirs.end()) {
          continue;
        }
        ++partners;
        SCOPED_TRACE(std::string(traits::name) + " with " + std::string(partner::name) + " on " +
                     std::string(to_string(b)));
        for (const bool as_sets : {false, true}) {
          SCOPED_TRACE(as_sets ? "set()" : "upsert_last_wins()");
          // A graph meeting both algorithms' requirements.
          graph_properties props;
          traits::require(props);
          partner::require(props);
          if (as_sets) {
            props.semantics = batch_semantics::set();
          }
          graph_shape shape = traits::shape(size_class::small);
          const graph_shape theirs_shape = partner::shape(size_class::small);
          shape.vertices = std::min(shape.vertices, theirs_shape.vertices);
          shape.edges = std::min(shape.edges, theirs_shape.edges);
          std::mt19937_64 rng(10000 + (as_sets ? 1U : 0U));
          graph_model<graph_t> model = random_model<graph_t>(shape, rng);
          model.num_weights =
              std::max(kit_detail::weight_columns<traits>(), kit_detail::weight_columns<partner>());
          const resources res = kit_detail::resources_for(b);
          // Both results on one graph (plus one left out of the updates), and each alone on a
          // graph of its own.
          kit_detail::chain<TypeParam> joint(res, model, props);
          std::optional<typename partner::template result<graph_t>> joint_partner;
          joint_partner.emplace(
              partner::template compute<graph_t>(res, *joint.g, engine::automatic));
          std::optional<typename traits::template result<graph_t>> left_out;
          left_out.emplace(traits::template compute<graph_t>(res, *joint.g, engine::automatic));
          kit_detail::chain<TypeParam> alone(res, model, props);
          kit_detail::chain<partner_case> alone_partner(res, model, props);
          for (int step = 0; step < 3; ++step) {
            SCOPED_TRACE("batch " + std::to_string(step));
            const batch_mix mix = step == 2 ? batch_mix::grow : batch_mix::mixed;
            const generated_batch<graph_t> gen = random_batch(model, mix, rng);
            typename traits::stats s_mine;
            typename partner::stats s_partner;
            if (step % 2 == 0) {
              std::tie(s_mine, s_partner) =
                  dyng::update(res, *joint.g, gen.batch.view(), *joint.r, *joint_partner);
            } else {
              std::tie(s_partner, s_mine) =
                  dyng::update(res, *joint.g, gen.batch.view(), *joint_partner, *joint.r);
            }
            const auto a_mine = alone.step(gen.batch);
            const auto a_partner = alone_partner.step(gen.batch);
            EXPECT_TRUE(kit_detail::same<traits>(alone.take(), joint.take()));
            EXPECT_TRUE(kit_detail::same<partner>(
                alone_partner.take(), partner::template take<graph_t>(res, *joint_partner)));
            EXPECT_EQ(kit_detail::counters<traits>(s_mine), kit_detail::counters<traits>(a_mine));
            EXPECT_EQ(kit_detail::counters<partner>(s_partner),
                      kit_detail::counters<partner>(a_partner));
            EXPECT_EQ(joint.r->graph_version(), joint.g->version());
            EXPECT_EQ(joint_partner->graph_version(), joint.g->version());
          }
          // C11 through composition: the result left out of the updates is stale.
          using batch_t = typename kit_detail::chain<TypeParam>::batch_type;
          EXPECT_THROW(
              (void)traits::template update<graph_t>(res, *joint.g, batch_t{}.view(), *left_out),
              stale_result_error);
          EXPECT_THROW((void)dyng::update(res, *joint.g, batch_t{}.view(), *joint.r, *left_out),
                       stale_result_error);
        }
      }
    }
  });
  if (partners == 0) {
    GTEST_SKIP() << "no other registered algorithm of this build shares this graph type and a "
                    "backend";
  }
}

// C11: stale results are detected.
TYPED_TEST_P(conformance, C11_StaleResultsAreDetected) {
  using traits = typename TypeParam::traits;
  using graph_t = typename TypeParam::graph_type;
  for (const backend b : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(b)));
    std::mt19937_64 rng(11000);
    graph_model<graph_t> model = kit_detail::model_of<TypeParam>(size_class::small, rng);
    const kit_detail::preset p = kit_detail::presets<traits>().front();
    kit_detail::chain<TypeParam> c(kit_detail::resources_for(b), model, p.props);
    const auto before = c.take();
    // A separate g.apply(): the result is stale, and update() changes nothing.
    const generated_batch<graph_t> gen = random_batch(model, batch_mix::mixed, rng);
    (void)c.g->apply(c.res, gen.batch.view());
    const std::uint64_t version = c.g->version();
    using batch_t = typename kit_detail::chain<TypeParam>::batch_type;
    EXPECT_THROW((void)c.step(batch_t{}), stale_result_error);
    EXPECT_THROW((void)dyng::update(c.res, *c.g, gen.batch.view(), *c.r), stale_result_error);
    EXPECT_EQ(c.g->version(), version);
    EXPECT_TRUE(kit_detail::same<traits>(before, c.take()));
    // A result of another graph (built from the same edges, at the same version) is stale for
    // it; a clone of its own graph is the same state (ADR 0006, "Graph identity") until the two
    // diverge.
    kit_detail::chain<TypeParam> x(c.res, model, p.props);
    kit_detail::chain<TypeParam> y(c.res, model, p.props);
    ASSERT_EQ(x.g->version(), y.g->version());
    EXPECT_THROW((void)traits::template update<graph_t>(c.res, *y.g, batch_t{}.view(), *x.r),
                 stale_result_error);
    graph_t copy = x.g->clone(c.res);
    EXPECT_NO_THROW((void)traits::template update<graph_t>(c.res, copy, batch_t{}.view(), *x.r));
    EXPECT_THROW((void)x.step(batch_t{}), stale_result_error);  // x.g has not changed with copy
    // Recomputing makes it current again.
    c.r.emplace(traits::template compute<graph_t>(c.res, *c.g, engine::automatic));
    EXPECT_NO_THROW((void)c.step(batch_t{}));
  }
}

// C12: stream ordering on non-default streams (CUDA).
TYPED_TEST_P(conformance, C12_NonDefaultStreamsOrderTheWork) {
  using traits = typename TypeParam::traits;
  using graph_t = typename TypeParam::graph_type;
#if defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA
  const std::vector<backend> mine = kit_detail::backends<traits>();
  if (std::find(mine.begin(), mine.end(), backend::cuda) == mine.end()) {
    GTEST_SKIP() << "no cuda backend of " << traits::name << " in this build";
  }
  cudaStream_t a = nullptr;
  cudaStream_t s = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), cudaSuccess);
  {
    const resources on_a = resources::cuda(0, stream_ref(a));
    const resources on_s = resources::cuda(0, stream_ref(s));
    for (const kit_detail::preset& p : kit_detail::presets<traits>()) {
      SCOPED_TRACE(p.label);
      std::mt19937_64 rng(12000);
      graph_model<graph_t> model = kit_detail::model_of<TypeParam>(size_class::medium, rng);
      kit_detail::chain<TypeParam> host(resources::sequential(), model, p.props);
      kit_detail::chain<TypeParam> device(on_a, model, p.props);
      for (int step = 0; step < 4; ++step) {
        SCOPED_TRACE("batch " + std::to_string(step));
        const generated_batch<graph_t> gen =
            random_batch(model, step == 3 ? batch_mix::grow : batch_mix::heavy, rng);
        (void)host.step(gen.batch);
        // Alternate the streams: each call orders its work on its own stream.
        device.res = step % 2 == 0 ? on_s : on_a;
        (void)device.step(gen.batch);
        EXPECT_TRUE(kit_detail::same<traits>(host.take(), device.take()));
      }
    }
  }
  EXPECT_EQ(cudaStreamDestroy(a), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(s), cudaSuccess);
#else
  (void)sizeof(graph_t);
  GTEST_SKIP() << "CUDA only: runs in the gpu conformance executable of " << traits::name;
#endif
}

REGISTER_TYPED_TEST_SUITE_P(conformance, C0_TheTraitsAgreeWithTheRegistry,
                            C1_AnEmptyBatchChangesNothing, C2_UpdateChainsEqualTheOracle,
                            C3_BackendsAgree, C4_EnginesAgree, C5_ABatchAndItsInverseCancel,
                            C6_RunsAreDeterministic, C7_InvalidInputIsRejectedOrCounted,
                            C8_TheAlgorithmPhaseStaysWithinItsBudget, C9_StatsAreSane,
                            C10_OneUpdateOfSeveralResultsEqualsSeparateUpdates,
                            C11_StaleResultsAreDetected, C12_NonDefaultStreamsOrderTheWork);

}  // namespace dyng::conformance

/**
 * @brief Instantiate the conformance kit for one registered algorithm (its test_traits must be
 *        defined: include cpp/tests/conformance/registry.hpp, which includes them). Checks the
 *        registration rules at compile time (invariant I8).
 * @param algo The algorithm's name (its tag, dyng::conformance::tags::algo).
 */
#define DYNG_CONFORMANCE_SUITE(algo)                                                        \
  namespace dyng::conformance {                                                             \
  static_assert(registration_check<tags::algo>());                                          \
  INSTANTIATE_TYPED_TEST_SUITE_P(algo, conformance, cases_of<tags::algo>::type, case_name); \
  }                                                                                         \
  static_assert(true, "DYNG_CONFORMANCE_SUITE ends with a semicolon")
