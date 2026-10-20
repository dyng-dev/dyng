// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file conformance.hpp
 * @brief The conformance kit (PLAN Section 8.2): checks C0-C12 for every registered algorithm,
 *        every backend of the build and every instantiated container type (graph or hypergraph),
 *        from the algorithm's test_traits (conformance/test_traits.hpp) and one line
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
 * The checks reach the container through container_traits<container_t> (containers.hpp: the
 * graph's; hypergraph_generators.hpp: the hypergraph's; ADR 0043): its properties and presets,
 * its host model and batch mixes, its batch summary, its case names. What differs too much to be
 * data (C7's invalid batches and skip rules, C9's summary arithmetic) is container_checks<case_t,
 * container_t> below. A graph algorithm sees the kit of 0.1-0.2 unchanged.
 *
 * The suite source is compiled twice, like the algorithm suites (support/gtest_helpers.hpp): into
 * the host executable (label cpu: sequential and OpenMP) and with DYNG_TEST_CUDA=1 into the CUDA
 * executable (label gpu: cuda; the cross-backend check compares it with the host backends).
 * Every randomized check prints its seed; DYNG_TEST_SEED=<seed> replays one.
 */
#pragma once

#include "conformance/allocation_counter.hpp"
#include "conformance/containers.hpp"
#include "conformance/generators.hpp"
#include "conformance/hypergraph_generators.hpp"
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
#include <dyng/hypergraph/hyper_apply_summary.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/hypergraph/hypergraph_properties.hpp>
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

namespace kit_detail {

template <typename traits_t, typename = void>
struct has_container_types : std::false_type {};
template <typename traits_t>
struct has_container_types<traits_t, std::void_t<typename traits_t::container_types>>
    : std::true_type {};

/// The container types of an algorithm: test_traits::container_types, or graph_types (the name
/// of 0.1-0.2, which a graph algorithm keeps).
template <typename traits_t, bool = has_container_types<traits_t>::value>
struct container_types_of {
  using type = typename traits_t::graph_types;  ///< the list
};
/// @copydoc container_types_of
template <typename traits_t>
struct container_types_of<traits_t, true> {
  using type = typename traits_t::container_types;  ///< the list
};

template <typename traits_t, typename = void>
struct has_test_partners : std::false_type {};
template <typename traits_t>
struct has_test_partners<traits_t, std::void_t<typename traits_t::test_partners>> : std::true_type {
};

/// The partners of C10: the registered algorithms of this binary, and the test-only algorithms
/// the traits list (test_traits::test_partners; the kit's own hypergraph fakes).
template <typename traits_t, bool = has_test_partners<traits_t>::value>
struct partners_of {
  using type = registered_algorithms;  ///< the list
};
/// @copydoc partners_of
template <typename traits_t>
struct partners_of<traits_t, true> {
  using type = typename concat<registered_algorithms, typename traits_t::test_partners>::type;
};

}  // namespace kit_detail

/// One typed case of the suite: an algorithm on one of its container types.
template <typename tag_t, typename graph_t>
struct suite_case {
  using tag = tag_t;                  ///< the algorithm's tag
  using traits = test_traits<tag_t>;  ///< its traits
  using graph_type = graph_t;         ///< the container type (a graph or a hypergraph)
};

/// The cases of an algorithm: one per container type of its traits.
template <typename tag_t>
struct cases_of {
  template <typename graph_t>
  using make = suite_case<tag_t, graph_t>;  ///< the case of one container type
  /// ::testing::Types of the cases
  using type = typename as_gtest_types<typename transform<
      typename kit_detail::container_types_of<test_traits<tag_t>>::type, make>::type>::type;
};

/// The name of a type in a case name.
template <typename type_t>
std::string type_name() {
  return container_traits<graph<std::int32_t, std::int32_t, std::int32_t>>::type_name<type_t>();
}

/// GoogleTest names of the cases: container_traits::case_name(), e.g. v32_e64_unweighted for a
/// graph, hv32_unweighted for a hypergraph.
struct case_name {
  template <typename case_t>
  static std::string GetName(int /*index*/) {  // NOLINT(readability-identifier-naming): gtest API
    return container_traits<typename case_t::graph_type>::case_name();
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
                      std::declval<const typename container_traits<graph_t>::batch_view_type&>(),
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

/// Whether stats_t has `batch` of the container's summary type (apply_summary for graphs,
/// hyper_apply_summary for hypergraphs).
template <typename stats_t, typename summary_t, typename = void>
struct has_batch : std::false_type {};
template <typename stats_t, typename summary_t>
struct has_batch<stats_t, summary_t, std::void_t<decltype(std::declval<stats_t&>().batch)>>
    : std::is_same<std::decay_t<decltype(std::declval<stats_t&>().batch)>, summary_t> {};

/// The first type of a list.
template <typename list_t>
struct first_type;
/// @copydoc first_type
template <typename first_t, typename... rest_t>
struct first_type<type_list<first_t, rest_t...>> {
  using type = first_t;  ///< the type
};
/// @copydoc first_type
template <>
struct first_type<type_list<>> {
  using type = graph<std::int32_t, std::int32_t, std::int32_t>;  ///< none: the graph's summary
};

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
                "exist for every graph type of graph_types (container type of container_types; "
                "the batch is the container's batch view)");
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
  using containers_t = typename kit_detail::container_types_of<traits_t>::type;
  using summary_t =
      typename container_traits<typename kit_detail::first_type<containers_t>::type>::summary_type;
  static_assert(containers_t::size > 0,
                "dyng conformance kit: test_traits::graph_types lists no graph type");
  static_assert(kit_detail::has_oracle_kind<traits_t>::value,
                "dyng conformance kit: test_traits must declare its oracle kind, `static constexpr "
                "oracle_kind oracle = oracle_kind::compute;` (or reference); invariant I8");
  static_assert(kit_detail::has_level<traits_t>::value,
                "dyng conformance kit: test_traits must declare its determinism level, `static "
                "constexpr determinism level = ...;`");
  static_assert(kit_detail::has_batch<typename traits_t::stats, summary_t>::value,
                "dyng conformance kit: the algorithm's stats must derive from update_stats and "
                "contain `apply_summary batch` (`hyper_apply_summary batch` for a hypergraph "
                "algorithm; ADR 0006)");
  static_assert(!kit_detail::has_level<traits_t>::value ||
                    traits_t::level != determinism::tolerance ||
                    kit_detail::has_compare<traits_t>::value,
                "dyng conformance kit: an algorithm with determinism::tolerance must provide "
                "test_traits::compare(expected, actual) with its tolerance");
  return registration_check_graphs<traits_t>(containers_t{});
}

// ------------------------------------------------------------------------------------------------
// Helpers of the checks
// ------------------------------------------------------------------------------------------------

namespace kit_detail {

template <typename traits_t, typename = void>
struct has_registry_entry : std::false_type {};
template <typename traits_t>
struct has_registry_entry<traits_t, std::void_t<decltype(traits_t::registry_entry())>>
    : std::true_type {};

/// The algorithm's registry entry: dyng::find_algorithm(name), or, for a test-only algorithm that
/// the library does not contain (the kit's own fakes), test_traits::registry_entry().
template <typename traits_t>
const algorithm_info* info_of() {
  if constexpr (has_registry_entry<traits_t>::value) {
    return traits_t::registry_entry();
  } else {
    return find_algorithm(traits_t::name);
  }
}

/// The backends of the algorithm that this test binary runs (suite_backends() of the binary,
/// restricted to the manifest's backends).
template <typename traits_t>
std::vector<backend> backends() {
  std::vector<backend> out;
  const algorithm_info* info = info_of<traits_t>();
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
  const algorithm_info* info = info_of<traits_t>();
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

/// The OpenMP threads of the kit's resources: test_traits::kit_openmp_threads, else 4.
template <typename traits_t, typename = void>
struct kit_openmp_threads {
  static constexpr int value = 4;  ///< the default
};
/// The OpenMP threads a test_traits names (triad_count runs its kit on 8, M9a criterion 10).
template <typename traits_t>
struct kit_openmp_threads<traits_t, std::void_t<decltype(traits_t::kit_openmp_threads)>> {
  static constexpr int value = traits_t::kit_openmp_threads;  ///< the traits' count
};

/// The resources of a backend for the kit's case `case_t` (its traits' OpenMP thread count).
template <typename case_t>
inline resources resources_for(backend b) {
  return test::make_resources(
      b, b == backend::openmp ? kit_openmp_threads<typename case_t::traits>::value : 0);
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

/// The container axis of a case (containers.hpp).
template <typename case_t>
using axis = container_traits<typename case_t::graph_type>;

/// A property preset of the checks, for the container of `case_t`.
template <typename case_t>
using preset_of = kit_preset<typename axis<case_t>::properties_type>;

/// The presets of a case (container_traits::presets: for graphs the defaults, the defaults with
/// the insertions first, set() semantics and the traits' extras; each with require()).
template <typename case_t>
std::vector<preset_of<case_t>> presets() {
  return axis<case_t>::template presets<typename case_t::traits>();
}

/// One algorithm result kept up to date on one container (the unit of every check).
template <typename case_t>
struct chain {
  using traits = typename case_t::traits;                            ///< the traits
  using graph_type = typename case_t::graph_type;                    ///< the container type
  using result_type = typename traits::template result<graph_type>;  ///< the result type
  using stats_type = typename traits::stats;                         ///< the stats type
  using batch_type = typename axis<case_t>::batch_type;              ///< the batch type
  using model_type = typename axis<case_t>::model_type;              ///< the model type
  using properties_type = typename axis<case_t>::properties_type;    ///< the properties

  resources res;                 ///< the backend
  std::optional<graph_type> g;   ///< the container
  std::optional<result_type> r;  ///< the result

  /// compute() on the model's container.
  chain(resources on, const model_type& model, const properties_type& props,
        engine e = engine::automatic)
      : res(std::move(on)) {
    g.emplace(axis<case_t>::build(model, res, props));
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

/// The deterministic counters of a stats object, with the batch summary's (on container_t).
template <typename traits_t, typename container_t>
std::vector<std::int64_t> counters(const typename traits_t::stats& s) {
  std::vector<std::int64_t> out = traits_t::deterministic(s);
  out.push_back(s.affected);
  const std::vector<std::int64_t> b = container_traits<container_t>::summary_counters(s.batch);
  out.insert(out.end(), b.begin(), b.end());
  return out;
}

/// A random model of a size class.
template <typename case_t>
typename axis<case_t>::model_type model_of(size_class size, std::mt19937_64& rng) {
  return axis<case_t>::template random_model<typename case_t::traits>(size, rng);
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
  using ct = axis<case_t>;
  bool compared = false;
  for (const backend b : backends) {
    SCOPED_TRACE(std::string(to_string(b)));
    for (const auto& p : presets<case_t>()) {
      SCOPED_TRACE(p.label);
      for (const size_class size : {size_class::small, size_class::medium}) {
        for (const auto mix : ct::mixes(kit_check::c4)) {
          if (!ct::mix_applies(mix, p.props)) {
            continue;
          }
          SCOPED_TRACE("size " + std::to_string(static_cast<int>(size)) + ", mix " +
                       std::string(to_string(mix)));
          std::mt19937_64 rng(4000 + static_cast<std::uint64_t>(size) * 17 +
                              static_cast<std::uint64_t>(mix));
          typename ct::model_type model = model_of<case_t>(size, rng);
          std::optional<chain<case_t>> fused;
          std::optional<chain<case_t>> operators;
          try {
            fused.emplace(resources_for<case_t>(b), model, p.props, engine::fused);
            operators.emplace(resources_for<case_t>(b), model, p.props, engine::operators);
          } catch (const not_supported_error&) {
            break;  // this backend has one engine
          }
          if (!same<traits>(fused->take(), operators->take())) {
            ADD_FAILURE() << "compute(): the fused and the operators engine disagree";
            return true;
          }
          for (int step = 0; step < 3; ++step) {
            SCOPED_TRACE("batch " + std::to_string(step));
            const typename ct::generated_type gen = ct::random_batch(model, mix, rng, p.props);
            typename traits::stats sf;
            typename traits::stats so;
            try {
              sf = fused->step(ct::batch_of(gen));
              ct::observe(model, gen, sf.batch);
              so = operators->step(ct::batch_of(gen));
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
// The checks that differ per container: C7's invalid input and skip rules, C9's summary arithmetic
// ------------------------------------------------------------------------------------------------

/// The container-specific parts of C7 and C9; specialised per container.
template <typename case_t, typename container_t = typename case_t::graph_type>
struct container_checks;

/// The graph's: the kit of 0.1-0.2, unchanged.
template <typename case_t, typename vertex_t_, typename edge_t_, typename weight_t_>
struct container_checks<case_t, graph<vertex_t_, edge_t_, weight_t_>> {
  /// C7 under one preset on one backend: invalid input is rejected with the documented exception
  /// (and changes nothing), or counted as skipped per the batch semantics.
  static void invalid_input(backend be, const kit_preset<graph_properties>& p) {
    using traits = typename case_t::traits;
    using graph_t = typename case_t::graph_type;
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
    std::mt19937_64 rng(7000);
    graph_model<graph_t> model = kit_detail::model_of<case_t>(size_class::small, rng);
    kit_detail::chain<case_t> c(kit_detail::resources_for<case_t>(be), model, p.props);
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
        if (x != y && !model.has(x, y)) {
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
      EXPECT_EQ(s.batch.ignored_deletions, model.directions())
          << "a deletion of a missing edge is counted (once per stored direction)";
      EXPECT_EQ(s.affected, 0);
      EXPECT_TRUE(kit_detail::same<traits>(before, c.take()));
    }
    if (p.props.semantics.as_sets && !model.weights.empty()) {
      const auto existing = model.weights.begin()->first;
      batch_t b = make_batch();
      insert(b, existing.first, existing.second);
      insert(b, vertex_t{0}, vertex_t{0});
      const auto s = c.step(b);
      EXPECT_EQ(s.batch.ignored_insertions, model.directions())
          << "an insertion of an existing edge is counted (once per stored direction)";
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
      kit_detail::chain<case_t> s(kit_detail::resources_for<case_t>(be), model, strict);
      const auto strict_before = s.take();
      batch_t b = make_batch();
      b.delete_edge(u, v);
      const std::uint64_t version = s.g->version();
      EXPECT_THROW((void)s.step(b), invalid_argument_error);
      EXPECT_EQ(s.g->version(), version);
      EXPECT_TRUE(kit_detail::same<traits>(strict_before, s.take()));
    }
  }

  /// C9 after one batch: stats sanity.
  template <typename stats_t, typename generated_t, typename model_t>
  static void stats_sane(const resources& /*res*/, const stats_t& s, const generated_t& gen,
                         const model_t& model, const graph<vertex_t_, edge_t_, weight_t_>& g) {
    const apply_summary& a = s.batch;
    EXPECT_GE(s.affected, 0);
    EXPECT_LE(s.affected, std::max<std::int64_t>(a.num_vertices_after, 1));
    EXPECT_EQ(a.num_vertices_after, static_cast<std::int64_t>(g.num_vertices()));
    const auto requested =
        static_cast<std::int64_t>(gen.batch.num_insertions() + gen.batch.num_deletions()) *
        model.directions();  // an undirected graph counts each direction
    EXPECT_EQ(a.inserted_edges + a.updated_edges + a.ignored_insertions + a.deleted_edges +
                  a.ignored_deletions + a.dropped_self_loops,
              requested)
        << "applied + skipped != requested";
  }
};

/// The hypergraph's (ADR 0043).
template <typename case_t, typename vertex_t_, typename weight_t_>
struct container_checks<case_t, hypergraph<vertex_t_, weight_t_>> {
  using traits = typename case_t::traits;                        ///< the traits
  using hypergraph_t = hypergraph<vertex_t_, weight_t_>;         ///< the container
  using ct = container_traits<hypergraph_t>;                     ///< its axis
  using batch_t = hyperedge_batch<vertex_t_, weight_t_>;         ///< the owning batch
  static constexpr bool weighted = !is_unweighted_v<weight_t_>;  ///< has weights

  /// A batch view whose arrays point into `offsets`, `members`, `weights`, `deletions` and the
  /// incidence arrays (raw: the builder of hyperedge_batch does not make malformed batches).
  struct raw_batch {
    std::vector<std::int64_t> deletions;            ///< delete_hyperedges
    std::vector<std::int64_t> offsets;              ///< insert_offsets
    std::vector<vertex_t_> members;                 ///< insert_members
    std::vector<weight_t_> weights;                 ///< insert_weights
    std::vector<std::int64_t> incidence_hyperedge;  ///< insert_incidence_hyperedge
    std::vector<vertex_t_> incidence_vertex;        ///< insert_incidence_vertex

    /// The view.
    [[nodiscard]] hyperedge_batch_view<vertex_t_, weight_t_> view() const {
      hyperedge_batch_view<vertex_t_, weight_t_> v;
      v.delete_hyperedges = host_view(deletions);
      v.insert_offsets = host_view(offsets);
      v.insert_members = host_view(members);
      v.insert_weights = host_view(weights);
      v.insert_incidence_hyperedge = host_view(incidence_hyperedge);
      v.insert_incidence_vertex = host_view(incidence_vertex);
      return v;
    }
  };

  /// C7 under one preset on one backend: invalid batches are rejected before anything changes
  /// (invalid_argument_error; incidence edits not_supported_error, planned for 0.4); skipped
  /// deletions and dropped members are counted; duplicates still give compute().
  static void invalid_input(backend be, const kit_preset<hypergraph_properties>& p) {
    std::mt19937_64 rng(7000);
    typename ct::model_type model = kit_detail::model_of<case_t>(size_class::small, rng);
    kit_detail::chain<case_t> c(kit_detail::resources_for<case_t>(be), model, p.props);
    const auto before = c.take();
    const vertex_t_ n = c.g->num_vertices();
    ASSERT_GT(n, 1) << "the small shape needs two vertices";
    // Rejected: nothing changes (version, state, contents), and the result stays current.
    const auto expect_rejected = [&](const hyperedge_batch_view<vertex_t_, weight_t_>& b,
                                     const char* what, bool not_supported) {
      SCOPED_TRACE(what);
      const std::uint64_t version = c.g->version();
      const std::int64_t live = c.g->num_hyperedges();
      const std::int64_t bound = c.g->hyperedge_id_bound();
      if (not_supported) {
        EXPECT_THROW((void)traits::template update<hypergraph_t>(c.res, *c.g, b, *c.r),
                     not_supported_error);
      } else {
        EXPECT_THROW((void)traits::template update<hypergraph_t>(c.res, *c.g, b, *c.r),
                     invalid_argument_error);
      }
      EXPECT_EQ(c.g->version(), version);
      EXPECT_EQ(c.g->num_hyperedges(), live);
      EXPECT_EQ(c.g->hyperedge_id_bound(), bound);
      EXPECT_EQ(c.r->graph_version(), c.g->version());
      EXPECT_TRUE(kit_detail::same<traits>(before, c.take()));
    };
    const auto one = [&](std::vector<vertex_t_> members, std::int64_t weight) {
      raw_batch b;
      b.offsets = {0, static_cast<std::int64_t>(members.size())};
      b.members = std::move(members);
      if constexpr (weighted) b.weights = {static_cast<weight_t_>(weight)};
      return b;
    };
    {
      const raw_batch b = one({0, -1}, 1);
      expect_rejected(b.view(), "a member with a negative id", false);
    }
    {
      const raw_batch b = one({0, n}, 1);
      expect_rejected(b.view(), "a member at num_vertices", false);
    }
    {
      raw_batch b = one({}, 1);
      expect_rejected(b.view(), "an empty hyperedge", false);
    }
    {
      raw_batch b = one({0, 1}, 1);
      b.offsets = {0, 3};  // three members announced, two given
      expect_rejected(b.view(), "malformed offsets", false);
    }
    {
      raw_batch b = one({0, 1}, 1);
      b.deletions = {0};  // the deletion must not happen either
      if constexpr (weighted) {
        b.weights = {weight_t_{0}};
        expect_rejected(b.view(), "a weight of 0", false);
      } else {
        b.weights.resize(1);
        expect_rejected(b.view(), "weights on an unweighted batch", false);
      }
    }
    {
      raw_batch b = one({0, 1}, 1);
      b.incidence_hyperedge = {0};
      b.incidence_vertex = {1};
      expect_rejected(b.view(), "an incidence edit (planned for 0.4)", true);
    }
    // Bad options: each must throw invalid_argument_error.
    for (const std::function<void()>& bad : traits::invalid_options(c.res, *c.g)) {
      EXPECT_THROW(bad(), invalid_argument_error) << "an invalid option was accepted";
    }
    // The result still updates.
    EXPECT_NO_THROW((void)c.step(batch_t{}));
    // Counted as skipped: dead, out-of-range and negative ids; nothing else changes.
    std::int64_t dead = -1;
    for (std::int64_t id = 0; id < c.g->hyperedge_id_bound() && dead < 0; ++id) {
      if (model.live.count(id) == 0) dead = id;
    }
    {
      batch_t b;
      b.delete_hyperedge(c.g->hyperedge_id_bound());
      b.delete_hyperedge(-7);
      b.delete_hyperedge(std::numeric_limits<std::int64_t>::max());
      if (dead >= 0) b.delete_hyperedge(dead);
      const auto s = c.step(b);
      EXPECT_EQ(s.batch.ignored_deletions, dead >= 0 ? 4 : 3)
          << "deletions of dead, out-of-range and negative ids are counted";
      EXPECT_EQ(s.batch.deleted_hyperedges, 0);
      EXPECT_EQ(s.batch.skipped_ops, s.batch.ignored_deletions + s.batch.dropped_members);
      EXPECT_EQ(s.affected, 0);
      EXPECT_TRUE(kit_detail::same<traits>(before, c.take()));
    }
    // A repeated deletion: the first deletes, the second is counted.
    if (!model.live.empty()) {
      const std::int64_t id = model.live.begin()->first;
      batch_t b;
      b.delete_hyperedge(id);
      b.delete_hyperedge(id);
      const auto s = c.step(b);
      EXPECT_EQ(s.batch.deleted_hyperedges, 1);
      EXPECT_EQ(s.batch.ignored_deletions, 1) << "a repeated deletion is counted";
      model.live.erase(id);
      kit_detail::expect_oracle(c);
    }
    // Repeated members: dropped and counted; the result still equals compute().
    {
      batch_t b;
      const std::vector<vertex_t_> members = {1, 0, 1, 0, 1};
      if constexpr (weighted) {
        b.insert_hyperedge(host_view(members), weight_t_{3});
      } else {
        b.insert_hyperedge(host_view(members));
      }
      const auto s = c.step(b);
      EXPECT_EQ(s.batch.dropped_members, 3) << "repeated members are dropped and counted";
      EXPECT_EQ(s.batch.inserted_hyperedges, 1);
      kit_detail::expect_oracle(c);
    }
  }

  /// C9 after one batch: stats sanity (the summary's arithmetic, the inserted ids, the counts the
  /// generator expects).
  template <typename stats_t>
  static void stats_sane(const resources& res, const stats_t& s,
                         const typename ct::generated_type& gen,
                         const typename ct::model_type& /*model*/, const hypergraph_t& h) {
    const hyper_apply_summary& a = s.batch;
    EXPECT_GE(s.affected, 0);
    EXPECT_LE(s.affected,
              std::max<std::int64_t>({static_cast<std::int64_t>(h.num_vertices()), a.id_bound_after,
                                      a.inserted_hyperedges + a.deleted_hyperedges, 1}));
    const auto requested = static_cast<std::int64_t>(gen.batch.view().num_insertions() +
                                                     gen.batch.view().num_deletions());
    EXPECT_EQ(a.inserted_hyperedges + a.deleted_hyperedges + a.ignored_deletions, requested)
        << "applied + skipped != requested";
    EXPECT_EQ(a.ignored_deletions, gen.expected_ignored);
    EXPECT_EQ(a.dropped_members, gen.expected_dropped);
    EXPECT_EQ(a.skipped_ops, a.ignored_deletions + a.dropped_members);
    EXPECT_EQ(a.deleted_hyperedges, static_cast<std::int64_t>(gen.deleted.size()));
    EXPECT_EQ(a.inserted_incidences, 0);
    EXPECT_EQ(a.deleted_incidences, 0);
    EXPECT_EQ(a.num_hyperedges_after, h.num_hyperedges());
    EXPECT_EQ(a.id_bound_after, h.hyperedge_id_bound());
    ASSERT_EQ(static_cast<std::int64_t>(a.inserted_ids.size()), a.inserted_hyperedges);
    // The inserted ids are distinct, live, below the bound and hold the inserted hyperedges.
    std::vector<std::int64_t> sorted = a.inserted_ids;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_TRUE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end())
        << "an id was given twice";
    const auto view = h.h2v();
    const std::vector<std::int64_t> ptr = to_vector(res, view.row_ptr);
    const std::vector<vertex_t_> cols = to_vector(res, view.col_ind);
    for (std::size_t i = 0; i < a.inserted_ids.size(); ++i) {
      const std::int64_t id = a.inserted_ids[i];
      ASSERT_TRUE(id >= 0 && id < a.id_bound_after) << "inserted id " << id << " is out of range";
      const std::vector<vertex_t_> row(cols.begin() + ptr[static_cast<std::size_t>(id)],
                                       cols.begin() + ptr[static_cast<std::size_t>(id) + 1]);
      EXPECT_EQ(row, gen.inserted[i].members) << "inserted hyperedge " << i << " (id " << id << ")";
    }
  }
};

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
  using ct = kit_detail::axis<TypeParam>;
  const algorithm_info* info = kit_detail::info_of<traits>();
  ASSERT_NE(info, nullptr) << traits::name << " is not in dyng::algorithms() (its manifest, "
                           << "scripts/regen.py, or DYNG_ALGORITHMS)";
  EXPECT_EQ(info->oracle, traits::oracle) << "the manifest's oracle differs from the traits'";
  EXPECT_EQ(info->determinism_level, traits::level) << "the manifest's determinism differs";
  ASSERT_FALSE(info->backends.empty());
  EXPECT_EQ(info->backends.front(), backend::sequential)
      << "invariant I8: every algorithm has the sequential reference backend";
  EXPECT_EQ(info->container, ct::kind)
      << "the manifest's container differs from the traits' container types";
  // The kit runs the manifest's backends only: a backend the library runs but the manifest omits
  // would lose its conformance coverage silently (and the registry would misreport it).
  using graph_t = typename TypeParam::graph_type;
  for (const backend b : test::suite_backends()) {
    if (std::find(info->backends.begin(), info->backends.end(), b) != info->backends.end()) {
      continue;
    }
    SCOPED_TRACE(std::string(to_string(b)) + " (not in the manifest)");
    std::mt19937_64 rng(10);
    const resources res = kit_detail::resources_for<TypeParam>(b);
    const graph_t g = ct::build(kit_detail::model_of<TypeParam>(size_class::tiny, rng), res,
                                kit_detail::presets<TypeParam>().front().props);
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
  using ct = kit_detail::axis<TypeParam>;
  for (const backend b : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(b)));
    for (const auto& p : kit_detail::presets<TypeParam>()) {
      SCOPED_TRACE(p.label);
      for (const size_class size : {size_class::tiny, size_class::small}) {
        std::mt19937_64 rng(11);
        kit_detail::chain<TypeParam> c(kit_detail::resources_for<TypeParam>(b),
                                       kit_detail::model_of<TypeParam>(size, rng), p.props);
        const auto before = c.take();
        const std::uint64_t version = c.g->version();
        const typename kit_detail::chain<TypeParam>::batch_type empty;
        const auto s = c.step(empty);
        EXPECT_EQ(s.affected, 0);
        EXPECT_TRUE(kit_detail::same<traits>(before, c.take()));
        EXPECT_EQ(ct::changed(s.batch), 0);
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
  using ct = kit_detail::axis<TypeParam>;
  for (const backend b : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(b)));
    for (const auto& p : kit_detail::presets<TypeParam>()) {
      SCOPED_TRACE(p.label);
      for (const std::uint64_t seed : test::test_seeds(2000, 2)) {
        SCOPED_TRACE(test::seed_trace(seed));
        for (const size_class size : {size_class::tiny, size_class::small, size_class::medium}) {
          for (const auto mix : ct::mixes(kit_check::c2)) {
            if (!ct::mix_applies(mix, p.props)) {
              continue;
            }
            SCOPED_TRACE("size " + std::to_string(static_cast<int>(size)) + ", mix " +
                         std::string(to_string(mix)));
            std::mt19937_64 rng(seed * 131 + static_cast<std::uint64_t>(size) * 17 +
                                static_cast<std::uint64_t>(mix));
            typename ct::model_type model = kit_detail::model_of<TypeParam>(size, rng);
            kit_detail::chain<TypeParam> c(kit_detail::resources_for<TypeParam>(b), model, p.props);
            kit_detail::expect_oracle(c);
            for (int step = 0; step < 3; ++step) {
              SCOPED_TRACE("batch " + std::to_string(step));
              const typename ct::generated_type gen = ct::random_batch(model, mix, rng, p.props);
              ct::observe(model, gen, c.step(ct::batch_of(gen)).batch);
              ASSERT_TRUE(ct::matches(*c.g, model));
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
  using ct = kit_detail::axis<TypeParam>;
  const std::vector<backend> backends = kit_detail::comparison_backends<traits>();
  if (backends.size() < 2) {
    GTEST_SKIP() << "only one backend of " << traits::name << " in this build";
  }
  for (const auto& p : kit_detail::presets<TypeParam>()) {
    SCOPED_TRACE(p.label);
    for (const std::uint64_t seed : test::test_seeds(3000, 2)) {
      SCOPED_TRACE(test::seed_trace(seed));
      for (const auto mix : ct::mixes(kit_check::c3)) {
        if (!ct::mix_applies(mix, p.props)) {
          continue;
        }
        SCOPED_TRACE("mix " + std::string(to_string(mix)));
        std::mt19937_64 rng(seed * 7 + static_cast<std::uint64_t>(mix));
        typename ct::model_type model = kit_detail::model_of<TypeParam>(size_class::small, rng);
        std::vector<kit_detail::chain<TypeParam>> chains;
        chains.reserve(backends.size());
        for (const backend b : backends) {
          chains.emplace_back(kit_detail::resources_for<TypeParam>(b), model, p.props);
        }
        for (std::size_t i = 1; i < chains.size(); ++i) {
          EXPECT_TRUE(kit_detail::same<traits>(chains[0].take(), chains[i].take()))
              << "compute() on " << to_string(backends[i]) << " differs from sequential";
        }
        for (int step = 0; step < 3; ++step) {
          SCOPED_TRACE("batch " + std::to_string(step));
          const typename ct::generated_type gen = ct::random_batch(model, mix, rng, p.props);
          const auto first = chains[0].step(ct::batch_of(gen));
          ct::observe(model, gen, first.batch);
          const auto reference = kit_detail::counters<traits, graph_t>(first);
          const auto expected = chains[0].take();
          for (std::size_t i = 1; i < chains.size(); ++i) {
            SCOPED_TRACE(std::string(to_string(backends[i])));
            EXPECT_EQ((kit_detail::counters<traits, graph_t>(chains[i].step(ct::batch_of(gen)))),
                      reference);
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
  using ct = kit_detail::axis<TypeParam>;
  for (const backend b : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(b)));
    for (const auto& p : kit_detail::presets<TypeParam>()) {
      SCOPED_TRACE(p.label);
      for (const std::uint64_t seed : test::test_seeds(5000, 2)) {
        SCOPED_TRACE(test::seed_trace(seed));
        for (const auto mix : ct::mixes(kit_check::c5)) {
          if (!ct::mix_applies(mix, p.props)) {
            continue;
          }
          SCOPED_TRACE("mix " + std::string(to_string(mix)));
          std::mt19937_64 rng(seed * 3 + static_cast<std::uint64_t>(mix));
          typename ct::model_type model = kit_detail::model_of<TypeParam>(size_class::small, rng);
          kit_detail::chain<TypeParam> c(kit_detail::resources_for<TypeParam>(b), model, p.props);
          const auto before = c.take();
          const typename ct::generated_type gen = ct::random_batch(model, mix, rng, p.props);
          ASSERT_TRUE(ct::invertible(gen));
          const auto forward = c.step(ct::batch_of(gen));
          ct::observe(model, gen, forward.batch);
          // Graphs: the inverse the generator recorded (with the old weights); hypergraphs: built
          // from the forward batch's hyper_apply_summary (its inserted ids) and the deleted
          // member sets.
          const typename ct::generated_type inverse = ct::inverse(model, gen, forward.batch);
          ct::observe(model, inverse, c.step(ct::batch_of(inverse)).batch);
          EXPECT_TRUE(kit_detail::same<traits>(before, c.take()));
          (void)sizeof(graph_t);
        }
      }
    }
  }
}

// C6: two runs give identical output (and deterministic counters).
TYPED_TEST_P(conformance, C6_RunsAreDeterministic) {
  using traits = typename TypeParam::traits;
  using graph_t = typename TypeParam::graph_type;
  using ct = kit_detail::axis<TypeParam>;
  for (const backend b : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(b)));
    for (const auto& p : kit_detail::presets<TypeParam>()) {
      SCOPED_TRACE(p.label);
      for (const size_class size : {size_class::small, size_class::medium}) {
        for (const auto mix : ct::mixes(kit_check::c6)) {
          SCOPED_TRACE("mix " + std::string(to_string(mix)));
          std::mt19937_64 rng(6000 + static_cast<std::uint64_t>(mix));
          typename ct::model_type model = kit_detail::model_of<TypeParam>(size, rng);
          kit_detail::chain<TypeParam> first(kit_detail::resources_for<TypeParam>(b), model,
                                             p.props);
          kit_detail::chain<TypeParam> second(kit_detail::resources_for<TypeParam>(b), model,
                                              p.props);
          EXPECT_TRUE(kit_detail::same<traits>(first.take(), second.take()));
          for (int step = 0; step < 3; ++step) {
            const typename ct::generated_type gen = ct::random_batch(model, mix, rng, p.props);
            const auto s1 = first.step(ct::batch_of(gen));
            ct::observe(model, gen, s1.batch);
            EXPECT_EQ((kit_detail::counters<traits, graph_t>(s1)),
                      (kit_detail::counters<traits, graph_t>(second.step(ct::batch_of(gen)))));
            EXPECT_TRUE(kit_detail::same<traits>(first.take(), second.take()));
          }
        }
      }
    }
  }
}

// C7: invalid input is rejected with the documented exception (and changes nothing), or counted
// as skipped per the batch semantics (container_checks::invalid_input).
TYPED_TEST_P(conformance, C7_InvalidInputIsRejectedOrCounted) {
  using traits = typename TypeParam::traits;
  for (const backend be : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(be)));
    for (const auto& p : kit_detail::presets<TypeParam>()) {
      SCOPED_TRACE(p.label);
      container_checks<TypeParam>::invalid_input(be, p);
      if (::testing::Test::HasFatalFailure()) {
        return;
      }
    }
  }
}

// C8: budgets of the algorithm phase once reserved (DYNG_DEBUG_BUDGETS builds; invariant I9),
// with engine::automatic and, where a backend has a second engine, engine::operators (a device
// without cooperative launch runs that engine under engine::automatic: M7).
TYPED_TEST_P(conformance, C8_TheAlgorithmPhaseStaysWithinItsBudget) {
  using traits = typename TypeParam::traits;
  using ct = kit_detail::axis<TypeParam>;
  if (!detail::budgets_enabled()) {
    GTEST_SKIP() << "not a DYNG_DEBUG_BUDGETS build (the dev presets are)";
  }
  namespace fw = ::dyng::detail::framework;
  // Strict budgets: an excess throws internal_error here (a Debug build of the library only logs
  // it, framework/budgets.hpp).
  const fw::strict_budgets_scope strict;
  for (const backend b : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(b)));
    for (const auto& p : kit_detail::presets<TypeParam>()) {
      SCOPED_TRACE(p.label);
      for (const size_class size : {size_class::small, size_class::medium}) {
        for (const auto mix : ct::mixes(kit_check::c8)) {
          if (!ct::mix_applies(mix, p.props)) {
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
            typename ct::model_type model = kit_detail::model_of<TypeParam>(size, rng);
            const typename ct::model_type base = model;
            const typename ct::generated_type gen = ct::random_batch(model, mix, rng, p.props);
            const resources res = kit_detail::resources_for<TypeParam>(b);
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
              (void)warm->step(ct::batch_of(gen));
              measured.emplace(res, base, p.props, e);
              host_allocation_counter armed;
              counted = armed.counting();
              try {
                stats = measured->step(ct::batch_of(gen));
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
  using ct = kit_detail::axis<TypeParam>;
  for (const backend b : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(b)));
    for (const auto& p : kit_detail::presets<TypeParam>()) {
      SCOPED_TRACE(p.label);
      for (const auto mix : ct::mixes(kit_check::c9)) {
        if (!ct::mix_applies(mix, p.props)) {
          continue;
        }
        SCOPED_TRACE("mix " + std::string(to_string(mix)));
        std::mt19937_64 rng(9000 + static_cast<std::uint64_t>(mix));
        typename ct::model_type model = kit_detail::model_of<TypeParam>(size_class::small, rng);
        kit_detail::chain<TypeParam> c(kit_detail::resources_for<TypeParam>(b), model, p.props);
        for (int step = 0; step < 3; ++step) {
          const typename ct::generated_type gen = ct::random_batch(model, mix, rng, p.props);
          const auto s = c.step(ct::batch_of(gen));
          ct::observe(model, gen, s.batch);
          container_checks<TypeParam>::stats_sane(c.res, s, gen, model, *c.g);
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
  using ct = kit_detail::axis<TypeParam>;
  int partners = 0;
  for_each_type(typename kit_detail::partners_of<traits>::type{}, [&](auto* partner_tag) {
    using partner_tag_t = std::remove_pointer_t<decltype(partner_tag)>;
    using partner = test_traits<partner_tag_t>;
    if constexpr (!std::is_same_v<partner_tag_t, typename TypeParam::tag> &&
                  contains_v<typename kit_detail::container_types_of<partner>::type, graph_t>) {
      using partner_case = suite_case<partner_tag_t, graph_t>;
      for (const backend b : kit_detail::backends<traits>()) {
        const std::vector<backend> theirs = kit_detail::backends<partner>();
        if (std::find(theirs.begin(), theirs.end(), b) == theirs.end()) {
          continue;
        }
        ++partners;
        SCOPED_TRACE(std::string(traits::name) + " with " + std::string(partner::name) + " on " +
                     std::string(to_string(b)));
        // A container meeting both algorithms' requirements, under each composition preset (graphs:
        // upsert_last_wins(), set(), upsert_last_wins() with the insertions first).
        const auto composition = ct::template composition_presets<traits, partner>();
        for (std::size_t variant = 0; variant < composition.size(); ++variant) {
          SCOPED_TRACE(composition[variant].label);
          const auto& props = composition[variant].props;
          std::mt19937_64 rng(10000 + static_cast<std::uint64_t>(variant));
          typename ct::model_type model =
              ct::template composition_model<traits, partner>(props, rng);
          const resources res = kit_detail::resources_for<TypeParam>(b);
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
            const auto mix = ct::composition_mix(step);
            const typename ct::generated_type gen = ct::random_batch(model, mix, rng, props);
            const auto& batch = ct::batch_of(gen);
            typename traits::stats s_mine;
            typename partner::stats s_partner;
            if (step % 2 == 0) {
              std::tie(s_mine, s_partner) =
                  dyng::update(res, *joint.g, batch.view(), *joint.r, *joint_partner);
            } else {
              std::tie(s_partner, s_mine) =
                  dyng::update(res, *joint.g, batch.view(), *joint_partner, *joint.r);
            }
            ct::observe(model, gen, s_mine.batch);
            const auto a_mine = alone.step(batch);
            const auto a_partner = alone_partner.step(batch);
            EXPECT_TRUE(kit_detail::same<traits>(alone.take(), joint.take()));
            EXPECT_TRUE(kit_detail::same<partner>(
                alone_partner.take(), partner::template take<graph_t>(res, *joint_partner)));
            EXPECT_EQ((kit_detail::counters<traits, graph_t>(s_mine)),
                      (kit_detail::counters<traits, graph_t>(a_mine)));
            EXPECT_EQ((kit_detail::counters<partner, graph_t>(s_partner)),
                      (kit_detail::counters<partner, graph_t>(a_partner)));
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
  using ct = kit_detail::axis<TypeParam>;
  for (const backend b : kit_detail::backends<traits>()) {
    SCOPED_TRACE(std::string(to_string(b)));
    std::mt19937_64 rng(11000);
    typename ct::model_type model = kit_detail::model_of<TypeParam>(size_class::small, rng);
    const kit_detail::preset_of<TypeParam> p = kit_detail::presets<TypeParam>().front();
    kit_detail::chain<TypeParam> c(kit_detail::resources_for<TypeParam>(b), model, p.props);
    const auto before = c.take();
    // A separate g.apply(): the result is stale, and update() changes nothing.
    const typename ct::generated_type gen = ct::random_batch(model, ct::stale_mix(), rng);
    ct::observe(model, gen, c.g->apply(c.res, ct::batch_of(gen).view()));
    const std::uint64_t version = c.g->version();
    using batch_t = typename kit_detail::chain<TypeParam>::batch_type;
    EXPECT_THROW((void)c.step(batch_t{}), stale_result_error);
    EXPECT_THROW((void)dyng::update(c.res, *c.g, ct::batch_of(gen).view(), *c.r),
                 stale_result_error);
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
    using ct = kit_detail::axis<TypeParam>;
    for (const auto& p : kit_detail::presets<TypeParam>()) {
      SCOPED_TRACE(p.label);
      std::mt19937_64 rng(12000);
      typename ct::model_type model = kit_detail::model_of<TypeParam>(size_class::medium, rng);
      kit_detail::chain<TypeParam> host(resources::sequential(), model, p.props);
      kit_detail::chain<TypeParam> device(on_a, model, p.props);
      for (int step = 0; step < 4; ++step) {
        SCOPED_TRACE("batch " + std::to_string(step));
        const typename ct::generated_type gen = ct::random_batch(model, ct::stream_mix(step), rng);
        ct::observe(model, gen, host.step(ct::batch_of(gen)).batch);
        // Alternate the streams: each call orders its work on its own stream.
        device.res = step % 2 == 0 ? on_s : on_a;
        (void)device.step(ct::batch_of(gen));
        (void)sizeof(graph_t);
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
