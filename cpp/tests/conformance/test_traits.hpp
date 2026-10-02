// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file test_traits.hpp
 * @brief What an algorithm tells the conformance kit (PLAN Section 8.2): the primary template
 *        test_traits<tag> and the vocabulary its specialisations use.
 *
 * An algorithm's traits (cpp/tests/algorithms/<name>/<name>_traits.hpp, 20-60 lines) specialise
 * test_traits<tags::<name>> with:
 *
 *     static constexpr std::string_view name;            // as in its manifest
 *     static constexpr oracle_kind oracle;               // compute | reference (as the manifest)
 *     static constexpr determinism level;                // bitwise | exact_value | tolerance
 *     static constexpr bool history_independent;         // C5: a batch and its inverse cancel
 *     using graph_types = type_list<graph<V, E, W>...>;  // the instantiated graph types
 *     template <typename graph_t> using result = ...;    // the result type on graph_t
 *     using stats = ...;                                 // update()'s stats (with `batch`)
 *     using snapshot = ...;                              // a host copy of a result: ==, PrintTo
 *     static void require(graph_properties&);            // the graph requirements
 *     static graph_shape shape(size_class);              // the graphs the kit generates
 *     template <typename graph_t>
 *     static result<graph_t> compute(const resources&, const graph_t&, engine);
 *     template <typename graph_t>
 *     static stats update(const resources&, graph_t&, const edge_batch_view<V, W>&,
 *                         result<graph_t>&);
 *     template <typename graph_t>
 *     static snapshot take(const resources&, const result<graph_t>&);
 *     static std::vector<std::int64_t> deterministic(const stats&);   // counters equal across runs
 *                                                                     // and backends
 *     template <typename graph_t>
 *     static std::vector<std::function<void()>> invalid_options(const resources&, const graph_t&);
 *     static std::int64_t host_sync_budget(backend);    // C8: syncs of the algorithm phase
 *
 * and optionally:
 *
 *     static ::testing::AssertionResult compare(const snapshot&, const snapshot&);  // default ==;
 *                                                     // required for determinism::tolerance
 *     template <typename graph_t> static snapshot oracle_of(const graph_t&);  // an independent
 *                                                     // host oracle (dyng::testing), checked in C2
 *     static std::vector<graph_properties> extra_properties();  // more presets for C2 (e.g. MOSP's)
 *     template <typename graph_t>                    // oracle_kind::reference only: the converged
 *     static ::testing::AssertionResult near_reference(const graph_t&, const snapshot&);
 *     static constexpr int num_weights;              // weight columns of the kit's graphs (default
 *                                                    // 1; C7 uses num_weights + 1, C10 the larger
 *                                                    // of a pair; mosp: K)
 *     static std::int64_t host_sync_budget(backend, const stats&);  // C8, preferred over the
 *                                                    // one-argument form: a budget that depends on
 *                                                    // the engine that ran (stats::engine_used) and
 *                                                    // its counters, or run_dependent_budget
 *
 * C8 runs with engine::automatic and, where a backend has a second engine, with
 * engine::operators (on a device without cooperative launch engine::automatic runs that engine).
 *
 * The kit reads everything else (the backends, the maturity) from the registry
 * (dyng::algorithms(), generated from the manifest), and C0 checks that the traits agree with it.
 */
#pragma once

#include "conformance/type_list.hpp"

#include <dyng/core/registry.hpp>
#include <dyng/core/types.hpp>

#include <cstdint>
#include <type_traits>

namespace dyng::conformance {

/**
 * @brief The traits of one algorithm (see the file comment); specialised per tag.
 * @tparam tag_t The algorithm's tag (tags::<name>, conformance/registry.hpp).
 */
template <typename tag_t>
struct test_traits;

/// host_sync_budget(backend, stats) of a run whose budget depends on counters its stats do not
/// carry: C8 then checks the phase against the problem's own (bounded) budget.
inline constexpr std::int64_t run_dependent_budget = -2;

/// The sizes of the generated graphs.
enum class size_class : std::uint8_t {
  tiny,    ///< a handful of vertices (edge cases: empty rows, one cycle)
  small,   ///< tens of vertices
  medium,  ///< a few hundred vertices
};

/// The shape of a generated graph.
struct graph_shape {
  std::int64_t vertices = 0;  ///< n
  std::int64_t edges = 0;     ///< m (distinct, no self-loops)
  int max_weight = 9;         ///< weights are drawn from [1, max_weight]
};

/// Whether test_traits<tag_t> is defined (the algorithm is built into this test binary).
template <typename tag_t, typename = void>
struct has_traits : std::false_type {};
/// @copydoc has_traits
template <typename tag_t>
struct has_traits<tag_t, std::void_t<decltype(sizeof(test_traits<tag_t>))>> : std::true_type {};

/// The one-parameter form of has_traits, for template template parameters such as filter's
/// predicate. Clang < 19 does not apply C++17's relaxed template template matching (P0522) by
/// default, so has_traits itself (with its defaulted SFINAE parameter) is not accepted there.
template <typename tag_t>
using has_test_traits = has_traits<tag_t>;

}  // namespace dyng::conformance
