// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file conformance_registration.cpp
 * @brief Compile-failure tests of the conformance kit's registration rules (invariant I8, PLAN
 *        Section 8.2): DYNG_CONFORMANCE_SUITE stops at a static_assert with a plain-English message
 *        when the algorithm's test_traits break a rule. Compiled with -fsyntax-only and one of the
 *        defines below (cpp/tests/CMakeLists.txt checks the message); without a define it compiles.
 *
 *   DYNG_CF_NO_TRAITS          no test_traits specialisation
 *   DYNG_CF_NO_COMPUTE         no compute()
 *   DYNG_CF_NO_UPDATE          no update()
 *   DYNG_CF_NO_ORACLE          no oracle kind
 *   DYNG_CF_NO_LEVEL           no determinism level
 *   DYNG_CF_STATS_NO_BATCH     stats without `apply_summary batch`
 *   DYNG_CF_TOLERANCE_NO_CMP   determinism::tolerance without compare()
 *   DYNG_CF_REFERENCE_NO_NEAR  oracle_kind::reference without near_reference()
 *
 * The rule "a sequential backend" is not a compile-time check: the kit reads the backends from
 * the registry, so C0 checks it at run time and scripts/regen.py when it reads the manifest (ADR
 * 0022).
 */
#include "conformance/conformance.hpp"

#include <dyng/core/registry.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stats.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <cstdint>
#include <string_view>
#include <vector>

namespace dyng::conformance {

namespace tags {
struct cf_algo;
}  // namespace tags

namespace cf {
using graph_type = graph<std::int32_t, std::int32_t, unweighted>;
struct result {};
struct stats : update_stats {
#if !defined(DYNG_CF_STATS_NO_BATCH)
  apply_summary batch;
#endif
};
}  // namespace cf

#if !defined(DYNG_CF_NO_TRAITS)
template <>
struct test_traits<tags::cf_algo> {
  static constexpr std::string_view name = "cf_algo";
#if !defined(DYNG_CF_NO_ORACLE)
#if defined(DYNG_CF_REFERENCE_NO_NEAR)
  static constexpr oracle_kind oracle = oracle_kind::reference;
#else
  static constexpr oracle_kind oracle = oracle_kind::compute;
#endif
#endif
#if !defined(DYNG_CF_NO_LEVEL)
#if defined(DYNG_CF_TOLERANCE_NO_CMP)
  static constexpr determinism level = determinism::tolerance;
#else
  static constexpr determinism level = determinism::exact_value;
#endif
#endif
  using graph_types = type_list<cf::graph_type>;
  template <typename graph_t>
  using result = cf::result;
  using stats = cf::stats;
  using snapshot = std::vector<int>;
#if !defined(DYNG_CF_NO_COMPUTE)
  template <typename graph_t>
  static result<graph_t> compute(const resources&, const graph_t&, engine) {
    return {};
  }
#endif
#if !defined(DYNG_CF_NO_UPDATE)
  template <typename graph_t>
  static stats update(
      const resources&, graph_t&,
      const edge_batch_view<typename graph_t::vertex_type, typename graph_t::weight_type>&,
      result<graph_t>&) {
    return {};
  }
#endif
};
#endif

static_assert(registration_check<tags::cf_algo>());

}  // namespace dyng::conformance
