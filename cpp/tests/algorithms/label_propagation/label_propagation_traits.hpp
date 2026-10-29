// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file label_propagation_traits.hpp
 * @brief The conformance-kit traits of label_propagation (PLAN Section 8.2; cpp/tests/conformance):
 *        the registration (M10a step 3) without the kit's checks yet.
 *
 * label_propagation runs on float graphs with row_layout::slotted and vertex batches, which the
 * kit's graph axis (compact graphs with edge batches) does not generate. Its container axis (the
 * float slotted graphs: arrivals with labels, departures, mixed batches, Ins and Del in one batch,
 * edges between present vertices), the oracle kind "reference" with harmonic_solution as
 * near_reference and the checks C2-C10 arrive in M10a step 5 (ADR 0072); until then these traits
 * only register the algorithm with the kit (its name, oracle kind, determinism and container
 * types: the registry test compares them with the manifest, and no other algorithm composes with
 * float graphs in C10), and label_propagation_conformance_test.cpp states that the suite is not
 * run. Its hand cases (label_propagation_test.cpp) and the replays against the original
 * (compare.py label_propagation) check it meanwhile.
 */
#pragma once

#include "conformance/test_traits.hpp"
#include "conformance/type_list.hpp"

#include <dyng/core/types.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/label_propagation.hpp>

#include <cstdint>
#include <string_view>

namespace dyng::conformance {

/// label_propagation: registered with the kit; its checks arrive in M10a step 5 (see above).
template <>
struct test_traits<tags::label_propagation> {
  static constexpr std::string_view name = "label_propagation";  ///< as in the manifest
  static constexpr oracle_kind oracle = oracle_kind::reference;  ///< the harmonic solution
  static constexpr determinism level = determinism::tolerance;   ///< in_place; damped bitwise
  static constexpr bool history_independent = false;  ///< C5 does not apply (single-use ids)
  /// The instantiated graph types (float similarity graphs).
  using graph_types =
      type_list<graph<std::int32_t, std::int32_t, float>, graph<std::int32_t, std::int64_t, float>>;
  template <typename graph_t>
  using result = label_propagation::result<std::int32_t>;  ///< the result
  using stats = label_propagation::stats;                  ///< update()'s stats
};

}  // namespace dyng::conformance
