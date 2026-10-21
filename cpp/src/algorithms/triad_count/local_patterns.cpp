// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file local_patterns.cpp
 * @brief experimental::count_local_patterns (M9b, ADR 0058): the non-template parts of the
 *        experimental extension point that libdyng compiles: the shapes' names and the row
 *        accessor (experimental::access_rows) for the hypergraph types of triad_count.
 *
 * The driver of count_local_patterns is triad_count's (triad_count.cpp: local_pattern_problem
 * with the flavor local_patterns_flavor, local_patterns_compute / _update / _participant and the
 * result's accessors); the per-Op kernels and host work are the user's
 * (<dyng/experimental/local_patterns.cuh>).
 */
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/instantiate.hpp"
#include "hypergraph/row_views.hpp"

#include <dyng/core/error.hpp>
#include <dyng/experimental/local_patterns.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <cstdint>
#include <string_view>

namespace dyng::experimental {

std::string_view to_string(shape s) noexcept {
  switch (s) {
    case shape::triangles:
      return "triangles";
    case shape::pairs:
      return "pairs";
    case shape::keys:
      return "keys";
  }
  return "unknown";
}

}  // namespace dyng::experimental

namespace dyng::detail {

template <typename vertex_t, typename weight_t>
experimental::row_access local_patterns_access_rows(const hypergraph<vertex_t, weight_t>& h) {
  DYNG_EXPECTS(!hypergraph_access::impl(h).poisoned,
               "experimental::access_rows: the hypergraph was poisoned by a failed apply(); "
               "rebuild it with from_hyperedges()");
  const hyper::row_views v = hypergraph_access::views(h);
  const local_patterns::line_rows line =
      v.has_h2h
          ? local_patterns::line_rows{v.h2h.flat, v.h2h.loc, v.h2h.len, v.h2h.out, v.h2h.max_key}
          : local_patterns::line_rows{};
  return local_patterns::rows_factory::make_access(
      local_patterns::rows_factory::make(v.h2v.flat, v.h2v.loc, v.h2v.len), line, v.h2v.max_key,
      v.id_bound, v.space, v.device, v.has_h2h);
}

#define DYNG_INSTANTIATE_LOCAL_PATTERNS_ROWS(V, W) \
  template experimental::row_access local_patterns_access_rows<V, W>(const hypergraph<V, W>&);
DYNG_FOR_EACH_HYPERGRAPH_TYPE(DYNG_INSTANTIATE_LOCAL_PATTERNS_ROWS)
#undef DYNG_INSTANTIATE_LOCAL_PATTERNS_ROWS

}  // namespace dyng::detail
