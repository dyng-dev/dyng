// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file testing.cpp
 * @brief dyng::testing: the host oracles (Dijkstra with lowest-id ties, the sssp tree check and
 *        the simple-cycle oracles), which share no code with the algorithms they check.
 */
#include "types.hpp"

#include <dyng/testing/check_sssp.hpp>
#include <dyng/testing/cycle_oracle.hpp>
#include <dyng/testing/dijkstra.hpp>

#include <nanobind/stl/string.h>

namespace dyng::python {
namespace {

template <typename vertex_t, typename edge_t, typename weight_t>
void bind_testing_type(nb::module_& m) {
  using graph_t = graph_holder<vertex_t, edge_t, weight_t>;
  if constexpr (detail::sssp_supported_v<vertex_t, edge_t, weight_t>) {
    m.def(
        "testing_dijkstra",
        [](const graph_t& g, std::int64_t source, int objective) {
          DYNG_EXPECTS(source >= 0 && source < static_cast<std::int64_t>(g.value.num_vertices()),
                       "dijkstra: the source ", source, " is out of range");
          auto tree = without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, false);
            locks.lock();
            return testing::dijkstra(g.value, static_cast<vertex_t>(source), objective);
          });
          return nb::make_tuple(to_numpy(std::move(tree.distances)),
                                to_numpy(std::move(tree.parents)));
        },
        nb::arg("graph"), nb::arg("source"), nb::arg("objective"));
    m.def(
        "testing_check_sssp_tree",
        [](const graph_t& g, const sssp_holder<vertex_t>& r, bool require_canonical) {
          return without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, false);
            locks.add(r.mutex, false);
            locks.lock();
            return testing::check_sssp_tree(g.value, r.value, require_canonical);
          });
        },
        nb::arg("graph"), nb::arg("result"), nb::arg("require_canonical"));
  }
  m.def(
      "testing_simple_cycles",
      [](const graph_t& g, int max_length, bool brute_force) {
        auto hist = without_gil([&] {
          lock_set locks;
          locks.add(g.mutex, false);
          locks.lock();
          const auto out = g.value.view().out;
          return brute_force ? testing::brute_force_simple_cycles(out, max_length)
                             : testing::oracle_simple_cycles(out, max_length);
        });
        return to_numpy(std::move(hist));
      },
      nb::arg("graph"), nb::arg("max_length"), nb::arg("brute_force"));
}

}  // namespace

void bind_testing(nb::module_& m) {
  nb::class_<testing::sssp_tree_check>(m, "SsspTreeCheck", "testing::sssp_tree_check.")
      .def_ro("distance_mismatches", &testing::sssp_tree_check::distance_mismatches)
      .def_ro("inconsistent_parents", &testing::sssp_tree_check::inconsistent_parents)
      .def_ro("non_canonical_parents", &testing::sssp_tree_check::non_canonical_parents)
      .def_ro("parent_mismatches", &testing::sssp_tree_check::parent_mismatches)
      .def_ro("require_canonical", &testing::sssp_tree_check::require_canonical)
      .def("ok", &testing::sssp_tree_check::ok)
      .def("summary", &testing::sssp_tree_check::summary);
#define DYNG_PY_BIND_TESTING(V, E, W) bind_testing_type<V, E, W>(m);
  DYNG_PY_FOR_EACH_GRAPH_TYPE(DYNG_PY_BIND_TESTING)
#undef DYNG_PY_BIND_TESTING
}

}  // namespace dyng::python
