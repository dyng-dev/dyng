// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file testing.cpp
 * @brief dyng::testing: the host oracles (Dijkstra with lowest-id ties, the sssp tree check, the
 *        simple-cycle oracles and the mosp references), which share no code with the algorithms
 *        they check.
 */
#include "types.hpp"

#include <dyng/testing/check_sssp.hpp>
#include <dyng/testing/cycle_oracle.hpp>
#include <dyng/testing/dijkstra.hpp>
#include <dyng/testing/mosp_oracle.hpp>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <vector>

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
            return testing::check_sssp_tree(g.value, r.get(), require_canonical);
          });
        },
        nb::arg("graph"), nb::arg("result"), nb::arg("require_canonical"));
    m.def(
        "testing_mosp_path_costs",
        [](const graph_t& g, const in_array<vertex_t>& parents, std::int64_t source,
           int num_objectives) {
          DYNG_EXPECTS(source >= 0 && source < static_cast<std::int64_t>(g.value.num_vertices()),
                       "mosp_path_costs: the source ", source, " is out of range");
          std::vector<vertex_t> tree(parents.data(), parents.data() + parents.shape(0));
          std::size_t cols = 0;
          auto costs = without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, false);
            locks.lock();
            const auto out = g.value.view().out;
            cols = static_cast<std::size_t>(num_objectives > 0 ? num_objectives : out.num_weights);
            return testing::mosp_path_costs_reference(out, tree, static_cast<vertex_t>(source),
                                                      num_objectives);
          });
          const std::size_t rows = cols == 0 ? 0 : costs.size() / cols;
          return to_numpy_2d(std::move(costs), rows, cols);
        },
        nb::arg("graph"), nb::arg("parents").noconvert(), nb::arg("source"),
        nb::arg("num_objectives"));
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

template <typename vertex_t>
void bind_combined_graph(nb::module_& m) {
  m.def(
      suffixed("testing_combined_graph", {type_code<vertex_t>::lower}, "_").c_str(),
      [](const std::vector<in_array<vertex_t>>& parents, std::int64_t source,
         const std::vector<std::int32_t>& preferences) {
        std::vector<std::vector<vertex_t>> trees;
        trees.reserve(parents.size());
        for (const auto& p : parents) {
          trees.emplace_back(p.data(), p.data() + p.shape(0));
        }
        auto c = without_gil([&] {
          return testing::combined_graph_reference(trees, static_cast<vertex_t>(source),
                                                   preferences);
        });
        return nb::make_tuple(to_numpy(std::move(c.row_ptr)), to_numpy(std::move(c.col_ind)),
                              to_numpy(std::move(c.weights)));
      },
      nb::arg("parents"), nb::arg("source"), nb::arg("preferences"));
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
  bind_combined_graph<std::int32_t>(m);
  bind_combined_graph<std::int64_t>(m);
}

}  // namespace dyng::python
