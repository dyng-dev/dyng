// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file testing.cpp
 * @brief dyng::testing: the host oracles (Dijkstra with lowest-id ties, the sssp tree check, the
 *        simple-cycle oracles, the mosp references, the line-graph oracle of hypergraphs and, in
 *        0.4, label_propagation's harmonic solution), which share no code with the algorithms they
 *        check.
 */
#include "types.hpp"

#include <dyng/testing/check_sssp.hpp>
#include <dyng/testing/cycle_oracle.hpp>
#include <dyng/testing/dijkstra.hpp>
#include <dyng/testing/harmonic_solution.hpp>
#include <dyng/testing/line_graph_oracle.hpp>
#include <dyng/testing/mosp_oracle.hpp>
#include <dyng/testing/triad_reference.hpp>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace dyng::python {
namespace {

/// testing::harmonic_solution of a float graph's current edges (0.4, label_propagation's oracle):
/// the present vertices are the active ones (every vertex of compact rows), the edges those of
/// to_csr(), and the seed labels one per vertex (-1: none).
template <typename vertex_t, typename edge_t>
void bind_harmonic_solution(nb::module_& m) {
  using graph_t = graph_holder<vertex_t, edge_t, float>;
  m.def(
      "testing_harmonic_solution",
      [](const resources& res, const graph_t& g, const in_array<std::int8_t>& seed_labels,
         double relative_tolerance, std::int64_t max_iterations, int num_threads) {
        auto out = without_gil([&] {
          lock_set locks;
          locks.add(g.mutex, false);
          locks.lock();
          const std::size_t n = static_cast<std::size_t>(g.value.num_vertices());
          DYNG_EXPECTS(seed_labels.shape(0) == n, "harmonic_solution: ", seed_labels.shape(0),
                       " seed labels for ", n, " vertices");
          auto csr = g.value.to_csr(res);
          std::vector<std::uint8_t> present(n, 1);
          const auto status = g.value.view().status;
          if (!status.empty()) {
            for (std::size_t v = 0; v < n; ++v) {
              present[v] = status[v] == vertex_status::active ? 1 : 0;
            }
          }
          std::vector<std::int8_t> seeds(seed_labels.data(), seed_labels.data() + n);
          testing::harmonic_options opt;
          opt.relative_tolerance = relative_tolerance;
          opt.max_iterations = max_iterations;
          opt.num_threads = num_threads;
          std::vector<std::int32_t> col(csr.col_ind.begin(), csr.col_ind.end());
          return testing::harmonic_solution(csr.row_ptr, col, csr.weights, present, seeds, opt);
        });
        std::vector<std::uint8_t> determined = std::move(out.determined);
        nb::dict d;
        d["converged"] = out.converged;
        d["relative_residual"] = out.relative_residual;
        d["iterations"] = out.iterations;
        d["num_seeds"] = out.num_seeds;
        d["num_unlabelled"] = out.num_unlabelled;
        d["num_determined"] = out.num_determined;
        d["num_excluded"] = out.num_excluded;
        return nb::make_tuple(to_numpy(std::move(out.solution)), to_numpy(std::move(determined)),
                              d);
      },
      nb::arg("resources"), nb::arg("graph"), nb::arg("seed_labels").noconvert(),
      nb::arg("relative_tolerance"), nb::arg("max_iterations"), nb::arg("num_threads"),
      "testing::harmonic_solution of the graph's current edges: (solution float64, determined "
      "uint8, the counters).");
}

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
  if constexpr (std::is_same_v<weight_t, float>) {
    bind_harmonic_solution<vertex_t, edge_t>(m);
  } else {
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
  m.def(
      "testing_rebuild_line_graph",
      [](const in_array<std::int64_t>& offsets, const in_array<std::int32_t>& members,
         std::int64_t num_vertices) {
        DYNG_EXPECTS(num_vertices >= 0 && num_vertices <= std::numeric_limits<std::int32_t>::max(),
                     "rebuild_line_graph: num_vertices ", num_vertices, " does not fit int32");
        std::vector<std::int64_t> o(offsets.data(), offsets.data() + offsets.shape(0));
        std::vector<std::int32_t> mem(members.data(), members.data() + members.shape(0));
        auto rows = without_gil([&] {
          return testing::rebuild_line_graph(o, mem, static_cast<std::int32_t>(num_vertices));
        });
        return nb::make_tuple(to_numpy(std::move(rows.offsets)),
                              to_numpy(std::move(rows.neighbours)));
      },
      nb::arg("offsets").noconvert(), nb::arg("members").noconvert(), nb::arg("num_vertices"),
      "MOSP_ESCHER's rebuildLineGraph over h2v rows: (offsets, neighbours).");
  m.def(
      "testing_triad_reference",
      [](const in_array<std::int64_t>& offsets, const in_array<std::int32_t>& members,
         std::int64_t num_vertices, const std::string& pattern, int threads) {
        DYNG_EXPECTS(num_vertices >= 0 && num_vertices <= std::numeric_limits<std::int32_t>::max(),
                     "triad_reference: num_vertices ", num_vertices, " does not fit int32");
        const testing::triad_pattern all[] = {
            testing::triad_pattern::h_motif, testing::triad_pattern::stathyper_type1,
            testing::triad_pattern::stathyper_type2, testing::triad_pattern::stathyper_type3};
        const testing::triad_pattern* p = nullptr;
        for (const testing::triad_pattern& q : all) {
          if (pattern == testing::triad_pattern_name(q)) {
            p = &q;
          }
        }
        DYNG_EXPECTS(p != nullptr, "triad_reference: unknown pattern '", pattern, "'");
        std::vector<std::int64_t> o(offsets.data(), offsets.data() + offsets.shape(0));
        std::vector<std::int32_t> mem(members.data(), members.data() + members.shape(0));
        testing::triad_reference_options opt;
        opt.threads = threads;
        auto counts = without_gil([&] {
          return testing::triad_reference(o, mem, static_cast<std::int32_t>(num_vertices), *p, opt);
        });
        return to_numpy(std::move(counts));
      },
      nb::arg("offsets").noconvert(), nb::arg("members").noconvert(), nb::arg("num_vertices"),
      nb::arg("pattern"), nb::arg("threads"),
      "ESCHER-GPU's CPU reference triad counts of a hyperedge list (testing::triad_reference).");
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
