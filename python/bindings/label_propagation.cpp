// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file label_propagation.cpp
 * @brief dyng::label_propagation (0.4): options, stats, result<int32>, compute() and update() for
 *        the float graphs (int32 vertex ids, int32 or int64 edge offsets), and its oracle
 *        dyng::testing::harmonic_solution (testing_harmonic_solution, re-exported by dyng.testing).
 */
#include "types.hpp"

#include <dyng/label_propagation.hpp>
#include <dyng/testing/harmonic_solution.hpp>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace dyng::python {
namespace {

namespace lp = label_propagation;

/// The holder of a label_propagation result.
using lp_holder = result_holder<lp::result<std::int32_t>>;

void bind_result(nb::module_& m) {
  nb::class_<lp_holder>(m, "LabelPropagationResult", "A label_propagation::result (private).",
                        nb::is_weak_referenceable())
      .def("labels",
           [](const lp_holder& h) {
             return export_array(h, [](const auto& r) { return r.labels(); });
           })
      .def("seed_labels",
           [](const lp_holder& h) {
             return export_array(h, [](const auto& r) { return r.seed_labels(); });
           })
      .def_prop_ro(
          "writer", [](const lp_holder& h) { return writer_of(h); },
          "The resources of the call that last wrote this result (orders its device arrays).")
      .def_prop_ro("generation", &lp_holder::generation,
                   "The number of updates of this result (dyng.Array's staleness check).")
      .def_prop_ro("options",
                   [](const lp_holder& h) -> lp::options {
                     return read_result(h, [](const auto& r) { return r.get_options(); });
                   })
      .def(
          "set_options",
          [](lp_holder& h, const lp::options& opt) {
            without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, true);
              locks.lock();
              h.for_options().set_options(opt);
            });
          },
          nb::arg("options"))
      .def_prop_ro("graph_version",
                   [](const lp_holder& h) {
                     return read_result(h, [](const auto& r) { return r.graph_version(); });
                   })
      .def_prop_ro("space",
                   [](const lp_holder& h) {
                     return read_result(h, [](const auto& r) { return r.space(); });
                   })
      .def(
          "clone",
          [](const lp_holder& h, const resources& res) {
            return without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, false);
              locks.lock();
              return new lp_holder(h.get().clone(res), res);
            });
          },
          nb::arg("resources"), nb::rv_policy::take_ownership)
      .def(
          "reserve",
          [](lp_holder& h, const resources& res, std::int64_t max_batch_vertices,
             std::int64_t max_batch_edges) {
            DYNG_EXPECTS(max_batch_vertices >= 0 && max_batch_edges >= 0,
                         "label_propagation::result::reserve: negative sizes (", max_batch_vertices,
                         ", ", max_batch_edges, ")");
            without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, true);
              locks.lock();
              // The workspaces only: the labels (and every array exported from them) are kept.
              h.for_options().reserve(res, static_cast<std::size_t>(max_batch_vertices),
                                      static_cast<std::size_t>(max_batch_edges));
            });
          },
          nb::arg("resources"), nb::arg("max_batch_vertices"), nb::arg("max_batch_edges"));
}

template <typename vertex_t, typename edge_t, typename weight_t>
void bind_functions(nb::module_& m) {
  if constexpr (detail::label_propagation_supported_v<vertex_t, edge_t, weight_t>) {
    using graph_t = graph_holder<vertex_t, edge_t, weight_t>;
    m.def(
        "label_propagation_compute",
        [](const resources& res, const graph_t& g, const in_array<vertex_t>& seed_vertices,
           const in_array<std::int8_t>& seed_labels, const lp::options& opt) {
          return without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, false);
            locks.lock();
            return new lp_holder(
                lp::compute(res, g.value, view_of(seed_vertices), view_of(seed_labels), opt), res);
          });
        },
        nb::arg("resources"), nb::arg("graph"), nb::arg("seed_vertices").noconvert(),
        nb::arg("seed_labels").noconvert(), nb::arg("options"), nb::rv_policy::take_ownership);
    m.def(
        "label_propagation_update",
        [](const resources& res, graph_t& g, const batch_arrays<vertex_t, weight_t>& b,
           lp_holder& r) {
          return without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, true);
            locks.add(r.mutex, true);
            locks.lock();
            return lp::update(res, g.value, b.view(), r.for_update(res));
          });
        },
        nb::arg("resources"), nb::arg("graph"), nb::arg("batch"), nb::arg("result"));
  } else {
    (void)m;
  }
}

/// testing::harmonic_solution of a float graph's current edges (0.4, label_propagation's oracle):
/// the present vertices are the active ones (every vertex of compact rows), the edges those of
/// to_csr(), and the seed labels one per vertex (-1: none).
template <typename vertex_t, typename edge_t, typename weight_t>
void bind_harmonic_solution(nb::module_& m) {
  if constexpr (!std::is_same_v<weight_t, float>) {
    (void)m;
  } else {
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
}

}  // namespace

void bind_label_propagation(nb::module_& m) {
  nb::enum_<lp::schedule>(m, "LpSchedule", "label_propagation::schedule.")
      .value("in_place", lp::schedule::in_place)
      .value("damped", lp::schedule::damped);
  nb::enum_<lp::on_limit>(m, "LpOnLimit", "label_propagation::on_limit.")
      .value("error", lp::on_limit::error)
      .value("report", lp::on_limit::report)
      .value("fallback_recompute", lp::on_limit::fallback_recompute);
  nb::class_<lp::options>(m, "LpOptions", "label_propagation::options.")
      .def(nb::init<>())
      .def_rw("similarity_threshold", &lp::options::similarity_threshold, nb::arg().none())
      .def_rw("tolerance", &lp::options::tolerance)
      .def_rw("max_iterations", &lp::options::max_iterations)
      .def_rw("on_iteration_limit", &lp::options::on_iteration_limit)
      .def_rw("update_schedule", &lp::options::update_schedule)
      .def_rw("cuda_engine", &lp::options::cuda_engine);
  nb::class_<lp::stats>(m, "LpStats", "label_propagation::stats.")
      .def_ro("affected", &update_stats::affected)
      .def_ro("iterations", &update_stats::iterations)
      .def_ro("frontier_visits", &update_stats::frontier_visits)
      .def_ro("fallback_used", &update_stats::fallback_used)
      .def_ro("converged", &update_stats::converged)
      .def_ro("engine_used", &update_stats::engine_used)
      .def_ro("batch", &lp::stats::batch)
      .def_ro("components", &lp::stats::components)
      .def_ro("sparsified_edges", &lp::stats::sparsified_edges)
      .def_ro("initial_frontier", &lp::stats::initial_frontier);
  bind_result(m);
#define DYNG_PY_BIND_LABEL_PROPAGATION(V, E, W) \
  bind_functions<V, E, W>(m);                   \
  bind_harmonic_solution<V, E, W>(m);
  DYNG_PY_FOR_EACH_GRAPH_TYPE(DYNG_PY_BIND_LABEL_PROPAGATION)
#undef DYNG_PY_BIND_LABEL_PROPAGATION
}

}  // namespace dyng::python
