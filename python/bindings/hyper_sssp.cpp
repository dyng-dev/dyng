// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hyper_sssp.cpp
 * @brief dyng::hyper_sssp: the budget action and fallback reason enumerations, options, stats,
 *        result<int32_t>, compute() and update() on hypergraph<int32_t, int32_t> (the one type
 *        hyper_sssp supports).
 *
 * The distance and parent arrays are exported like sssp's (zero-copy views of the result's
 * state, host or device memory). distance_to() and vertex_distances() read the hypergraph the
 * result matches and fill a cache of the result's state, so the binding holds the result's lock
 * exclusively and the hypergraph's (passed by the typed layer, which keeps a weak reference to
 * it) shared while they run: a concurrent apply() of the hypergraph or a second accessor call
 * cannot race with them.
 */
#include "types.hpp"

#include <dyng/hyper_sssp.hpp>

#include <nanobind/stl/string.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

namespace dyng::python {
namespace {

using hgraph_t = hypergraph_holder<std::int32_t, std::int32_t>;

/// Run `f(result)` with the result's lock exclusive and, if given, its hypergraph's shared (the
/// accessors that read the hypergraph and fill the state's caches).
template <typename function_t>
decltype(auto) with_hypergraph(const hyper_sssp_holder& h, const nb::handle& graph,
                               function_t&& f) {
  const hgraph_t* g = graph.is_none() ? nullptr : &nb::cast<const hgraph_t&>(graph);
  return without_gil([&] {
    lock_set locks;
    locks.add(h.mutex, true);
    if (g != nullptr) {
      locks.add(g->mutex, false);
    }
    locks.lock();
    return f(h);
  });
}

void bind_result(nb::module_& m) {
  using result_t = hyper_sssp::result<std::int32_t>;
  nb::class_<hyper_sssp_holder>(m, "HyperSsspResult", "A hyper_sssp::result (private).",
                                nb::is_weak_referenceable())
      .def_prop_ro("source",
                   [](const hyper_sssp_holder& h) {
                     return read_result(h, [](const result_t& r) {
                       return static_cast<std::int64_t>(r.source());
                     });
                   })
      .def("hyperedge_distances",
           [](const hyper_sssp_holder& h) {
             return export_array(h, [](const result_t& r) { return r.hyperedge_distances(); });
           })
      .def("hyperedge_parents",
           [](const hyper_sssp_holder& h) {
             return export_array(h, [](const result_t& r) { return r.hyperedge_parents(); });
           })
      .def(
          "distance_to",
          [](const hyper_sssp_holder& h, const nb::handle& graph, std::int64_t target) {
            DYNG_EXPECTS(target >= 0 && target <= std::numeric_limits<std::int32_t>::max(),
                         "hyper_sssp::result::distance_to: the vertex ", target,
                         " is outside [0, num_vertices)");
            return with_hypergraph(h, graph, [&](const hyper_sssp_holder& held) {
              return held.get().distance_to(static_cast<std::int32_t>(target));
            });
          },
          nb::arg("hypergraph").none(), nb::arg("target"),
          "The distance of a vertex (the hypergraph the result matches, or None).")
      .def(
          "vertex_distances",
          [](const hyper_sssp_holder& h, const nb::handle& graph) {
            std::shared_ptr<const result_t> state;
            const auto v = with_hypergraph(h, graph, [&](const hyper_sssp_holder& held) {
              state = held.share();
              return state->vertex_distances();
            });
            return owned_view(v, std::shared_ptr<const void>(std::move(state)));
          },
          nb::arg("hypergraph").none(),
          "The distance of every vertex (host memory; the hypergraph the result matches, or "
          "None).")
      .def_prop_ro(
          "writer", [](const hyper_sssp_holder& h) { return writer_of(h); },
          "The resources of the call that last wrote this result (orders its device arrays).")
      .def_prop_ro("generation", &hyper_sssp_holder::generation,
                   "The number of updates of this result (dyng.Array's staleness check).")
      .def_prop_ro("options",
                   [](const hyper_sssp_holder& h) -> hyper_sssp::options {
                     return read_result(h, [](const result_t& r) { return r.get_options(); });
                   })
      .def(
          "set_options",
          [](hyper_sssp_holder& h, const hyper_sssp::options& opt) {
            without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, true);
              locks.lock();
              h.for_options().set_options(opt);
            });
          },
          nb::arg("options"))
      .def_prop_ro("graph_version",
                   [](const hyper_sssp_holder& h) {
                     return read_result(h, [](const result_t& r) { return r.graph_version(); });
                   })
      .def_prop_ro("space",
                   [](const hyper_sssp_holder& h) {
                     return read_result(h, [](const result_t& r) { return r.space(); });
                   })
      .def(
          "clone",
          [](const hyper_sssp_holder& h, const resources& res) {
            return without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, false);
              locks.lock();
              return new hyper_sssp_holder(h.get().clone(res), res);
            });
          },
          nb::arg("resources"), nb::rv_policy::take_ownership);
}

void bind_functions(nb::module_& m) {
  using batch_t = hyper_batch_arrays<std::int32_t, std::int32_t>;
  m.def(
      "hyper_sssp_compute",
      [](const resources& res, const hgraph_t& h, std::int64_t source,
         const hyper_sssp::options& opt) {
        DYNG_EXPECTS(source >= 0 && source < static_cast<std::int64_t>(h.value.num_vertices()),
                     "hyper_sssp::compute: the source vertex ", source, " is outside [0, ",
                     h.value.num_vertices(), ")");
        return without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, false);
          locks.lock();
          return new hyper_sssp_holder(
              hyper_sssp::compute(res, h.value, static_cast<std::int32_t>(source), opt), res);
        });
      },
      nb::arg("resources"), nb::arg("hypergraph"), nb::arg("source"), nb::arg("options"),
      nb::rv_policy::take_ownership);
  m.def(
      "hyper_sssp_update",
      [](const resources& res, hgraph_t& h, const batch_t& b, hyper_sssp_holder& r) {
        return without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, true);
          locks.add(r.mutex, true);
          locks.lock();
          return hyper_sssp::update(res, h.value, b.view(), r.for_update(res));
        });
      },
      nb::arg("resources"), nb::arg("hypergraph"), nb::arg("batch"), nb::arg("result"));
}

}  // namespace

void bind_hyper_sssp(nb::module_& m) {
  nb::enum_<hyper_sssp::budget_action>(m, "HyperSsspBudgetAction", "hyper_sssp::budget_action.")
      .value("recompute", hyper_sssp::budget_action::recompute)
      .value("error", hyper_sssp::budget_action::error);
  nb::enum_<hyper_sssp::fallback_reason>(m, "HyperSsspFallbackReason",
                                         "hyper_sssp::fallback_reason.")
      .value("none", hyper_sssp::fallback_reason::none)
      .value("work_budget", hyper_sssp::fallback_reason::work_budget)
      .value("iterations", hyper_sssp::fallback_reason::iterations)
      .value("distance_width", hyper_sssp::fallback_reason::distance_width)
      .value("invalidation_depth", hyper_sssp::fallback_reason::invalidation_depth);
  nb::class_<hyper_sssp::options>(m, "HyperSsspOptions", "hyper_sssp::options.")
      .def(nb::init<>())
      .def_rw("work_budget", &hyper_sssp::options::work_budget)
      .def_rw("max_iterations", &hyper_sssp::options::max_iterations)
      .def_rw("on_budget", &hyper_sssp::options::on_budget)
      .def_rw("cuda_engine", &hyper_sssp::options::cuda_engine)
      .def_rw("validate_inputs", &hyper_sssp::options::validate_inputs);
  nb::class_<hyper_sssp::stats>(m, "HyperSsspStats", "hyper_sssp::stats.")
      .def_ro("affected", &update_stats::affected)
      .def_ro("iterations", &update_stats::iterations)
      .def_ro("frontier_visits", &update_stats::frontier_visits)
      .def_ro("fallback_used", &update_stats::fallback_used)
      .def_ro("converged", &update_stats::converged)
      .def_ro("engine_used", &update_stats::engine_used)
      .def_ro("batch", &hyper_sssp::stats::batch)
      .def_ro("line_graph_insertions", &hyper_sssp::stats::line_graph_insertions)
      .def_ro("line_graph_deletions", &hyper_sssp::stats::line_graph_deletions)
      .def_ro("invalidated", &hyper_sssp::stats::invalidated)
      .def_ro("relaxations", &hyper_sssp::stats::relaxations)
      .def_ro("fallback_reason", &hyper_sssp::stats::fallback_reason);
  bind_result(m);
  bind_functions(m);
}

}  // namespace dyng::python
