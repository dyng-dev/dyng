// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triangle_delta.cpp
 * @brief dyng::triangle_delta (a tutorial algorithm): options, stats, result, compute() and
 *        update() for every graph type triangle_delta supports.
 */
#include "types.hpp"

#include <dyng/triangle_delta.hpp>

namespace dyng::python {
namespace {

/// The holder of a triangle_delta result.
using triangle_delta_holder = result_holder<triangle_delta::result>;

template <typename vertex_t, typename edge_t, typename weight_t>
void bind_functions(nb::module_& m) {
  if constexpr (detail::triangle_delta_supported_v<vertex_t, edge_t, weight_t>) {
    using graph_t = graph_holder<vertex_t, edge_t, weight_t>;
    m.def(
        "triangle_delta_compute",
        [](const resources& res, const graph_t& g, const triangle_delta::options& opt) {
          return without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, false);
            locks.lock();
            return new triangle_delta_holder(triangle_delta::compute(res, g.value, opt), res);
          });
        },
        nb::arg("resources"), nb::arg("graph"), nb::arg("options"), nb::rv_policy::take_ownership);
    m.def(
        "triangle_delta_update",
        [](const resources& res, graph_t& g, const batch_arrays<vertex_t, weight_t>& b,
           triangle_delta_holder& r) {
          return without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, true);
            locks.add(r.mutex, true);
            locks.lock();
            return triangle_delta::update(res, g.value, b.view(), r.for_update(res));
          });
        },
        nb::arg("resources"), nb::arg("graph"), nb::arg("batch"), nb::arg("result"));
  } else {
    (void)m;
  }
}

}  // namespace

void bind_triangle_delta(nb::module_& m) {
  nb::class_<triangle_delta::options>(m, "TriangleDeltaOptions", "triangle_delta::options.")
      .def(nb::init<>());
  nb::class_<triangle_delta::stats>(m, "TriangleDeltaStats", "triangle_delta::stats.")
      .def_ro("affected", &update_stats::affected)
      .def_ro("iterations", &update_stats::iterations)
      .def_ro("frontier_visits", &update_stats::frontier_visits)
      .def_ro("fallback_used", &update_stats::fallback_used)
      .def_ro("converged", &update_stats::converged)
      .def_ro("engine_used", &update_stats::engine_used)
      .def_ro("batch", &triangle_delta::stats::batch)
      .def_ro("deletions", &triangle_delta::stats::deletions)
      .def_ro("insertions", &triangle_delta::stats::insertions)
      .def_ro("triangles_removed", &triangle_delta::stats::triangles_removed)
      .def_ro("triangles_added", &triangle_delta::stats::triangles_added);

  nb::class_<triangle_delta_holder>(m, "TriangleDeltaResult", "A triangle_delta::result (private).",
                                    nb::is_weak_referenceable())
      .def_prop_ro("count",
                   [](const triangle_delta_holder& h) {
                     return read_result(h,
                                        [](const triangle_delta::result& r) { return r.count(); });
                   })
      .def_prop_ro(
          "writer", [](const triangle_delta_holder& h) { return writer_of(h); },
          "The resources of the call that last wrote this result (orders its device memory).")
      .def_prop_ro("generation", &triangle_delta_holder::generation,
                   "The number of updates of this result.")
      .def_prop_ro("options",
                   [](const triangle_delta_holder& h) -> triangle_delta::options {
                     return read_result(
                         h, [](const triangle_delta::result& r) { return r.get_options(); });
                   })
      .def_prop_ro("graph_version",
                   [](const triangle_delta_holder& h) {
                     return read_result(
                         h, [](const triangle_delta::result& r) { return r.graph_version(); });
                   })
      .def_prop_ro("space",
                   [](const triangle_delta_holder& h) {
                     return read_result(h,
                                        [](const triangle_delta::result& r) { return r.space(); });
                   })
      .def(
          "clone",
          [](const triangle_delta_holder& h, const resources& res) {
            return without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, false);
              locks.lock();
              return new triangle_delta_holder(h.get().clone(res), res);
            });
          },
          nb::arg("resources"), nb::rv_policy::take_ownership);

#define DYNG_PY_BIND_TRIANGLE_DELTA(V, E, W) bind_functions<V, E, W>(m);
  DYNG_PY_FOR_EACH_GRAPH_TYPE(DYNG_PY_BIND_TRIANGLE_DELTA)
#undef DYNG_PY_BIND_TRIANGLE_DELTA
}

}  // namespace dyng::python
