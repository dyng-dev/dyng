// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file dynamic_bfs.cpp
 * @brief dyng::dynamic_bfs (a tutorial algorithm): options, stats, result, compute() and update()
 *        for every graph type dynamic_bfs supports.
 */
#include "types.hpp"

#include <dyng/dynamic_bfs.hpp>

namespace dyng::python {
namespace {

/// The holder of a dynamic_bfs result.
using dynamic_bfs_holder = result_holder<dynamic_bfs::result>;

template <typename vertex_t, typename edge_t, typename weight_t>
void bind_functions(nb::module_& m) {
  if constexpr (detail::dynamic_bfs_supported_v<vertex_t, edge_t, weight_t>) {
    using graph_t = graph_holder<vertex_t, edge_t, weight_t>;
    m.def(
        "dynamic_bfs_compute",
        [](const resources& res, const graph_t& g, const dynamic_bfs::options& opt) {
          return without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, false);
            locks.lock();
            return new dynamic_bfs_holder(dynamic_bfs::compute(res, g.value, opt), res);
          });
        },
        nb::arg("resources"), nb::arg("graph"), nb::arg("options"), nb::rv_policy::take_ownership);
    m.def(
        "dynamic_bfs_update",
        [](const resources& res, graph_t& g, const batch_arrays<vertex_t, weight_t>& b,
           dynamic_bfs_holder& r) {
          return without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, true);
            locks.add(r.mutex, true);
            locks.lock();
            return dynamic_bfs::update(res, g.value, b.view(), r.for_update(res));
          });
        },
        nb::arg("resources"), nb::arg("graph"), nb::arg("batch"), nb::arg("result"));
  } else {
    (void)m;
  }
}

}  // namespace

void bind_dynamic_bfs(nb::module_& m) {
  nb::class_<dynamic_bfs::options>(m, "DynamicBfsOptions", "dynamic_bfs::options.")
      .def(nb::init<>())
      .def_rw("source", &dynamic_bfs::options::source);
  nb::class_<dynamic_bfs::stats>(m, "DynamicBfsStats", "dynamic_bfs::stats.")
      .def_ro("affected", &update_stats::affected)
      .def_ro("iterations", &update_stats::iterations)
      .def_ro("frontier_visits", &update_stats::frontier_visits)
      .def_ro("fallback_used", &update_stats::fallback_used)
      .def_ro("converged", &update_stats::converged)
      .def_ro("engine_used", &update_stats::engine_used)
      .def_ro("batch", &dynamic_bfs::stats::batch)
      .def_ro("invalidated", &dynamic_bfs::stats::invalidated)
      .def_ro("invalidation_rounds", &dynamic_bfs::stats::invalidation_rounds);

  nb::class_<dynamic_bfs_holder>(m, "DynamicBfsResult", "A dynamic_bfs::result (private).",
                                 nb::is_weak_referenceable())
      .def("levels",
           [](const dynamic_bfs_holder& h) {
             return export_array(h, [](const dynamic_bfs::result& r) { return r.levels(); });
           })
      .def_prop_ro(
          "writer", [](const dynamic_bfs_holder& h) { return writer_of(h); },
          "The resources of the call that last wrote this result (orders its device arrays).")
      .def_prop_ro("generation", &dynamic_bfs_holder::generation,
                   "The number of updates of this result (dyng.Array's staleness check).")
      .def_prop_ro("options",
                   [](const dynamic_bfs_holder& h) -> dynamic_bfs::options {
                     return read_result(
                         h, [](const dynamic_bfs::result& r) { return r.get_options(); });
                   })
      .def(
          "set_options",
          [](dynamic_bfs_holder& h, const dynamic_bfs::options& opt) {
            without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, true);
              locks.lock();
              h.for_options().set_options(opt);
            });
          },
          nb::arg("options"))
      .def_prop_ro("graph_version",
                   [](const dynamic_bfs_holder& h) {
                     return read_result(
                         h, [](const dynamic_bfs::result& r) { return r.graph_version(); });
                   })
      .def_prop_ro("space",
                   [](const dynamic_bfs_holder& h) {
                     return read_result(h, [](const dynamic_bfs::result& r) { return r.space(); });
                   })
      .def(
          "clone",
          [](const dynamic_bfs_holder& h, const resources& res) {
            return without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, false);
              locks.lock();
              return new dynamic_bfs_holder(h.get().clone(res), res);
            });
          },
          nb::arg("resources"), nb::rv_policy::take_ownership);

#define DYNG_PY_BIND_DYNAMIC_BFS(V, E, W) bind_functions<V, E, W>(m);
  DYNG_PY_FOR_EACH_GRAPH_TYPE(DYNG_PY_BIND_DYNAMIC_BFS)
#undef DYNG_PY_BIND_DYNAMIC_BFS
}

}  // namespace dyng::python
