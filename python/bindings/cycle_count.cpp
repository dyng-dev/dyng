// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_count.cpp
 * @brief dyng::cycle_count: the enumerations, options, stats, result, compute() and update() for
 *        every graph type cycle_count supports.
 */
#include "types.hpp"

#include <dyng/cycle_count.hpp>

namespace dyng::python {
namespace {

template <typename vertex_t, typename edge_t, typename weight_t>
void bind_functions(nb::module_& m) {
  if constexpr (detail::cycle_count_supported_v<vertex_t, edge_t, weight_t>) {
    using graph_t = graph_holder<vertex_t, edge_t, weight_t>;
    m.def(
        "cycle_count_compute",
        [](const resources& res, const graph_t& g, const cycle_count::options& opt) {
          return without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, false);
            locks.lock();
            return new cycle_count_holder(cycle_count::compute(res, g.value, opt), res);
          });
        },
        nb::arg("resources"), nb::arg("graph"), nb::arg("options"), nb::rv_policy::take_ownership);
    m.def(
        "cycle_count_update",
        [](const resources& res, graph_t& g, const batch_arrays<vertex_t, weight_t>& b,
           cycle_count_holder& r) {
          return without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, true);
            locks.add(r.mutex, true);
            locks.lock();
            return cycle_count::update(res, g.value, b.view(), r.for_update(res));
          });
        },
        nb::arg("resources"), nb::arg("graph"), nb::arg("batch"), nb::arg("result"));
  } else {
    (void)m;
  }
}

}  // namespace

void bind_cycle_count(nb::module_& m) {
  nb::enum_<cycle_count::search_method>(m, "SearchMethod", "cycle_count::search_method.")
      .value("johnson", cycle_count::search_method::johnson);
  nb::enum_<cycle_count::cycle_mode>(m, "CycleMode", "cycle_count::cycle_mode.")
      .value("simple", cycle_count::cycle_mode::simple);
  nb::enum_<cycle_count::cuda_scheduler>(m, "CudaScheduler", "cycle_count::cuda_scheduler.")
      .value("work_queue", cycle_count::cuda_scheduler::work_queue)
      .value("naive", cycle_count::cuda_scheduler::naive);
  nb::enum_<cycle_count::cuda_work_items>(m, "CudaWorkItems", "cycle_count::cuda_work_items.")
      .value("automatic", cycle_count::cuda_work_items::automatic)
      .value("roots", cycle_count::cuda_work_items::roots)
      .value("edges", cycle_count::cuda_work_items::edges)
      .value("two_hop", cycle_count::cuda_work_items::two_hop);

  nb::class_<cycle_count::options>(m, "CycleCountOptions", "cycle_count::options.")
      .def(nb::init<>())
      .def_rw("max_length", &cycle_count::options::max_length)
      .def_rw("method", &cycle_count::options::method)
      .def_rw("mode", &cycle_count::options::mode)
      .def_rw("cuda_engine", &cycle_count::options::cuda_engine)
      .def_rw("scheduler", &cycle_count::options::scheduler)
      .def_rw("work_items", &cycle_count::options::work_items);
  nb::class_<cycle_count::stats>(m, "CycleCountStats", "cycle_count::stats.")
      .def_ro("affected", &update_stats::affected)
      .def_ro("iterations", &update_stats::iterations)
      .def_ro("frontier_visits", &update_stats::frontier_visits)
      .def_ro("fallback_used", &update_stats::fallback_used)
      .def_ro("converged", &update_stats::converged)
      .def_ro("engine_used", &update_stats::engine_used)
      .def_ro("batch", &cycle_count::stats::batch)
      .def_ro("deletions", &cycle_count::stats::deletions)
      .def_ro("insertions", &cycle_count::stats::insertions)
      .def_ro("cycles_removed", &cycle_count::stats::cycles_removed)
      .def_ro("cycles_added", &cycle_count::stats::cycles_added);

  nb::class_<cycle_count_holder>(m, "CycleCountResult", "A cycle_count::result (private).",
                                 nb::is_weak_referenceable())
      .def("counts",
           [](const cycle_count_holder& h) {
             return export_array(h, [](const cycle_count::result& r) { return r.counts(); });
           })
      .def(
          "count",
          [](const cycle_count_holder& h, std::int64_t length) {
            return read_result(h, [&](const cycle_count::result& r) { return r.count(length); });
          },
          nb::arg("length"))
      .def_prop_ro("total",
                   [](const cycle_count_holder& h) {
                     return read_result(h, [](const cycle_count::result& r) { return r.total(); });
                   })
      .def_prop_ro("bound",
                   [](const cycle_count_holder& h) {
                     return read_result(h, [](const cycle_count::result& r) { return r.bound(); });
                   })
      .def_prop_ro(
          "writer", [](const cycle_count_holder& h) { return writer_of(h); },
          "The resources of the call that last wrote this result (orders its device arrays).")
      .def_prop_ro("generation", &cycle_count_holder::generation,
                   "The number of updates of this result (dyng.Array's staleness check).")
      .def_prop_ro("options",
                   [](const cycle_count_holder& h) -> cycle_count::options {
                     return read_result(
                         h, [](const cycle_count::result& r) { return r.get_options(); });
                   })
      .def(
          "set_options",
          [](cycle_count_holder& h, const cycle_count::options& opt) {
            without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, true);
              locks.lock();
              h.for_options().set_options(opt);
            });
          },
          nb::arg("options"))
      .def_prop_ro("graph_version",
                   [](const cycle_count_holder& h) {
                     return read_result(
                         h, [](const cycle_count::result& r) { return r.graph_version(); });
                   })
      .def_prop_ro("space",
                   [](const cycle_count_holder& h) {
                     return read_result(h, [](const cycle_count::result& r) { return r.space(); });
                   })
      .def(
          "clone",
          [](const cycle_count_holder& h, const resources& res) {
            return without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, false);
              locks.lock();
              return new cycle_count_holder(h.get().clone(res), res);
            });
          },
          nb::arg("resources"), nb::rv_policy::take_ownership);

#define DYNG_PY_BIND_CYCLE_COUNT(V, E, W) bind_functions<V, E, W>(m);
  DYNG_PY_FOR_EACH_GRAPH_TYPE(DYNG_PY_BIND_CYCLE_COUNT)
#undef DYNG_PY_BIND_CYCLE_COUNT
}

}  // namespace dyng::python
