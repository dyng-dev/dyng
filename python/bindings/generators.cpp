// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file generators.cpp
 * @brief dyng::generators::legacy: MOSP's change generator and CycleEnumeration-GPU's batch
 *        generator, bit-exact for a fixed seed.
 */
#include "types.hpp"

#include <dyng/generators/legacy.hpp>

#include <nanobind/stl/string.h>

namespace dyng::python {
namespace {

namespace legacy = generators::legacy;

template <typename vertex_t, typename edge_t, typename weight_t>
void bind_generator_type(nb::module_& m) {
  using graph_t = graph_holder<vertex_t, edge_t, weight_t>;
  if constexpr (!is_unweighted_v<weight_t>) {
    m.def(
        "mosp_changes",
        [](const graph_t& g, const legacy::mosp_change_options& opt) {
          legacy::mosp_change_report report;
          auto batch = without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, false);
            locks.lock();
            return legacy::mosp_changes(g.value.view().out, opt, &report);
          });
          return nb::make_tuple(batch_to_numpy(batch), report);
        },
        nb::arg("graph"), nb::arg("options"));
  }
  m.def(
      "cycle_enum_batch",
      [](const graph_t& g, const legacy::cycle_enum_batch_options& opt) {
        auto batch = without_gil([&] {
          lock_set locks;
          locks.add(g.mutex, false);
          locks.lock();
          return legacy::cycle_enum_batch(g.value.view().out, opt);
        });
        return batch_to_numpy(batch);
      },
      nb::arg("graph"), nb::arg("options"));
}

}  // namespace

void bind_generators(nb::module_& m) {
  nb::enum_<legacy::mosp_change_mode>(m, "MospChangeMode", "generators::legacy::mosp_change_mode.")
      .value("uniform", legacy::mosp_change_mode::uniform)
      .value("targeted", legacy::mosp_change_mode::targeted)
      .value("reweight", legacy::mosp_change_mode::reweight)
      .value("increase", legacy::mosp_change_mode::increase);
  nb::class_<legacy::mosp_change_options>(m, "MospChangeOptions",
                                          "generators::legacy::mosp_change_options.")
      .def(nb::init<>())
      .def_rw("num_changes", &legacy::mosp_change_options::num_changes)
      .def_rw("insertion_percentage", &legacy::mosp_change_options::insertion_percentage)
      .def_rw("mode", &legacy::mosp_change_options::mode)
      .def_rw("weight_min", &legacy::mosp_change_options::weight_min)
      .def_rw("weight_max", &legacy::mosp_change_options::weight_max)
      .def_rw("seed", &legacy::mosp_change_options::seed)
      .def_rw("local_hops", &legacy::mosp_change_options::local_hops)
      .def_rw("safe_deletions", &legacy::mosp_change_options::safe_deletions)
      .def_rw("source", &legacy::mosp_change_options::source);
  nb::class_<legacy::mosp_change_report>(m, "MospChangeReport",
                                         "generators::legacy::mosp_change_report.")
      .def_ro("inserts", &legacy::mosp_change_report::inserts)
      .def_ro("reweights", &legacy::mosp_change_report::reweights)
      .def_ro("deletes", &legacy::mosp_change_report::deletes)
      .def_ro("requested_deletes", &legacy::mosp_change_report::requested_deletes)
      .def_ro("safe_rounds", &legacy::mosp_change_report::safe_rounds)
      .def_ro("safe", &legacy::mosp_change_report::safe)
      .def_ro("centre", &legacy::mosp_change_report::centre)
      .def_ro("local_hops", &legacy::mosp_change_report::local_hops)
      .def_ro("region", &legacy::mosp_change_report::region)
      .def("summary", &legacy::mosp_change_report::summary);
  nb::class_<legacy::cycle_enum_batch_options>(m, "CycleEnumBatchOptions",
                                               "generators::legacy::cycle_enum_batch_options.")
      .def(nb::init<>())
      .def_rw("num_deletions", &legacy::cycle_enum_batch_options::num_deletions)
      .def_rw("num_insertions", &legacy::cycle_enum_batch_options::num_insertions)
      .def_rw("seed", &legacy::cycle_enum_batch_options::seed)
      .def_rw("locality_window", &legacy::cycle_enum_batch_options::locality_window);

#define DYNG_PY_BIND_GENERATORS(V, E, W) bind_generator_type<V, E, W>(m);
  DYNG_PY_FOR_EACH_GRAPH_TYPE(DYNG_PY_BIND_GENERATORS)
#undef DYNG_PY_BIND_GENERATORS
}

}  // namespace dyng::python
