// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sssp.cpp
 * @brief dyng::sssp: options, stats, result<V> (one class per vertex type), compute() and
 *        update() for every graph type sssp supports.
 */
#include "types.hpp"

#include <dyng/sssp.hpp>

#include <nanobind/stl/string.h>

#include <string>

namespace dyng::python {
namespace {

template <typename vertex_t>
void bind_result(nb::module_& m) {
  using holder_t = sssp_holder<vertex_t>;
  const std::string name = suffixed("SsspResult", {type_code<vertex_t>::upper}, "");
  nb::class_<holder_t>(m, name.c_str(), "An sssp::result (private).", nb::is_weak_referenceable())
      .def_prop_ro("source",
                   [](const holder_t& h) { return static_cast<std::int64_t>(h.value.source()); })
      .def("distances",
           [](nb::handle self) {
             const holder_t& h = nb::cast<const holder_t&>(self);
             const auto v = without_gil([&] {
               std::shared_lock<std::shared_mutex> lock(h.mutex);
               return h.value.distances();
             });
             return owned_view(v, self);
           })
      .def("parents",
           [](nb::handle self) {
             const holder_t& h = nb::cast<const holder_t&>(self);
             const auto v = without_gil([&] {
               std::shared_lock<std::shared_mutex> lock(h.mutex);
               return h.value.parents();
             });
             return owned_view(v, self);
           })
      .def_prop_ro("options",
                   [](const holder_t& h) -> sssp::options { return h.value.get_options(); })
      .def(
          "set_options",
          [](holder_t& h, const sssp::options& opt) {
            without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, true);
              locks.lock();
              h.value.set_options(opt);
            });
          },
          nb::arg("options"))
      .def_prop_ro("graph_version", [](const holder_t& h) { return h.value.graph_version(); })
      .def_prop_ro("space", [](const holder_t& h) { return h.value.space(); })
      .def(
          "clone",
          [](const holder_t& h, const resources& res) {
            return without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, false);
              locks.lock();
              return new holder_t(h.value.clone(res));
            });
          },
          nb::arg("resources"), nb::rv_policy::take_ownership);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void bind_functions(nb::module_& m) {
  if constexpr (detail::sssp_supported_v<vertex_t, edge_t, weight_t>) {
    using graph_t = graph_holder<vertex_t, edge_t, weight_t>;
    using result_t = sssp_holder<vertex_t>;
    m.def(
        "sssp_compute",
        [](const resources& res, const graph_t& g, std::int64_t source, const sssp::options& opt) {
          DYNG_EXPECTS(source >= 0 && source < static_cast<std::int64_t>(g.value.num_vertices()),
                       "sssp: the source ", source, " is out of range [0, ", g.value.num_vertices(),
                       ")");
          return without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, false);
            locks.lock();
            return new result_t(sssp::compute(res, g.value, static_cast<vertex_t>(source), opt));
          });
        },
        nb::arg("resources"), nb::arg("graph"), nb::arg("source"), nb::arg("options"),
        nb::rv_policy::take_ownership);
    m.def(
        "sssp_update",
        [](const resources& res, graph_t& g, const batch_arrays<vertex_t, weight_t>& b,
           result_t& r) {
          return without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, true);
            locks.add(r.mutex, true);
            locks.lock();
            return sssp::update(res, g.value, b.view(), r.value);
          });
        },
        nb::arg("resources"), nb::arg("graph"), nb::arg("batch"), nb::arg("result"));
    m.def(
        "sssp_from_arrays",
        [](const resources& res, const graph_t& g, std::int64_t source,
           const in_array<std::int64_t>& distances, const in_array<vertex_t>& parents,
           bool canonicalize, const sssp::options& opt) {
          DYNG_EXPECTS(source >= 0 && source < static_cast<std::int64_t>(g.value.num_vertices()),
                       "sssp::result::from_arrays: the source ", source, " is out of range [0, ",
                       g.value.num_vertices(), ")");
          return without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, false);
            locks.lock();
            return new result_t(sssp::result<vertex_t>::from_arrays(
                res, g.value, static_cast<vertex_t>(source), view_of(distances), view_of(parents),
                canonicalize, opt));
          });
        },
        nb::arg("resources"), nb::arg("graph"), nb::arg("source"), nb::arg("distances").noconvert(),
        nb::arg("parents").noconvert(), nb::arg("canonicalize"), nb::arg("options"),
        nb::rv_policy::take_ownership);
  } else {
    (void)m;
  }
}

}  // namespace

void bind_sssp(nb::module_& m) {
  nb::class_<sssp::options>(m, "SsspOptions", "sssp::options.")
      .def(nb::init<>())
      .def_rw("delta", &sssp::options::delta)
      .def_rw("objective", &sssp::options::objective)
      .def_rw("cuda_engine", &sssp::options::cuda_engine)
      .def_rw("validate_inputs", &sssp::options::validate_inputs);
  nb::class_<sssp::stats>(m, "SsspStats", "sssp::stats.")
      .def_ro("affected", &update_stats::affected)
      .def_ro("iterations", &update_stats::iterations)
      .def_ro("frontier_visits", &update_stats::frontier_visits)
      .def_ro("fallback_used", &update_stats::fallback_used)
      .def_ro("converged", &update_stats::converged)
      .def_ro("engine_used", &update_stats::engine_used)
      .def_ro("batch", &sssp::stats::batch)
      .def_ro("invalidated", &sssp::stats::invalidated)
      .def_ro("epochs", &sssp::stats::epochs)
      .def_ro("pushes", &sssp::stats::pushes)
      .def_ro("packed_parents", &sssp::stats::packed_parents);
  bind_result<std::int32_t>(m);
  bind_result<std::int64_t>(m);
#define DYNG_PY_BIND_SSSP(V, E, W) bind_functions<V, E, W>(m);
  DYNG_PY_FOR_EACH_GRAPH_TYPE(DYNG_PY_BIND_SSSP)
#undef DYNG_PY_BIND_SSSP
}

}  // namespace dyng::python
