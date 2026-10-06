// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file mosp.cpp
 * @brief dyng::mosp: options, stats, result<V> (one class per vertex type), compute(), update()
 *        and result::from_arrays() for every graph type mosp supports.
 */
#include "types.hpp"

#include <dyng/mosp.hpp>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

namespace dyng::python {
namespace {

template <typename vertex_t>
void bind_result(nb::module_& m) {
  using holder_t = mosp_holder<vertex_t>;
  using result_t = mosp::result<vertex_t>;
  const std::string name = suffixed("MospResult", {type_code<vertex_t>::upper}, "");
  nb::class_<holder_t>(m, name.c_str(), "A mosp::result (private).", nb::is_weak_referenceable())
      .def_prop_ro("source",
                   [](const holder_t& h) {
                     return read_result(h, [](const result_t& r) {
                       return static_cast<std::int64_t>(r.source());
                     });
                   })
      .def_prop_ro("num_objectives",
                   [](const holder_t& h) {
                     return read_result(h, [](const result_t& r) { return r.num_objectives(); });
                   })
      .def(
          "distances",
          [](const holder_t& h, int objective) {
            return export_array(h, [&](const result_t& r) { return r.distances(objective); });
          },
          nb::arg("objective"))
      .def(
          "parents",
          [](const holder_t& h, int objective) {
            return export_array(h, [&](const result_t& r) { return r.parents(objective); });
          },
          nb::arg("objective"))
      .def("combined_distances",
           [](const holder_t& h) {
             return export_array(h, [](const result_t& r) { return r.combined_distances(); });
           })
      .def("combined_parents",
           [](const holder_t& h) {
             return export_array(h, [](const result_t& r) { return r.combined_parents(); });
           })
      .def("path_costs",
           [](const holder_t& h) {
             // n * K values, vertex-major: an (n, K) array.
             std::shared_ptr<const result_t> state;
             int k = 0;
             const auto v = without_gil([&] {
               std::shared_lock<std::shared_mutex> lock(h.mutex);
               state = h.share();
               k = state->num_objectives();
               return state->path_costs();
             });
             return owned_view_2d(v, static_cast<std::size_t>(k),
                                  std::shared_ptr<const void>(std::move(state)));
           })
      .def_prop_ro("preference_scale",
                   [](const holder_t& h) {
                     return read_result(h, [](const result_t& r) { return r.preference_scale(); });
                   })
      .def_prop_ro(
          "writer", [](const holder_t& h) { return writer_of(h); },
          "The resources of the call that last wrote this result (orders its device arrays).")
      .def_prop_ro("generation", &holder_t::generation,
                   "The number of updates of this result (dyng.Array's staleness check).")
      .def_prop_ro("options",
                   [](const holder_t& h) -> mosp::options {
                     return read_result(h, [](const result_t& r) { return r.get_options(); });
                   })
      .def(
          "set_options",
          [](holder_t& h, const mosp::options& opt) {
            without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, true);
              locks.lock();
              h.for_options().set_options(opt);
            });
          },
          nb::arg("options"))
      .def_prop_ro("graph_version",
                   [](const holder_t& h) {
                     return read_result(h, [](const result_t& r) { return r.graph_version(); });
                   })
      .def_prop_ro("space",
                   [](const holder_t& h) {
                     return read_result(h, [](const result_t& r) { return r.space(); });
                   })
      .def(
          "clone",
          [](const holder_t& h, const resources& res) {
            return without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, false);
              locks.lock();
              return new holder_t(h.get().clone(res), res);
            });
          },
          nb::arg("resources"), nb::rv_policy::take_ownership);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void bind_functions(nb::module_& m) {
  if constexpr (detail::mosp_supported_v<vertex_t, edge_t, weight_t>) {
    using graph_t = graph_holder<vertex_t, edge_t, weight_t>;
    using result_t = mosp_holder<vertex_t>;
    m.def(
        "mosp_compute",
        [](const resources& res, const graph_t& g, std::int64_t source, const mosp::options& opt) {
          DYNG_EXPECTS(source >= 0 && source < static_cast<std::int64_t>(g.value.num_vertices()),
                       "mosp: the source ", source, " is out of range [0, ", g.value.num_vertices(),
                       ")");
          return without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, false);
            locks.lock();
            return new result_t(mosp::compute(res, g.value, static_cast<vertex_t>(source), opt),
                                res);
          });
        },
        nb::arg("resources"), nb::arg("graph"), nb::arg("source"), nb::arg("options"),
        nb::rv_policy::take_ownership);
    m.def(
        "mosp_update",
        [](const resources& res, graph_t& g, const batch_arrays<vertex_t, weight_t>& b,
           result_t& r) {
          return without_gil([&] {
            lock_set locks;
            locks.add(g.mutex, true);
            locks.add(r.mutex, true);
            locks.lock();
            return mosp::update(res, g.value, b.view(), r.for_update(res));
          });
        },
        nb::arg("resources"), nb::arg("graph"), nb::arg("batch"), nb::arg("result"));
    m.def(
        "mosp_from_arrays",
        [](const resources& res, const graph_t& g, std::int64_t source,
           const std::vector<in_array<std::int64_t>>& distances,
           const std::vector<in_array<vertex_t>>& parents, bool canonicalize,
           const mosp::options& opt) {
          DYNG_EXPECTS(source >= 0 && source < static_cast<std::int64_t>(g.value.num_vertices()),
                       "mosp::result::from_arrays: the source ", source, " is out of range [0, ",
                       g.value.num_vertices(), ")");
          return without_gil([&] {
            std::vector<array_view<const std::int64_t>> dv;
            std::vector<array_view<const vertex_t>> pv;
            dv.reserve(distances.size());
            pv.reserve(parents.size());
            for (const auto& d : distances) {
              dv.push_back(view_of(d));
            }
            for (const auto& p : parents) {
              pv.push_back(view_of(p));
            }
            lock_set locks;
            locks.add(g.mutex, false);
            locks.lock();
            return new result_t(
                mosp::result<vertex_t>::from_arrays(
                    res, g.value, static_cast<vertex_t>(source), host_view(std::as_const(dv)),
                    host_view(std::as_const(pv)), canonicalize, opt),
                res);
          });
        },
        nb::arg("resources"), nb::arg("graph"), nb::arg("source"), nb::arg("distances"),
        nb::arg("parents"), nb::arg("canonicalize"), nb::arg("options"),
        nb::rv_policy::take_ownership);
  } else {
    (void)m;
  }
}

}  // namespace

void bind_mosp(nb::module_& m) {
  m.attr("MOSP_MAX_OBJECTIVES") = mosp::max_objectives;
  m.attr("MOSP_MAX_PREFERENCE_SCALE") = mosp::max_preference_scale;
  nb::class_<mosp::options>(m, "MospOptions", "mosp::options.")
      .def(nb::init<>())
      .def_rw("preferences", &mosp::options::preferences)
      .def_rw("delta", &mosp::options::delta)
      .def_rw("cuda_engine", &mosp::options::cuda_engine)
      .def_rw("compute_path_costs", &mosp::options::compute_path_costs)
      .def_rw("validate_inputs", &mosp::options::validate_inputs)
      .def_rw("num_objectives", &mosp::options::num_objectives);
  nb::class_<mosp::stats>(m, "MospStats", "mosp::stats.")
      .def_ro("affected", &update_stats::affected)
      .def_ro("iterations", &update_stats::iterations)
      .def_ro("frontier_visits", &update_stats::frontier_visits)
      .def_ro("fallback_used", &update_stats::fallback_used)
      .def_ro("converged", &update_stats::converged)
      .def_ro("engine_used", &update_stats::engine_used)
      .def_ro("batch", &mosp::stats::batch)
      .def_ro("objectives", &mosp::stats::objectives)
      .def_ro("combined_edges", &mosp::stats::combined_edges)
      .def_ro("preference_scale", &mosp::stats::preference_scale);
  bind_result<std::int32_t>(m);
  bind_result<std::int64_t>(m);
#define DYNG_PY_BIND_MOSP(V, E, W) bind_functions<V, E, W>(m);
  DYNG_PY_FOR_EACH_GRAPH_TYPE(DYNG_PY_BIND_MOSP)
#undef DYNG_PY_BIND_MOSP
}

}  // namespace dyng::python
