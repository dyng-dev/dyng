// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_count.cpp
 * @brief dyng::triad_count: the pattern enumeration, options, stats, result, compute() and
 *        update() for both hypergraph types (int32 vertices, unweighted or int32 weights).
 *
 * The counts are a host array on every backend (triad_count::result::counts()), so the result
 * exports them as NumPy copies; stats::delta (a fixed-size array of max_bins) is exported whole
 * and the typed layer keeps the result's num_bins() values.
 */
#include "types.hpp"

#include <dyng/triad_count.hpp>

#include <cstdint>
#include <vector>

namespace dyng::python {
namespace {

template <typename vertex_t, typename weight_t>
void bind_functions(nb::module_& m) {
  using hgraph_t = hypergraph_holder<vertex_t, weight_t>;
  using batch_t = hyper_batch_arrays<vertex_t, weight_t>;
  m.def(
      "triad_count_compute",
      [](const resources& res, const hgraph_t& h, const triad_count::options& opt) {
        return without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, false);
          locks.lock();
          return new triad_count_holder(triad_count::compute(res, h.value, opt), res);
        });
      },
      nb::arg("resources"), nb::arg("hypergraph"), nb::arg("options"),
      nb::rv_policy::take_ownership);
  m.def(
      "triad_count_update",
      [](const resources& res, hgraph_t& h, const batch_t& b, triad_count_holder& r) {
        return without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, true);
          locks.add(r.mutex, true);
          locks.lock();
          return triad_count::update(res, h.value, b.view(), r.for_update(res));
        });
      },
      nb::arg("resources"), nb::arg("hypergraph"), nb::arg("batch"), nb::arg("result"));
}

}  // namespace

void bind_triad_count(nb::module_& m) {
  nb::enum_<triad_count::pattern>(m, "TriadPattern", "triad_count::pattern.")
      .value("h_motif", triad_count::pattern::h_motif)
      .value("stathyper_type1", triad_count::pattern::stathyper_type1)
      .value("stathyper_type2", triad_count::pattern::stathyper_type2)
      .value("stathyper_type3", triad_count::pattern::stathyper_type3);
  m.attr("triad_count_max_bins") = triad_count::max_bins;

  nb::class_<triad_count::options>(m, "TriadCountOptions", "triad_count::options.")
      .def(nb::init<>())
      .def_rw("kind", &triad_count::options::kind)
      .def_rw("cuda_engine", &triad_count::options::cuda_engine);
  nb::class_<triad_count::stats>(m, "TriadCountStats", "triad_count::stats.")
      .def_ro("affected", &update_stats::affected)
      .def_ro("iterations", &update_stats::iterations)
      .def_ro("frontier_visits", &update_stats::frontier_visits)
      .def_ro("fallback_used", &update_stats::fallback_used)
      .def_ro("converged", &update_stats::converged)
      .def_ro("engine_used", &update_stats::engine_used)
      .def_ro("batch", &triad_count::stats::batch)
      .def_ro("deleted_frontier", &triad_count::stats::deleted_frontier)
      .def_ro("inserted_frontier", &triad_count::stats::inserted_frontier)
      .def_ro("triads_removed", &triad_count::stats::triads_removed)
      .def_ro("triads_added", &triad_count::stats::triads_added)
      .def_prop_ro(
          "delta",
          [](const triad_count::stats& s) {
            std::vector<std::int64_t> d(s.delta.begin(), s.delta.end());
            return to_numpy(std::move(d));
          },
          nb::rv_policy::automatic,
          "The change of every bin (an int64 copy of max_bins values; 0 at and above the "
          "result's num_bins()).");

  nb::class_<triad_count_holder>(m, "TriadCountResult", "A triad_count::result (private).",
                                 nb::is_weak_referenceable())
      .def("counts",
           [](const triad_count_holder& h) {
             std::vector<std::uint64_t> c = read_result(h, [](const triad_count::result& r) {
               const auto v = r.counts();  // host memory on every backend
               return std::vector<std::uint64_t>(v.begin(), v.end());
             });
             return to_numpy(std::move(c));
           })
      .def(
          "count",
          [](const triad_count_holder& h, int bin) {
            return read_result(h, [&](const triad_count::result& r) { return r.count(bin); });
          },
          nb::arg("bin"))
      .def_prop_ro("total",
                   [](const triad_count_holder& h) {
                     return read_result(h, [](const triad_count::result& r) { return r.total(); });
                   })
      .def_prop_ro(
          "num_bins",
          [](const triad_count_holder& h) {
            return read_result(h, [](const triad_count::result& r) { return r.num_bins(); });
          })
      .def_prop_ro("kind",
                   [](const triad_count_holder& h) {
                     return read_result(h, [](const triad_count::result& r) { return r.kind(); });
                   })
      .def_prop_ro(
          "writer", [](const triad_count_holder& h) { return writer_of(h); },
          "The resources of the call that last wrote this result.")
      .def_prop_ro("generation", &triad_count_holder::generation,
                   "The number of updates of this result.")
      .def_prop_ro("options",
                   [](const triad_count_holder& h) -> triad_count::options {
                     return read_result(
                         h, [](const triad_count::result& r) { return r.get_options(); });
                   })
      .def(
          "set_options",
          [](triad_count_holder& h, const triad_count::options& opt) {
            without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, true);
              locks.lock();
              h.for_options().set_options(opt);
            });
          },
          nb::arg("options"))
      .def_prop_ro("graph_version",
                   [](const triad_count_holder& h) {
                     return read_result(
                         h, [](const triad_count::result& r) { return r.graph_version(); });
                   })
      .def_prop_ro("space",
                   [](const triad_count_holder& h) {
                     return read_result(h, [](const triad_count::result& r) { return r.space(); });
                   })
      .def(
          "clone",
          [](const triad_count_holder& h, const resources& res) {
            return without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, false);
              locks.lock();
              return new triad_count_holder(h.get().clone(res), res);
            });
          },
          nb::arg("resources"), nb::rv_policy::take_ownership);

#define DYNG_PY_BIND_TRIAD_COUNT(V, W) bind_functions<V, W>(m);
  DYNG_PY_FOR_EACH_HYPERGRAPH_TYPE(DYNG_PY_BIND_TRIAD_COUNT)
#undef DYNG_PY_BIND_TRIAD_COUNT
}

}  // namespace dyng::python
