// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph.cpp
 * @brief dyng::hypergraph<V,W> (one class per instantiation), its properties, the capacity policy,
 *        the summary of an applied batch and the storage counters.
 *
 * The views (h2v, v2h, h2h, the weights) are returned as host copies: a zero-copy view would be
 * invalidated by the next apply() (the hypergraph's views are rebuilt per version), the rule of
 * graph::view() in docs/developer/python_gaps.md.
 */
#include "types.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/hypergraph/hyper_apply_summary.hpp>
#include <dyng/hypergraph/hypergraph_properties.hpp>
#include <dyng/hypergraph/hypergraph_view.hpp>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace dyng::python {
namespace {

/// A host copy of an array view in any memory space (read with the hypergraph's resources).
template <typename value_t>
std::vector<value_t> host_copy(const resources& res, array_view<const value_t> v) {
  if (v.empty()) {
    return {};
  }
  return to_vector(res, v);
}

template <typename vertex_t, typename weight_t>
void bind_hypergraph_type(nb::module_& m) {
  using hgraph = hypergraph<vertex_t, weight_t>;
  using holder_t = hypergraph_holder<vertex_t, weight_t>;
  using batch_t = hyper_batch_arrays<vertex_t, weight_t>;
  using weights_arg = std::optional<in_array<weight_scalar_t<weight_t>>>;
  constexpr bool weighted = !is_unweighted_v<weight_t>;
  const std::string name =
      suffixed("Hypergraph", {type_code<vertex_t>::upper, type_code<weight_t>::upper}, "");

  auto cls = nb::class_<holder_t>(m, name.c_str(), "A dyng::hypergraph instantiation (private).");
  cls.def_static(
      "from_hyperedges",
      [](const resources& res, std::int64_t num_vertices, const in_array<std::int64_t>& offsets,
         const in_array<vertex_t>& members, const weights_arg& weights,
         const hypergraph_properties& props) {
        const auto list =
            hyperedge_list_view_of<vertex_t, weight_t>(num_vertices, offsets, members, weights);
        return without_gil([&] { return new holder_t(hgraph::from_hyperedges(res, list, props)); });
      },
      nb::arg("resources"), nb::arg("num_vertices"), nb::arg("offsets").noconvert(),
      nb::arg("members").noconvert(), nb::arg("weights").noconvert().none(), nb::arg("properties"),
      nb::rv_policy::take_ownership, "Build a hypergraph from a CSR of member lists.");
  cls.def_prop_ro("num_vertices", [](const holder_t& h) {
    return static_cast<std::int64_t>(h.value.num_vertices());
  });
  cls.def_prop_ro("num_hyperedges", [](const holder_t& h) { return h.value.num_hyperedges(); });
  cls.def_prop_ro("hyperedge_id_bound",
                  [](const holder_t& h) { return h.value.hyperedge_id_bound(); });
  cls.def_prop_ro("version", [](const holder_t& h) { return h.value.version(); });
  cls.def_prop_ro("space", [](const holder_t& h) { return h.value.space(); });
  cls.def_prop_ro("properties",
                  [](const holder_t& h) -> hypergraph_properties { return h.value.properties(); });
  cls.def(
      "h2v",
      [](const holder_t& h, const resources& res) {
        auto [ptr, col, version] = without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, false);
          locks.lock();
          const auto v = h.value.h2v();
          return std::make_tuple(host_copy(res, v.row_ptr), host_copy(res, v.col_ind), v.version);
        });
        if (ptr.empty()) {
          ptr.push_back(0);
        }
        return nb::make_tuple(to_numpy(std::move(ptr)), to_numpy(std::move(col)), version);
      },
      nb::arg("resources"), "(row_ptr, col_ind, version): a host copy of hyperedge -> members.");
  cls.def(
      "v2h",
      [](const holder_t& h, const resources& res) {
        auto [ptr, col, version] = without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, false);
          locks.lock();
          const auto v = h.value.v2h();
          return std::make_tuple(host_copy(res, v.row_ptr), host_copy(res, v.col_ind), v.version);
        });
        if (ptr.empty()) {
          ptr.push_back(0);
        }
        return nb::make_tuple(to_numpy(std::move(ptr)), to_numpy(std::move(col)), version);
      },
      nb::arg("resources"), "(row_ptr, col_ind, version): a host copy of vertex -> hyperedges.");
  cls.def(
      "h2h",
      [](const holder_t& h, const resources& res) {
        auto [ptr, col, out, version] = without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, false);
          locks.lock();
          const auto v = h.value.h2h();
          return std::make_tuple(host_copy(res, v.row_ptr), host_copy(res, v.col_ind),
                                 host_copy(res, v.out), v.version);
        });
        if (ptr.empty()) {
          ptr.push_back(0);
        }
        return nb::make_tuple(to_numpy(std::move(ptr)), to_numpy(std::move(col)),
                              to_numpy(std::move(out)), version);
      },
      nb::arg("resources"),
      "(row_ptr, col_ind, out, version): a host copy of the line graph (hyperedge -> overlapping "
      "hyperedges).");
  cls.def(
      "weights",
      [](const holder_t& h) -> nb::object {
        if constexpr (!weighted) {
          (void)h;
          return nb::none();
        } else {
          auto w = without_gil([&] {
            lock_set locks;
            locks.add(h.mutex, false);
            locks.lock();
            const auto v = h.value.weights();
            return std::vector<weight_t>(v.begin(), v.end());  // host memory on every backend
          });
          return nb::cast(to_numpy(std::move(w)));
        }
      },
      "The weights by id (a host copy; 0 for a dead id), or None for an unweighted hypergraph.");
  cls.def(
      "apply",
      [](holder_t& h, const resources& res, const batch_t& b) {
        return without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, true);
          locks.lock();
          return h.value.apply(res, b.view());
        });
      },
      nb::arg("resources"), nb::arg("batch"), "Apply a batch to the hypergraph.");
  cls.def(
      "to_hyperedge_list",
      [](const holder_t& h, const resources& res) {
        auto list = without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, false);
          locks.lock();
          return h.value.to_hyperedge_list(res);
        });
        return hyperedge_list_to_numpy(std::move(list));
      },
      nb::arg("resources"),
      "(num_vertices, offsets, members, weights or None): the live hyperedges in id order.");
  cls.def(
      "clone",
      [](const holder_t& h, const resources& res) {
        return without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, false);
          locks.lock();
          return new holder_t(h.value.clone(res));
        });
      },
      nb::arg("resources"), nb::rv_policy::take_ownership, "A deep copy for `resources`.");
  cls.def(
      "to_backend",
      [](const holder_t& h, const resources& res) {
        return without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, false);
          locks.lock();
          return new holder_t(h.value.to_backend(res));
        });
      },
      nb::arg("resources"), nb::rv_policy::take_ownership,
      "The hypergraph for the backend of `resources`.");
  cls.def("store_stats", [](const holder_t& h) {
    return without_gil([&] {
      lock_set locks;
      locks.add(h.mutex, false);
      locks.lock();
      return h.value.store_stats();
    });
  });
  cls.def(
      "check_integrity",
      [](const holder_t& h, const resources& res) {
        without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, false);
          locks.lock();
          h.value.check_integrity(res);
        });
      },
      nb::arg("resources"));
}

}  // namespace

void bind_hypergraph(nb::module_& m) {
  nb::enum_<hypergraph_storage>(m, "HypergraphStorage", "How the incidences are stored.")
      .value("escher", hypergraph_storage::escher)
      .value("slack_csr", hypergraph_storage::slack_csr);
  nb::enum_<line_graph_policy>(m, "LineGraphPolicy", "How the line graph is kept.")
      .value("none", line_graph_policy::none)
      .value("automatic", line_graph_policy::automatic)
      .value("sorted_merge", line_graph_policy::sorted_merge)
      .value("slack_csr", line_graph_policy::slack_csr);
  nb::enum_<id_reuse>(m, "IdReuse", "The order of a batch's deletions and insertions.")
      .value("erase_first", id_reuse::erase_first)
      .value("insert_first", id_reuse::insert_first);

  nb::class_<capacity_policy>(m, "CapacityPolicy", "How a hypergraph's storage is sized and grows.")
      .def(nb::init<>())
      .def_rw("initial_headroom", &capacity_policy::initial_headroom)
      .def_rw("initial_slack", &capacity_policy::initial_slack)
      .def_rw("min_initial_payload", &capacity_policy::min_initial_payload)
      .def_rw("grow_factor", &capacity_policy::grow_factor)
      .def_rw("compact_before_grow", &capacity_policy::compact_before_grow)
      .def_rw("max_payload", &capacity_policy::max_payload)
      .def_rw("row_alignment", &capacity_policy::row_alignment)
      .def_rw("compaction_headroom", &capacity_policy::compaction_headroom)
      .def_rw("exact_scratch", &capacity_policy::exact_scratch)
      .def_static("escher_gpu", &capacity_policy::escher_gpu, nb::arg("min_initial_payload") = 0)
      .def_static("mosp_escher", &capacity_policy::mosp_escher,
                  nb::arg("extra") = std::int64_t{1} << 20);

  nb::class_<hypergraph_properties>(m, "HypergraphProperties", "The properties of a hypergraph.")
      .def(nb::init<>())
      .def_rw("storage", &hypergraph_properties::storage)
      .def_rw("line_graph", &hypergraph_properties::line_graph)
      .def_rw("reuse", &hypergraph_properties::reuse)
      .def_rw("capacity", &hypergraph_properties::capacity)
      .def_rw("store_v2h", &hypergraph_properties::store_v2h);

  nb::class_<hyper_apply_summary>(m, "HyperApplySummary", "What applying a batch did.")
      .def_ro("inserted_hyperedges", &hyper_apply_summary::inserted_hyperedges)
      .def_ro("deleted_hyperedges", &hyper_apply_summary::deleted_hyperedges)
      .def_ro("inserted_incidences", &hyper_apply_summary::inserted_incidences)
      .def_ro("deleted_incidences", &hyper_apply_summary::deleted_incidences)
      .def_ro("skipped_ops", &hyper_apply_summary::skipped_ops)
      .def_ro("ignored_deletions", &hyper_apply_summary::ignored_deletions)
      .def_ro("dropped_members", &hyper_apply_summary::dropped_members)
      .def_ro("num_hyperedges_after", &hyper_apply_summary::num_hyperedges_after)
      .def_ro("id_bound_after", &hyper_apply_summary::id_bound_after)
      .def_prop_ro(
          "inserted_ids",
          [](const hyper_apply_summary& s) {
            std::vector<std::int64_t> ids = s.inserted_ids;
            return to_numpy(std::move(ids));
          },
          nb::rv_policy::automatic, "The id of every inserted hyperedge (an int64 copy).");

  nb::class_<storage_path_counters>(m, "StoragePathCounters", "Kernel paths of an op family.")
      .def_ro("thread", &storage_path_counters::thread)
      .def_ro("warp", &storage_path_counters::warp)
      .def_ro("block", &storage_path_counters::block);
  nb::class_<storage_stats>(m, "StorageStats", "The storage counters of a hypergraph.")
      .def_ro("device_bytes", &storage_stats::device_bytes)
      .def_ro("view_bytes", &storage_stats::view_bytes)
      .def_ro("host_bytes", &storage_stats::host_bytes)
      .def_ro("payload_capacity", &storage_stats::payload_capacity)
      .def_ro("payload_used", &storage_stats::payload_used)
      .def_ro("growths", &storage_stats::growths)
      .def_ro("compactions", &storage_stats::compactions)
      .def_ro("fill", &storage_stats::fill)
      .def_ro("reuse", &storage_stats::reuse)
      .def_ro("unfill", &storage_stats::unfill);

#define DYNG_PY_BIND_HYPERGRAPH(V, W) bind_hypergraph_type<V, W>(m);
  DYNG_PY_FOR_EACH_HYPERGRAPH_TYPE(DYNG_PY_BIND_HYPERGRAPH)
#undef DYNG_PY_BIND_HYPERGRAPH
}

}  // namespace dyng::python
