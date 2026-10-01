// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file graph.cpp
 * @brief dyng::graph<V,E,W> (one class per instantiation), graph_properties, batch_semantics and
 *        apply_summary.
 */
#include "types.hpp"

#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <memory>
#include <optional>

namespace dyng::python {
namespace {

template <typename vertex_t, typename edge_t, typename weight_t>
void bind_graph_type(nb::module_& m) {
  using graph_t = graph<vertex_t, edge_t, weight_t>;
  using holder_t = graph_holder<vertex_t, edge_t, weight_t>;
  using weights_arg = std::optional<in_array<weight_scalar_t<weight_t>>>;
  constexpr bool weighted = !is_unweighted_v<weight_t>;
  const std::string name = suffixed(
      "Graph", {type_code<vertex_t>::upper, type_code<edge_t>::upper, type_code<weight_t>::upper},
      "");

  auto cls = nb::class_<holder_t>(m, name.c_str(), "A dyng::graph instantiation (private).");
  cls.def_static(
      "from_edges",
      [](const resources& res, std::int64_t num_vertices, const in_array<vertex_t>& src,
         const in_array<vertex_t>& dst, const weights_arg& weights, int num_weights,
         const graph_properties& props) {
        DYNG_EXPECTS(num_vertices >= 0 && num_vertices <= static_cast<std::int64_t>(
                                                              std::numeric_limits<vertex_t>::max()),
                     "num_vertices ", num_vertices, " does not fit the vertex id type");
        edge_list_view<vertex_t, weight_t> edges;
        edges.num_vertices = static_cast<vertex_t>(num_vertices);
        edges.src = view_of(src);
        edges.dst = view_of(dst);
        if constexpr (weighted) {
          if (weights) {
            edges.weights = view_of(*weights);
          }
        }
        edges.num_weights = num_weights;
        return without_gil([&] { return new holder_t(graph_t::from_edges(res, edges, props)); });
      },
      nb::arg("resources"), nb::arg("num_vertices"), nb::arg("src").noconvert(),
      nb::arg("dst").noconvert(), nb::arg("weights").noconvert().none(), nb::arg("num_weights"),
      nb::arg("properties"), nb::rv_policy::take_ownership,
      "Build a graph from an edge list (edge-major weights).");
  cls.def_static(
      "from_csr",
      [](const resources& res, const in_array<edge_t>& row_ptr, const in_array<vertex_t>& col_ind,
         const weights_arg& weights, int num_weights, const graph_properties& props) {
        csr_view<vertex_t, edge_t, weight_t> csr;
        csr.row_ptr = view_of(row_ptr);
        csr.col_ind = view_of(col_ind);
        if constexpr (weighted) {
          if (weights) {
            csr.weights = view_of(*weights);
          }
        }
        csr.num_weights = num_weights;
        return without_gil([&] { return new holder_t(graph_t::from_csr(res, csr, props)); });
      },
      nb::arg("resources"), nb::arg("row_ptr").noconvert(), nb::arg("col_ind").noconvert(),
      nb::arg("weights").noconvert().none(), nb::arg("num_weights"), nb::arg("properties"),
      nb::rv_policy::take_ownership,
      "Build a graph from a CSR of the out-edges (objective-major weights).");
  cls.def_prop_ro("num_vertices", [](const holder_t& h) {
    return static_cast<std::int64_t>(h.value.num_vertices());
  });
  cls.def_prop_ro("num_edges",
                  [](const holder_t& h) { return static_cast<std::int64_t>(h.value.num_edges()); });
  cls.def_prop_ro("num_weights", [](const holder_t& h) { return h.value.num_weights(); });
  cls.def_prop_ro("directed", [](const holder_t& h) { return h.value.is_directed(); });
  cls.def_prop_ro("has_transposed", [](const holder_t& h) { return h.value.has_transposed(); });
  cls.def_prop_ro("space", [](const holder_t& h) { return h.value.space(); });
  cls.def_prop_ro("version", [](const holder_t& h) { return h.value.version(); });
  cls.def_prop_ro("properties",
                  [](const holder_t& h) -> graph_properties { return h.value.properties(); });
  cls.def(
      "to_csr",
      [](const holder_t& h, const resources& res) {
        auto csr = without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, false);
          locks.lock();
          return h.value.to_csr(res);
        });
        const std::size_t m_edges = csr.col_ind.size();
        const int k = csr.num_weights;
        nb::object weights = nb::none();
        if constexpr (weighted) {
          weights =
              nb::cast(to_numpy_2d(std::move(csr.weights), static_cast<std::size_t>(k), m_edges));
        }
        return nb::make_tuple(to_numpy(std::move(csr.row_ptr)), to_numpy(std::move(csr.col_ind)),
                              weights);
      },
      nb::arg("resources"),
      "(row_ptr, col_ind, weights of shape (K, m) or None): a host copy of the out-edges.");
  cls.def(
      "apply",
      [](holder_t& h, const resources& res, const batch_arrays<vertex_t, weight_t>& b) {
        return without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, true);
          locks.lock();
          return h.value.apply(res, b.view());
        });
      },
      nb::arg("resources"), nb::arg("batch"), "Apply a batch to the graph.");
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
      "The graph for the backend of `resources`.");
  cls.def(
      "reserve",
      [](holder_t& h, const resources& res, std::int64_t edge_capacity) {
        DYNG_EXPECTS(edge_capacity >= 0 && edge_capacity <= static_cast<std::int64_t>(
                                                                std::numeric_limits<edge_t>::max()),
                     "edge_capacity ", edge_capacity, " does not fit the edge offset type");
        without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, true);
          locks.lock();
          h.value.reserve(res, static_cast<edge_t>(edge_capacity));
        });
      },
      nb::arg("resources"), nb::arg("edge_capacity"));
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

void bind_graph(nb::module_& m) {
  nb::enum_<row_layout>(m, "RowLayout", "The storage layout of the rows.")
      .value("compact", row_layout::compact)
      .value("slotted", row_layout::slotted)
      .value("slack", row_layout::slack);
  nb::enum_<row_order>(m, "RowOrder", "The order of the neighbours in a row.")
      .value("sorted", row_order::sorted)
      .value("append", row_order::append);
  nb::enum_<multi_edges>(m, "MultiEdges", "Whether parallel edges may exist.")
      .value("forbid", multi_edges::forbid)
      .value("allow", multi_edges::allow);
  nb::enum_<batch_semantics::existing_insert>(m, "ExistingInsert",
                                              "What an insertion of an existing edge does.")
      .value("upsert", batch_semantics::existing_insert::upsert)
      .value("error", batch_semantics::existing_insert::error)
      .value("ignore", batch_semantics::existing_insert::ignore);
  nb::enum_<batch_semantics::missing_delete>(m, "MissingDelete",
                                             "What a deletion of a missing edge does.")
      .value("ignore", batch_semantics::missing_delete::ignore)
      .value("error", batch_semantics::missing_delete::error);
  nb::enum_<batch_semantics::self_loop>(m, "SelfLoop", "What a self-loop in a batch does.")
      .value("keep", batch_semantics::self_loop::keep)
      .value("drop", batch_semantics::self_loop::drop)
      .value("error", batch_semantics::self_loop::error);

  nb::class_<batch_semantics>(m, "BatchSemantics", "How a batch of edge changes is interpreted.")
      .def(nb::init<>())
      .def_rw("on_existing_insert", &batch_semantics::on_existing_insert)
      .def_rw("on_missing_delete", &batch_semantics::on_missing_delete)
      .def_rw("on_self_loop", &batch_semantics::on_self_loop)
      .def_rw("deletions_first", &batch_semantics::deletions_first)
      .def_rw("allow_vertex_growth", &batch_semantics::allow_vertex_growth)
      .def_rw("as_sets", &batch_semantics::as_sets)
      .def_static("upsert_last_wins", &batch_semantics::upsert_last_wins)
      .def_static("set", &batch_semantics::set);

  nb::class_<graph_properties>(m, "GraphProperties", "The properties of a graph.")
      .def(nb::init<>())
      .def_rw("directed", &graph_properties::directed)
      .def_rw("store_transposed", &graph_properties::store_transposed)
      .def_rw("num_weights", &graph_properties::num_weights)
      .def_rw("layout", &graph_properties::layout)
      .def_rw("headroom", &graph_properties::headroom)
      .def_rw("order", &graph_properties::order)
      .def_rw("parallel_edges", &graph_properties::parallel_edges)
      .def_rw("semantics", &graph_properties::semantics)
      .def_static("mosp_compatible", &graph_properties::mosp_compatible)
      .def_static("cycle_enum_compatible", &graph_properties::cycle_enum_compatible);

  nb::class_<apply_summary>(m, "ApplySummary", "What applying a batch did to a graph.")
      .def_ro("inserted_edges", &apply_summary::inserted_edges)
      .def_ro("updated_edges", &apply_summary::updated_edges)
      .def_ro("deleted_edges", &apply_summary::deleted_edges)
      .def_ro("ignored_deletions", &apply_summary::ignored_deletions)
      .def_ro("dropped_self_loops", &apply_summary::dropped_self_loops)
      .def_ro("cancelled_pairs", &apply_summary::cancelled_pairs)
      .def_ro("inserted_vertices", &apply_summary::inserted_vertices)
      .def_ro("deleted_vertices", &apply_summary::deleted_vertices)
      .def_ro("num_vertices_after", &apply_summary::num_vertices_after)
      .def_ro("ignored_insertions", &apply_summary::ignored_insertions);

#define DYNG_PY_BIND_GRAPH(V, E, W) bind_graph_type<V, E, W>(m);
  DYNG_PY_FOR_EACH_GRAPH_TYPE(DYNG_PY_BIND_GRAPH)
#undef DYNG_PY_BIND_GRAPH
}

}  // namespace dyng::python
