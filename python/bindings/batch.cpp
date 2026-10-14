// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file batch.cpp
 * @brief The batches over Python arrays: edge changes (one class per (vertex, weight) type) and
 *        hypergraph changes (one class per hypergraph type).
 */
#include "types.hpp"

#include <nanobind/stl/optional.h>

#include <optional>
#include <string>

namespace dyng::python {
namespace {

template <typename vertex_t, typename weight_t>
void bind_batch_type(nb::module_& m) {
  using batch_t = batch_arrays<vertex_t, weight_t>;
  using weights_arg = std::optional<in_array<weight_scalar_t<weight_t>>>;
  const std::string name =
      suffixed("EdgeBatch", {type_code<vertex_t>::upper, type_code<weight_t>::upper}, "");
  nb::class_<batch_t>(m, name.c_str(), "A batch of edge changes over arrays (private).")
      .def(
          "__init__",
          [](batch_t* self, const in_array<vertex_t>& insert_src,
             const in_array<vertex_t>& insert_dst, const weights_arg& insert_weights,
             const in_array<vertex_t>& delete_src, const in_array<vertex_t>& delete_dst,
             int num_weights, const std::optional<in_array<vertex_t>>& insert_vertices,
             const std::optional<in_array<std::int8_t>>& insert_vertex_labels,
             const std::optional<in_array<vertex_t>>& delete_vertices) {
            if constexpr (is_unweighted_v<weight_t>) {
              DYNG_EXPECTS(!insert_weights && num_weights == 0,
                           "a batch of an unweighted graph carries no weights");
            }
            new (self) batch_t{insert_src,           insert_dst,      insert_weights,
                               delete_src,           delete_dst,      insert_vertices,
                               insert_vertex_labels, delete_vertices, num_weights};
          },
          nb::arg("insert_src").noconvert(), nb::arg("insert_dst").noconvert(),
          nb::arg("insert_weights").noconvert().none(), nb::arg("delete_src").noconvert(),
          nb::arg("delete_dst").noconvert(), nb::arg("num_weights"),
          nb::arg("insert_vertices").noconvert().none(),
          nb::arg("insert_vertex_labels").noconvert().none(),
          nb::arg("delete_vertices").noconvert().none())
      .def_prop_ro("num_insertions", [](const batch_t& b) { return b.insert_src.shape(0); })
      .def_prop_ro("num_deletions", [](const batch_t& b) { return b.delete_src.shape(0); })
      .def_prop_ro("num_weights", [](const batch_t& b) { return b.num_weights; });
}

template <typename vertex_t, typename weight_t>
void bind_hyper_batch_type(nb::module_& m) {
  using batch_t = hyper_batch_arrays<vertex_t, weight_t>;
  using weights_arg = std::optional<in_array<weight_scalar_t<weight_t>>>;
  constexpr bool weighted = !is_unweighted_v<weight_t>;
  const std::string batch_name =
      suffixed("HyperedgeBatch", {type_code<vertex_t>::upper, type_code<weight_t>::upper}, "");
  nb::class_<batch_t>(m, batch_name.c_str(), "A batch of hypergraph changes over arrays (private).")
      .def(
          "__init__",
          [](batch_t* self, const in_array<std::int64_t>& delete_hyperedges,
             const in_array<std::int64_t>& insert_offsets, const in_array<vertex_t>& insert_members,
             const weights_arg& insert_weights,
             const std::optional<in_array<std::int64_t>>& delete_incidence_hyperedge,
             const std::optional<in_array<vertex_t>>& delete_incidence_vertex,
             const std::optional<in_array<std::int64_t>>& insert_incidence_hyperedge,
             const std::optional<in_array<vertex_t>>& insert_incidence_vertex) {
            if constexpr (!weighted) {
              DYNG_EXPECTS(!insert_weights,
                           "a batch of an unweighted hypergraph carries no weights");
            }
            new (self) batch_t{delete_hyperedges,          insert_offsets,
                               insert_members,             insert_weights,
                               delete_incidence_hyperedge, delete_incidence_vertex,
                               insert_incidence_hyperedge, insert_incidence_vertex};
          },
          nb::arg("delete_hyperedges").noconvert(), nb::arg("insert_offsets").noconvert(),
          nb::arg("insert_members").noconvert(), nb::arg("insert_weights").noconvert().none(),
          nb::arg("delete_incidence_hyperedge").noconvert().none(),
          nb::arg("delete_incidence_vertex").noconvert().none(),
          nb::arg("insert_incidence_hyperedge").noconvert().none(),
          nb::arg("insert_incidence_vertex").noconvert().none())
      .def_prop_ro("num_insertions",
                   [](const batch_t& b) { return static_cast<std::int64_t>(b.view().num_insertions()); })
      .def_prop_ro("num_deletions",
                   [](const batch_t& b) { return static_cast<std::int64_t>(b.view().num_deletions()); });

}

}  // namespace

void bind_batch(nb::module_& m) {
#define DYNG_PY_BIND_BATCH(V, W) bind_batch_type<V, W>(m);
  DYNG_PY_FOR_EACH_BATCH_TYPE(DYNG_PY_BIND_BATCH)
#undef DYNG_PY_BIND_BATCH
#define DYNG_PY_BIND_HYPER_BATCH(V, W) bind_hyper_batch_type<V, W>(m);
  DYNG_PY_FOR_EACH_HYPERGRAPH_TYPE(DYNG_PY_BIND_HYPER_BATCH)
#undef DYNG_PY_BIND_HYPER_BATCH
}

}  // namespace dyng::python
