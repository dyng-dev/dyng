// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file batch.cpp
 * @brief The batch of edge changes over Python arrays (one class per (vertex, weight) type).
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

}  // namespace

void bind_batch(nb::module_& m) {
#define DYNG_PY_BIND_BATCH(V, W) bind_batch_type<V, W>(m);
  DYNG_PY_FOR_EACH_BATCH_TYPE(DYNG_PY_BIND_BATCH)
#undef DYNG_PY_BIND_BATCH
}

}  // namespace dyng::python
