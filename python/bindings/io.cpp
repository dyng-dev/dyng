// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file io.cpp
 * @brief dyng::io: the readers and writers of the 0.1 formats (docs/api/file_formats.md).
 *
 * Readers return NumPy arrays that own the library's vectors (no copy); the typed layer builds
 * graphs and batches from them. Every call releases the GIL.
 */
#include "types.hpp"

#include <dyng/io/batch_io.hpp>
#include <dyng/io/csr_triplet.hpp>
#include <dyng/io/edge_list_io.hpp>
#include <dyng/io/matrix_market.hpp>
#include <dyng/io/result_io.hpp>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace dyng::python {
namespace {

/// (num_vertices, src, dst, weights of shape (m, K) or None, K) of an edge list.
template <typename vertex_t, typename weight_t>
nb::tuple edges_to_numpy(edge_list<vertex_t, weight_t>&& e) {
  const std::size_t m = e.src.size();
  const int k = e.num_weights;
  nb::object weights = nb::none();
  if constexpr (!is_unweighted_v<weight_t>) {
    weights = nb::cast(to_numpy_2d(std::move(e.weights), m, static_cast<std::size_t>(k)));
  }
  return nb::make_tuple(static_cast<std::int64_t>(e.num_vertices), to_numpy(std::move(e.src)),
                        to_numpy(std::move(e.dst)), weights, k);
}

/// The edge_list_view of arrays passed by the typed layer.
template <typename vertex_t, typename weight_t>
edge_list_view<vertex_t, weight_t> edges_view(
    std::int64_t num_vertices, const in_array<vertex_t>& src, const in_array<vertex_t>& dst,
    const std::optional<in_array<weight_scalar_t<weight_t>>>& weights, int num_weights) {
  edge_list_view<vertex_t, weight_t> v;
  v.num_vertices = static_cast<vertex_t>(num_vertices);
  v.src = view_of(src);
  v.dst = view_of(dst);
  if constexpr (!is_unweighted_v<weight_t>) {
    if (weights) {
      v.weights = view_of(*weights);
    }
  }
  v.num_weights = num_weights;
  return v;
}

template <typename vertex_t, typename weight_t>
void bind_edge_list(nb::module_& m) {
  using weights_arg = std::optional<in_array<weight_scalar_t<weight_t>>>;
  const char* v = type_code<vertex_t>::lower;
  const char* w = type_code<weight_t>::lower;
  m.def(
      suffixed("read_edge_list", {v, w}, "_").c_str(),
      [](const std::string& path, const io::edge_list_options& opt) {
        io::edge_list_info info;
        auto edges =
            without_gil([&] { return io::read_edge_list<vertex_t, weight_t>(path, opt, &info); });
        nb::dict d;
        d["external_ids"] = to_numpy(std::move(info.external_ids));
        d["timestamps"] = to_numpy(std::move(info.timestamps));
        d["matrix_market"] = info.matrix_market;
        d["symmetric"] = info.symmetric;
        return nb::make_tuple(edges_to_numpy(std::move(edges)), d);
      },
      nb::arg("path"), nb::arg("options"));
  m.def(
      suffixed("write_edge_list", {v, w}, "_").c_str(),
      [](const std::string& path, std::int64_t num_vertices, const in_array<vertex_t>& src,
         const in_array<vertex_t>& dst, const weights_arg& weights, int num_weights) {
        const auto view =
            edges_view<vertex_t, weight_t>(num_vertices, src, dst, weights, num_weights);
        without_gil([&] { io::write_edge_list<vertex_t, weight_t>(path, view); });
      },
      nb::arg("path"), nb::arg("num_vertices"), nb::arg("src").noconvert(),
      nb::arg("dst").noconvert(), nb::arg("weights").noconvert().none(), nb::arg("num_weights"));
}

template <typename vertex_t>
void bind_weighted_io(nb::module_& m) {
  using weight_t = std::int32_t;
  const char* v = type_code<vertex_t>::lower;
  const char* w = type_code<weight_t>::lower;
  m.def(
      suffixed("read_matrix_market", {v, w}, "_").c_str(),
      [](const std::string& path, const io::matrix_market_options& opt) {
        auto edges =
            without_gil([&] { return io::read_matrix_market<vertex_t, weight_t>(path, opt); });
        return edges_to_numpy(std::move(edges));
      },
      nb::arg("path"), nb::arg("options"));
  m.def(
      suffixed("write_matrix_market", {v, w}, "_").c_str(),
      [](const std::string& path, std::int64_t num_vertices, const in_array<vertex_t>& src,
         const in_array<vertex_t>& dst, const std::optional<in_array<weight_t>>& weights,
         int num_weights, int weight_column) {
        const auto view =
            edges_view<vertex_t, weight_t>(num_vertices, src, dst, weights, num_weights);
        without_gil(
            [&] { io::write_matrix_market<vertex_t, weight_t>(path, view, weight_column); });
      },
      nb::arg("path"), nb::arg("num_vertices"), nb::arg("src").noconvert(),
      nb::arg("dst").noconvert(), nb::arg("weights").noconvert().none(), nb::arg("num_weights"),
      nb::arg("weight_column"));
  m.def(
      suffixed("read_legacy_batch", {v, w}, "_").c_str(),
      [](const std::string& insert_path, const std::string& delete_path, int num_weights,
         std::int64_t num_vertices, bool mosp_lenient) {
        io::legacy_batch_options opt;
        opt.num_weights = num_weights;
        opt.num_vertices = num_vertices;
        opt.mosp_lenient = mosp_lenient;
        auto batch = without_gil([&] {
          return io::read_legacy_batch<vertex_t, weight_t>(insert_path, delete_path, opt);
        });
        return batch_to_numpy(batch);
      },
      nb::arg("insert_path"), nb::arg("delete_path"), nb::arg("num_weights"),
      nb::arg("num_vertices"), nb::arg("mosp_lenient"));
  m.def(
      suffixed("write_legacy_batch", {v, w}, "_").c_str(),
      [](const std::string& insert_path, const std::string& delete_path,
         const batch_arrays<vertex_t, weight_t>& b) {
        without_gil([&] {
          io::write_legacy_batch<vertex_t, weight_t>(insert_path, delete_path, b.view());
        });
      },
      nb::arg("insert_path"), nb::arg("delete_path"), nb::arg("batch"));
  m.def(
      suffixed("read_batches", {v, w}, "_").c_str(),
      [](const std::string& path, int num_weights, std::int64_t num_vertices) {
        io::batch_file_options opt;
        opt.num_weights = num_weights;
        opt.num_vertices = num_vertices;
        auto batches = without_gil([&] { return io::read_batches<vertex_t, weight_t>(path, opt); });
        nb::list out;
        for (const auto& b : batches) {
          out.append(batch_to_numpy(b));
        }
        return out;
      },
      nb::arg("path"), nb::arg("num_weights"), nb::arg("num_vertices"),
      "The batches of a .dgt file, each as the tuple of batch_to_numpy().");
  m.def(
      suffixed("write_batches", {v, w}, "_").c_str(),
      [](const std::string& path, const nb::list& batches) {
        // The views read the batches' arrays, which the list keeps alive during the call.
        std::vector<edge_batch_view<vertex_t, weight_t>> views;
        views.reserve(nb::len(batches));
        for (nb::handle b : batches) {
          views.push_back(nb::cast<const batch_arrays<vertex_t, weight_t>&>(b).view());
        }
        without_gil([&] { io::write_batches<vertex_t, weight_t>(path, views); });
      },
      nb::arg("path"), nb::arg("batches"), "Write native batches as a .dgt file.");
  m.def(
      suffixed("read_parents", {v}, "_").c_str(),
      [](const std::string& path, std::int64_t num_vertices) {
        return to_numpy(
            without_gil([&] { return io::read_parents<vertex_t>(path, num_vertices); }));
      },
      nb::arg("path"), nb::arg("num_vertices"));
  m.def(
      "write_parents",
      [](const std::string& path, const in_array<vertex_t>& parents) {
        without_gil([&] { io::write_parents<vertex_t>(path, view_of(parents)); });
      },
      nb::arg("path"), nb::arg("parents").noconvert());
}

template <typename vertex_t, typename edge_t, typename weight_t>
void bind_csr_triplet(nb::module_& m) {
  m.def(
      suffixed("read_csr_triplet",
               {type_code<vertex_t>::lower, type_code<edge_t>::lower, type_code<weight_t>::lower},
               "_")
          .c_str(),
      [](const std::string& prefix, int num_weights) {
        io::csr_triplet_options opt;
        opt.num_weights = num_weights;
        auto csr = without_gil(
            [&] { return io::read_csr_triplet<vertex_t, edge_t, weight_t>(prefix, opt); });
        const std::size_t m_edges = csr.col_ind.size();
        const int k = csr.num_weights;
        return nb::make_tuple(
            to_numpy(std::move(csr.row_ptr)), to_numpy(std::move(csr.col_ind)),
            to_numpy_2d(std::move(csr.weights), static_cast<std::size_t>(k), m_edges), k);
      },
      nb::arg("prefix"), nb::arg("num_weights"));
  m.def(
      "write_csr_triplet",
      [](const std::string& prefix, const in_array<edge_t>& row_ptr,
         const in_array<vertex_t>& col_ind, const in_array<weight_t>& weights, int num_weights) {
        csr_view<vertex_t, edge_t, weight_t> csr;
        csr.row_ptr = view_of(row_ptr);
        csr.col_ind = view_of(col_ind);
        csr.weights = view_of(weights);
        csr.num_weights = num_weights;
        without_gil([&] { io::write_csr_triplet(prefix, csr); });
      },
      nb::arg("prefix"), nb::arg("row_ptr").noconvert(), nb::arg("col_ind").noconvert(),
      nb::arg("weights").noconvert(), nb::arg("num_weights"));
}

}  // namespace

void bind_io(nb::module_& m) {
  nb::enum_<io::vertex_ids>(m, "VertexIds", "io::vertex_ids.")
      .value("compact", io::vertex_ids::compact)
      .value("as_is", io::vertex_ids::as_is);
  nb::enum_<io::duplicate_edges>(m, "DuplicateEdges", "io::duplicate_edges.")
      .value("merge", io::duplicate_edges::merge)
      .value("keep", io::duplicate_edges::keep);
  nb::enum_<io::matrix_market_weights>(m, "MatrixMarketWeights", "io::matrix_market_weights.")
      .value("automatic", io::matrix_market_weights::automatic)
      .value("none", io::matrix_market_weights::none)
      .value("from_file", io::matrix_market_weights::from_file)
      .value("random", io::matrix_market_weights::random);

  nb::class_<io::edge_list_options>(m, "EdgeListOptions", "io::edge_list_options.")
      .def(nb::init<>())
      .def_rw("num_weights", &io::edge_list_options::num_weights)
      .def_rw("ids", &io::edge_list_options::ids)
      .def_rw("index_base", &io::edge_list_options::index_base)
      .def_rw("symmetrize", &io::edge_list_options::symmetrize)
      .def_rw("drop_self_loops", &io::edge_list_options::drop_self_loops)
      .def_rw("duplicates", &io::edge_list_options::duplicates)
      .def_rw("threads", &io::edge_list_options::threads);
  nb::class_<io::matrix_market_options>(m, "MatrixMarketOptions", "io::matrix_market_options.")
      .def(nb::init<>())
      .def_rw("weights", &io::matrix_market_options::weights)
      .def_rw("drop_self_loops", &io::matrix_market_options::drop_self_loops)
      .def_rw("sort_and_dedupe", &io::matrix_market_options::sort_and_dedupe)
      .def(
          "set_random",
          [](io::matrix_market_options& o, int num_weights, std::int64_t min, std::int64_t max,
             std::uint32_t seed) {
            o.random.num_weights = num_weights;
            o.random.min = min;
            o.random.max = max;
            o.random.seed = seed;
          },
          nb::arg("num_weights"), nb::arg("min"), nb::arg("max"), nb::arg("seed"));

  bind_edge_list<std::int32_t, std::int32_t>(m);
  bind_edge_list<std::int64_t, std::int32_t>(m);
  bind_edge_list<std::int32_t, unweighted>(m);
  bind_edge_list<std::int64_t, unweighted>(m);
  bind_weighted_io<std::int32_t>(m);
  bind_weighted_io<std::int64_t>(m);
  bind_csr_triplet<std::int32_t, std::int32_t, std::int32_t>(m);
  bind_csr_triplet<std::int32_t, std::int64_t, std::int32_t>(m);
  bind_csr_triplet<std::int64_t, std::int64_t, std::int32_t>(m);

  m.def(
      "read_distances",
      [](const std::string& path, std::int64_t num_vertices) {
        return to_numpy(
            without_gil([&] { return io::read_distances<std::int64_t>(path, num_vertices); }));
      },
      nb::arg("path"), nb::arg("num_vertices"));
  m.def(
      "write_distances",
      [](const std::string& path, const in_array<std::int64_t>& distances) {
        without_gil([&] { io::write_distances<std::int64_t>(path, view_of(distances)); });
      },
      nb::arg("path"), nb::arg("distances").noconvert());
  m.def(
      "histogram_csv",
      [](const in_array<std::uint64_t>& counts, bool include_total) {
        return without_gil([&] {
          std::ostringstream out;
          io::write_histogram_csv(out, view_of(counts), include_total);
          return out.str();
        });
      },
      nb::arg("counts").noconvert(), nb::arg("include_total"),
      "The cycle histogram as CycleEnumeration-GPU's CSV text.");
}

}  // namespace dyng::python
