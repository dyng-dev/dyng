// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file io.cpp
 * @brief dyng::io: the readers and writers of the file formats (docs/api/file_formats.md): graphs,
 *        hypergraphs (.hg), batches (.dgt text, .dgb binary, MOSP's legacy files) and results.
 *
 * Readers return NumPy arrays that own the library's vectors (no copy); the typed layer builds
 * graphs and batches from them. Every call releases the GIL.
 */
#include "types.hpp"

#include <dyng/io/batch_io.hpp>
#include <dyng/io/csr_triplet.hpp>
#include <dyng/io/edge_list_io.hpp>
#include <dyng/io/hypergraph_io.hpp>
#include <dyng/io/matrix_market.hpp>
#include <dyng/io/result_io.hpp>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <memory>
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

// ---- hypergraphs: .hg files and hyperedge batches (.dgt, .dgb) -------------------------------

template <typename vertex_t, typename weight_t>
void bind_hypergraph_io(nb::module_& m) {
  using weights_arg = std::optional<in_array<weight_scalar_t<weight_t>>>;
  using batch_t = hyper_batch_arrays<vertex_t, weight_t>;
  const char* v = type_code<vertex_t>::lower;
  const char* w = type_code<weight_t>::lower;
  m.def(
      suffixed("read_hypergraph", {v, w}, "_").c_str(),
      [](const std::string& path, const io::hypergraph_read_options& opt) {
        io::hypergraph_file_info info;
        auto list = without_gil(
            [&] { return io::read_hypergraph<vertex_t, weight_t>(path, opt, &info); });
        nb::dict d;
        d["external_ids"] = to_numpy(std::move(info.external_ids));
        d["lines"] = info.lines;
        d["repeated_members"] = info.repeated_members;
        d["over_cardinality"] = info.over_cardinality;
        d["repeated_hyperedges"] = info.repeated_hyperedges;
        return nb::make_tuple(hyperedge_list_to_numpy(std::move(list)), d);
      },
      nb::arg("path"), nb::arg("options"),
      "((num_vertices, offsets, members, weights or None), info) of a .hg file.");
  m.def(
      suffixed("write_hypergraph", {v, w}, "_").c_str(),
      [](const std::string& path, std::int64_t num_vertices, const in_array<std::int64_t>& offsets,
         const in_array<vertex_t>& members, const weights_arg& weights, std::int64_t index_base) {
        const auto view =
            hyperedge_list_view_of<vertex_t, weight_t>(num_vertices, offsets, members, weights);
        without_gil([&] { io::write_hypergraph<vertex_t, weight_t>(path, view, index_base); });
      },
      nb::arg("path"), nb::arg("num_vertices"), nb::arg("offsets").noconvert(),
      nb::arg("members").noconvert(), nb::arg("weights").noconvert().none(),
      nb::arg("index_base"));
  m.def(
      suffixed("read_hyperedge_batches", {v, w}, "_").c_str(),
      [](const std::string& path, std::int64_t num_vertices) {
        io::hyperedge_batch_file_options opt;
        opt.num_vertices = num_vertices;
        auto batches = without_gil(
            [&] { return io::read_hyperedge_batches<vertex_t, weight_t>(path, opt); });
        nb::list out;
        for (const auto& b : batches) {
          out.append(hyper_batch_to_numpy(b));
        }
        return out;
      },
      nb::arg("path"), nb::arg("num_vertices"),
      "The hypergraph batches of a .dgt file, each as the tuple of hyper_batch_to_numpy().");
  m.def(
      suffixed("write_hyperedge_batches", {v, w}, "_").c_str(),
      [](const std::string& path, const nb::list& batches) {
        std::vector<hyperedge_batch_view<vertex_t, weight_t>> views;
        views.reserve(nb::len(batches));
        for (nb::handle b : batches) {
          views.push_back(nb::cast<const batch_t&>(b).view());
        }
        without_gil([&] { io::write_hyperedge_batches<vertex_t, weight_t>(path, views); });
      },
      nb::arg("path"), nb::arg("batches"), "Write native hyperedge batches as a .dgt file.");
  m.def(
      suffixed("write_binary_hyperedge_batches", {v, w}, "_").c_str(),
      [](const std::string& path, const nb::list& batches, std::uint64_t flags,
         const nb::list& deleted_offsets, const nb::list& deleted_members,
         const nb::list& expected_ids) {
        const std::size_t n = nb::len(batches);
        const bool with_members = (flags & io::batch_file_flags::deleted_members) != 0;
        const bool with_ids = (flags & io::batch_file_flags::expected_ids) != 0;
        DYNG_EXPECTS(!with_members || (nb::len(deleted_offsets) == n && nb::len(deleted_members) == n),
                     "write_binary_hyperedge_batches: one deleted member set list per batch");
        DYNG_EXPECTS(!with_ids || nb::len(expected_ids) == n,
                     "write_binary_hyperedge_batches: one expected id array per batch");
        std::vector<hyperedge_batch_view<vertex_t, weight_t>> views;
        std::vector<io::hyperedge_batch_annotations<vertex_t>> notes(n);
        // The arrays of the annotations (the lists keep their memory alive during the call).
        std::vector<in_array<std::int64_t>> offsets_of(n), ids_of(n);
        std::vector<in_array<vertex_t>> members_of(n);
        views.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
          views.push_back(nb::cast<const batch_t&>(batches[i]).view());
          if (with_members) {
            offsets_of[i] = nb::cast<in_array<std::int64_t>>(deleted_offsets[i], false);
            members_of[i] = nb::cast<in_array<vertex_t>>(deleted_members[i], false);
            notes[i].deleted_offsets = view_of(offsets_of[i]);
            notes[i].deleted_members = view_of(members_of[i]);
          }
          if (with_ids) {
            ids_of[i] = nb::cast<in_array<std::int64_t>>(expected_ids[i], false);
            notes[i].expected_ids = view_of(ids_of[i]);
          }
        }
        without_gil([&] {
          io::batch_writer out(path, io::hyperedge_batch_file_header<vertex_t, weight_t>(flags));
          for (std::size_t i = 0; i < n; ++i) {
            out.write(views[i], notes[i]);
          }
          out.close();
        });
      },
      nb::arg("path"), nb::arg("batches"), nb::arg("flags"), nb::arg("deleted_offsets"),
      nb::arg("deleted_members"), nb::arg("expected_ids"),
      "Write native hyperedge batches (and their annotations) as a .dgb file.");
}

/// The batch_reader of a .dgb file, its current batch copied into NumPy arrays on request.
struct binary_reader {
  std::unique_ptr<io::batch_reader> reader;
};

/// (kind, vertex dtype, weight dtype, K, flags, batches, format version) of a .dgb header.
nb::dict header_dict(const io::batch_file_header& h) {
  nb::dict d;
  d["format_version"] = h.format_version;
  d["kind"] = std::string(to_string(h.kind));
  d["vertex_type"] = std::string(io::to_string(h.vertex_type));
  d["weight_type"] = std::string(io::to_string(h.weight_type));
  d["num_weights"] = h.num_weights;
  d["flags"] = h.flags;
  d["num_batches"] = h.num_batches;
  return d;
}

template <typename value_t>
nb::ndarray<nb::numpy, value_t, nb::ndim<1>> copied(array_view<const value_t> v) {
  return to_numpy(std::vector<value_t>(v.begin(), v.end()));
}

/// The current batch of a reader, by the header's kind and types.
nb::tuple current_batch(const io::batch_reader& r) {
  const io::batch_file_header& h = r.header();
  if (h.kind == container_kind::hypergraph) {
    DYNG_EXPECTS(h.vertex_type == io::batch_dtype::int32,
                 "the Python package reads hypergraph .dgb files with int32 vertex ids");
    nb::object batch;
    if (h.weight_type == io::batch_dtype::none) {
      const auto b = r.hyperedge_batch<std::int32_t, unweighted>();
      batch = nb::make_tuple(copied(b.delete_hyperedges), copied(b.insert_offsets),
                             copied(b.insert_members), nb::none());
    } else {
      DYNG_EXPECTS(h.weight_type == io::batch_dtype::int32,
                   "the Python package reads hypergraph .dgb files with int32 weights");
      const auto b = r.hyperedge_batch<std::int32_t, std::int32_t>();
      batch = nb::make_tuple(copied(b.delete_hyperedges), copied(b.insert_offsets),
                             copied(b.insert_members), copied(b.insert_weights));
    }
    const auto a = r.annotations<std::int32_t>();
    nb::object members = nb::none();
    nb::object ids = nb::none();
    if ((h.flags & io::batch_file_flags::deleted_members) != 0) {
      members = nb::make_tuple(copied(a.deleted_offsets), copied(a.deleted_members));
    }
    if ((h.flags & io::batch_file_flags::expected_ids) != 0) {
      ids = nb::cast(copied(a.expected_ids));
    }
    return nb::make_tuple(r.batch_id(), batch, members, ids);
  }
  // A graph file: (insert_src, insert_dst, weights (n, K) or None, delete_src, delete_dst, K).
  auto graph_batch = [&](auto vtag, auto wtag) {
    using vertex_t = decltype(vtag);
    using weight_t = decltype(wtag);
    const auto b = r.edge_batch<vertex_t, weight_t>();
    nb::object weights = nb::none();
    if constexpr (!is_unweighted_v<weight_t>) {
      const std::size_t n = b.insert_src.size();
      weights = nb::cast(to_numpy_2d(std::vector<weight_t>(b.insert_weights.begin(),
                                                           b.insert_weights.end()),
                                     n, static_cast<std::size_t>(b.num_weights)));
    }
    return nb::make_tuple(copied(b.insert_src), copied(b.insert_dst), weights,
                          copied(b.delete_src), copied(b.delete_dst),
                          is_unweighted_v<weight_t> ? 0 : b.num_weights);
  };
  DYNG_EXPECTS(h.weight_type != io::batch_dtype::int64,
               "the Python package reads graph .dgb files with int32 weights or none");
  nb::object batch;
  const bool v32 = h.vertex_type == io::batch_dtype::int32;
  if (h.weight_type == io::batch_dtype::none) {
    batch = v32 ? graph_batch(std::int32_t{}, unweighted{})
                : graph_batch(std::int64_t{}, unweighted{});
  } else {
    batch = v32 ? graph_batch(std::int32_t{}, std::int32_t{})
                : graph_batch(std::int64_t{}, std::int32_t{});
  }
  return nb::make_tuple(r.batch_id(), batch, nb::none(), nb::none());
}

void bind_binary_batches(nb::module_& m) {
  nb::class_<binary_reader>(m, "BatchReader", "io::batch_reader over a .dgb file (private).")
      .def(
          "__init__",
          [](binary_reader* self, const std::string& path, std::int64_t num_vertices) {
            io::batch_reader_options opt;
            opt.num_vertices = num_vertices;
            auto r = without_gil([&] { return std::make_unique<io::batch_reader>(path, opt); });
            new (self) binary_reader{std::move(r)};
          },
          nb::arg("path"), nb::arg("num_vertices"))
      .def_prop_ro("header",
                   [](const binary_reader& r) { return header_dict(r.reader->header()); })
      .def(
          "next",
          [](binary_reader& r) -> nb::object {
            const bool more = without_gil([&] { return r.reader->next(); });
            if (!more) {
              return nb::none();
            }
            return current_batch(*r.reader);
          },
          "The next record as (batch_id, batch tuple, deleted members or None, expected ids or "
          "None), or None at the end of the file.");
  // Graph batches (the edge batch types of the package): a writer over every batch.
  auto bind_edges = [&m](auto vtag, auto wtag) {
    using vertex_t = decltype(vtag);
    using weight_t = decltype(wtag);
    m.def(
        suffixed("write_binary_batches",
                 {type_code<vertex_t>::lower, type_code<weight_t>::lower}, "_")
            .c_str(),
        [](const std::string& path, int num_weights, const nb::list& batches) {
          std::vector<edge_batch_view<vertex_t, weight_t>> views;
          views.reserve(nb::len(batches));
          for (nb::handle b : batches) {
            views.push_back(nb::cast<const batch_arrays<vertex_t, weight_t>&>(b).view());
          }
          without_gil([&] {
            io::batch_writer out(path, io::edge_batch_file_header<vertex_t, weight_t>(num_weights));
            for (const auto& v : views) {
              out.write(v);
            }
            out.close();
          });
        },
        nb::arg("path"), nb::arg("num_weights"), nb::arg("batches"),
        "Write native edge batches as a .dgb file.");
  };
  bind_edges(std::int32_t{}, std::int32_t{});
  bind_edges(std::int64_t{}, std::int32_t{});
  bind_edges(std::int32_t{}, unweighted{});
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

  nb::enum_<io::vertex_relabel>(m, "VertexRelabel", "io::vertex_relabel.")
      .value("none", io::vertex_relabel::none)
      .value("dense_sorted", io::vertex_relabel::dense_sorted)
      .value("first_appearance", io::vertex_relabel::first_appearance);
  nb::class_<io::weight_spec>(m, "WeightSpec", "io::weight_spec.")
      .def(nb::init<>())
      .def_rw("lo", &io::weight_spec::lo)
      .def_rw("hi", &io::weight_spec::hi)
      .def_rw("seed", &io::weight_spec::seed)
      .def_rw("hsosp_target_draw", &io::weight_spec::hsosp_target_draw);
  nb::class_<io::hypergraph_read_options>(m, "HypergraphReadOptions",
                                          "io::hypergraph_read_options.")
      .def(nb::init<>())
      .def_rw("index_base", &io::hypergraph_read_options::index_base)
      .def_rw("dedupe_hyperedges", &io::hypergraph_read_options::dedupe_hyperedges)
      .def_rw("max_cardinality", &io::hypergraph_read_options::max_cardinality)
      .def_rw("relabel", &io::hypergraph_read_options::relabel)
      .def_rw("random_weights", &io::hypergraph_read_options::random_weights, nb::arg().none())
      .def_rw("num_vertices", &io::hypergraph_read_options::num_vertices)
      .def_static("escher_gpu", &io::hypergraph_read_options::escher_gpu)
      .def_static("hsosp", &io::hypergraph_read_options::hsosp, nb::arg("seed") = 20260725U,
                  nb::arg("max_cardinality") = 25);
  m.attr("BATCH_FLAG_DELETED_MEMBERS") = io::batch_file_flags::deleted_members;
  m.attr("BATCH_FLAG_EXPECTED_IDS") = io::batch_file_flags::expected_ids;
  m.attr("BATCH_FLAG_INSERT_FIRST") = io::batch_file_flags::insert_first;
#define DYNG_PY_BIND_HYPERGRAPH_IO(V, W) bind_hypergraph_io<V, W>(m);
  DYNG_PY_FOR_EACH_HYPERGRAPH_TYPE(DYNG_PY_BIND_HYPERGRAPH_IO)
#undef DYNG_PY_BIND_HYPERGRAPH_IO
  bind_binary_batches(m);

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
      "write_path_costs",
      [](const std::string& path, const in_array<std::int64_t>& costs, int num_objectives) {
        without_gil(
            [&] { io::write_path_costs<std::int64_t>(path, view_of(costs), num_objectives); });
      },
      nb::arg("path"), nb::arg("costs").noconvert(), nb::arg("num_objectives"));
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
