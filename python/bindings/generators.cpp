// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file generators.cpp
 * @brief dyng::generators: the hypergraph generators (library-owned distributions), and
 *        dyng::generators::legacy: MOSP's change generator, CycleEnumeration-GPU's batch generator
 *        and ESCHER-GPU's hypergraph and batch generators, bit-exact for a fixed seed, and MOSP's
 *        weight stream (for `dyng prep widen`).
 */
#include "types.hpp"
#include "util/rng.hpp"  // private header of libdyng (cpp/src): the libstdc++ draws

#include <dyng/generators/hypergraphs.hpp>
#include <dyng/generators/legacy.hpp>

#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <cstdint>
#include <random>
#include <vector>

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

template <typename vertex_t, typename weight_t>
void bind_hypergraph_generator_type(nb::module_& m) {
  using holder_t = hypergraph_holder<vertex_t, weight_t>;
  const char* v = type_code<vertex_t>::lower;
  const char* w = type_code<weight_t>::lower;
  m.def(
      suffixed("uniform_hypergraph", {v, w}, "_").c_str(),
      [](const generators::uniform_hypergraph_options& opt) {
        auto list =
            without_gil([&] { return generators::uniform_hypergraph<vertex_t, weight_t>(opt); });
        return hyperedge_list_to_numpy(std::move(list));
      },
      nb::arg("options"), "(num_vertices, offsets, members, weights or None).");
  m.def(
      "random_hyperedge_batch",
      [](const resources& res, const holder_t& h, const generators::hyperedge_batch_options& opt) {
        auto batch = without_gil([&] {
          lock_set locks;
          locks.add(h.mutex, false);
          locks.lock();
          return generators::random_hyperedge_batch(res, h.value, opt);
        });
        return hyper_batch_to_numpy(batch);
      },
      nb::arg("resources"), nb::arg("hypergraph"), nb::arg("options"),
      "(delete_hyperedges, insert_offsets, insert_members, insert_weights or None).");
  m.def(
      suffixed("escher_hypergraph", {v, w}, "_").c_str(),
      [](std::mt19937_64& rng, const legacy::escher_hypergraph_options& opt) {
        auto list =
            without_gil([&] { return legacy::escher_hypergraph<vertex_t, weight_t>(rng, opt); });
        return hyperedge_list_to_numpy(std::move(list));
      },
      nb::arg("engine"), nb::arg("options"));
}

/// A generator over an engine owned by a Python object (kept alive by the generator).
struct escher_batches {
  legacy::escher_batch_generator generator;
};

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

  // MOSP's weight stream: `count` draws of std::uniform_int_distribution<int>(lo, hi) (libstdc++'s
  // algorithm, reproduced by detail::legacy_uniform_int) from std::mt19937(seed), the weights of
  // `mospPrep mtx2csr` and `mospPrep widen` in draw order (the CLI's `dyng prep widen`).
  m.def(
      "legacy_mosp_weights",
      [](std::int64_t count, std::int32_t lo, std::int32_t hi, std::uint32_t seed) {
        DYNG_EXPECTS(count >= 0, "legacy_mosp_weights: count must be >= 0, got ", count);
        DYNG_EXPECTS(lo <= hi, "legacy_mosp_weights: min ", lo, " > max ", hi);
        std::vector<std::int32_t> out = without_gil([&] {
          std::vector<std::int32_t> w(static_cast<std::size_t>(count));
          std::mt19937 engine(seed);
          for (std::int32_t& x : w) {
            x = static_cast<std::int32_t>(detail::legacy_uniform_int(engine, lo, hi));
          }
          return w;
        });
        return to_numpy(std::move(out));
      },
      nb::arg("count"), nb::arg("min"), nb::arg("max"), nb::arg("seed"),
      "MOSP's weight stream: `count` draws in [min, max] from std::mt19937(seed).");

  // ---- hypergraphs (library-owned distributions) --------------------------------------------
  nb::enum_<generators::hyperedge_batch_mix>(m, "HyperedgeBatchMix",
                                             "generators::hyperedge_batch_mix.")
      .value("insert_only", generators::hyperedge_batch_mix::insert_only)
      .value("delete_only", generators::hyperedge_batch_mix::delete_only)
      .value("mixed", generators::hyperedge_batch_mix::mixed)
      .value("reuse", generators::hyperedge_batch_mix::reuse)
      .value("heavy_delete", generators::hyperedge_batch_mix::heavy_delete)
      .value("delete_all", generators::hyperedge_batch_mix::delete_all)
      .value("reinsert", generators::hyperedge_batch_mix::reinsert)
      .value("larger", generators::hyperedge_batch_mix::larger)
      .value("duplicates", generators::hyperedge_batch_mix::duplicates)
      .value("dead_ids", generators::hyperedge_batch_mix::dead_ids)
      .value("skewed", generators::hyperedge_batch_mix::skewed)
      .value("limits", generators::hyperedge_batch_mix::limits)
      .value("empty", generators::hyperedge_batch_mix::empty);
  nb::class_<generators::uniform_hypergraph_options>(m, "UniformHypergraphOptions",
                                                     "generators::uniform_hypergraph_options.")
      .def(nb::init<>())
      .def_rw("num_vertices", &generators::uniform_hypergraph_options::num_vertices)
      .def_rw("num_hyperedges", &generators::uniform_hypergraph_options::num_hyperedges)
      .def_rw("min_cardinality", &generators::uniform_hypergraph_options::min_cardinality)
      .def_rw("max_cardinality", &generators::uniform_hypergraph_options::max_cardinality)
      .def_rw("min_weight", &generators::uniform_hypergraph_options::min_weight)
      .def_rw("max_weight", &generators::uniform_hypergraph_options::max_weight)
      .def_rw("seed", &generators::uniform_hypergraph_options::seed);
  nb::class_<generators::hyperedge_batch_options>(m, "HyperedgeBatchOptions",
                                                  "generators::hyperedge_batch_options.")
      .def(nb::init<>())
      .def_rw("mix", &generators::hyperedge_batch_options::mix)
      .def_rw("size", &generators::hyperedge_batch_options::size)
      .def_rw("min_cardinality", &generators::hyperedge_batch_options::min_cardinality)
      .def_rw("max_cardinality", &generators::hyperedge_batch_options::max_cardinality)
      .def_rw("skewed_cardinality", &generators::hyperedge_batch_options::skewed_cardinality)
      .def_rw("min_weight", &generators::hyperedge_batch_options::min_weight)
      .def_rw("max_weight", &generators::hyperedge_batch_options::max_weight)
      .def_rw("seed", &generators::hyperedge_batch_options::seed);

  // ---- ESCHER-GPU's generators (one std::mt19937_64 shared in A's draw order) ----------------
  nb::class_<std::mt19937_64>(m, "Mt19937_64", "std::mt19937_64 (the shared engine of ESCHER-GPU).")
      .def(nb::init<std::uint64_t>(), nb::arg("seed"))
      .def("__call__", [](std::mt19937_64& e) { return static_cast<std::uint64_t>(e()); });
  nb::enum_<legacy::escher_insert_mode>(m, "EscherInsertMode", "generators::legacy::escher_insert_mode.")
      .value("random", legacy::escher_insert_mode::random)
      .value("sample", legacy::escher_insert_mode::sample);
  nb::class_<legacy::escher_hypergraph_options>(m, "EscherHypergraphOptions",
                                                "generators::legacy::escher_hypergraph_options.")
      .def(nb::init<>())
      .def_rw("num_hyperedges", &legacy::escher_hypergraph_options::num_hyperedges)
      .def_rw("max_cardinality", &legacy::escher_hypergraph_options::max_cardinality)
      .def_rw("min_vertex", &legacy::escher_hypergraph_options::min_vertex)
      .def_rw("max_vertex", &legacy::escher_hypergraph_options::max_vertex);
  nb::class_<legacy::escher_batch_options>(m, "EscherBatchOptions",
                                           "generators::legacy::escher_batch_options.")
      .def(nb::init<>())
      .def_rw("num_deletions", &legacy::escher_batch_options::num_deletions)
      .def_rw("num_insertions", &legacy::escher_batch_options::num_insertions)
      .def_rw("insert_mode", &legacy::escher_batch_options::insert_mode)
      .def_rw("graph", &legacy::escher_batch_options::graph)
      .def_static("batch_size", &legacy::escher_batch_options::batch_size, nb::arg("size"),
                  nb::arg("mode"), nb::arg("graph"));
  nb::class_<escher_batches>(m, "EscherBatchGenerator",
                             "generators::legacy::escher_batch_generator (private).")
      .def(
          "__init__",
          [](escher_batches* self, std::mt19937_64& rng, const legacy::escher_batch_options& opt) {
            new (self) escher_batches{legacy::escher_batch_generator(rng, opt)};
          },
          nb::arg("engine"), nb::arg("options"), nb::keep_alive<1, 2>())
      .def(
          "next",
          [](escher_batches& g, const resources& res,
             const hypergraph_holder<std::int32_t, unweighted>& h) {
            auto batch = without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, false);
              locks.lock();
              return g.generator.next(res, h.value);
            });
            return hyper_batch_to_numpy(batch);
          },
          nb::arg("resources"), nb::arg("hypergraph"))
      .def(
          "next",
          [](escher_batches& g, const resources& res,
             const hypergraph_holder<std::int32_t, std::int32_t>& h) {
            auto batch = without_gil([&] {
              lock_set locks;
              locks.add(h.mutex, false);
              locks.lock();
              return g.generator.next(res, h.value);
            });
            return hyper_batch_to_numpy(batch);
          },
          nb::arg("resources"), nb::arg("hypergraph"));
  m.def("standard_library_matches_legacy_streams", [] {
    return without_gil([] { return legacy::standard_library_matches_legacy_streams(); });
  });
#define DYNG_PY_BIND_HYPERGRAPH_GENERATORS(V, W) bind_hypergraph_generator_type<V, W>(m);
  DYNG_PY_FOR_EACH_HYPERGRAPH_TYPE(DYNG_PY_BIND_HYPERGRAPH_GENERATORS)
#undef DYNG_PY_BIND_HYPERGRAPH_GENERATORS

#define DYNG_PY_BIND_GENERATORS(V, E, W) bind_generator_type<V, E, W>(m);
  DYNG_PY_FOR_EACH_GRAPH_TYPE(DYNG_PY_BIND_GENERATORS)
#undef DYNG_PY_BIND_GENERATORS
}

}  // namespace dyng::python
