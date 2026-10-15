// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file containers.hpp
 * @brief The container axis of the conformance kit (ADR 0043): container_traits<container_t>, what
 *        the checks C0-C12 need to know about the container an algorithm runs on, so that one
 *        typed suite serves graph and hypergraph algorithms alike.
 *
 * container_traits<container_t> provides:
 *
 *     using container_type, properties_type, shape_type, model_type, batch_type,
 *           batch_view_type, summary_type, generated_type, mix_type;
 *     static constexpr container_kind kind;               // C0: the manifest's container
 *     static std::string case_name();                     // the typed case's name
 *     template <typename traits_t> static std::vector<kit_preset<properties_type>> presets();
 *     static std::vector<mix_type> mixes(kit_check);      // the batch mixes of each check
 *     static bool mix_applies(mix_type, const properties_type&);
 *     template <typename traits_t>
 *     static model_type random_model(size_class, std::mt19937_64&);   // traits_t::shape(size)
 *     static container_type build(const model_type&, const resources&, const properties_type&);
 *     static generated_type random_batch(model_type&, mix_type, std::mt19937_64&,
 *                                        const properties_type&);
 *     static generated_type random_batch(model_type&, mix_type, std::mt19937_64&);
 *     static void observe(model_type&, const generated_type&, const summary_type&);
 *     static const batch_type& batch_of(const generated_type&);
 *     static bool invertible(const generated_type&);
 *     static generated_type inverse(model_type&, const generated_type&, const summary_type&);
 *     static std::vector<std::int64_t> summary_counters(const summary_type&);   // C3, C6, C10
 *     static std::int64_t changed(const summary_type&);   // C1: what an empty batch must not do
 *     static ::testing::AssertionResult matches(const container_type&, const model_type&);
 *     template <typename traits_t, typename partner_t>
 *     static std::vector<kit_preset<properties_type>> composition_presets();     // C10
 *     template <typename traits_t, typename partner_t>
 *     static model_type composition_model(const properties_type&, std::mt19937_64&);  // C10
 *     static mix_type composition_mix(int step);           // C10
 *     static mix_type stale_mix();                         // C11
 *     static mix_type stream_mix(int step);                // C12
 *
 * The model is the kit's host copy of the container, from which every batch is generated (a
 * kit tests only what its generator makes: M6b). random_batch() applies the batch's deletions to
 * the model at once; observe() completes it with what only the container decides (a hypergraph's
 * inserted ids); inverse() builds the batch that undoes one (C5). The graph's specialization below
 * keeps the kit's behaviour of 0.1-0.2 exactly (generators.hpp); the hypergraph's is in
 * hypergraph_generators.hpp.
 */
#pragma once

#include "conformance/generators.hpp"
#include "conformance/test_traits.hpp"

#include <dyng/core/registry.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace dyng::conformance {

/// The checks whose batch mixes differ (container_traits::mixes).
enum class kit_check : std::uint8_t {
  c2,  ///< the oracle over every mix
  c3,  ///< backends agree
  c4,  ///< engines agree
  c5,  ///< a batch and its inverse cancel
  c6,  ///< determinism
  c8,  ///< budgets
  c9,  ///< stats sanity
};

/// A property preset of the checks: a label for traces and the container's properties (the
/// algorithm's requirements applied).
template <typename properties_t>
struct kit_preset {
  std::string label;   ///< for traces
  properties_t props;  ///< the properties
};

/// The test_traits options the container axes read.
namespace kit_detail {

template <typename traits_t, typename = void>
struct has_extra_properties : std::false_type {};
template <typename traits_t>
struct has_extra_properties<traits_t, std::void_t<decltype(traits_t::extra_properties())>>
    : std::true_type {};

template <typename traits_t, typename = void>
struct has_num_weights : std::false_type {};
template <typename traits_t>
struct has_num_weights<traits_t, std::void_t<decltype(traits_t::num_weights)>> : std::true_type {};

/// The weight columns of the kit's weighted graphs: test_traits::num_weights, default 1.
template <typename traits_t>
constexpr int weight_columns() {
  if constexpr (has_num_weights<traits_t>::value) {
    return traits_t::num_weights;
  } else {
    return 1;
  }
}

}  // namespace kit_detail

/// The kit's view of a container type; specialised per container (see the file comment).
template <typename container_t>
struct container_traits;

/// The graph's container axis: the kit of 0.1-0.2, unchanged.
template <typename vertex_t, typename edge_t, typename weight_t>
struct container_traits<graph<vertex_t, edge_t, weight_t>> {
  using container_type = graph<vertex_t, edge_t, weight_t>;      ///< the container
  using properties_type = graph_properties;                      ///< its properties
  using shape_type = graph_shape;                                ///< test_traits::shape()
  using model_type = graph_model<container_type>;                ///< the host model
  using batch_type = edge_batch<vertex_t, weight_t>;             ///< the owning batch
  using batch_view_type = edge_batch_view<vertex_t, weight_t>;   ///< update()'s batch argument
  using summary_type = apply_summary;                            ///< stats::batch
  using generated_type = generated_batch<container_type>;        ///< a generated batch
  using mix_type = batch_mix;                                    ///< the batch mixes
  static constexpr container_kind kind = container_kind::graph;  ///< the manifest's container

  /// The name of a type in a case name.
  template <typename type_t>
  static std::string type_name() {
    if constexpr (is_unweighted_v<type_t>) {
      return "unweighted";
    } else {
      return (std::is_signed_v<type_t> ? "i" : "u") + std::to_string(8 * sizeof(type_t));
    }
  }

  /// "v<V>_e<E>_w<W>", e.g. v32_e64_unweighted.
  static std::string case_name() {
    return "v" + type_name<vertex_t>().substr(1) + "_e" + type_name<edge_t>().substr(1) + "_w" +
           (is_unweighted_v<weight_t> ? std::string("unweighted")
                                      : type_name<weight_t>().substr(1));
  }

  /// The presets: the defaults, the defaults with the insertions applied before the deletions,
  /// set() semantics, and the traits' extras; each with require().
  template <typename traits_t>
  static std::vector<kit_preset<graph_properties>> presets() {
    std::vector<kit_preset<graph_properties>> out;
    graph_properties plain;
    traits_t::require(plain);
    out.push_back({"upsert_last_wins()", plain});
    // applied_batch::delta lists the requested changes: with the insertions first, an edge a
    // batch inserts and deletes is listed but gone (batch_mix::cancel), which an algorithm must
    // not trust.
    graph_properties inserts_first;
    inserts_first.semantics.deletions_first = false;
    traits_t::require(inserts_first);
    if (!inserts_first.semantics.as_sets && inserts_first.semantics.deletions_first == false) {
      out.push_back({"upsert_last_wins(), insertions first", inserts_first});
    }
    graph_properties sets = plain;
    sets.semantics = batch_semantics::set();
    out.push_back({"set()", sets});
    if constexpr (kit_detail::has_extra_properties<traits_t>::value) {
      int i = 0;
      for (graph_properties p : traits_t::extra_properties()) {
        traits_t::require(p);
        out.push_back({"extra properties " + std::to_string(i++), p});
      }
    }
    return out;
  }

  /// The mixes of each check.
  static std::vector<batch_mix> mixes(kit_check check) {
    switch (check) {
      case kit_check::c2:
      case kit_check::c4:
      case kit_check::c9:
        return all_mixes();
      case kit_check::c3:
        return {batch_mix::mixed, batch_mix::heavy, batch_mix::reweight, batch_mix::grow,
                batch_mix::cancel};
      case kit_check::c5:
        return {batch_mix::insert_only, batch_mix::delete_only, batch_mix::mixed,
                batch_mix::local,       batch_mix::heavy,       batch_mix::reweight};
      case kit_check::c6:
        return {batch_mix::mixed, batch_mix::heavy, batch_mix::grow};
      case kit_check::c8:
        return {batch_mix::mixed, batch_mix::local, batch_mix::reweight};
    }
    return {};
  }

  /// Whether a mix applies to a preset (reweight: weighted graphs, upsert only).
  static bool mix_applies(batch_mix mix, const graph_properties& props) {
    if (mix == batch_mix::reweight) {
      return !is_unweighted_v<weight_t> && !props.semantics.as_sets &&
             props.semantics.on_existing_insert == batch_semantics::existing_insert::upsert;
    }
    return true;
  }

  /// Whether the algorithm's graphs are directed (its requirements may ask for an undirected
  /// graph, graph_properties::directed = false; the kit's models follow).
  template <typename traits_t>
  static bool directed_graphs() {
    graph_properties props;
    traits_t::require(props);
    return props.directed;
  }

  /// A random model of a size class.
  template <typename traits_t>
  static model_type random_model(size_class size, std::mt19937_64& rng) {
    model_type m = conformance::random_model<container_type>(traits_t::shape(size), rng,
                                                             directed_graphs<traits_t>());
    m.num_weights = kit_detail::weight_columns<traits_t>();
    return m;
  }

  /// The graph of the model.
  static container_type build(const model_type& m, const resources& res,
                              const graph_properties& props) {
    return m.build(res, props);
  }

  /// A batch of `mix` from the model (applied to the model) under the preset's semantics.
  static generated_type random_batch(model_type& m, batch_mix mix, std::mt19937_64& rng,
                                     const graph_properties& props) {
    return conformance::random_batch(m, mix, rng, props.semantics);
  }

  /// A batch of `mix` under the default semantics (upsert_last_wins()).
  static generated_type random_batch(model_type& m, batch_mix mix, std::mt19937_64& rng) {
    return conformance::random_batch(m, mix, rng);
  }

  /// The model needs nothing from the commit (random_batch() applied the whole batch).
  static void observe(model_type& /*m*/, const generated_type& /*gen*/,
                      const summary_type& /*summary*/) {}

  /// The batch of a generated batch.
  static const batch_type& batch_of(const generated_type& gen) {
    return gen.batch;
  }

  /// Whether C5 may invert it (no vertex growth, no cancelled pairs).
  static bool invertible(const generated_type& gen) {
    return gen.invertible;
  }

  /// The inverse the generator recorded (with the old weights).
  static generated_type inverse(model_type& /*m*/, const generated_type& gen,
                                const summary_type& /*summary*/) {
    generated_type out;
    out.batch = gen.inverse;
    out.invertible = false;
    return out;
  }

  /// The deterministic counters of the batch summary.
  static std::vector<std::int64_t> summary_counters(const summary_type& b) {
    return {b.inserted_edges,    b.updated_edges,      b.deleted_edges,
            b.ignored_deletions, b.dropped_self_loops, b.cancelled_pairs,
            b.inserted_vertices, b.num_vertices_after, b.ignored_insertions};
  }

  /// What an empty batch must leave at zero.
  static std::int64_t changed(const summary_type& b) {
    return b.inserted_edges + b.deleted_edges + b.updated_edges;
  }

  /// The graph has the model's vertex and edge counts.
  static ::testing::AssertionResult matches(const container_type& g, const model_type& m) {
    if (static_cast<std::int64_t>(g.num_vertices()) != static_cast<std::int64_t>(m.num_vertices)) {
      return ::testing::AssertionFailure()
             << "the graph has " << static_cast<std::int64_t>(g.num_vertices())
             << " vertices, the model " << static_cast<std::int64_t>(m.num_vertices);
    }
    if (static_cast<std::int64_t>(g.num_edges()) !=
        static_cast<std::int64_t>(m.weights.size()) * m.directions()) {
      return ::testing::AssertionFailure()
             << "the graph has " << static_cast<std::int64_t>(g.num_edges()) << " edges, the model "
             << static_cast<std::int64_t>(m.weights.size()) * m.directions();
    }
    return ::testing::AssertionSuccess();
  }

  /// C10: a graph meeting both algorithms' requirements under the three semantics
  /// (upsert_last_wins(), set(), upsert_last_wins() with the insertions first).
  template <typename traits_t, typename partner_t>
  static std::vector<kit_preset<graph_properties>> composition_presets() {
    std::vector<kit_preset<graph_properties>> out;
    for (const int variant : {0, 1, 2}) {
      graph_properties props;
      props.semantics.deletions_first = variant != 2;
      traits_t::require(props);
      partner_t::require(props);
      if (variant == 1) {
        props.semantics = batch_semantics::set();
      }
      out.push_back({variant == 0   ? "upsert_last_wins()"
                     : variant == 1 ? "set()"
                                    : "upsert_last_wins(), insertions first",
                     props});
    }
    return out;
  }

  /// C10: the model of a composition preset (the smaller of the two small shapes).
  template <typename traits_t, typename partner_t>
  static model_type composition_model(const graph_properties& props, std::mt19937_64& rng) {
    graph_shape shape = traits_t::shape(size_class::small);
    const graph_shape theirs_shape = partner_t::shape(size_class::small);
    shape.vertices = std::min(shape.vertices, theirs_shape.vertices);
    shape.edges = std::min(shape.edges, theirs_shape.edges);
    model_type model = conformance::random_model<container_type>(shape, rng, props.directed);
    model.num_weights =
        std::max(kit_detail::weight_columns<traits_t>(), kit_detail::weight_columns<partner_t>());
    return model;
  }

  /// C10's mix of each of its three batches.
  static batch_mix composition_mix(int step) {
    return step == 2 ? batch_mix::grow : step == 1 ? batch_mix::cancel : batch_mix::mixed;
  }

  /// C11's batch (applied by a separate g.apply()).
  static batch_mix stale_mix() {
    return batch_mix::mixed;
  }

  /// C12's mix of each of its four batches.
  static batch_mix stream_mix(int step) {
    return step == 3 ? batch_mix::grow : batch_mix::heavy;
  }
};

}  // namespace dyng::conformance
