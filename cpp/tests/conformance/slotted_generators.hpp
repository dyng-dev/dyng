// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file slotted_generators.hpp
 * @brief The float slotted axis of the conformance kit (ADR 0076, the ADR 0043 pattern): a host
 *        model of a row_layout::slotted similarity graph with vertex batches under
 *        batch_semantics::net_effect(), the vertex-batch mixes generated from it, and
 *        container_traits<slotted_axis<graph<V, E, W>>> (containers.hpp has the vocabulary).
 *
 * The full graph of a model is LabelPropagation-CUDA's (generators::legacy::
 * dynlp_vertex_arrival_stream at average degree 5: weights in [0.01, 1], intra-class edges in
 * [0.5, 1]); the graph is graph::with_capacity() of its row capacities. The model keeps the state
 * of every vertex (inactive, active, deleted) and of every edge of the full graph (not delivered,
 * delivered, pending: withheld from an arrival batch while both endpoints are active, so that the
 * mix edges_between_present has edges to deliver, or dropped: delivered to an endpoint that is no
 * longer active, which net_effect() drops and counts). A vertex is a seed when kit_seed_label()
 * gives it a class: the rule depends on the vertex id only, so a test_traits' compute() can name
 * the seeds of any graph of the axis without the model (label_propagation's seeds are not stored
 * in the graph; ADR 0069).
 *
 * The mixes (M10a criterion 15): arrivals with labels (with their edges to present and past
 * vertices and among themselves, at least one labelled arrival), departures, mixed (arrivals and
 * departures), insert_and_delete_same_batch (some unlabelled arrivals deleted in the same batch:
 * they end deleted, as in the original) and edges_between_present (no vertex operation: half of
 * the pending edges). Departures and the deleted arrivals are unlabelled vertices, as in the
 * original generator (its phase C). Edge deletions exist only as C7's not_supported_error. Every model starts
 * with 70 % of its vertices present (graph::apply of the model's state, build()).
 */
#pragma once

#include "conformance/containers.hpp"
#include "conformance/test_traits.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/registry.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/generators/vertex_arrival.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/graph/graph_view.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dyng::conformance {

/// The key of the float slotted axis of a graph type (test_traits::container_axis).
template <typename graph_t>
struct slotted_axis {};

/// The vertex-batch mixes of the slotted axis.
enum class vertex_mix : std::uint8_t {
  arrival_with_labels,           ///< arrivals (some labelled) with their edges
  departure,                     ///< departures of present vertices
  mixed,                         ///< arrivals and departures
  insert_and_delete_same_batch,  ///< arrivals, some of them deleted in the same batch
  edges_between_present,         ///< pending edges between present vertices, no vertex operation
};

/// The name of a mix.
inline std::string_view to_string(vertex_mix mix) noexcept {
  switch (mix) {
    case vertex_mix::arrival_with_labels:
      return "arrival_with_labels";
    case vertex_mix::departure:
      return "departure";
    case vertex_mix::mixed:
      return "mixed";
    case vertex_mix::insert_and_delete_same_batch:
      return "insert_and_delete_same_batch";
    case vertex_mix::edges_between_present:
      return "edges_between_present";
  }
  return "?";
}

/// Every mix.
inline std::vector<vertex_mix> all_vertex_mixes() {
  return {vertex_mix::arrival_with_labels, vertex_mix::departure, vertex_mix::mixed,
          vertex_mix::insert_and_delete_same_batch, vertex_mix::edges_between_present};
}

/// The shape of a slotted model (test_traits::shape()).
struct slotted_shape {
  std::int64_t vertices = 5000;     ///< vertices of the full graph (the DynLP generator's)
  std::int32_t average_degree = 5;  ///< its average degree
};

/**
 * @brief The kit's seed rule: the class (0 or 1) of a seed, or -1. About one vertex in 16 is a
 *        seed; the rule depends on the id only (a test_traits' compute() reads it).
 * @param[in] v The vertex.
 * @return -1, 0 or 1.
 */
inline std::int8_t kit_seed_label(std::int64_t v) noexcept {
  std::uint64_t x = static_cast<std::uint64_t>(v) + 0x9E3779B97F4A7C15ULL;  // splitmix64
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  x ^= x >> 31;
  if ((x & 15U) != 0) {
    return -1;
  }
  return static_cast<std::int8_t>((x >> 8) & 1U);
}

/// The kit's host model of a slotted graph.
template <typename vertex_t, typename edge_t>
struct slotted_model {
  /// The state of a vertex (vertex_status's, by name).
  enum vstate : std::uint8_t { inactive = 0, active = 1, deleted = 2 };
  /// The state of an edge of the full graph.
  enum estate : std::uint8_t { undelivered = 0, delivered = 1, pending = 2, dropped = 3 };

  std::int64_t n = 0;                                       ///< the vertex capacity
  std::vector<std::int64_t> row_ptr;                        ///< the full graph's rows
  std::vector<std::int32_t> col;                            ///< the full graph's columns
  std::vector<std::int64_t> entry_edge;                     ///< the undirected edge of each entry
  std::vector<std::pair<std::int32_t, std::int32_t>> ends;  ///< each undirected edge (u < v)
  std::vector<float> weight;                                ///< its weight
  std::vector<std::uint8_t> edge_state;                     ///< its estate
  std::vector<edge_t> capacity;     ///< the row capacities (the full degrees)
  std::vector<std::uint8_t> state;  ///< the vstate of each vertex
  std::int64_t active_count = 0;    ///< active vertices
  std::int64_t live_edges = 0;      ///< delivered edges with both endpoints active

  /// Whether an edge's two endpoints are active.
  [[nodiscard]] bool live(std::int64_t e) const {
    const auto& p = ends[static_cast<std::size_t>(e)];
    return state[static_cast<std::size_t>(p.first)] == active &&
           state[static_cast<std::size_t>(p.second)] == active;
  }
};

/// A generated vertex batch and what its commit must report.
template <typename vertex_t, typename weight_t>
struct slotted_batch {
  edge_batch<vertex_t, weight_t> batch{1};  ///< the batch
  std::int64_t arrivals = 0;                ///< insert_vertices
  std::int64_t labelled_arrivals = 0;       ///< of them with a seed label
  std::int64_t departures = 0;              ///< delete_vertices
  std::int64_t inserted_and_deleted = 0;    ///< ids in both lists
  std::int64_t expected_inserted = 0;       ///< apply_summary::inserted_vertices (|Ins \ Del|)
  std::int64_t expected_deleted = 0;        ///< apply_summary::deleted_vertices (|Del ∩ V_t|)
  std::int64_t expected_live = 0;           ///< edges appended (undirected; x2 entries)
  std::int64_t expected_dropped = 0;        ///< edges dropped (undirected; x2 entries)
  std::int64_t delivered_pending = 0;       ///< pending edges delivered by this batch
};

namespace slotted_detail {

/// A uniform index below `k` (k > 0) from the kit's engine (the same in every build).
inline std::size_t below(std::mt19937_64& rng, std::size_t k) {
  return static_cast<std::size_t>(rng() % static_cast<std::uint64_t>(k));
}

/// A Bernoulli draw with probability num / den.
inline bool chance(std::mt19937_64& rng, std::uint64_t num, std::uint64_t den) {
  return rng() % den < num;
}

/// The model of a full DynLP graph with no vertex present.
template <typename vertex_t, typename edge_t>
slotted_model<vertex_t, edge_t> empty_model(const slotted_shape& shape, std::mt19937_64& rng) {
  generators::vertex_arrival_options opt;
  opt.total_vertices = shape.vertices;
  opt.average_degree = shape.average_degree;
  opt.num_batches = 1;
  opt.seed = static_cast<std::uint32_t>(rng() & 0x7fffffffU);
  const auto stream = generators::legacy::dynlp_vertex_arrival_stream<std::int32_t, std::int64_t>(
      resources::sequential(), opt);
  slotted_model<vertex_t, edge_t> m;
  m.n = shape.vertices;
  const auto& g = stream.graph;
  m.row_ptr.assign(g.row_ptr.begin(), g.row_ptr.end());
  m.col.assign(g.col_ind.begin(), g.col_ind.end());
  m.entry_edge.assign(m.col.size(), -1);
  for (std::int64_t u = 0; u < m.n; ++u) {
    for (auto e = static_cast<std::size_t>(m.row_ptr[static_cast<std::size_t>(u)]);
         e < static_cast<std::size_t>(m.row_ptr[static_cast<std::size_t>(u) + 1]); ++e) {
      const std::int32_t v = m.col[e];
      if (u < v) {
        m.entry_edge[e] = static_cast<std::int64_t>(m.ends.size());
        m.ends.emplace_back(static_cast<std::int32_t>(u), v);
        m.weight.push_back(g.weights[e]);
      }
    }
  }
  // The other direction of each edge (rows are sorted by column: a binary search).
  for (std::int64_t u = 0; u < m.n; ++u) {
    for (auto e = static_cast<std::size_t>(m.row_ptr[static_cast<std::size_t>(u)]);
         e < static_cast<std::size_t>(m.row_ptr[static_cast<std::size_t>(u) + 1]); ++e) {
      const std::int32_t v = m.col[e];
      if (u > v) {
        const auto begin = m.col.begin() + m.row_ptr[static_cast<std::size_t>(v)];
        const auto end = m.col.begin() + m.row_ptr[static_cast<std::size_t>(v) + 1];
        const auto at = std::lower_bound(begin, end, static_cast<std::int32_t>(u));
        m.entry_edge[e] = m.entry_edge[static_cast<std::size_t>(at - m.col.begin())];
      }
    }
  }
  m.edge_state.assign(m.ends.size(), slotted_model<vertex_t, edge_t>::undelivered);
  m.capacity.resize(static_cast<std::size_t>(m.n));
  for (std::int64_t u = 0; u < m.n; ++u) {
    m.capacity[static_cast<std::size_t>(u)] = static_cast<edge_t>(
        m.row_ptr[static_cast<std::size_t>(u) + 1] - m.row_ptr[static_cast<std::size_t>(u)]);
  }
  m.state.assign(static_cast<std::size_t>(m.n), slotted_model<vertex_t, edge_t>::inactive);
  return m;
}

/// The departure of an active vertex from the model: its live edges stop being live, its pending
/// edges can no longer be delivered (dropped).
template <typename vertex_t, typename edge_t>
void depart(slotted_model<vertex_t, edge_t>& m, std::int64_t x) {
  using model_t = slotted_model<vertex_t, edge_t>;
  for (auto e = static_cast<std::size_t>(m.row_ptr[static_cast<std::size_t>(x)]);
       e < static_cast<std::size_t>(m.row_ptr[static_cast<std::size_t>(x) + 1]); ++e) {
    const auto id = static_cast<std::size_t>(m.entry_edge[e]);
    if (m.edge_state[id] == model_t::delivered && m.live(static_cast<std::int64_t>(id))) {
      --m.live_edges;
    } else if (m.edge_state[id] == model_t::pending) {
      m.edge_state[id] = model_t::dropped;
    }
  }
  m.state[static_cast<std::size_t>(x)] = model_t::deleted;
  --m.active_count;
}

/**
 * @brief A vertex batch of `arrivals` arrivals (`ins_del` of them unlabelled and deleted in the
 *        same batch), `departures` departures of active vertices and `deliver` pending edges,
 *        applied to the model. Arriving edges whose endpoints both end active are withheld as
 *        pending with probability `withhold` / 8.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
slotted_batch<vertex_t, weight_t> make_batch(slotted_model<vertex_t, edge_t>& m,
                                             std::mt19937_64& rng, std::int64_t arrivals,
                                             std::int64_t ins_del, std::int64_t departures,
                                             std::int64_t deliver, bool want_label,
                                             std::uint64_t withhold) {
  using model_t = slotted_model<vertex_t, edge_t>;
  slotted_batch<vertex_t, weight_t> out;
  // Departures first: active unlabelled vertices before the batch (the original generator's rule,
  // phase C; AD7's bound and the class margin m are calibrated on that flow, ADR 0076).
  std::vector<std::int64_t> present;
  for (std::int64_t v = 0; v < m.n; ++v) {
    if (m.state[static_cast<std::size_t>(v)] == model_t::active && kit_seed_label(v) < 0) {
      present.push_back(v);
    }
  }
  std::vector<std::int64_t> leaving;
  departures = std::min<std::int64_t>(departures, static_cast<std::int64_t>(present.size()));
  for (std::int64_t i = 0; i < departures; ++i) {
    const std::size_t at = i + below(rng, present.size() - static_cast<std::size_t>(i));
    std::swap(present[static_cast<std::size_t>(i)], present[at]);
    leaving.push_back(present[static_cast<std::size_t>(i)]);
  }
  // Arrivals: inactive vertices, at least one labelled when asked.
  std::vector<std::int64_t> idle;
  for (std::int64_t v = 0; v < m.n; ++v) {
    if (m.state[static_cast<std::size_t>(v)] == model_t::inactive) {
      idle.push_back(v);
    }
  }
  arrivals = std::min<std::int64_t>(arrivals, static_cast<std::int64_t>(idle.size()));
  std::vector<std::int64_t> coming;
  for (std::int64_t i = 0; i < arrivals; ++i) {
    const std::size_t at = i + below(rng, idle.size() - static_cast<std::size_t>(i));
    std::swap(idle[static_cast<std::size_t>(i)], idle[at]);
    coming.push_back(idle[static_cast<std::size_t>(i)]);
  }
  if (want_label && !coming.empty() &&
      std::none_of(coming.begin(), coming.end(),
                   [](std::int64_t v) { return kit_seed_label(v) >= 0; })) {
    for (std::size_t i = coming.size(); i < idle.size(); ++i) {
      if (kit_seed_label(idle[i]) >= 0) {
        coming.back() = idle[i];
        break;
      }
    }
  }
  // Ins ∩ Del: unlabelled arrivals deleted in the same batch.
  std::vector<std::uint8_t> transient(coming.size(), 0);
  std::int64_t marked = 0;
  for (std::size_t i = 0; i < coming.size() && marked < ins_del; ++i) {
    if (kit_seed_label(coming[i]) < 0) {
      transient[i] = 1;
      ++marked;
    }
  }
  // The batch's vertex operations.
  for (std::size_t i = 0; i < coming.size(); ++i) {
    const std::int8_t label = transient[i] != 0 ? std::int8_t{-1} : kit_seed_label(coming[i]);
    out.batch.insert_vertex(static_cast<vertex_t>(coming[i]), label);
    out.labelled_arrivals += label >= 0 ? 1 : 0;
  }
  for (const std::int64_t x : leaving) {
    out.batch.delete_vertex(static_cast<vertex_t>(x));
  }
  for (std::size_t i = 0; i < coming.size(); ++i) {
    if (transient[i] != 0) {
      out.batch.delete_vertex(static_cast<vertex_t>(coming[i]));
    }
  }
  out.arrivals = static_cast<std::int64_t>(coming.size());
  out.departures = static_cast<std::int64_t>(leaving.size()) + marked;
  out.inserted_and_deleted = marked;
  out.expected_inserted = out.arrivals - marked;
  out.expected_deleted = static_cast<std::int64_t>(leaving.size());
  // The model after the vertex operations (Ins \ Del active, Del deleted).
  for (const std::int64_t x : leaving) {
    depart(m, x);
  }
  for (std::size_t i = 0; i < coming.size(); ++i) {
    const auto v = static_cast<std::size_t>(coming[i]);
    if (transient[i] != 0) {
      m.state[v] = model_t::deleted;
    } else {
      m.state[v] = model_t::active;
      ++m.active_count;
    }
  }
  // The arrivals' edges: to every vertex that arrived before or in this batch (each edge once,
  // from its later endpoint in the arrival order).
  const auto add_edge = [&](std::size_t id) {
    const auto& p = m.ends[id];
    out.batch.insert_edge(static_cast<vertex_t>(p.first), static_cast<vertex_t>(p.second),
                          {static_cast<weight_t>(m.weight[id])});
    if (m.live(static_cast<std::int64_t>(id))) {
      m.edge_state[id] = model_t::delivered;
      ++m.live_edges;
      ++out.expected_live;
    } else {
      m.edge_state[id] = model_t::dropped;
      ++out.expected_dropped;
    }
  };
  for (std::size_t i = 0; i < coming.size(); ++i) {
    const std::int64_t x = coming[i];
    for (auto e = static_cast<std::size_t>(m.row_ptr[static_cast<std::size_t>(x)]);
         e < static_cast<std::size_t>(m.row_ptr[static_cast<std::size_t>(x) + 1]); ++e) {
      const auto id = static_cast<std::size_t>(m.entry_edge[e]);
      if (m.edge_state[id] != model_t::undelivered) {
        continue;  // the other endpoint arrived earlier in this batch and took it
      }
      const auto y = static_cast<std::size_t>(m.col[e]);
      if (m.state[y] == model_t::inactive) {
        continue;  // it comes with y (an arrival of this batch is active or deleted by now)
      }
      if (m.live(static_cast<std::int64_t>(id)) && chance(rng, withhold, 8)) {
        m.edge_state[id] = model_t::pending;
        continue;
      }
      add_edge(id);
    }
  }
  // Pending edges between present vertices.
  if (deliver > 0) {
    std::vector<std::size_t> waiting;
    for (std::size_t id = 0; id < m.ends.size(); ++id) {
      if (m.edge_state[id] == model_t::pending) {
        waiting.push_back(id);
      }
    }
    deliver = std::min<std::int64_t>(deliver, static_cast<std::int64_t>(waiting.size()));
    for (std::int64_t i = 0; i < deliver; ++i) {
      const std::size_t at = i + below(rng, waiting.size() - static_cast<std::size_t>(i));
      std::swap(waiting[static_cast<std::size_t>(i)], waiting[at]);
      add_edge(waiting[static_cast<std::size_t>(i)]);
      ++out.delivered_pending;
    }
  }
  return out;
}

/// The number of pending edges of a model.
template <typename vertex_t, typename edge_t>
std::int64_t pending_edges(const slotted_model<vertex_t, edge_t>& m) {
  return static_cast<std::int64_t>(std::count(m.edge_state.begin(), m.edge_state.end(),
                                              slotted_model<vertex_t, edge_t>::pending));
}

}  // namespace slotted_detail

/// The float slotted axis (ADR 0076): label_propagation's similarity graphs with vertex batches.
template <typename vertex_t, typename edge_t, typename weight_t>
struct container_traits<slotted_axis<graph<vertex_t, edge_t, weight_t>>> {
  using container_type = graph<vertex_t, edge_t, weight_t>;      ///< the container
  using properties_type = graph_properties;                      ///< its properties
  using shape_type = slotted_shape;                              ///< test_traits::shape()
  using model_type = slotted_model<vertex_t, edge_t>;            ///< the host model
  using batch_type = edge_batch<vertex_t, weight_t>;             ///< the owning batch
  using batch_view_type = edge_batch_view<vertex_t, weight_t>;   ///< update()'s batch argument
  using summary_type = apply_summary;                            ///< stats::batch
  using generated_type = slotted_batch<vertex_t, weight_t>;      ///< a generated batch
  using mix_type = vertex_mix;                                   ///< the batch mixes
  static constexpr container_kind kind = container_kind::graph;  ///< the manifest's container

  /// "slotted_v32_e64_float".
  static std::string case_name() {
    return "slotted_v" + std::to_string(8 * sizeof(vertex_t)) + "_e" +
           std::to_string(8 * sizeof(edge_t)) + "_float";
  }

  /// The slotted properties (row_layout::slotted, net_effect(); ADR 0069).
  static graph_properties slotted_properties() {
    graph_properties p;
    p.directed = false;
    p.layout = row_layout::slotted;
    p.order = row_order::append;
    p.parallel_edges = multi_edges::allow;
    p.semantics = batch_semantics::net_effect();
    return p;
  }

  /// One preset: net_effect() is the only semantics of slotted rows in 0.4 (AD11).
  template <typename traits_t>
  static std::vector<kit_preset<graph_properties>> presets() {
    graph_properties p = slotted_properties();
    traits_t::require(p);
    return {{"slotted, net_effect()", p}};
  }

  /// The mixes of each check.
  static std::vector<vertex_mix> mixes(kit_check check) {
    switch (check) {
      case kit_check::c2:
      case kit_check::c4:
      case kit_check::c9:
        return all_vertex_mixes();
      case kit_check::c3:
        return {vertex_mix::mixed, vertex_mix::insert_and_delete_same_batch,
                vertex_mix::edges_between_present};
      case kit_check::c5:
        return {};
      case kit_check::c6:
        return {vertex_mix::arrival_with_labels, vertex_mix::mixed};
      case kit_check::c8:
        return {vertex_mix::arrival_with_labels, vertex_mix::mixed};
    }
    return {};
  }

  /// Every mix applies to the one preset.
  static bool mix_applies(vertex_mix /*mix*/, const graph_properties& /*props*/) {
    return true;
  }

  /// A model of a size class: 70 % of the full graph's vertices present (with their edges, one in
  /// eight withheld as pending).
  template <typename traits_t>
  static model_type random_model(size_class size, std::mt19937_64& rng) {
    model_type m = slotted_detail::empty_model<vertex_t, edge_t>(traits_t::shape(size), rng);
    (void)slotted_detail::make_batch<vertex_t, edge_t, weight_t>(m, rng, (7 * m.n) / 10, 0, 0, 0,
                                                                 true, 1);
    return m;
  }

  /// The graph of the model's state: with_capacity(), then one batch with every vertex that
  /// arrived (the seeds labelled) and every delivered edge, then one with the departed vertices.
  static container_type build(const model_type& m, const resources& res,
                              const graph_properties& props) {
    const std::vector<edge_t>& cap = m.capacity;
    container_type g =
        container_type::with_capacity(res, static_cast<vertex_t>(m.n), host_view(cap), props);
    batch_type arrive(1);
    batch_type leave(1);
    for (std::int64_t v = 0; v < m.n; ++v) {
      const std::uint8_t st = m.state[static_cast<std::size_t>(v)];
      if (st == model_type::inactive) {
        continue;
      }
      arrive.insert_vertex(static_cast<vertex_t>(v), kit_seed_label(v));
      if (st == model_type::deleted) {
        leave.delete_vertex(static_cast<vertex_t>(v));
      }
    }
    for (std::size_t id = 0; id < m.ends.size(); ++id) {
      if (m.edge_state[id] == model_type::delivered) {
        arrive.insert_edge(static_cast<vertex_t>(m.ends[id].first),
                           static_cast<vertex_t>(m.ends[id].second),
                           {static_cast<weight_t>(m.weight[id])});
      }
    }
    (void)g.apply(res, arrive.view());
    if (!leave.empty()) {
      (void)g.apply(res, leave.view());
    }
    return g;
  }

  /// A batch of `mix` from the model (applied to the model).
  static generated_type random_batch(model_type& m, vertex_mix mix, std::mt19937_64& rng,
                                     const graph_properties& /*props*/) {
    return random_batch(m, mix, rng);
  }

  /// A batch of `mix` from the model (applied to the model).
  static generated_type random_batch(model_type& m, vertex_mix mix, std::mt19937_64& rng) {
    const std::int64_t arrivals = std::max<std::int64_t>(m.n / 25, 2);
    const std::int64_t departures = std::max<std::int64_t>(m.n / 70, 1);
    switch (mix) {
      case vertex_mix::arrival_with_labels:
        return slotted_detail::make_batch<vertex_t, edge_t, weight_t>(m, rng, arrivals, 0, 0, 0,
                                                                      true, 2);
      case vertex_mix::departure:
        return slotted_detail::make_batch<vertex_t, edge_t, weight_t>(m, rng, 0, 0, departures, 0,
                                                                      false, 0);
      case vertex_mix::mixed:
        return slotted_detail::make_batch<vertex_t, edge_t, weight_t>(m, rng, arrivals, 0,
                                                                      departures, 0, true, 1);
      case vertex_mix::insert_and_delete_same_batch:
        return slotted_detail::make_batch<vertex_t, edge_t, weight_t>(
            m, rng, arrivals, std::max<std::int64_t>(arrivals / 4, 1), departures / 2, 0, true, 1);
      case vertex_mix::edges_between_present: {
        const std::int64_t pending = slotted_detail::pending_edges(m);
        return slotted_detail::make_batch<vertex_t, edge_t, weight_t>(
            m, rng, 0, 0, 0, std::max<std::int64_t>((pending + 1) / 2, 1), false, 0);
      }
    }
    return {};
  }

  /// The model needs nothing from the commit (random_batch() applied the whole batch).
  static void observe(model_type& /*m*/, const generated_type& /*gen*/,
                      const summary_type& /*summary*/) {}

  /// The batch of a generated batch.
  static const batch_type& batch_of(const generated_type& gen) {
    return gen.batch;
  }

  /// No vertex batch is inverted (single-use ids; C5 does not apply).
  static bool invertible(const generated_type& /*gen*/) {
    return false;
  }

  /// Not used (invertible() is false).
  static generated_type inverse(model_type& /*m*/, const generated_type& /*gen*/,
                                const summary_type& /*summary*/) {
    return {};
  }

  /// The deterministic counters of the batch summary (the vertex counts included).
  static std::vector<std::int64_t> summary_counters(const summary_type& b) {
    return {b.inserted_edges,     b.dropped_insertions, b.inserted_vertices, b.deleted_vertices,
            b.num_vertices_after, b.ignored_deletions,  b.updated_edges,     b.deleted_edges};
  }

  /// What an empty batch must leave at zero.
  static std::int64_t changed(const summary_type& b) {
    return b.inserted_edges + b.inserted_vertices + b.deleted_vertices + b.deleted_edges;
  }

  /// The graph has the model's capacity, active vertices and live entries.
  static ::testing::AssertionResult matches(const container_type& g, const model_type& m) {
    if (static_cast<std::int64_t>(g.num_vertices()) != m.n) {
      return ::testing::AssertionFailure()
             << "the graph has a capacity of " << static_cast<std::int64_t>(g.num_vertices())
             << " vertices, the model " << m.n;
    }
    if (static_cast<std::int64_t>(g.num_active_vertices()) != m.active_count) {
      return ::testing::AssertionFailure()
             << "the graph has " << static_cast<std::int64_t>(g.num_active_vertices())
             << " active vertices, the model " << m.active_count;
    }
    if (static_cast<std::int64_t>(g.num_edges()) != 2 * m.live_edges) {
      return ::testing::AssertionFailure()
             << "the graph has " << static_cast<std::int64_t>(g.num_edges())
             << " live entries, the model " << 2 * m.live_edges;
    }
    return ::testing::AssertionSuccess();
  }

  /// C10: the one preset with both algorithms' requirements.
  template <typename traits_t, typename partner_t>
  static std::vector<kit_preset<graph_properties>> composition_presets() {
    graph_properties p = slotted_properties();
    traits_t::require(p);
    partner_t::require(p);
    return {{"slotted, net_effect()", p}};
  }

  /// C10: the model of the composition (the small shape).
  template <typename traits_t, typename partner_t>
  static model_type composition_model(const graph_properties& /*props*/, std::mt19937_64& rng) {
    return random_model<traits_t>(size_class::small, rng);
  }

  /// C10's mix of each of its three batches.
  static vertex_mix composition_mix(int step) {
    return step == 2   ? vertex_mix::insert_and_delete_same_batch
           : step == 1 ? vertex_mix::mixed
                       : vertex_mix::arrival_with_labels;
  }

  /// C11's batch (applied by a separate g.apply()).
  static vertex_mix stale_mix() {
    return vertex_mix::mixed;
  }

  /// C11: a batch valid for the graph after the separate g.apply() (vertex ids are single-use, so
  /// the batch applied cannot be given again; ADR 0076): a departure from a copy of the model.
  static batch_type stale_probe(const model_type& m, std::mt19937_64& rng) {
    model_type copy = m;
    return random_batch(copy, vertex_mix::departure, rng).batch;
  }

  /// C12's mix of each of its four batches.
  static vertex_mix stream_mix(int step) {
    return step == 3 ? vertex_mix::edges_between_present : vertex_mix::mixed;
  }
};

}  // namespace dyng::conformance
