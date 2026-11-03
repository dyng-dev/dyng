// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/dynLP.cu (runDynLP: the steps of a batch),
// src/sparsifySubgraph.cu (markDeletionsKernel, buildSubgraphKernel, markInsertedKernel),
// src/connectedComponents.cu (the component numbering) and src/labelInitialization.cu
// (computeSupernodeWeightsKernel, initializeLabelsKernel)
/**
 * @file label_propagation.cpp
 * @brief label_propagation: argument validation, the result, compute() and update() through the
 *        framework's enactors, the hooks of the problem (problem.hpp), the stage trace (trace.hpp)
 *        and the explicit instantiations (ADRs 0071-0073).
 */
#include "algorithms/label_propagation/fused.hpp"
#include "algorithms/label_propagation/problem.hpp"
#include "algorithms/label_propagation/trace.hpp"
#include "core/budget_counters.hpp"
#include "core/resources_access.hpp"
#include "core/staging.hpp"
#include "framework/composition.hpp"
#include "framework/context.hpp"
#include "framework/enactor.hpp"
#include "framework/entry_check.hpp"
#include "framework/views.hpp"
#include "graph/graph_impl.hpp"
#include "util/allocation.hpp"

#include <dyng/config.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/label_propagation.hpp>
#include <dyng/update.hpp>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dyng::detail {

namespace {

/// The fixed-point scale of the supernode weights (labelInitialization.cu: WEIGHT_SCALE).
constexpr double weight_scale = 4294967296.0;  // 2^32

/// __double2ull_rn(x) for 0 <= x < 2^64: round to nearest, ties to even.
std::uint64_t round_to_u64(double x) {
  const int mode = std::fegetround();
  if (mode != FE_TONEAREST) {
    std::fesetround(FE_TONEAREST);
  }
  const double r = std::nearbyint(x);
  if (mode != FE_TONEAREST) {
    std::fesetround(mode);
  }
  return static_cast<std::uint64_t>(r);
}

/// Whether the backend `b` is one of label_propagation's (its manifest: sequential and cuda).
bool implemented(backend b) {
  return b == backend::sequential || b == backend::cuda;
}

/// Throws not_supported_error unless the backend of `res` is label_propagation's and built (the
/// backend sentence of ADR 0070: the message names the backends of the manifest and the fix).
void expect_backend(const resources& res, const char* function) {
  const backend b = res.get_backend();
  if (implemented(b) && backend_available(b)) {
    return;
  }
  throw not_supported_error(concat_message(
      "dyng: ", function, ": the ", to_string(b),
      " backend is not available for label_propagation (its backends: sequential and cuda; there "
      "is no OpenMP backend in 0.4); use dyng::resources::sequential() (Python: "
      "dyng.Resources(\"sequential\")) or dyng::resources::cuda()"));
}

/// Throws invalid_argument_error unless the result lives where `res` runs (its labels on the host
/// for the host backends, on the device of `res` for cuda).
template <typename vertex_t>
void expect_result_placement(const label_propagation_state<vertex_t>& st, const resources& res,
                             const char* function) {
  const bool on_cuda = res.get_backend() == backend::cuda;
  if (on_cuda && st.cuda == nullptr) {
    throw invalid_argument_error(concat_message(
        "dyng: ", function,
        ": the result belongs to the host backends but the resources are cuda; move it with "
        "result.clone(res)"));
  }
  if (!on_cuda && st.cuda != nullptr) {
    throw invalid_argument_error(
        concat_message("dyng: ", function, ": the result is resident on CUDA device ",
                       st.cuda->device, " but the resources are ", to_string(res.get_backend()),
                       "; move it with result.clone(res)"));
  }
  if (on_cuda && st.cuda->device != res.device()) {
    throw invalid_argument_error(concat_message(
        "dyng: ", function, ": the result is resident on CUDA device ", st.cuda->device,
        " but the resources use device ", res.device(), "; move it with result.clone(res)"));
  }
}

/// Throws not_supported_error for options::cuda_engine = engine::operators on CUDA (the operators
/// engine of label_propagation arrives in M10b).
void expect_cuda_engine(const label_propagation::options& opt, const resources& res,
                        const char* function) {
  if (res.get_backend() == backend::cuda && opt.cuda_engine == engine::operators) {
    throw not_supported_error(concat_message(
        "dyng: ", function,
        ": label_propagation has no operators engine on CUDA in this release (it arrives in M10b); "
        "use engine::fused or engine::automatic"));
  }
}

/// The propagation queues' capacities from the row capacities (createPropagationWorkspace).
template <typename edge_t>
std::array<int, lp_cuda::num_degree_bins> bin_capacities(const std::vector<edge_t>& row_ptr) {
  std::array<int, lp_cuda::num_degree_bins> c{};
  const std::size_t n = row_ptr.empty() ? 0 : row_ptr.size() - 1;
  c[lp_cuda::bin_small] = static_cast<int>(n);
  for (std::size_t v = 0; v < n; ++v) {
    const int top = lp_cuda::degree_bin_of(static_cast<std::int64_t>(row_ptr[v + 1] - row_ptr[v]));
    c[lp_cuda::bin_medium] += top >= lp_cuda::bin_medium ? 1 : 0;
    c[lp_cuda::bin_large] += top >= lp_cuda::bin_large ? 1 : 0;
  }
  return c;
}

/// Throws stale_result_error for a poisoned result.
template <typename vertex_t>
void expect_not_poisoned(const label_propagation_state<vertex_t>& st, const char* function) {
  if (st.poisoned) {
    throw stale_result_error(concat_message(
        "dyng: ", function, ": a failed update left this result unusable; recompute it"));
  }
}

/// The state of a result, or invalid_argument_error for a moved-from one.
template <typename state_t>
state_t& checked(state_t* st, const char* function) {
  DYNG_EXPECTS(st != nullptr, function, ": the result was moved from");
  expect_not_poisoned(*st, function);
  return *st;
}

/// The options' ranges.
void expect_options(const label_propagation::options& opt, const char* function) {
  DYNG_EXPECTS(opt.tolerance > 0.0F, function, ": options.tolerance must be > 0, got ",
               opt.tolerance);
  DYNG_EXPECTS(opt.max_iterations > 0, function, ": options.max_iterations must be > 0, got ",
               opt.max_iterations);
  DYNG_EXPECTS(!opt.similarity_threshold || !std::isnan(*opt.similarity_threshold), function,
               ": options.similarity_threshold must not be NaN (std::nullopt: the graph's average "
               "edge weight)");
}

/// The graph requirements (ADR 0071): undirected (float weights are the type).
template <typename vertex_t, typename edge_t>
void expect_graph(const graph<vertex_t, edge_t, float>& g, const char* function) {
  DYNG_EXPECTS(!g.properties().directed, function,
               ": label_propagation needs an undirected graph (graph_properties::directed = "
               "false): it averages over the neighbours of both directions");
  const row_layout layout = g.properties().layout;
  if (layout != row_layout::compact && layout != row_layout::slotted) {
    throw not_supported_error(concat_message(
        "dyng: ", function, ": row_layout::", to_string(layout), " is not supported yet"));
  }
}

/// The original's status codes of the current graph and seeds (by name, ADR 0069).
template <typename edge_t>
std::vector<std::int32_t> original_statuses(const lp_rows<edge_t>& rows,
                                            const std::vector<std::int8_t>& seeds) {
  std::vector<std::int32_t> out(static_cast<std::size_t>(rows.n));
  for (std::int64_t v = 0; v < rows.n; ++v) {
    const vertex_status st = rows.status != nullptr ? rows.status[v] : vertex_status::active;
    const auto vv = static_cast<std::size_t>(v);
    switch (st) {
      case vertex_status::active:
        out[vv] = seeds[vv] >= 0 ? lp_trace::original_ground_truth : lp_trace::original_active;
        break;
      case vertex_status::deleted:
        out[vv] = lp_trace::original_deleted;
        break;
      case vertex_status::inactive:
        out[vv] = lp_trace::original_inactive;
        break;
    }
  }
  return out;
}

}  // namespace

template <typename vertex_t>
label_propagation_state<vertex_t>& label_propagation_access::state(
    label_propagation::result<vertex_t>& r) {
  DYNG_EXPECTS(r.impl_ != nullptr, "label_propagation: the result was moved from");
  return *r.impl_;
}

template <typename vertex_t>
const label_propagation_state<vertex_t>& label_propagation_access::state(
    const label_propagation::result<vertex_t>& r) {
  DYNG_EXPECTS(r.impl_ != nullptr, "label_propagation: the result was moved from");
  return *r.impl_;
}

template <typename vertex_t>
label_propagation::result<vertex_t> label_propagation_access::make(
    std::unique_ptr<label_propagation_state<vertex_t>> st) {
  return label_propagation::result<vertex_t>(std::move(st));
}

void label_propagation_workspace::reserve(std::size_t n, std::size_t vertices, std::size_t edges) {
  const auto grow = [](auto& v, std::size_t size) {
    if (v.capacity() < size) {
      note_reservation();  // invariant I9: a reserving run (core/budget_counters.hpp)
      v.reserve(size + size / 2);
    }
  };
  // The per-vertex arrays (exactly n; the flags stay clear between updates).
  if (affected.size() != n) {
    if (affected.capacity() < n || inserted.capacity() < n) {
      note_reservation();
    }
    affected.assign(n, 0);
    inserted.assign(n, 0);
  }
  grow(parent, n);
  parent.resize(std::max(parent.size(), n));
  grow(component, n);
  component.resize(std::max(component.size(), n));
  grow(before, n);
  grow(frontier, n);
  grow(next, n);
  grow(halves, n);
  grow(side, n);
  // The per-batch arrays.
  grow(w0, vertices);
  grow(w1, vertices);
  grow(sub_edges, 2 * edges);
}

std::size_t label_propagation_workspace::bytes() const noexcept {
  return affected.capacity() + inserted.capacity() +
         (parent.capacity() + component.capacity() + frontier.capacity() + next.capacity() +
          halves.capacity() + sub_edges.capacity()) *
             sizeof(std::int32_t) +
         (before.capacity() + side.capacity() + irlp_next.capacity()) * sizeof(float) +
         (w0.capacity() + w1.capacity()) * sizeof(std::uint64_t) +
         snapshot.row_ptr.capacity() * sizeof(std::int64_t) +
         snapshot.col_ind.capacity() * sizeof(std::int32_t) +
         snapshot.weights.capacity() * sizeof(float) +
         snapshot_count.capacity() * sizeof(std::int64_t);
}

// ------------------------------------------------------------------------------------------------
// The problem
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
label_propagation_workspace& label_propagation_problem<vertex_t, edge_t, weight_t>::workspace(
    framework::context& ctx, const container_type& g, std::size_t vertices, std::size_t edges) {
  if (!ws_) {
    ws_.emplace(ctx.lease_workspace<label_propagation_workspace>());
  }
  label_propagation_workspace& ws = ws_->get();
  ws.reserve(static_cast<std::size_t>(g.num_vertices()),
             std::max(vertices, state_->reserved_vertices),
             std::max(edges, state_->reserved_edges));
  return ws;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void label_propagation_problem<vertex_t, edge_t, weight_t>::check_entry(
    const resources& res, const container_type& graph) {
  expect_backend(res, "label_propagation::update");
  graph_access::expect_placement_any_layout(res, graph, "label_propagation::update");
  expect_not_poisoned(*state_, "label_propagation::update");
  expect_graph(graph, "label_propagation::update");
  if (graph.properties().layout != row_layout::slotted) {
    throw not_supported_error(
        "dyng: label_propagation::update: the graph needs row_layout::slotted (vertex batches "
        "under batch_semantics::net_effect(); build it with graph::with_capacity()); compute() "
        "also reads compact graphs");
  }
  framework::expect_current_result("label_propagation::update", state_->version,
                                   state_->graph_state, graph);
  expect_options(state_->opt, "label_propagation::update");
  expect_result_placement(*state_, res, "label_propagation::update");
  expect_cuda_engine(state_->opt, res, "label_propagation::update");
}

template <typename vertex_t, typename edge_t, typename weight_t>
void label_propagation_problem<vertex_t, edge_t, weight_t>::begin_update(framework::context& ctx,
                                                                         old_graph g,
                                                                         const requested& batch) {
  const container_type& graph = g.get();
  check_entry(ctx.res(), graph);  // again: run_update() makes it before Step 0 for slotted rows
  const std::size_t labels =
      state_->cuda != nullptr ? static_cast<std::size_t>(state_->cuda->n) : state_->labels.size();
  DYNG_EXPECTS(labels == static_cast<std::size_t>(graph.num_vertices()),
               "label_propagation::update: the result has ", labels, " labels but the graph has ",
               graph.num_vertices(), " vertices");
  DYNG_EXPECTS(batch.normalized != nullptr && batch.normalized->vertex_step,
               "label_propagation::update: the framework's Step 0 of the vertex batch is missing");
  batch_ = &batch.edges;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void label_propagation_problem<vertex_t, edge_t, weight_t>::before_apply(
    framework::context& ctx, old_graph g, const requested& batch, frontier_type& /*frontier*/) {
  const container_type& graph = g.get();
  const normalized_batch<vertex_t>& step0 = *batch.normalized;
  if (ctx.chosen_engine() == engine::fused) {
    // The fused CUDA engine (ADR 0075): the affected flags cleared and launchMarkDeletions on the
    // device rows of G_t (Step 0 uploaded the batch).
    if (!cuda_ws_) {
      cuda_ws_.emplace(ctx.lease_workspace<lp_cuda::workspace>());
    }
    lp_cuda::before_apply(ctx.res(), graph, *state_->cuda, cuda_ws_->get(),
                          static_cast<std::int64_t>(state_->reserved_vertices),
                          static_cast<std::int64_t>(state_->reserved_edges),
                          state_->opt.update_schedule == label_propagation::schedule::damped);
    return;
  }
  label_propagation_workspace& ws =
      workspace(ctx, graph, step0.inserted_vertices.size(), batch.edges.num_insertions());
  std::fill(ws.affected.begin(), ws.affected.end(), std::uint8_t{0});
  // markDeletionsKernel: the neighbours of Del ∩ V_t on the rows of G_t.
  const lp_rows<edge_t> rows = host_rows(graph);
  for (const vertex_t u : step0.deleted_vertices) {
    for (auto e = static_cast<std::size_t>(rows.row_ptr[u]);
         e < static_cast<std::size_t>(rows.end(u)); ++e) {
      const std::int32_t v = rows.col_ind[e];
      if (v >= 0) {
        ws.affected[static_cast<std::size_t>(v)] = 1;
      }
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void label_propagation_problem<vertex_t, edge_t, weight_t>::resume(framework::context& ctx,
                                                                   new_graph g,
                                                                   const applied& applied) {
  applied_ = &applied;
  if (ctx.chosen_engine() == engine::fused) {
    return;  // the fused engine's workspace was leased and sized in before_apply
  }
  label_propagation_workspace& ws = workspace(ctx, g.get(), applied.delta.inserted_vertices.size(),
                                              applied.delta.insert_src.size());
  ws.before.assign(state_->labels.begin(), state_->labels.end());
}

template <typename vertex_t, typename edge_t, typename weight_t>
void label_propagation_problem<vertex_t, edge_t, weight_t>::identify_affected(
    framework::context& ctx, new_graph g, const applied& applied, frontier_type& /*frontier*/) {
  const container_type& graph = g.get();
  label_propagation_workspace& ws = ws_->get();
  const apply_delta<vertex_t>& delta = applied.delta;
  const lp_rows<edge_t> rows = host_rows(graph);
  // The batch's seeds join the result (setGroundTruthKernel; kept when the vertex leaves).
  for (std::size_t i = 0; i < delta.inserted_vertices.size(); ++i) {
    if (delta.inserted_labels[i] >= 0) {
      state_->seeds[static_cast<std::size_t>(delta.inserted_vertices[i])] =
          delta.inserted_labels[i];
    }
  }
  // V_aff: the inserted vertices (markInsertedKernel) and both endpoints of every appended edge
  // (buildSubgraphKernel's marks; also in a batch without insertions, ADR 0072).
  for (const vertex_t v : delta.inserted_vertices) {
    ws.affected[static_cast<std::size_t>(v)] = 1;
    ws.inserted[static_cast<std::size_t>(v)] = 1;
  }
  for (std::size_t i = 0; i < delta.insert_src.size(); ++i) {
#if defined(DYNG_MUTATION_LP_D_B_AFFECTED_TOO_SMALL)
    // MUTATION (D-b, AD9): the endpoints of a new edge that were present before are not marked.
    for (const vertex_t x : {delta.insert_src[i], delta.insert_dst[i]}) {
      if (ws.inserted[static_cast<std::size_t>(x)] != 0) {
        ws.affected[static_cast<std::size_t>(x)] = 1;
      }
    }
#else
    ws.affected[static_cast<std::size_t>(delta.insert_src[i])] = 1;
    ws.affected[static_cast<std::size_t>(delta.insert_dst[i])] = 1;
#endif
  }
  tau_ = state_->opt.similarity_threshold ? *state_->opt.similarity_threshold
                                          : average_live_weight(rows);
  num_components_ = 0;
  sparsified_ = 0;
  lp_trace::writer* trace = lp_trace::writer::get();
  const auto batch_no = static_cast<std::int32_t>(state_->updates);
  if (trace != nullptr) {
    const std::vector<std::int32_t> status = original_statuses(rows, state_->seeds);
    std::vector<std::int32_t> gt(state_->seeds.begin(), state_->seeds.end());
    lp_trace::record r = trace->begin("apply_vertex_changes", batch_no);
    trace->add_i32(r, "num_deleted", static_cast<std::int32_t>(batch_->delete_vertices.size()));
    trace->add_i32(r, "num_inserted", static_cast<std::int32_t>(delta.inserted_vertices.size()));
    trace->add_array(r, "inserted", lp_trace::i32, delta.inserted_vertices.data(),
                     delta.inserted_vertices.size());
    trace->add_array(r, "status", lp_trace::i32, status.data(), status.size());
    trace->add_array(r, "gt_class", lp_trace::i32, gt.data(), gt.size());
    trace->end_record(r);
    const auto n = static_cast<std::size_t>(rows.n);
    const auto slots = static_cast<std::size_t>(rows.row_ptr[n]);
    std::vector<std::int32_t> row_end(n);
    for (std::size_t v = 0; v < n; ++v) {
      row_end[v] = static_cast<std::int32_t>(rows.end(static_cast<std::int64_t>(v)));
    }
    lp_trace::record a = trace->begin("append_edges", batch_no);
    trace->add_i32(a, "num_new_edges", static_cast<std::int32_t>(batch_->num_insertions()));
    trace->add_i64(a, "capacity", static_cast<std::int64_t>(slots));
    if (batch_no == 0) {
      std::vector<std::int32_t> row_ptr(rows.row_ptr, rows.row_ptr + n + 1);
      trace->add_array(a, "row_ptr", lp_trace::i32, row_ptr.data(), row_ptr.size());
    }
    trace->add_array(a, "row_end", lp_trace::i32, row_end.data(), row_end.size());
    trace->add_array(a, "slot_col_ind", lp_trace::i32, rows.col_ind, slots);
    trace->add_array(a, "slot_values", lp_trace::f32, rows.weights, slots);
    const std::int32_t flags[2] = {0, 0};  // the host path refuses an overflow before the batch
    trace->add(a, "error_flags", lp_trace::i32, flags, 2);
    trace->end_record(a);
  }
  if (delta.inserted_vertices.empty()) {
    return;  // the original runs G', the components and the initialization only then
  }
  {
    // G': the appended edges of weight > tau between two inserted vertices (buildSubgraphKernel).
    scoped_stage stage(ctx.res(), "label_propagation.identify_affected.sparsify");
    std::vector<std::int32_t>& sub = ws.sub_edges;
    sub.clear();
    const std::size_t k = static_cast<std::size_t>(batch_->num_weights);
    for (std::size_t i = 0; i < delta.insert_src.size(); ++i) {
      const vertex_t u = delta.insert_src[i];
      const vertex_t v = delta.insert_dst[i];
      const float w = batch_->insert_weights[delta.insert_index[i] * k];
#if defined(DYNG_MUTATION_LP_D_A_UNFILTERED_COMPONENTS)
      // MUTATION (D-a, AD9): G' and its components over the unfiltered new edges
      if ((w > tau_ || true) && ws.inserted[static_cast<std::size_t>(u)] != 0 &&
          ws.inserted[static_cast<std::size_t>(v)] != 0) {
#else
      if (w > tau_ && ws.inserted[static_cast<std::size_t>(u)] != 0 &&
          ws.inserted[static_cast<std::size_t>(v)] != 0) {
#endif
        sub.push_back(std::min(u, v));
        sub.push_back(std::max(u, v));
      }
    }
    sparsified_ = static_cast<std::int64_t>(sub.size() / 2);
    if (trace != nullptr) {
      std::vector<std::pair<std::int32_t, std::int32_t>> pairs;
      for (std::size_t i = 0; i < sub.size(); i += 2) {
        pairs.emplace_back(sub[i], sub[i + 1]);
      }
      std::sort(pairs.begin(), pairs.end());
      std::vector<std::int32_t> flat;
      for (const auto& [a, b] : pairs) {
        flat.push_back(a);
        flat.push_back(b);
      }
      lp_trace::record r = trace->begin("sparsify", batch_no);
      trace->add_f32(r, "tau", tau_);
      trace->add_i32(r, "batch_stamp", batch_no + 1);
      trace->add_i32(r, "num_new_edges", static_cast<std::int32_t>(batch_->num_insertions()));
      trace->add_i32(r, "num_sub_edges", static_cast<std::int32_t>(sparsified_));
      trace->add_array(r, "sub_edges_sorted", lp_trace::i32, flat.data(), flat.size());
      trace->end_record(r);
    }
  }
  {
    // The components of G' over Ins_eff (a root is its smallest vertex; its id the rank of the
    // root in insertion order, the original's numbering).
    scoped_stage stage(ctx.res(), "label_propagation.identify_affected.components");
    std::vector<std::int32_t>& parent = ws.parent;
    for (const vertex_t v : delta.inserted_vertices) {
      parent[static_cast<std::size_t>(v)] = v;
    }
    const auto find = [&](std::int32_t x) {
      while (parent[static_cast<std::size_t>(x)] != x) {
        const std::int32_t p = parent[static_cast<std::size_t>(x)];
        parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(p)];
        x = p;
      }
      return x;
    };
    const std::vector<std::int32_t>& sub = ws.sub_edges;
    for (std::size_t i = 0; i < sub.size(); i += 2) {
      const std::int32_t a = find(sub[i]);
      const std::int32_t b = find(sub[i + 1]);
      if (a < b) {
        parent[static_cast<std::size_t>(b)] = a;
      } else if (b < a) {
        parent[static_cast<std::size_t>(a)] = b;
      }
    }
    std::vector<std::int32_t>& component = ws.component;
    std::int32_t count = 0;
    for (const vertex_t v : delta.inserted_vertices) {
      if (find(v) == v) {
        component[static_cast<std::size_t>(v)] = count++;
      }
    }
    for (const vertex_t v : delta.inserted_vertices) {
      component[static_cast<std::size_t>(v)] = component[static_cast<std::size_t>(find(v))];
    }
    num_components_ = count;
    if (trace != nullptr) {
      std::vector<std::int32_t> ids;
      ids.reserve(delta.inserted_vertices.size());
      for (const vertex_t v : delta.inserted_vertices) {
        ids.push_back(component[static_cast<std::size_t>(v)]);
      }
      lp_trace::record r = trace->begin("connected_components", batch_no);
      trace->add_i32(r, "num_components", count);
      trace->add_i32(r, "num_inserted", static_cast<std::int32_t>(ids.size()));
      trace->add_array(r, "component_of_inserted", lp_trace::i32, ids.data(), ids.size());
      trace->end_record(r);
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void label_propagation_problem<vertex_t, edge_t, weight_t>::seed(framework::context& /*ctx*/,
                                                                 new_graph g,
                                                                 frontier_type& /*frontier*/) {
  if (computing_) {
    return;
  }
  const apply_delta<vertex_t>& delta = applied_->delta;
  if (delta.inserted_vertices.empty()) {
    return;
  }
  label_propagation_workspace& ws = ws_->get();
  const lp_rows<edge_t> rows = host_rows(g.get());
  const auto components = static_cast<std::size_t>(num_components_);
  ws.w0.assign(components, 0);
  ws.w1.assign(components, 0);
  // computeSupernodeWeightsKernel: per inserted vertex, its entries to seeds (of any state).
  for (const vertex_t u : delta.inserted_vertices) {
    const auto c = static_cast<std::size_t>(ws.component[static_cast<std::size_t>(u)]);
    for (auto e = static_cast<std::size_t>(rows.row_ptr[u]);
         e < static_cast<std::size_t>(rows.end(u)); ++e) {
      const std::int32_t v = rows.col_ind[e];
      if (v < 0) {
        continue;
      }
      const std::int8_t s = state_->seeds[static_cast<std::size_t>(v)];
      if (s < 0) {
        continue;
      }
      const std::uint64_t w = round_to_u64(static_cast<double>(rows.weights[e]) * weight_scale);
      (s == 0 ? ws.w0[c] : ws.w1[c]) += w;
    }
  }
  // initializeLabelsKernel.
  std::vector<float>& labels = state_->labels;
  for (const vertex_t u : delta.inserted_vertices) {
    const auto uu = static_cast<std::size_t>(u);
    const std::int8_t s = state_->seeds[uu];
    if (s == 0 || s == 1) {
      labels[uu] = s == 0 ? 0.0F : 1.0F;
      continue;
    }
    const auto c = static_cast<std::size_t>(ws.component[uu]);
    const double l0 = static_cast<double>(ws.w0[c]) / weight_scale;
    const double l1 = static_cast<double>(ws.w1[c]) / weight_scale;
    const double denominator = l0 + l1;
    labels[uu] = denominator > 1e-10 ? static_cast<float>(l1 / denominator) : 0.5F;
  }
  if (lp_trace::writer* trace = lp_trace::writer::get()) {
    std::vector<float> init;
    init.reserve(delta.inserted_vertices.size());
    for (const vertex_t u : delta.inserted_vertices) {
      init.push_back(labels[static_cast<std::size_t>(u)]);
    }
    lp_trace::record r = trace->begin("label_init", static_cast<std::int32_t>(state_->updates));
    trace->add_i32(r, "num_components", static_cast<std::int32_t>(num_components_));
    trace->add_array(r, "w0", lp_trace::u64, ws.w0.data(), ws.w0.size());
    trace->add_array(r, "w1", lp_trace::u64, ws.w1.data(), ws.w1.size());
    trace->add_array(r, "initial_labels_of_inserted", lp_trace::f32, init.data(), init.size());
    trace->end_record(r);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void label_propagation_problem<vertex_t, edge_t, weight_t>::loop(framework::context& ctx,
                                                                 new_graph g, frontier_type& /*in*/,
                                                                 frontier_type& /*out*/) {
  const container_type& graph = g.get();
  const lp_rows<edge_t> rows = host_rows(graph);
  const label_propagation::options& opt = state_->opt;
  if (computing_) {
    label_propagation_workspace& ws = workspace(ctx, graph, 0, 0);
    irlp_ =
        irlp_sequential(rows, state_->seeds, state_->labels, opt.tolerance, opt.max_iterations, ws);
    if (profiler* p = ctx.res().get_profiler()) {
      p->add_counter("label_propagation.irlp_iterations", irlp_.iterations);  // results.csv
    }
    if (lp_trace::writer* trace = lp_trace::writer::get()) {
      lp_trace::record r = trace->begin("irlp_snapshot", -1);
      trace->add_i32(r, "returned", 1);
      trace->add_i32(r, "iterations", static_cast<std::int32_t>(irlp_.iterations));
      trace->add_f32(r, "delta", opt.tolerance);
      trace->add_array(r, "labels", lp_trace::f32, state_->labels.data(), state_->labels.size());
      trace->end_record(r);
    }
    if (irlp_.capped && opt.on_iteration_limit != label_propagation::on_limit::report) {
      throw convergence_error(concat_message(
          "dyng: label_propagation::compute: IrLP did not converge in ", opt.max_iterations,
          " iteration(s) (options.max_iterations); raise the cap, or choose on_limit::report"));
    }
    return;
  }
  label_propagation_workspace& ws = ws_->get();
  lp_trace::writer* trace = lp_trace::writer::get();
  const auto batch_no = static_cast<std::int32_t>(state_->updates);
  if (trace != nullptr) {
    std::vector<std::int32_t> frontier;
    for (std::int64_t v = 0; v < rows.n; ++v) {
      if (ws.affected[static_cast<std::size_t>(v)] != 0 && rows.active(v) &&
          state_->seeds[static_cast<std::size_t>(v)] < 0) {
        frontier.push_back(static_cast<std::int32_t>(v));
      }
    }
    lp_trace::record r = trace->begin("propagation_begin", batch_no);
    trace->add_i32(r, "num_nodes", static_cast<std::int32_t>(rows.n));
    trace->add_f32(r, "delta", opt.tolerance);
    trace->add_i32(r, "max_iterations", static_cast<std::int32_t>(opt.max_iterations));
    trace->add_i32(r, "update_rule",
                   opt.update_schedule == label_propagation::schedule::damped ? 1 : 0);
    trace->add_i32(r, "next_tag", static_cast<std::int32_t>(state_->next_tag));
    trace->add_i32(r, "frontier_size", static_cast<std::int32_t>(frontier.size()));
    trace->add_array(r, "initial_frontier", lp_trace::i32, frontier.data(), frontier.size());
    trace->add_array(r, "labels", lp_trace::f32, state_->labels.data(), state_->labels.size());
    trace->end_record(r);
  }
  propagation_ =
      propagate_sequential(rows, state_->seeds, state_->labels, state_->stamp, state_->next_tag,
                           ws.affected, opt.update_schedule == label_propagation::schedule::damped,
                           opt.tolerance, opt.max_iterations, ws);
  if (propagation_.capped) {
    switch (opt.on_iteration_limit) {
      case label_propagation::on_limit::report:
        break;
      case label_propagation::on_limit::error:
        throw convergence_error(concat_message(
            "dyng: label_propagation::update: the propagation did not converge in ",
            opt.max_iterations,
            " iteration(s) (options.max_iterations); raise the cap, or choose on_limit::report "
            "or on_limit::fallback_recompute"));
      case label_propagation::on_limit::fallback_recompute: {
        scoped_stage stage(ctx.res(), "label_propagation.recompute");
        irlp_ = irlp_sequential(rows, state_->seeds, state_->labels, opt.tolerance,
                                opt.max_iterations, ws);
        fallback_ = true;
        break;
      }
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void label_propagation_problem<vertex_t, edge_t, weight_t>::finalize(framework::context& /*ctx*/,
                                                                     stats_type& stats) {
  if (computing_) {
    return;
  }
  label_propagation_workspace& ws = ws_->get();
  std::vector<float>& labels = state_->labels;
  // enforceGroundTruthKernel: every seed's label is its class (deleted seeds included).
  for (std::size_t v = 0; v < labels.size(); ++v) {
    const std::int8_t s = state_->seeds[v];
    if (s == 0) {
      labels[v] = 0.0F;
    } else if (s == 1) {
      labels[v] = 1.0F;
    }
  }
  std::int64_t changed = 0;
  for (std::size_t v = 0; v < labels.size(); ++v) {
    changed += std::memcmp(&labels[v], &ws.before[v], sizeof(float)) != 0 ? 1 : 0;
  }
  for (const vertex_t v : applied_->delta.inserted_vertices) {
    ws.inserted[static_cast<std::size_t>(v)] = 0;  // clear for the next update
  }
  stats.affected = changed;
  stats.iterations = propagation_.iterations;
  stats.frontier_visits = propagation_.frontier_visits;
  // The fallback's recomputation converged only if IrLP did not reach the cap too (compute()
  // raises convergence_error there; the update keeps IrLP's labels and reports it).
  stats.converged = !propagation_.capped || (fallback_ && !irlp_.capped);
  stats.fallback_used = fallback_;
  stats.components = num_components_;
  stats.sparsified_edges = sparsified_;
  stats.initial_frontier = propagation_.initial_frontier;
  if (lp_trace::writer* trace = lp_trace::writer::get()) {
    lp_trace::record r =
        trace->begin("propagation_end", static_cast<std::int32_t>(state_->updates));
    trace->add_i32(r, "iterations", static_cast<std::int32_t>(propagation_.iterations));
    trace->add_i64(r, "frontier_visits", propagation_.frontier_visits);
    trace->add_i32(r, "next_tag", static_cast<std::int32_t>(state_->next_tag));
    trace->add_array(r, "labels", lp_trace::f32, labels.data(), labels.size());
    trace->end_record(r);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void label_propagation_problem<vertex_t, edge_t, weight_t>::end_update(
    framework::context& /*ctx*/, new_graph g, const stats_type& /*stats*/) {
  framework::stamp_result(state_->version, state_->graph_state, g.get());
  ++state_->updates;
  ws_.reset();
  cuda_ws_.reset();
}

template <typename vertex_t, typename edge_t, typename weight_t>
framework::budget label_propagation_problem<vertex_t, edge_t, weight_t>::algorithm_budget(
    framework::context& ctx) const noexcept {
  if (!ctx.on_cuda()) {
    return framework::budget::steady_state(0);
  }
  return cuda_syncs_ < 0 ? framework::budget::unchecked()
                         : framework::budget::steady_state(cuda_syncs_);
}

template <typename vertex_t, typename edge_t, typename weight_t>
engine label_propagation_problem<vertex_t, edge_t, weight_t>::select_engine(
    framework::context& ctx) const noexcept {
  return ctx.on_cuda() ? state_->opt.cuda_engine : engine::automatic;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void label_propagation_problem<vertex_t, edge_t, weight_t>::enact_fused(framework::context& ctx,
                                                                        new_graph g,
                                                                        const applied& applied,
                                                                        stats_type& stats) {
  const container_type& graph = g.get();
  const label_propagation::options& opt = state_->opt;
  const apply_delta<vertex_t>& delta = applied.delta;
  // The batch's seeds join the result (the host copy of the seed labels; the device classes are
  // set by the engine).
  for (std::size_t i = 0; i < delta.inserted_vertices.size(); ++i) {
    if (delta.inserted_labels[i] >= 0) {
      state_->seeds[static_cast<std::size_t>(delta.inserted_vertices[i])] =
          delta.inserted_labels[i];
    }
  }
  lp_cuda::batch_args a;
  // tau: the published one, or the average weight of the current graph's live entries (the
  // host sum of the sequential backend, over the downloaded rows: bitwise the same).
  a.tau =
      opt.similarity_threshold ? *opt.similarity_threshold : average_live_weight(host_rows(graph));
  a.delta = opt.tolerance;
  a.max_iterations = static_cast<int>(std::min<std::int64_t>(opt.max_iterations, 2147483647));
  a.damped = opt.update_schedule == label_propagation::schedule::damped;
  a.fallback = opt.on_iteration_limit == label_propagation::on_limit::fallback_recompute;
  a.batch_stamp = static_cast<int>(state_->updates + 1);
  a.batch_no = static_cast<int>(state_->updates);
  a.inserted = delta.inserted_vertices.data();
  a.num_inserted = delta.inserted_vertices.size();
  a.host_seeds = state_->seeds.data();
  a.num_deleted = static_cast<std::int32_t>(batch_ == nullptr ? 0 : batch_->delete_vertices.size());
  const lp_cuda::batch_result r =
      lp_cuda::after_commit(ctx.res(), graph, *state_->cuda, cuda_ws_->get(), a);
  cuda_syncs_ = lp_trace::writer::get() != nullptr ? -1 : r.host_syncs;
  stats.batch.inserted_edges = r.appended;
  stats.batch.dropped_insertions = r.dropped;
  if (r.graph_errors != 0) {
    ctx.raise_device_error(r.graph_errors, r.error_detail);  // thrown after this hook
  }
  stats.affected = r.affected;
  stats.iterations = r.propagation.iterations;
  stats.frontier_visits = r.propagation.frontier_visits;
  stats.components = r.components;
  stats.sparsified_edges = r.sparsified;
  stats.initial_frontier = r.propagation.initial_frontier;
  stats.converged = !r.capped || (r.recomputed && !r.irlp_capped);  // as on the host
  stats.fallback_used = r.recomputed;
  if (profiler* p = ctx.res().get_profiler()) {
    // The original's prof:: counters (cc.hookRounds, prop.persistentIterations).
    p->add_counter("label_propagation.hook_rounds", r.hook_rounds);
    p->add_counter("label_propagation.persistent_iterations", r.propagation.persistent_iterations);
    if (r.recomputed) {
      p->add_counter("label_propagation.irlp_iterations", r.irlp_iterations);
    }
  }
  if (r.capped && opt.on_iteration_limit == label_propagation::on_limit::error) {
    throw convergence_error(concat_message(
        "dyng: label_propagation::update: the propagation did not converge in ", opt.max_iterations,
        " iteration(s) (options.max_iterations); raise the cap, or choose on_limit::report or "
        "on_limit::fallback_recompute"));
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void label_propagation_problem<vertex_t, edge_t, weight_t>::compute_fused(framework::context& ctx,
                                                                          new_graph g,
                                                                          stats_type& /*stats*/) {
  const label_propagation::options& opt = state_->opt;
  auto cuda = std::make_shared<lp_cuda::state>();
  bool capped = false;
  const int iterations = lp_cuda::compute(
      ctx.res(), g.get(), state_->seeds.data(), opt.tolerance,
      static_cast<int>(std::min<std::int64_t>(opt.max_iterations, 2147483647)), *cuda, capped);
  if (profiler* p = ctx.res().get_profiler()) {
    p->add_counter("label_propagation.irlp_iterations", iterations);  // results.csv
  }
  if (capped && opt.on_iteration_limit != label_propagation::on_limit::report) {
    throw convergence_error(concat_message(
        "dyng: label_propagation::compute: IrLP did not converge in ", opt.max_iterations,
        " iteration(s) (options.max_iterations); raise the cap, or choose on_limit::report"));
  }
  state_->cuda = std::move(cuda);
  state_->labels.clear();  // on the device (state::labels)
  state_->stamp.clear();
}

template <typename vertex_t, typename edge_t, typename weight_t>
void label_propagation_problem<vertex_t, edge_t, weight_t>::reset(framework::context& /*ctx*/) {
  computing_ = true;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void label_propagation_problem<vertex_t, edge_t, weight_t>::seed_static(framework::context& /*ctx*/,
                                                                        new_graph /*g*/,
                                                                        frontier_type& /*f*/) {}

}  // namespace dyng::detail

// ------------------------------------------------------------------------------------------------
// The public functions
// ------------------------------------------------------------------------------------------------

namespace dyng::label_propagation {

template <typename vertex_t>
result<vertex_t>::result(std::unique_ptr<detail::label_propagation_state<vertex_t>> state) noexcept
    : impl_(std::move(state)) {}
template <typename vertex_t>
result<vertex_t>::result(result&& other) noexcept = default;
template <typename vertex_t>
result<vertex_t>& result<vertex_t>::operator=(result&& other) noexcept = default;
template <typename vertex_t>
result<vertex_t>::~result() = default;

template <typename vertex_t>
array_view<const float> result<vertex_t>::labels() const {
  const auto& st = detail::checked(impl_.get(), "label_propagation::result::labels");
  if (st.cuda != nullptr) {
    return array_view<const float>(st.cuda->labels.data(), static_cast<std::size_t>(st.cuda->n),
                                   memory_space::device, st.cuda->device);
  }
  return host_view(st.labels);
}

template <typename vertex_t>
array_view<const std::int8_t> result<vertex_t>::seed_labels() const {
  const auto& st = detail::checked(impl_.get(), "label_propagation::result::seed_labels");
  return host_view(st.seeds);
}

template <typename vertex_t>
const options& result<vertex_t>::get_options() const {
  return detail::checked(impl_.get(), "label_propagation::result::get_options").opt;
}

template <typename vertex_t>
void result<vertex_t>::set_options(const options& opt) {
  auto& st = detail::checked(impl_.get(), "label_propagation::result::set_options");
  detail::expect_options(opt, "label_propagation::result::set_options");
  st.opt = opt;
}

template <typename vertex_t>
std::uint64_t result<vertex_t>::graph_version() const noexcept {
  return impl_ == nullptr ? 0 : impl_->version;
}

template <typename vertex_t>
memory_space result<vertex_t>::space() const noexcept {
  return impl_ != nullptr && impl_->cuda != nullptr ? memory_space::device : memory_space::host;
}

template <typename vertex_t>
result<vertex_t> result<vertex_t>::clone(const resources& res) const try {
  const auto& st = detail::checked(impl_.get(), "label_propagation::result::clone");
  detail::expect_backend(res, "label_propagation::result::clone");
  auto copy = std::make_unique<detail::label_propagation_state<vertex_t>>(st);
  const bool to_cuda = res.get_backend() == backend::cuda;
  if (to_cuda) {
    auto cuda = std::make_shared<detail::lp_cuda::state>();
    if (st.cuda != nullptr) {
      detail::lp_cuda::copy_state(res, *st.cuda, *cuda);
    } else {
      int capacity[detail::lp_cuda::num_degree_bins] = {st.bin_capacity[0], st.bin_capacity[1],
                                                        st.bin_capacity[2]};
      detail::lp_cuda::upload_state(res, static_cast<std::int64_t>(st.labels.size()), capacity,
                                    st.seeds.data(), st.labels.data(), st.stamp.data(), st.next_tag,
                                    *cuda);
    }
    copy->cuda = std::move(cuda);
    copy->labels.clear();
    copy->stamp.clear();
  } else if (st.cuda != nullptr) {
    detail::lp_cuda::download_state(*st.cuda, copy->labels, copy->stamp);
    copy->next_tag = st.cuda->next_tag;
    copy->cuda = nullptr;
  }
  return result(std::move(copy));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("label_propagation::result::clone (",
                                  impl_ != nullptr ? impl_->seeds.size() : 0, " vertices)")

template <typename vertex_t>
void result<vertex_t>::reserve(const resources& res, std::size_t max_batch_vertices,
                               std::size_t max_batch_edges) try {
  auto& st = detail::checked(impl_.get(), "label_propagation::result::reserve");
  detail::expect_backend(res, "label_propagation::result::reserve");
  st.reserved_vertices = std::max(st.reserved_vertices, max_batch_vertices);
  st.reserved_edges = std::max(st.reserved_edges, max_batch_edges);
  // The pooled workspace of `res`, sized now (the original's setup sizes its buffers once).
  if (res.get_backend() == backend::cuda) {
    detail::expect_result_placement(st, res, "label_propagation::result::reserve");
    auto lease =
        detail::resources_access::workspaces(res).template acquire<detail::lp_cuda::workspace>(res);
    lease->reserve(
        res, st.cuda->n, st.cuda->capacity, static_cast<std::int64_t>(st.reserved_vertices),
        static_cast<std::int64_t>(st.reserved_edges), st.opt.update_schedule == schedule::damped,
        /*exact=*/true);
    return;
  }
  auto lease = detail::resources_access::workspaces(res)
                   .template acquire<detail::label_propagation_workspace>(res);
  lease->reserve(st.labels.size(), st.reserved_vertices, st.reserved_edges);
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("label_propagation::result::reserve (", max_batch_vertices,
                                  " vertices, ", max_batch_edges, " edges per batch)")

}  // namespace dyng::label_propagation

namespace dyng::detail {

template <typename vertex_t, typename edge_t>
label_propagation::result<vertex_t> label_propagation_compute(
    const resources& res, const graph<vertex_t, edge_t, float>& g,
    array_view<const vertex_t> seed_vertices, array_view<const std::int8_t> seed_labels,
    const label_propagation::options& opt) try {
  using problem_t = label_propagation_problem<vertex_t, edge_t, float>;
  expect_backend(res, "label_propagation::compute");
  graph_access::expect_placement_any_layout(res, g, "label_propagation::compute");
  expect_graph(g, "label_propagation::compute");
  expect_options(opt, "label_propagation::compute");
  expect_cuda_engine(opt, res, "label_propagation::compute");
  const host_input<vertex_t> seeds_in(res, seed_vertices,
                                      "label_propagation::compute: seed_vertices");
  const host_input<std::int8_t> labels_in(res, seed_labels,
                                          "label_propagation::compute: seed_labels");
  const array_view<const vertex_t> sv = seeds_in.view();
  const array_view<const std::int8_t> sl = labels_in.view();
  DYNG_EXPECTS(sv.size() == sl.size(), "label_propagation::compute: ", sv.size(),
               " seed vertices but ", sl.size(), " seed labels");
  const auto n = static_cast<std::size_t>(g.num_vertices());
  // The states only (current on the host also when the rows are on the device: no download).
  const auto& gi = graph_access::impl(g);
  const auto active = [&gi](vertex_t v) {
    return !gi.is_slotted() ||
           gi.slotted.status[static_cast<std::size_t>(v)] == vertex_status::active;
  };
  auto st = std::make_unique<label_propagation_state<vertex_t>>();
  st->opt = opt;
  st->seeds.assign(n, -1);
  for (std::size_t i = 0; i < sv.size(); ++i) {
    const vertex_t v = sv[i];
    DYNG_EXPECTS(v >= 0 && static_cast<std::size_t>(v) < n, "label_propagation::compute: seed ", i,
                 " (vertex ", v, ") is out of range [0, ", n, ")");
    DYNG_EXPECTS(sl[i] == 0 || sl[i] == 1, "label_propagation::compute: seed ", i, " (vertex ", v,
                 ") has the label ", static_cast<int>(sl[i]), "; a seed's class is 0 or 1");
    DYNG_EXPECTS(active(v), "label_propagation::compute: seed ", i, " (vertex ", v,
                 ") is not an active vertex of the graph");
    DYNG_EXPECTS(st->seeds[static_cast<std::size_t>(v)] < 0, "label_propagation::compute: vertex ",
                 v, " is listed twice among the seeds");
    st->seeds[static_cast<std::size_t>(v)] = sl[i];
  }
  st->labels.assign(n, 0.5F);
  st->stamp.assign(n, 0);
  st->bin_capacity = bin_capacities(graph_access::impl(g).host_storage().row_ptr);
  const framework::hook_stages& names = framework::stages_of<problem_t>();
  scoped_stage stage(res, names.compute);
  problem_t p(*st);
  framework::context ctx(res, problem_t::name);
  framework::static_enactor<problem_t> enactor(p);
  (void)enactor.run(ctx, framework::new_view<graph<vertex_t, edge_t, float>>(g));
  framework::stamp_result(st->version, st->graph_state, g);
  return label_propagation_access::make(std::move(st));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("label_propagation::compute (", g.num_vertices(), " vertices, ",
                                  g.num_edges(), " edges)")

template <typename vertex_t, typename edge_t>
label_propagation::stats label_propagation_update(const resources& res,
                                                  graph<vertex_t, edge_t, float>& g,
                                                  const edge_batch_view<vertex_t, float>& batch,
                                                  label_propagation::result<vertex_t>& r) {
  using problem_t = label_propagation_problem<vertex_t, edge_t, float>;
  return framework::update_one_checked<problem_t>(res, g, batch, r);  // entry checks first
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<update_participant<vertex_t, edge_t, weight_t>> make_label_propagation_participant(
    label_propagation::result<vertex_t>& r, label_propagation::stats& out) {
  using problem_t = label_propagation_problem<vertex_t, edge_t, weight_t>;
  return framework::make_checked_participant<problem_t>(out, r);  // entry checks first
}

// Explicit instantiations (PLAN Section 4.4.3): float graphs with 32-bit ids.
template struct label_propagation_state<std::int32_t>;
template label_propagation_state<std::int32_t>& label_propagation_access::state<std::int32_t>(
    label_propagation::result<std::int32_t>&);
template const label_propagation_state<std::int32_t>& label_propagation_access::state<std::int32_t>(
    const label_propagation::result<std::int32_t>&);

#define DYNG_INSTANTIATE_LABEL_PROPAGATION(E)                                                  \
  template class label_propagation_problem<std::int32_t, E, float>;                            \
  template std::unique_ptr<update_participant<std::int32_t, E, float>>                         \
  make_label_propagation_participant<std::int32_t, E, float>(                                  \
      label_propagation::result<std::int32_t>&, label_propagation::stats&);                    \
  template label_propagation::result<std::int32_t> label_propagation_compute<std::int32_t, E>( \
      const resources&, const graph<std::int32_t, E, float>&, array_view<const std::int32_t>,  \
      array_view<const std::int8_t>, const label_propagation::options&);                       \
  template label_propagation::stats label_propagation_update<std::int32_t, E>(                 \
      const resources&, graph<std::int32_t, E, float>&,                                        \
      const edge_batch_view<std::int32_t, float>&, label_propagation::result<std::int32_t>&);
DYNG_INSTANTIATE_LABEL_PROPAGATION(std::int32_t)
DYNG_INSTANTIATE_LABEL_PROPAGATION(std::int64_t)
#undef DYNG_INSTANTIATE_LABEL_PROPAGATION

}  // namespace dyng::detail

namespace dyng::label_propagation {
template class result<std::int32_t>;
}  // namespace dyng::label_propagation

#if !DYNG_HAS_CUDA
// Without CUDA the fused engine does not exist: the backend check refuses cuda resources first
// (backend_available), so these are never reached.
namespace dyng::detail::lp_cuda {
namespace {
[[noreturn]] void no_cuda() {
  throw not_supported_error(
      "dyng: label_propagation: the cuda backend is not built (configure with "
      "DYNG_ENABLE_CUDA=ON)");
}
}  // namespace
void workspace::reserve(const resources& /*res*/, std::int64_t /*n*/,
                        const int (& /*capacity*/)[num_degree_bins], std::int64_t /*inserted*/,
                        std::int64_t /*edges*/, bool /*damped*/, bool /*exact*/) {
  no_cuda();
}
std::size_t workspace::bytes() const noexcept {
  return 0;
}
std::size_t irlp_workspace::bytes() const noexcept {
  return 0;
}
void copy_state(const resources& /*res*/, const state& /*in*/, state& /*out*/) {
  no_cuda();
}
void upload_state(const resources& /*res*/, std::int64_t /*n*/,
                  const int (& /*capacity*/)[num_degree_bins], const std::int8_t* /*seeds*/,
                  const float* /*labels*/, const std::int64_t* /*stamp*/, std::int64_t /*next_tag*/,
                  state& /*out*/) {
  no_cuda();
}
void download_state(const state& /*in*/, std::vector<float>& /*labels*/,
                    std::vector<std::int64_t>& /*stamp*/) {
  no_cuda();
}
template <typename edge_t>
void before_apply(const resources& /*res*/, const graph<std::int32_t, edge_t, float>& /*g*/,
                  const state& /*st*/, workspace& /*ws*/, std::int64_t /*inserted*/,
                  std::int64_t /*edges*/, bool /*damped*/) {
  no_cuda();
}
template <typename edge_t>
batch_result after_commit(const resources& /*res*/, const graph<std::int32_t, edge_t, float>& /*g*/,
                          state& /*st*/, workspace& /*ws*/, const batch_args& /*a*/) {
  no_cuda();
}
template <typename edge_t>
int compute(const resources& /*res*/, const graph<std::int32_t, edge_t, float>& /*g*/,
            const std::int8_t* /*seeds*/, float /*delta*/, int /*max_iterations*/, state& /*st*/,
            bool& /*capped*/) {
  no_cuda();
}
#define DYNG_INSTANTIATE_LP_CUDA_STUBS(E)                                                       \
  template void before_apply<E>(const resources&, const graph<std::int32_t, E, float>&,         \
                                const state&, workspace&, std::int64_t, std::int64_t, bool);    \
  template batch_result after_commit<E>(const resources&, const graph<std::int32_t, E, float>&, \
                                        state&, workspace&, const batch_args&);                 \
  template int compute<E>(const resources&, const graph<std::int32_t, E, float>&,               \
                          const std::int8_t*, float, int, state&, bool&);
DYNG_INSTANTIATE_LP_CUDA_STUBS(std::int32_t)
DYNG_INSTANTIATE_LP_CUDA_STUBS(std::int64_t)
#undef DYNG_INSTANTIATE_LP_CUDA_STUBS
}  // namespace dyng::detail::lp_cuda
#endif
