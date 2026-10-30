// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDevice.cu (hsospUpdate's and hsospRecompute's
// argument checks and fallback order; the host emulation's driver of
// hypergraph/src/HostHypergraph.cpp)
/**
 * @file hyper_sssp.cpp
 * @brief hyper_sssp: argument validation, the result, compute() and update() through the
 *        framework's enactors on the hypergraph, the hooks of the problem (problem.hpp), the
 *        internal virtual-endpoint mode (virtual_endpoints.hpp) and the explicit instantiations
 *        (PLAN Section 4.8, file 4; M11a step 4, ADRs 0090-0092).
 */
#include "algorithms/hyper_sssp/problem.hpp"
#include "algorithms/hyper_sssp/virtual_endpoints.hpp"
#include "core/budget_counters.hpp"
#include "framework/composition.hpp"
#include "framework/context.hpp"
#include "framework/enactor.hpp"
#include "framework/views.hpp"
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/instantiate.hpp"
#include "util/allocation.hpp"

#include <dyng/config.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/hyper_sssp.hpp>
#include <dyng/update.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dyng::hyper_sssp {

std::string_view to_string(budget_action a) noexcept {
  switch (a) {
    case budget_action::recompute:
      return "recompute";
    case budget_action::error:
      return "error";
  }
  return "unknown";
}

std::string_view to_string(fallback_reason r) noexcept {
  switch (r) {
    case fallback_reason::none:
      return "none";
    case fallback_reason::work_budget:
      return "work_budget";
    case fallback_reason::iterations:
      return "iterations";
    case fallback_reason::distance_width:
      return "distance_width";
    case fallback_reason::invalidation_depth:
      return "invalidation_depth";
  }
  return "unknown";
}

}  // namespace dyng::hyper_sssp

namespace dyng::detail {

namespace {

using impl_t = hypergraph_impl<std::int32_t, std::int32_t>;

/**
 * The backends of the manifest (sequential, cuda; HD14, joint with AD19): the rule "an algorithm
 * accepts the backends its manifest lists; any other raises not_supported_error naming them" is
 * M10a's guard-ADR sentence (ADR 0090 cites it).
 */
void expect_backend(const resources& res, const char* what) {
  const backend b = res.get_backend();
  if (b == backend::openmp) {
    throw not_supported_error(concat_message(
        "dyng: ", what,
        ": hyper_sssp runs on the sequential and cuda backends in 0.4, not on openmp; use "
        "dyng::resources::sequential() (Python: dyng.Resources(\"sequential\")) or "
        "dyng::resources::cuda()"));
  }
  if (!backend_available(b)) {
    throw not_supported_error(concat_message("dyng: ", what, ": the ", to_string(b),
                                             " backend is not built (hyper_sssp runs on the "
                                             "sequential and cuda backends)"));
  }
}

/// Throws stale_result_error for a poisoned result.
void expect_not_poisoned(const hyper_sssp_state& st, const char* what) {
  if (st.poisoned) {
    throw stale_result_error(concat_message(
        "dyng: ", what, ": a failed update left this result unusable; recompute it"));
  }
}

/// The state of a result, or invalid_argument_error for a moved-from one (stale_result_error for
/// a poisoned one).
template <typename state_t>
state_t& checked(state_t* st, const char* what) {
  DYNG_EXPECTS(st != nullptr, what, ": the result was moved from");
  expect_not_poisoned(*st, what);
  return *st;
}

/// The options' values (invalid_argument_error) and the CUDA engine (not_supported_error on cuda).
void expect_options(const hyper_sssp::options& opt, bool on_cuda, const char* what) {
  DYNG_EXPECTS(!std::isnan(opt.work_budget) && opt.work_budget >= 0.0, what,
               ": options::work_budget must be >= 0 (or +infinity), got ", opt.work_budget);
  DYNG_EXPECTS(opt.max_iterations >= 1, what, ": options::max_iterations must be >= 1, got ",
               opt.max_iterations);
  DYNG_EXPECTS(opt.on_budget == hyper_sssp::budget_action::recompute ||
                   opt.on_budget == hyper_sssp::budget_action::error,
               what, ": options::on_budget holds the invalid value ",
               static_cast<int>(opt.on_budget));
  DYNG_EXPECTS(opt.cuda_engine == engine::automatic || opt.cuda_engine == engine::fused ||
                   opt.cuda_engine == engine::operators,
               what, ": options::cuda_engine holds the invalid value ",
               static_cast<int>(opt.cuda_engine));
  if (on_cuda && opt.cuda_engine == engine::operators) {
    throw not_supported_error(concat_message(
        "dyng: ", what,
        ": hyper_sssp has no operators engine in this release (the shared sssp engine arrives in "
        "M11b); use engine::automatic or engine::fused (MOSP_ESCHER's kernels)"));
  }
}

/// MOSP_ESCHER's block-size check of hsospUpdate and hsospRecompute (both entry points, before
/// anything changes): the recompute kernels run one warp per node.
void expect_block_size(int block_size, const char* what) {
  DYNG_EXPECTS(block_size >= 32 && block_size <= 1024 && block_size % 32 == 0, what,
               ": blockSize must be a multiple of 32 in [32, 1024], got ", block_size);
}

/// What hyper_sssp needs of the hypergraph (HD9): slack_csr storage with its line graph and the
/// vertex-to-hyperedge rows, not poisoned.
template <typename vertex_t, typename weight_t>
void expect_hypergraph(const hypergraph<vertex_t, weight_t>& h, const char* what) {
  const hypergraph_properties& p = h.properties();
  DYNG_EXPECTS(!hypergraph_access::impl(h).poisoned, what,
               ": the hypergraph was poisoned by a failed apply(); rebuild it with "
               "from_hyperedges()");
  if (p.storage != hypergraph_storage::slack_csr) {
    throw not_supported_error(concat_message(
        "dyng: ", what, ": the hypergraph has hypergraph_storage::", to_string(p.storage),
        "; hyper_sssp reads the slack_csr line graph in 0.4: build the hypergraph with "
        "hypergraph_storage::slack_csr and line_graph_policy::slack_csr (or automatic)"));
  }
  if (p.line_graph == line_graph_policy::none) {
    throw not_supported_error(concat_message(
        "dyng: ", what,
        ": the hypergraph keeps no line graph (line_graph_policy::none); hyper_sssp reads the "
        "slack_csr line graph: build it with line_graph_policy::slack_csr or automatic"));
  }
  DYNG_EXPECTS(p.store_v2h, what,
               ": hyper_sssp reads the vertex-to-hyperedge rows; build the hypergraph with "
               "store_v2h = true");
}

/// The node rows of the slack line graph (built under line_graph_policy::automatic).
template <typename vertex_t, typename weight_t>
const hyper::slack_access& slack_of(const hypergraph<vertex_t, weight_t>& h, const char* what) {
  const hyper::slack_access* s = hypergraph_access::slack(h);
  DYNG_EXPECTS(s != nullptr, what, ": the hypergraph has no slack_csr store");
  return *s;
}

/// Link a result to the hypergraph it now matches (distance_to()'s weak reference).
template <typename vertex_t, typename weight_t>
void link(hyper_sssp_state& st, const hypergraph<vertex_t, weight_t>& h) {
  const auto& impl = hypergraph_access::impl(h);
  framework::stamp_result(st.version, st.graph_state, h);
  st.graph_alive = impl.lifetime;
  st.graph_impl = &impl;
  st.n = h.hyperedge_id_bound();
  st.invalidate_caches();
}

/// The weights of the live hyperedges in [1, 2^28] (compute(); the running maximum).
template <typename vertex_t, typename weight_t>
std::int64_t checked_weights(const hypergraph<vertex_t, weight_t>& h, const char* what) {
  const auto w = h.weights();
  std::int64_t largest = 0;
  for (std::size_t i = 0; i < w.size(); ++i) {
    const std::int64_t x = w[i];
    if (x > hyper_sssp_max_weight) {
      throw invalid_argument_error(concat_message(
          "dyng: ", what, ": the weight ", x, " of hyperedge ", i,
          " is above 2^28 (hyper_sssp takes weights in [1, 2^28]; options::validate_inputs = false "
          "accepts it)"));
    }
    largest = std::max(largest, x);
  }
  return largest;
}

/// The hypergraph a result matches, for its accessors (stale_result_error when it is gone or
/// changed).
const impl_t& matched_graph(const hyper_sssp_state& st, const char* what) {
  const std::shared_ptr<const int> alive = st.graph_alive.lock();
  if (!alive || st.graph_impl == nullptr) {
    throw stale_result_error(
        concat_message("dyng: ", what, ": the hypergraph this result matches no longer exists"));
  }
  const auto& impl = *static_cast<const impl_t*>(st.graph_impl);
  if (impl.version != st.version || impl.state_id != st.graph_state) {
    throw stale_result_error(concat_message(
        "dyng: ", what,
        ": the hypergraph has changed since the result's last compute() or "
        "update() (version ",
        impl.version, ", the result matches ", st.version, "); update the result with the batch"));
  }
  return impl;
}

/// The distances of a result on the host (the cuda arrays downloaded once per version).
const std::vector<std::int64_t>& host_distances(const resources& res, const hyper_sssp_state& st) {
  if (st.home != backend::cuda) return st.dist;
  if (!st.host_dist_valid) {
    std::vector<std::int32_t> unused;
    hyper_sssp_cuda_download(res, st, st.host_dist, unused);
    st.host_dist_valid = true;
  }
  return st.host_dist;
}

/// The resources of a result's accessors (its backend, its device).
resources accessor_resources(const hyper_sssp_state& st) {
  return st.home == backend::cuda ? resources::cuda(st.device < 0 ? 0 : st.device)
                                  : resources::sequential();
}

}  // namespace

hyper_sssp_state& hyper_sssp_access::state(hyper_sssp::result<std::int32_t>& r) {
  DYNG_EXPECTS(r.impl_ != nullptr, "hyper_sssp: the result was moved from");
  return *r.impl_;
}

const hyper_sssp_state& hyper_sssp_access::state(const hyper_sssp::result<std::int32_t>& r) {
  DYNG_EXPECTS(r.impl_ != nullptr, "hyper_sssp: the result was moved from");
  return *r.impl_;
}

hyper_sssp::result<std::int32_t> hyper_sssp_access::make(std::unique_ptr<hyper_sssp_state> st) {
  return hyper_sssp::result<std::int32_t>(std::move(st));
}

// ------------------------------------------------------------------------------------------------
// The CUDA engine: stubs until MOSP_ESCHER's engine is transplanted (cuda.cu)
// ------------------------------------------------------------------------------------------------

#if 1
namespace {
[[noreturn]] void no_cuda() {
  throw not_supported_error("dyng: hyper_sssp: the cuda backend is not built");
}
}  // namespace

void hyper_sssp_device_state_deleter::operator()(hsosp::device_state* /*st*/) const noexcept {}
void hyper_sssp_cuda_reserve(const resources& /*res*/, hyper_sssp_state& /*st*/,
                             std::int64_t /*nodes*/) {
  no_cuda();
}
hyper_sssp_engine_counters hyper_sssp_cuda_recompute(const resources& /*res*/,
                                                     hyper_sssp_state& /*st*/,
                                                     const hyper_sssp_graph& /*g*/,
                                                     const hyper_sssp_budget& /*budget*/) {
  no_cuda();
}
hyper_sssp_engine_counters hyper_sssp_cuda_update(const resources& /*res*/,
                                                  hyper_sssp_state& /*st*/,
                                                  const hyper_sssp_graph& /*g*/,
                                                  const hyper_sssp_batch& /*b*/,
                                                  const hyper_sssp_budget& /*budget*/,
                                                  const framework::hook_stages& /*names*/) {
  no_cuda();
}
void hyper_sssp_cuda_download(const resources& /*res*/, const hyper_sssp_state& /*st*/,
                              std::vector<std::int64_t>& /*dist*/,
                              std::vector<std::int32_t>& /*parent*/) {
  no_cuda();
}
void hyper_sssp_cuda_upload(const resources& /*res*/, hyper_sssp_state& /*st*/,
                            const std::vector<std::int64_t>& /*dist*/,
                            const std::vector<std::int32_t>& /*parent*/) {
  no_cuda();
}
const std::int64_t* hyper_sssp_cuda_distances(const hyper_sssp_state& /*st*/) noexcept {
  return nullptr;
}
const std::int32_t* hyper_sssp_cuda_parents(const hyper_sssp_state& /*st*/) noexcept {
  return nullptr;
}
void hyper_sssp_cuda_set_holds(const resources& /*res*/, hyper_sssp_state& /*st*/,
                               const std::vector<std::int64_t>& /*ids*/, std::uint8_t /*value*/) {
  no_cuda();
}
std::int64_t hyper_sssp_cuda_bytes(const hyper_sssp_state& /*st*/) noexcept {
  return 0;
}
#endif

// ------------------------------------------------------------------------------------------------
// The problem
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename weight_t>
hyper_sssp_graph hyper_sssp_problem<vertex_t, weight_t>::graph_of(const container_type& h) const {
  const hyper::slack_access& s = slack_of(h, "hyper_sssp");
  hyper_sssp_graph g;
  g.rows = s.rows();
  g.host_rows = &s.host_rows();
  g.host_v2h = &s.host_v2h();
  g.endpoints = state_->endpoints;
  g.holds = state_->endpoints.is_virtual() || state_->home == backend::cuda ? nullptr
                                                                            : &state_->holds_source;
  return g;
}

template <typename vertex_t, typename weight_t>
hyper_sssp_budget hyper_sssp_problem<vertex_t, weight_t>::budget_of() const {
  hyper_sssp_budget b;
  // MOSP_ESCHER: workLimit = workBudget x max(1, numEntries) (the live line-graph entries).
  b.work_limit = state_->opt.work_budget *
                 static_cast<double>(std::max<std::int64_t>(1, graph_.rows.num_entries));
  b.max_iterations = state_->opt.max_iterations;
  b.block_size = state_->block_size;
  b.on_budget = state_->opt.on_budget;
  return b;
}

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::size_arrays(framework::context& ctx,
                                                         const hyper_sssp_graph& g) {
  // The arrays cover the store's node capacity, as MOSP_ESCHER's state covers maxNodes: the ids a
  // batch adds need no growth until the store's node arrays grow (a reservation, invariant I9).
  const auto nodes = static_cast<std::size_t>(g.rows.num_nodes);
  const auto capacity =
      std::max(nodes, static_cast<std::size_t>(std::max<std::int64_t>(g.rows.node_capacity, 0)));
  const auto grow = [&](auto& v, auto fill) {
    if (v.capacity() < capacity) {
      note_reservation();
      v.reserve(capacity);
    }
    if (v.size() < nodes) v.resize(nodes, fill);
  };
  if (!state_->endpoints.is_virtual()) grow(state_->holds_source, std::uint8_t{0});
  if (ctx.on_cuda()) {
    hyper_sssp_cuda_reserve(ctx.res(), *state_, static_cast<std::int64_t>(capacity));
    return;
  }
  grow(state_->dist, hyper_sssp_infinity);
  grow(state_->parent, std::int32_t{-1});
}

template <typename vertex_t, typename weight_t>
framework::budget hyper_sssp_problem<vertex_t, weight_t>::algorithm_budget(
    framework::context& ctx) const noexcept {
  if (!ctx.on_cuda()) return framework::budget::steady_state(0);
  framework::budget b;
  b.allocations = 0;  // the device state is sized by compute() and grows only with the ids
  return b;           // host syncs: one per pointer-jumping and push round (unbounded)
}

template <typename vertex_t, typename weight_t>
engine hyper_sssp_problem<vertex_t, weight_t>::select_engine(
    framework::context& ctx) const noexcept {
  return ctx.on_cuda() ? engine::fused : engine::operators;
}

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::begin_update(framework::context& ctx, old_graph g,
                                                          const requested& batch) {
  const char* what = "hyper_sssp::update";
  const resources& res = ctx.res();
  expect_backend(res, what);
  expect_not_poisoned(*state_, what);
  framework::expect_current_result(what, state_->version, state_->graph_state, g.get());
  hypergraph_access::expect_placement(res, g.get(), what);
  expect_hypergraph(g.get(), what);
  expect_options(state_->opt, ctx.on_cuda(), what);
  expect_block_size(state_->block_size, what);
  DYNG_EXPECTS(batch.normalized != nullptr, what, ": Step 0 of the hypergraph is missing");
  // The weights of the insertions (HD11), before anything changes: the live weights were checked
  // by compute() and every earlier update, so the batch's are the only new ones (O(batch)).
  if (state_->opt.validate_inputs) {
    std::int64_t largest = state_->max_weight;
    const auto& w = batch.normalized->weights;
    for (std::size_t i = 0; i < w.size(); ++i) {
      const std::int64_t x = w[i];
      if (x > hyper_sssp_max_weight) {
        throw invalid_argument_error(
            concat_message("dyng: ", what, ": the weight ", x, " of inserted hyperedge ", i,
                           " of the batch is above 2^28 (hyper_sssp takes weights in [1, 2^28]; "
                           "options::validate_inputs = false accepts it)"));
      }
      largest = std::max(largest, x);
    }
    pending_max_ = largest;
  }
}

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::resume(framework::context& ctx, new_graph g,
                                                    const applied& a) {
  const hyper_applied& record = a.record;
  DYNG_EXPECTS(record.has_slack_record && record.has_line_graph_delta, "hyper_sssp::update",
               ": the commit did not record the line graph's pair delta (slack_csr storage with "
               "its line graph is required)");
  graph_ = graph_of(g.get());
  size_arrays(ctx, graph_);
  ws_.emplace(ctx.lease_workspace<hyper_sssp_workspace>());
  hyper_sssp_workspace& ws = ws_->get();
  ws.begin();
  batch_.root_ids = &ws.root_ids;
  batch_.lost_source = &ws.lost_source;
  batch_.gained_source = &ws.gained_source;
  // The pairs: on the device for the CUDA store (its last_delta()), in the record otherwise.
  if (record.pairs_on_device) {
    batch_.device = slack_of(g.get(), "hyper_sssp::update").last_delta();
    batch_.num_deleted = batch_.device.num_deleted;
    batch_.num_inserted = batch_.device.num_inserted;
  } else {
    const hyper::line_graph_delta& pairs = record.pairs;
    batch_.deleted_a = &pairs.deleted_a;
    batch_.deleted_b = &pairs.deleted_b;
    batch_.inserted_a = &pairs.inserted_a;
    batch_.inserted_b = &pairs.inserted_b;
    batch_.num_deleted = static_cast<std::int64_t>(pairs.deleted_a.size());
    batch_.num_inserted = static_cast<std::int64_t>(pairs.inserted_a.size());
    // MOSP_ESCHER's newHe (inserted ids, recycled ones included) and deadHe.
    reserve_scratch(ws.root_ids,
                    record.new_ids.size() + record.recreated_ids.size() + record.dead_ids.size());
    ws.root_ids.insert(ws.root_ids.end(), record.new_ids.begin(), record.new_ids.end());
    ws.root_ids.insert(ws.root_ids.end(), record.recreated_ids.begin(), record.recreated_ids.end());
    ws.root_ids.insert(ws.root_ids.end(), record.dead_ids.begin(), record.dead_ids.end());
  }
  // The public mode: the hyperedges whose membership of the source changed (the record's
  // incidence changes of the source vertex), and the source flags after the batch.
  if (!state_->endpoints.is_virtual()) {
    const std::int32_t s = state_->endpoints.source_vertex;
    const hyper_incidence_delta& inc = record.incidence_delta;
    std::vector<std::int64_t>& changed = ws.changed;
    reserve_scratch(changed, inc.vertices.size());
    for (std::size_t i = 0; i < inc.vertices.size(); ++i) {
      if (inc.vertices[i] == s) changed.push_back(inc.ids[i]);
    }
    std::sort(changed.begin(), changed.end());
    changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
    for (auto* list : {&ws.lost_source, &ws.gained_source, &ws.holding, &ws.not_holding}) {
      reserve_scratch(*list, changed.size());
    }
    const auto& rows = *graph_.host_rows;
    for (const std::int64_t id : changed) {
      const auto i = static_cast<std::size_t>(id);
      const bool after = i < rows.size() && std::binary_search(rows[i].begin(), rows[i].end(), s);
      const bool live = graph_.live(id);
      const bool before = before_holds(id);
      if (before && !after && live) ws.lost_source.push_back(id);
      if (!before && after && live) ws.gained_source.push_back(id);
      (after ? ws.holding : ws.not_holding).push_back(id);
    }
    for (const std::int64_t id : ws.holding) state_->holds_source[static_cast<std::size_t>(id)] = 1;
    for (const std::int64_t id : ws.not_holding)
      state_->holds_source[static_cast<std::size_t>(id)] = 0;
    if (ctx.on_cuda()) {
      // The device copy of the flags (the CUDA engine's pulls read them).
      if (!ws.holding.empty()) hyper_sssp_cuda_set_holds(ctx.res(), *state_, ws.holding, 1);
      if (!ws.not_holding.empty()) hyper_sssp_cuda_set_holds(ctx.res(), *state_, ws.not_holding, 0);
    }
  }
  state_->invalidate_caches();
}

template <typename vertex_t, typename weight_t>
bool hyper_sssp_problem<vertex_t, weight_t>::before_holds(std::int64_t id) const {
  const auto i = static_cast<std::size_t>(id);
  return i < state_->holds_source.size() && state_->holds_source[i] != 0;
}

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::end_update(framework::context& /*ctx*/, new_graph g,
                                                        const stats_type& /*stats*/) {
  link(*state_, g.get());
  if (state_->opt.validate_inputs) state_->max_weight = pending_max_;
  state_->last = counters_;
  ws_.reset();
}

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::fill_stats(const hyper_sssp_engine_counters& c,
                                                        stats_type& stats) const {
  stats.line_graph_insertions = batch_.num_inserted;
  stats.line_graph_deletions = batch_.num_deleted;
  stats.invalidated = c.seed_count;
  stats.affected = c.seed_count;
  stats.relaxations = c.work;
  stats.iterations = c.iterations;
  stats.frontier_visits = c.max_frontier;
  stats.fallback_used = c.fallback;
  stats.fallback_reason = c.reason;
  stats.converged = true;
}

// ---- Tier A: the host engine ------------------------------------------------------------------

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::identify_affected(framework::context& /*ctx*/,
                                                               new_graph /*g*/,
                                                               const applied& /*applied*/,
                                                               frontier_type& /*f*/) {
  hyper_sssp_workspace& ws = ws_->get();
  hyper_sssp_host_identify(graph_, batch_, state_->dist, state_->parent, ws);
  counters_.seed_count = ws.invalidated;
}

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::seed(framework::context& /*ctx*/, new_graph /*g*/,
                                                  frontier_type& /*f*/) {
  hyper_sssp_host_seed(graph_, batch_, state_->dist, state_->parent, ws_->get());
}

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::loop(framework::context& /*ctx*/, new_graph /*g*/,
                                                  frontier_type& /*in*/, frontier_type& /*out*/) {
  if (computing_) {
    // compute(): emulateSospRecompute, its round cap max(max_iterations, n + 1) (hsospRecompute).
    const std::int64_t cap = std::max(state_->opt.max_iterations, graph_.rows.num_nodes + 1);
    const std::int64_t rounds =
        hyper_sssp_host_recompute(graph_, state_->dist, state_->parent, cap, ws_->get());
    counters_.iterations = rounds < 0 ? cap : rounds;
    return;
  }
  hyper_sssp_workspace& ws = ws_->get();
  hyper_sssp_host_loop(graph_, state_->dist, state_->parent, budget_of().work_limit, ws);
  counters_.iterations = ws.expansions;
  counters_.max_frontier = ws.expansions;
  counters_.work = ws.work;
}

template <typename vertex_t, typename weight_t>
bool hyper_sssp_problem<vertex_t, weight_t>::over_budget(framework::context& /*ctx*/) {
  if (computing_ || !ws_->get().over_budget) return false;
  if (state_->opt.on_budget == hyper_sssp::budget_action::error) {
    throw convergence_error(concat_message(
        "dyng: hyper_sssp::update: the update exceeded its work budget (", ws_->get().work,
        " relaxations > options::work_budget ", state_->opt.work_budget, " x ",
        std::max<std::int64_t>(1, graph_.rows.num_entries),
        " live entries) under budget_action::error; the hypergraph holds the batch and the "
        "result is poisoned: recompute it"));
  }
  counters_.reason = hyper_sssp::fallback_reason::work_budget;
  return true;
}

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::recompute(framework::context& /*ctx*/, new_graph /*g*/,
                                                       stats_type& stats) {
  const std::int64_t cap = std::max(state_->opt.max_iterations, graph_.rows.num_nodes + 1);
  const std::int64_t rounds =
      hyper_sssp_host_recompute(graph_, state_->dist, state_->parent, cap, ws_->get());
  counters_.fallback = true;
  counters_.fallback_iterations = rounds < 0 ? cap : rounds;
  fill_stats(counters_, stats);
}

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::finalize(framework::context& /*ctx*/,
                                                      stats_type& stats) {
  if (computing_) return;
  fill_stats(counters_, stats);
}

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::reset(framework::context& /*ctx*/) {
  computing_ = true;
}

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::bind_static(framework::context& ctx,
                                                         const container_type& h) {
  graph_ = graph_of(h);
  size_arrays(ctx, graph_);
  if (!ctx.on_cuda()) ws_.emplace(ctx.lease_workspace<hyper_sssp_workspace>());
  if (!state_->endpoints.is_virtual()) {
    // The source flags of the public mode (the engines' base offers): the live hyperedges of s.
    std::fill(state_->holds_source.begin(), state_->holds_source.end(), std::uint8_t{0});
    const std::int32_t s = state_->endpoints.source_vertex;
    for (const std::int64_t id : (*graph_.host_v2h)[static_cast<std::size_t>(s)]) {
      if (graph_.live(id)) state_->holds_source[static_cast<std::size_t>(id)] = 1;
    }
  }
}

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::seed_static(framework::context& ctx, new_graph g,
                                                         frontier_type& /*f*/) {
  bind_static(ctx, g.get());
}

// ---- Tier B: the CUDA engine ------------------------------------------------------------------

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::enact_fused(framework::context& ctx, new_graph /*g*/,
                                                         const applied& /*applied*/,
                                                         stats_type& stats) {
  counters_ = hyper_sssp_cuda_update(ctx.res(), *state_, graph_, batch_, budget_of(),
                                     framework::stages_of<hyper_sssp_problem>());
  fill_stats(counters_, stats);
}

template <typename vertex_t, typename weight_t>
void hyper_sssp_problem<vertex_t, weight_t>::compute_fused(framework::context& ctx, new_graph g,
                                                           stats_type& /*stats*/) {
  bind_static(ctx, g.get());
  counters_ = hyper_sssp_cuda_recompute(ctx.res(), *state_, graph_, budget_of());
}

// ------------------------------------------------------------------------------------------------
// compute(), update(), the participant, the virtual-endpoint mode
// ------------------------------------------------------------------------------------------------

namespace {

/// The static solve into `st` through the static enactor (both modes).
template <typename vertex_t, typename weight_t>
void run_compute(const resources& res, const hypergraph<vertex_t, weight_t>& h,
                 hyper_sssp_state& st) {
  using problem_t = hyper_sssp_problem<vertex_t, weight_t>;
  const framework::hook_stages& names = framework::stages_of<problem_t>();
  scoped_stage stage(res, names.compute);
  problem_t p(st);
  framework::context ctx(res, problem_t::name);
  framework::static_enactor<problem_t> enactor(p);
  (void)enactor.run(ctx, framework::new_view<hypergraph<vertex_t, weight_t>>(h));
  st.last = p.counters();
  link(st, h);
}

/// The checks of compute() (both modes) and a new state.
template <typename vertex_t, typename weight_t>
std::unique_ptr<hyper_sssp_state> new_state(const resources& res,
                                            const hypergraph<vertex_t, weight_t>& h,
                                            const hyper_sssp::options& opt, int block_size,
                                            const char* what) {
  expect_backend(res, what);
  expect_options(opt, res.get_backend() == backend::cuda, what);
  expect_block_size(block_size, what);
  hypergraph_access::expect_placement(res, h, what);
  expect_hypergraph(h, what);
  // line_graph_policy::automatic builds the slack line graph here, once.
  hypergraph_access::require_line_graph(h, what);
  auto st = std::make_unique<hyper_sssp_state>();
  st->opt = opt;
  st->block_size = block_size;
  st->home = res.get_backend();
  st->device = res.get_backend() == backend::cuda ? res.device() : -1;
  if (opt.validate_inputs) st->max_weight = checked_weights(h, what);
  return st;
}

}  // namespace

template <typename vertex_t, typename weight_t>
hyper_sssp::result<vertex_t> hyper_sssp_compute(const resources& res,
                                                const hypergraph<vertex_t, weight_t>& h,
                                                vertex_t source,
                                                const hyper_sssp::options& opt) try {
  constexpr const char* what = "hyper_sssp::compute";
  DYNG_EXPECTS(source >= 0 && source < h.num_vertices(), what, ": the source vertex ", source,
               " is outside [0, ", h.num_vertices(), ")");
  auto st = new_state(res, h, opt, 256, what);
  st->endpoints.source_vertex = source;
  run_compute(res, h, *st);
  return hyper_sssp_access::make(std::move(st));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hyper_sssp::compute (", h.hyperedge_id_bound(),
                                  " hyperedge ids)")

template <typename vertex_t, typename weight_t>
hyper_sssp::stats hyper_sssp_update(const resources& res, hypergraph<vertex_t, weight_t>& h,
                                    const hyperedge_batch_view<vertex_t, weight_t>& batch,
                                    hyper_sssp::result<vertex_t>& r) try {
  using problem_t = hyper_sssp_problem<vertex_t, weight_t>;
  return framework::update_one<problem_t>(res, h, batch, r);
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hyper_sssp::update (", h.hyperedge_id_bound(),
                                  " hyperedge ids; batch of ", batch.num_insertions(),
                                  " insertions, ", batch.num_deletions(), " deletions)")

template <typename vertex_t, typename weight_t>
std::unique_ptr<hyper_update_participant<vertex_t, weight_t>> make_hyper_sssp_participant(
    hyper_sssp::result<vertex_t>& r, hyper_sssp::stats& out) {
  return framework::make_participant<hyper_sssp_problem<vertex_t, weight_t>>(out, r);
}

namespace hyper_sssp_virtual {

hyper_sssp::result<std::int32_t> compute(const resources& res,
                                         const hypergraph<std::int32_t, std::int32_t>& h,
                                         const hyper_sssp_endpoints& endpoints,
                                         const hyper_sssp::options& opt, int block_size) try {
  constexpr const char* what = "hyper_sssp::compute (virtual endpoints)";
  DYNG_EXPECTS(
      endpoints.source_hyperedge >= 0 && endpoints.source_hyperedge < h.hyperedge_id_bound(), what,
      ": the source row ", endpoints.source_hyperedge, " is outside [0, ", h.hyperedge_id_bound(),
      ")");
  auto st = new_state(res, h, opt, block_size, what);
  st->endpoints = endpoints;
  st->endpoints.source_vertex = -1;
  run_compute(res, h, *st);
  return hyper_sssp_access::make(std::move(st));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hyper_sssp::compute (", h.hyperedge_id_bound(),
                                  " hyperedge ids, virtual endpoints)")

void recompute(const resources& res, const hypergraph<std::int32_t, std::int32_t>& h,
               hyper_sssp::result<std::int32_t>& r) {
  constexpr const char* what = "hyper_sssp::recompute";
  hyper_sssp_state& st = hyper_sssp_access::state(r);
  expect_backend(res, what);
  expect_options(st.opt, res.get_backend() == backend::cuda, what);
  expect_block_size(st.block_size, what);
  hypergraph_access::expect_placement(res, h, what);
  expect_hypergraph(h, what);
  hypergraph_access::require_line_graph(h, what);
  DYNG_EXPECTS(st.home == res.get_backend(), what, ": the result belongs to the ",
               to_string(st.home), " backend");
  st.poisoned = false;
  run_compute(res, h, st);
}

void set_block_size(hyper_sssp::result<std::int32_t>& r, int block_size) {
  hyper_sssp_access::state(r).block_size = block_size;
}

const hyper_sssp_engine_counters& last_counters(const hyper_sssp::result<std::int32_t>& r) {
  return hyper_sssp_access::state(r).last;
}

std::int64_t state_bytes(const hyper_sssp::result<std::int32_t>& r) {
  const hyper_sssp_state& st = hyper_sssp_access::state(r);
  return st.home == backend::cuda ? hyper_sssp_cuda_bytes(st) : 0;
}

}  // namespace hyper_sssp_virtual

// Explicit instantiations (PLAN Section 4.4.3: hypergraph<int32_t, int32_t> only).
template class hyper_sssp_problem<std::int32_t, std::int32_t>;
template hyper_sssp::result<std::int32_t> hyper_sssp_compute<std::int32_t, std::int32_t>(
    const resources&, const hypergraph<std::int32_t, std::int32_t>&, std::int32_t,
    const hyper_sssp::options&);
template hyper_sssp::stats hyper_sssp_update<std::int32_t, std::int32_t>(
    const resources&, hypergraph<std::int32_t, std::int32_t>&,
    const hyperedge_batch_view<std::int32_t, std::int32_t>&, hyper_sssp::result<std::int32_t>&);
template std::unique_ptr<hyper_update_participant<std::int32_t, std::int32_t>>
make_hyper_sssp_participant<std::int32_t, std::int32_t>(hyper_sssp::result<std::int32_t>&,
                                                        hyper_sssp::stats&);

}  // namespace dyng::detail

// ------------------------------------------------------------------------------------------------
// The result
// ------------------------------------------------------------------------------------------------

namespace dyng::hyper_sssp {

template <typename vertex_t, typename distance_t>
result<vertex_t, distance_t>::result(std::unique_ptr<detail::hyper_sssp_state> state) noexcept
    : impl_(std::move(state)) {}
template <typename vertex_t, typename distance_t>
result<vertex_t, distance_t>::result(result&& other) noexcept = default;
template <typename vertex_t, typename distance_t>
result<vertex_t, distance_t>& result<vertex_t, distance_t>::operator=(result&& other) noexcept =
    default;
template <typename vertex_t, typename distance_t>
result<vertex_t, distance_t>::~result() = default;

template <typename vertex_t, typename distance_t>
vertex_t result<vertex_t, distance_t>::source() const noexcept {
  return impl_ == nullptr ? vertex_t{-1} : static_cast<vertex_t>(impl_->endpoints.source_vertex);
}

template <typename vertex_t, typename distance_t>
array_view<const distance_t> result<vertex_t, distance_t>::hyperedge_distances() const {
  const auto& st = detail::checked(impl_.get(), "hyper_sssp::result::hyperedge_distances");
  const auto n = static_cast<std::size_t>(st.n);
  if (st.home == backend::cuda) {
    return array_view<const distance_t>(detail::hyper_sssp_cuda_distances(st), n,
                                        memory_space::device, st.device);
  }
  return array_view<const distance_t>(st.dist.data(), n, memory_space::host);
}

template <typename vertex_t, typename distance_t>
array_view<const std::int32_t> result<vertex_t, distance_t>::hyperedge_parents() const {
  const auto& st = detail::checked(impl_.get(), "hyper_sssp::result::hyperedge_parents");
  const auto n = static_cast<std::size_t>(st.n);
  if (st.home == backend::cuda) {
    return array_view<const std::int32_t>(detail::hyper_sssp_cuda_parents(st), n,
                                          memory_space::device, st.device);
  }
  return array_view<const std::int32_t>(st.parent.data(), n, memory_space::host);
}

template <typename vertex_t, typename distance_t>
distance_t result<vertex_t, distance_t>::distance_to(vertex_t target) const {
  constexpr const char* what = "hyper_sssp::result::distance_to";
  const auto& st = detail::checked(impl_.get(), what);
  const detail::impl_t& g = detail::matched_graph(st, what);
  const std::int32_t n = g.store->num_vertices();
  DYNG_EXPECTS(target >= 0 && target < n, what, ": the vertex ", target, " is outside [0, ", n,
               ")");
  if (target == st.endpoints.source_vertex) return 0;
  const std::vector<std::int64_t>& d = detail::host_distances(detail::accessor_resources(st), st);
  const auto& v2h = g.store->slack()->host_v2h();
  distance_t best = detail::hyper_sssp_infinity;
  for (const std::int64_t h : v2h[static_cast<std::size_t>(target)]) {
    if (static_cast<std::size_t>(h) < d.size())
      best = std::min(best, d[static_cast<std::size_t>(h)]);
  }
  return best;
}

template <typename vertex_t, typename distance_t>
array_view<const distance_t> result<vertex_t, distance_t>::vertex_distances() const try {
  constexpr const char* what = "hyper_sssp::result::vertex_distances";
  const auto& st = detail::checked(impl_.get(), what);
  const detail::impl_t& g = detail::matched_graph(st, what);
  if (!st.vertex_dist_valid) {
    const std::vector<std::int64_t>& d = detail::host_distances(detail::accessor_resources(st), st);
    const auto& v2h = g.store->slack()->host_v2h();
    st.vertex_dist.assign(v2h.size(), detail::hyper_sssp_infinity);
    for (std::size_t v = 0; v < v2h.size(); ++v) {
      std::int64_t best = detail::hyper_sssp_infinity;
      for (const std::int64_t h : v2h[v]) {
        if (static_cast<std::size_t>(h) < d.size())
          best = std::min(best, d[static_cast<std::size_t>(h)]);
      }
      st.vertex_dist[v] = best;
    }
    if (st.endpoints.source_vertex >= 0)
      st.vertex_dist[static_cast<std::size_t>(st.endpoints.source_vertex)] = 0;
    st.vertex_dist_valid = true;
  }
  return array_view<const distance_t>(st.vertex_dist.data(), st.vertex_dist.size(),
                                      memory_space::host);
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hyper_sssp::result::vertex_distances")

template <typename vertex_t, typename distance_t>
const options& result<vertex_t, distance_t>::get_options() const {
  return detail::checked(impl_.get(), "hyper_sssp::result::get_options").opt;
}

template <typename vertex_t, typename distance_t>
void result<vertex_t, distance_t>::set_options(const options& opt) {
  constexpr const char* what = "hyper_sssp::result::set_options";
  auto& st = detail::checked(impl_.get(), what);
  detail::expect_options(opt, false, what);
  st.opt = opt;
}

template <typename vertex_t, typename distance_t>
std::uint64_t result<vertex_t, distance_t>::graph_version() const noexcept {
  return impl_ == nullptr ? 0 : impl_->version;
}

template <typename vertex_t, typename distance_t>
memory_space result<vertex_t, distance_t>::space() const noexcept {
  return impl_ != nullptr && impl_->home == backend::cuda ? memory_space::device
                                                          : memory_space::host;
}

template <typename vertex_t, typename distance_t>
result<vertex_t, distance_t> result<vertex_t, distance_t>::clone(const resources& res) const try {
  constexpr const char* what = "hyper_sssp::result::clone";
  const auto& st = detail::checked(impl_.get(), what);
  detail::expect_backend(res, what);
  auto copy = std::make_unique<detail::hyper_sssp_state>();
  copy->opt = st.opt;
  copy->endpoints = st.endpoints;
  copy->block_size = st.block_size;
  copy->home = res.get_backend();
  copy->device = copy->home == backend::cuda ? res.device() : -1;
  copy->n = st.n;
  copy->version = st.version;
  copy->graph_state = st.graph_state;
  copy->max_weight = st.max_weight;
  copy->graph_alive = st.graph_alive;
  copy->graph_impl = st.graph_impl;
  copy->last = st.last;
  // The distances and parents through the host (the copy's own arrays on its backend).
  std::vector<std::int64_t> dist;
  std::vector<std::int32_t> parent;
  copy->holds_source = st.holds_source;
  if (st.home == backend::cuda) {
    detail::hyper_sssp_cuda_download(detail::accessor_resources(st), st, dist, parent);
  } else {
    dist = st.dist;
    parent = st.parent;
  }
  if (copy->home == backend::cuda) {
    detail::hyper_sssp_cuda_reserve(res, *copy, static_cast<std::int64_t>(dist.size()));
    detail::hyper_sssp_cuda_upload(res, *copy, dist, parent);
  } else {
    copy->dist = std::move(dist);
    copy->parent = std::move(parent);
  }
  return result(std::move(copy));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hyper_sssp::result::clone")

template class result<std::int32_t, std::int64_t>;

}  // namespace dyng::hyper_sssp
