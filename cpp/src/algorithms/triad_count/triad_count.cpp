// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:src/driver.cu (run(): the subtract / maintain / add order of a
// batch, the counts after = the counts before + the delta of both phases)
/**
 * @file triad_count.cpp
 * @brief triad_count: argument validation, the result, compute() and update() through the
 *        framework's enactors on the hypergraph, the hooks of the problem (problem.hpp), and the
 *        explicit instantiations (PLAN Section 4.8, file 4).
 *
 * The problem is the aggregate-delta template of ESCHER-GPU@abcf9b2's dynamic counters
 * (src/driver.cu: subtract() on the snapshot before the data-structure update, add() on the
 * snapshot after it): count(-) on G_t's internal row views with the deleted ids as the frontier,
 * one commit, count(+) on G_{t+1}'s with the ASSIGNED inserted ids, each triad owned by its
 * smallest frontier member (ownership::min_member). The CUDA backend runs ESCHER-GPU's wedge
 * engine (cuda.cu, engine/), Tier B; the host backends run the same engine as loops (the host
 * engine, host_engine.hpp, through the Tier A hooks; ADR 0054).
 */
#include "algorithms/triad_count/problem.hpp"
#include "core/budget_counters.hpp"
#include "core/resources_access.hpp"
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
#include <dyng/triad_count.hpp>
#include <dyng/update.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dyng::triad_count {

std::string_view to_string(pattern p) noexcept {
  switch (p) {
    case pattern::h_motif:
      return "h_motif";
    case pattern::stathyper_type1:
      return "stathyper_type1";
    case pattern::stathyper_type2:
      return "stathyper_type2";
    case pattern::stathyper_type3:
      return "stathyper_type3";
  }
  return "unknown";
}

}  // namespace dyng::triad_count

namespace dyng::detail {

namespace {

/// Throws not_supported_error unless the backend of `res` is built.
void expect_backend(const resources& res, const char* what) {
  const backend b = res.get_backend();
  if (!backend_available(b)) {
    throw not_supported_error(
        concat_message("dyng: ", what, ": the ", to_string(b),
                       " backend is not available for triad_count (available: sequential",
                       backend_available(backend::openmp) ? ", openmp" : "",
                       backend_available(backend::cuda) ? ", cuda" : "", ")"));
  }
}

/// The CUDA engine of the options (PLAN Section 4.5.4): automatic and fused run the wedge engine;
/// there is no operators engine.
void expect_cuda_engine(const triad_count::options& opt, const char* what) {
  if (opt.cuda_engine == engine::operators) {
    throw not_supported_error(concat_message(
        "dyng: ", what,
        ": triad_count has no operators engine on the cuda backend; use engine::automatic or "
        "engine::fused (the wedge engine)"));
  }
  DYNG_EXPECTS(opt.cuda_engine == engine::automatic || opt.cuda_engine == engine::fused, what,
               ": options::cuda_engine holds the invalid value ",
               static_cast<int>(opt.cuda_engine));
}

/// Throws invalid_argument_error for an engine value outside the enumeration (on every backend).
void expect_engine_value(const triad_count::options& opt, const char* what) {
  DYNG_EXPECTS(opt.cuda_engine == engine::automatic || opt.cuda_engine == engine::fused ||
                   opt.cuda_engine == engine::operators,
               what, ": options::cuda_engine holds the invalid value ",
               static_cast<int>(opt.cuda_engine));
}

/// Throws invalid_argument_error for a value outside the enumeration.
void expect_pattern(triad_count::pattern kind, const char* what) {
  DYNG_EXPECTS(kind == triad_count::pattern::h_motif ||
                   kind == triad_count::pattern::stathyper_type1 ||
                   kind == triad_count::pattern::stathyper_type2 ||
                   kind == triad_count::pattern::stathyper_type3,
               what, ": options::kind holds the invalid value ", static_cast<int>(kind));
}

/// Throws stale_result_error for a poisoned result.
void expect_not_poisoned(const local_pattern_core& st, const char* what) {
  if (st.poisoned) {
    throw stale_result_error(concat_message(
        "dyng: ", what, ": a failed update left this result unusable; recompute it"));
  }
}

/// The state of a result, or invalid_argument_error for a moved-from one; stale_result_error for
/// a poisoned one.
template <typename state_t>
const state_t& readable(const state_t* st, const char* what) {
  DYNG_EXPECTS(st != nullptr, what, ": the result was moved from");
  expect_not_poisoned(*st, what);
  return *st;
}

/// The largest vertex count the classifying patterns accept: the rows hold vertex + 1 and the
/// three-row merge of the classification (ESCHER-GPU's venn3) ends a row at INT32_MAX, so the
/// largest vertex id must stay below 2^31 - 2 (2^31 - 3 at most).
constexpr std::int64_t max_classified_vertices = std::numeric_limits<std::int32_t>::max() - 1;

/// What a pattern needs of the hypergraph (checked by compute(), naming the property): the
/// merge limit of the vertex ids (h_motif, type 3 and every Op of a user), the line graph and the
/// vertex-to-hyperedge rows (the shapes triangles and pairs). `subject` names the pattern
/// ("pattern h_motif", "an Op of shape pairs").
template <typename vertex_t, typename weight_t>
void expect_requirements(const hypergraph<vertex_t, weight_t>& h, const local_patterns::kernels& k,
                         const std::string& subject, const char* what) {
  if (k.merge_limit) {
    DYNG_EXPECTS(static_cast<std::int64_t>(h.num_vertices()) <= max_classified_vertices, what, ": ",
                 subject,
                 k.builtin ? " classifies triads by a merge of their member rows, which "
                           : " is counted on rows that the set helpers merge, which ",
                 "supports vertex ids up to 2^31 - 3 (num_vertices() <= 2,147,483,646); the "
                 "hypergraph has ",
                 static_cast<std::int64_t>(h.num_vertices()), " vertices");
  }
  if (!reads_line_graph(k)) {
    return;
  }
  const hypergraph_properties& props = h.properties();
  DYNG_EXPECTS(props.line_graph != line_graph_policy::none, what, ": ", subject,
               " reads the line graph, but the hypergraph was built with "
               "line_graph_policy::none (hypergraph_properties::line_graph); build it with "
               "line_graph_policy::automatic or sorted_merge");
  DYNG_EXPECTS(props.store_v2h, what, ": ", subject,
               " reads the vertex-to-hyperedge rows, but the hypergraph was built with "
               "store_v2h = false (hypergraph_properties::store_v2h); build it with "
               "store_v2h = true");
  // line_graph_policy::automatic builds the line graph here, once, and maintains it from then on.
  hypergraph_access::require_line_graph(h, what);
}

/// a + b, or capacity_error past 2^64 - 1.
std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* what) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a) {
    throw capacity_error(concat_message("dyng: ", what, ": a count exceeds 2^64 - 1"));
  }
  return a + b;
}

/// The sum of the first n bins (capacity_error past 2^64 - 1).
std::uint64_t checked_total(const std::uint64_t* bins, int n, const char* what) {
  std::uint64_t total = 0;
  for (int k = 0; k < n; ++k) {
    total = checked_add(total, bins[k], what);
  }
  return total;
}

}  // namespace

triad_count_state& triad_count_access::state(triad_count::result& r) {
  DYNG_EXPECTS(r.impl_ != nullptr, "triad_count: the result was moved from");
  return *r.impl_;
}

const triad_count_state& triad_count_access::state(const triad_count::result& r) {
  DYNG_EXPECTS(r.impl_ != nullptr, "triad_count: the result was moved from");
  return *r.impl_;
}

triad_count::result triad_count_access::make(std::unique_ptr<triad_count_state> st) {
  return triad_count::result(std::move(st));
}

// ------------------------------------------------------------------------------------------------
// The CUDA engine without CUDA: the stubs (cuda.cu defines them in a CUDA build)
// ------------------------------------------------------------------------------------------------

#if !DYNG_HAS_CUDA
namespace {
[[noreturn]] void no_cuda() {
  throw not_supported_error("dyng: triad_count: the cuda backend is not built");
}
}  // namespace

void triad_cuda_open(const resources& /*res*/, const local_patterns::kernels& /*k*/,
                     const void* /*op*/, triad_cuda_engine& /*e*/) {
  no_cuda();
}
void triad_cuda_compute(const resources& /*res*/, triad_cuda_engine& /*e*/,
                        const hyper::row_views& /*views*/, std::uint64_t* /*counts*/,
                        const framework::hook_stages& /*names*/) {
  no_cuda();
}
void triad_cuda_count_minus(const resources& /*res*/, triad_cuda_engine& /*e*/,
                            const hyper::row_views& /*views*/,
                            const std::vector<std::int64_t>& /*deletions*/,
                            std::int64_t /*insertions*/, framework::sign /*s*/,
                            framework::ownership::min_member /*rule*/) {
  no_cuda();
}
void triad_cuda_identify_affected(const resources& /*res*/, triad_cuda_engine& /*e*/,
                                  const hyper::row_views& /*views*/,
                                  const triad_inserted_frontier& /*ids*/) {
  no_cuda();
}
void triad_cuda_count_plus(const resources& /*res*/, triad_cuda_engine& /*e*/,
                           const hyper::row_views& /*views*/, framework::sign /*s*/,
                           framework::ownership::min_member /*rule*/) {
  no_cuda();
}
void triad_cuda_finalize(const resources& /*res*/, triad_cuda_engine& /*e*/,
                         triad_phase_bins& /*minus*/, triad_phase_bins& /*plus*/,
                         const char* /*what*/) {
  no_cuda();
}
void triad_cuda_take_launches(const triad_cuda_engine& /*e*/,
                              triad_launch_counters& /*into*/) noexcept {}
std::int64_t triad_cuda_host_syncs(const local_patterns::kernels& /*k*/) noexcept {
  return 0;
}
local_patterns::launch_fn triad_device_launch(triad_count::pattern /*kind*/) noexcept {
  return nullptr;
}
#endif

// ------------------------------------------------------------------------------------------------
// What both engines share
// ------------------------------------------------------------------------------------------------

triad_inserted_frontier triad_inserted_ids(const hyper_applied& record,
                                           std::int64_t id_bound_before,
                                           std::vector<std::int64_t>& scratch) {
  triad_inserted_frontier f;
  f.size = static_cast<int>(record.inserted_ids.size());
#if defined(DYNG_MUTATION_TRIAD_INSERTED_REQUESTED_IDS)
  // MUTATION (M9a criterion 11, triad_inserted_requested_ids): the frontier of count(+) is the
  // ids the request would get without reuse (after the bound of G_t), not the ids the commit
  // assigned.
  scratch.clear();
  for (int i = 0; i < f.size; ++i) {
    scratch.push_back(id_bound_before + i);
  }
  f.host = scratch.data();
  f.device = nullptr;
#else
  (void)id_bound_before;
  (void)scratch;
  f.host = record.inserted_ids.data();
  f.device = record.inserted_ids_device.size() == record.inserted_ids.size() && f.size > 0
                 ? record.inserted_ids_device.data()
                 : nullptr;
#endif
  return f;
}

// ------------------------------------------------------------------------------------------------
// The problem
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename weight_t, typename flavor_t>
local_pattern_problem<vertex_t, weight_t, flavor_t>::local_pattern_problem(
    typename flavor_t::handle h)
    : handle_(h) {
  // The result's pattern decides whether the commit keeps the line graph (asked before
  // begin_update); a moved-from result fails in begin_update.
  const local_pattern_core* st = flavor_t::find(handle_);
  reads_line_graph_ = st == nullptr || reads_line_graph(st->kernels);
}

template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::begin_update(framework::context& ctx,
                                                                       old_graph g,
                                                                       const requested& batch) {
  const resources& res = ctx.res();
  const char* what = framework::stages_of<local_pattern_problem>().update_function.c_str();
  expect_backend(res, what);
  state_ = &flavor_t::state(handle_);
  expect_not_poisoned(*state_, what);
  framework::expect_current_result(what, state_->version, state_->graph_state, g.get());
  hypergraph_access::expect_placement(res, g.get(), what);
  flavor_t::check_update(*state_, ctx.on_cuda(), what);
  DYNG_EXPECTS(batch.normalized != nullptr, what, ": Step 0 of the hypergraph is missing");
  normalized_ = batch.normalized;
  deleted_ = static_cast<std::int64_t>(normalized_->deletions.size());
  inserted_ = normalized_->rows.size();
}

template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::resume(framework::context& /*ctx*/,
                                                                 new_graph g,
                                                                 const applied& /*applied*/) {
  graph_ = &g.get();
}

template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::end_update(framework::context& /*ctx*/,
                                                                     new_graph g,
                                                                     const stats_type& /*stats*/) {
  framework::stamp_result(state_->version, state_->graph_state, g.get());
  cuda_.reset();
  host_.reset();
}

template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::poison() noexcept {
  if (state_ != nullptr) {
    state_->poisoned = true;
  }
}

template <typename vertex_t, typename weight_t, typename flavor_t>
engine local_pattern_problem<vertex_t, weight_t, flavor_t>::select_engine(
    framework::context& ctx) const noexcept {
  return ctx.on_cuda() ? engine::fused : engine::operators;
}

template <typename vertex_t, typename weight_t, typename flavor_t>
framework::budget local_pattern_problem<vertex_t, weight_t, flavor_t>::algorithm_budget(
    framework::context& ctx) const noexcept {
  if (!ctx.on_cuda() || state_ == nullptr) {
    return framework::budget::steady_state(0);
  }
  return framework::budget::steady_state(triad_cuda_host_syncs(state_->kernels));
}

namespace {

/// Record one count's outcome (mode 0: the full count, 1: a frontier phase) in `c`.
void note_launch(triad_launch_counters& c, const local_patterns::kernels& k, int mode,
                 triad::wedge::launch outcome) noexcept {
  const int shape = shape_index(k);
  switch (outcome) {
    case triad::wedge::launch::kernel:
      ++c.kernels[mode][shape];
      break;
    case triad::wedge::launch::no_sources:
      ++c.no_sources[mode][shape];
      break;
    case triad::wedge::launch::no_wedges:
      ++c.no_wedges[mode][shape];
      break;
  }
}

/// Add the counters `from` to `into`.
void add_launches(const triad_launch_counters& from, triad_launch_counters& into) noexcept {
  for (int m = 0; m < 2; ++m) {
    for (int k = 0; k < 3; ++k) {
      into.kernels[m][k] += from.kernels[m][k];
      into.no_sources[m][k] += from.no_sources[m][k];
      into.no_wedges[m][k] += from.no_wedges[m][k];
    }
  }
}

/// One frontier phase of the host engine on the backend of `res`.
triad::wedge::launch host_phase(const resources& res, const local_patterns::kernels& k,
                                const void* op, const hyper::row_views& views, const int* keys,
                                int n, const unsigned char* flags, framework::sign s,
                                framework::ownership::min_member rule, std::uint64_t* scratch,
                                triad_phase_bins& out) {
  if (res.get_backend() == backend::openmp) {
    return triad_openmp_phase(res, k, op, views, keys, n, flags, s, rule, scratch, out);
  }
  return triad_sequential_phase(res, k, op, views, keys, n, flags, s, rule, scratch, out);
}

}  // namespace

template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::open_host(framework::context& ctx) {
  const resources& res = ctx.res();
  triad_host_engine& e = host_.emplace();
  e.k = &state_->kernels;
  e.op = op_value();
  e.ws.emplace(ctx.lease_workspace<triad_host_workspace>());
  e.ws->get().bins.reserve(res, triad_host_scratch_words(res));
}

template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::host_frontier(
    const hyper::row_views& views, const std::int64_t* ids, int n) {
  triad_host_engine& e = *host_;
  triad_host_workspace& ws = e.ws->get();
  if (n > e.max_keys) {
    DYNG_FAIL("triad_count: a frontier of ", n, " keys exceeds the ", e.max_keys,
              " the workspace reserved before the commit");
  }
  e.num_keys = n;
  int* keys = ws.keys.data();
  for (int i = 0; i < n; ++i) {
    // A frontier holds ids of the phase's hypergraph state: below its h2v view's bound (every
    // loop of the engine indexes the rows by these keys).
    if (ids[i] < 0 || ids[i] >= views.h2v.max_key) {
      DYNG_FAIL("triad_count: the frontier id ", ids[i], " is outside the hypergraph's ids [0, ",
                views.h2v.max_key, ")");
    }
    keys[i] = static_cast<int>(ids[i] + 1);
  }
  if (!reads_line_graph(*e.k)) {
    return;  // the keys shape (type 1) counts keys: no flags
  }
  const int max_key = views.h2h.max_key;
  if (static_cast<std::int64_t>(max_key) + 1 > e.max_flags) {
    DYNG_FAIL("triad_count: the frontier flags of ", max_key,
              " line-graph keys exceed the workspace reserved before the commit (", e.max_flags,
              " flags)");
  }
  unsigned char* flags = ws.flags.data();
  std::fill(flags, flags + max_key + 1, static_cast<unsigned char>(0));
  for (int i = 0; i < n; ++i) {
    if (keys[i] >= 0 && keys[i] <= max_key) {
      flags[keys[i]] = 1;
    }
  }
}

/// count(-) on G_t: the subtract phase with the frontier of Step 0's deletions (ESCHER-GPU's
/// subtract() on the snapshot before graph.update()), on cuda by the wedge engine, on the host
/// backends by the host engine; before it, everything both phases need is reserved (I9).
template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::count(framework::context& ctx,
                                                                old_graph g, frontier_type& /*f*/,
                                                                framework::sign s,
                                                                ownership_type rule) {
  const resources& res = ctx.res();
  const hyper::row_views views = hypergraph_access::views(g.get());
  if (ctx.chosen_engine() == engine::fused) {
    triad_cuda_open(res, state_->kernels, op_value(), cuda_.emplace());
    triad_cuda_count_minus(res, *cuda_, views, normalized_->deletions, inserted_,
                           triad::phase_sign(s), rule);
    return;
  }
  open_host(ctx);
  triad_host_engine& e = *host_;
  triad_host_workspace& ws = e.ws->get();
  const auto deleted = static_cast<int>(deleted_);
  const auto inserted = static_cast<int>(inserted_);
  // The frontier of the larger phase, and flags for every key the line graph can have after the
  // batch (its keys never shrink and grow by at most the insertions).
  // (With headroom, as on cuda: the flags follow a key bound that grows with most batches.)
  e.max_keys = std::max(deleted, inserted);
  ws.keys.reserve_with_headroom(res, static_cast<std::size_t>(std::max(e.max_keys, 1)));
  if (reads_line_graph(*e.k)) {
    e.max_flags = static_cast<std::int64_t>(views.h2h.max_key) + inserted_ + 1;
    ws.flags.reserve_with_headroom(res, static_cast<std::size_t>(e.max_flags));
  }
  host_frontier(views, normalized_->deletions.data(), deleted);
  note_launch(e.launches, *e.k, 1,
              host_phase(res, *e.k, e.op, views, ws.keys.data(), e.num_keys, ws.flags.data(),
                         triad::phase_sign(s), rule, ws.bins.data(), e.minus));
}

/// identify_affected (host backends): the add phase's frontier, the ids the commit ASSIGNED
/// (ESCHER-GPU's add() reads the keys the CBST insert chose).
template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::identify_affected(
    framework::context& /*ctx*/, new_graph g, const applied& applied, frontier_type& /*f*/) {
  const triad_inserted_frontier ids =
      triad_inserted_ids(applied.record, normalized_->id_bound, requested_);
  host_frontier(hypergraph_access::views(g.get()), ids.host, ids.size);
}

/// update(): count(+), the add phase on G_{t+1} (host backends); compute(): the full count.
template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::count(framework::context& ctx,
                                                                new_graph g, frontier_type& /*f*/,
                                                                framework::sign s,
                                                                ownership_type rule) {
  const resources& res = ctx.res();
  graph_ = &g.get();
  const hyper::row_views views = hypergraph_access::views(g.get());
  if (computing_) {
    open_host(ctx);
    triad_host_engine& e = *host_;
    std::uint64_t* scratch = e.ws->get().bins.data();
    pattern_bins full{};
    const triad::wedge::launch outcome =
        ctx.get_backend() == backend::openmp
            ? triad_openmp_full(res, *e.k, e.op, views, rule, scratch, full.data())
            : triad_sequential_full(res, *e.k, e.op, views, rule, scratch, full.data());
    for (int k = 0; k < state_->num_bins; ++k) {
      state_->counts[static_cast<std::size_t>(k)] = full[static_cast<std::size_t>(k)];
    }
    note_launch(state_->launches, *e.k, 0, outcome);
    host_.reset();
    return;
  }
  triad_host_engine& e = *host_;
  triad_host_workspace& ws = e.ws->get();
  note_launch(e.launches, *e.k, 1,
              host_phase(res, *e.k, e.op, views, ws.keys.data(), e.num_keys, ws.flags.data(),
                         triad::phase_sign(s), rule, ws.bins.data(), e.plus));
}

template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::finalize(framework::context& /*ctx*/,
                                                                   stats_type& stats) {
  if (computing_) {
    framework::stamp_result(state_->version, state_->graph_state, *graph_);
    return;
  }
  // As on cuda: the subtract and add phases (magnitudes and signs), applied once.
  add_launches(host_->launches, state_->launches);
  apply_delta(host_->minus, host_->plus, stats);
}

template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::fill_counts(stats_type& stats) const {
  stats.iterations = 0;
  stats.deleted_frontier = deleted_;
  stats.inserted_frontier = inserted_;
  stats.frontier_visits = deleted_ + inserted_;
}

template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::apply_delta(const triad_phase_bins& minus,
                                                                      const triad_phase_bins& plus,
                                                                      stats_type& stats) {
  const char* what = framework::stages_of<local_pattern_problem>().update_function.c_str();
  constexpr auto largest = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  const int n = state_->num_bins;
  // ADR 0057: each phase holds unsigned magnitudes (exact up to 2^64 - 1 per bin) and the sign
  // it counted with; the subtract phase must have counted with minus, the add phase with plus.
  // Check every bin first, so a failure writes nothing.
  std::array<std::int64_t, experimental::max_pattern_bins> delta{};
  pattern_bins after{};
  for (int k = 0; k < experimental::max_pattern_bins; ++k) {
    const auto i = static_cast<std::size_t>(k);
    const std::uint64_t removed = minus.counts[i];
    const std::uint64_t added = plus.counts[i];
    if ((removed != 0 && minus.sign != framework::sign::minus) ||
        (added != 0 && plus.sign != framework::sign::plus) ||
        (k >= n && (removed != 0 || added != 0))) {
      DYNG_FAIL(what, ": bin ", k, " of the engine holds ", removed,
                " triads of the subtract phase (",
                minus.sign == framework::sign::minus ? "subtracted" : "added", ") and ", added,
                " of the add phase (", plus.sign == framework::sign::plus ? "added" : "subtracted",
                "), outside the pattern's signs or bins");
    }
    // The count after, count + added - removed, without an intermediate past 2^64 - 1 (a batch
    // may remove and add 2^63 or more triads of a bin).
    const std::uint64_t before = k < n ? state_->counts[i] : 0;
    if (removed > before && added < removed - before) {
      DYNG_FAIL(what, " would make the count of bin ", k,
                " negative (the result does not match the hypergraph: ", before, " + ", added,
                " - ", removed, ")");
    }
    after[i] =
        removed <= before ? checked_add(before - removed, added, what) : added - (removed - before);
    // stats::delta is signed 64-bit (PLAN I7): a net change of 2^63 or more cannot be reported.
    if (added >= removed ? added - removed > largest : removed - added > largest + 1) {
      throw capacity_error(
          concat_message("dyng: ", what, ": the change of bin ", k, " (", added,
                         state_->kernels.builtin ? " triads added, " : " added, ", removed,
                         " removed) is outside the signed 64-bit range of ", flavor_t::delta_name));
    }
    delta[i] = added >= removed ? static_cast<std::int64_t>(added - removed)
                                : static_cast<std::int64_t>(std::uint64_t{0} - (removed - added));
  }
  const std::uint64_t removed = checked_total(minus.counts.data(), n, what);
  const std::uint64_t added = checked_total(plus.counts.data(), n, what);
  flavor_t::write(stats, removed, added, delta.data(), n);
  std::int64_t affected = 0;
  for (int k = 0; k < n; ++k) {
    const auto i = static_cast<std::size_t>(k);
    state_->counts[i] = after[i];
    affected += delta[i] != 0 ? 1 : 0;
  }
  stats.affected = affected;
  fill_counts(stats);
}

/// cuda, Tier B: identify_affected, count_plus and finalize on G_{t+1} (ESCHER-GPU's add() on the
/// snapshot after graph.update(), with the keys the CBST insert assigned).
template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::enact_fused(framework::context& ctx,
                                                                      new_graph g,
                                                                      const applied& applied,
                                                                      stats_type& stats) {
  const resources& res = ctx.res();
  const framework::hook_stages& names = framework::stages_of<local_pattern_problem>();
  const hyper::row_views views = hypergraph_access::views(g.get());
  {
    scoped_stage stage(res, names.identify_affected);
    triad_cuda_identify_affected(
        res, *cuda_, views, triad_inserted_ids(applied.record, normalized_->id_bound, requested_));
  }
  {
    scoped_stage stage(res, names.count_plus);
    triad_cuda_count_plus(res, *cuda_, views, triad::phase_sign(framework::sign::plus),
                          ownership_type{});
  }
  scoped_stage stage(res, names.finalize);
  triad_phase_bins minus;
  triad_phase_bins plus;
  triad_cuda_finalize(res, *cuda_, minus, plus, names.update_function.c_str());
  triad_cuda_take_launches(*cuda_, state_->launches);
  apply_delta(minus, plus, stats);
}

template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::reset(framework::context& /*ctx*/) {
  std::fill(state_->counts.begin(), state_->counts.end(), std::uint64_t{0});
}

template <typename vertex_t, typename weight_t, typename flavor_t>
void local_pattern_problem<vertex_t, weight_t, flavor_t>::compute_fused(framework::context& ctx,
                                                                        new_graph g,
                                                                        stats_type& /*stats*/) {
  const resources& res = ctx.res();
  const framework::hook_stages& names = framework::stages_of<local_pattern_problem>();
  graph_ = &g.get();
  {
    scoped_stage stage(res, names.reset);
    std::fill(state_->counts.begin(), state_->counts.end(), std::uint64_t{0});
    triad_cuda_open(res, state_->kernels, op_value(), cuda_.emplace());
  }
  // <name>.count (the full count) and <name>.finalize (the read of the bins) inside.
  triad_cuda_compute(res, *cuda_, hypergraph_access::views(g.get()), state_->counts.data(), names);
  triad_cuda_take_launches(*cuda_, state_->launches);
  cuda_.reset();
  framework::stamp_result(state_->version, state_->graph_state, *graph_);
}

// ------------------------------------------------------------------------------------------------
// compute(), update(), the participant
// ------------------------------------------------------------------------------------------------

namespace {

/// compute() of any flavor on a state whose table, Op value and bins are set: the static enactor
/// (the full count), then the total's range check.
template <typename vertex_t, typename weight_t, typename flavor_t>
void run_compute(const resources& res, const hypergraph<vertex_t, weight_t>& h,
                 local_pattern_core& st, const char* what) {
  using problem_t = local_pattern_problem<vertex_t, weight_t, flavor_t>;
  const framework::hook_stages& names = framework::stages_of<problem_t>();
  scoped_stage stage(res, names.compute);
  problem_t p(st);
  framework::context ctx(res, problem_t::name);
  framework::static_enactor<problem_t> enactor(p);
  (void)enactor.run(ctx, framework::new_view<hypergraph<vertex_t, weight_t>>(h));
  (void)checked_total(st.counts.data(), st.num_bins, what);
}

/// Throws invalid_argument_error for a hypergraph poisoned by a failed apply().
template <typename vertex_t, typename weight_t>
void expect_not_poisoned_hypergraph(const hypergraph<vertex_t, weight_t>& h, const char* what) {
  DYNG_EXPECTS(!hypergraph_access::impl(h).poisoned, what,
               ": the hypergraph was poisoned by a failed apply(); rebuild it with "
               "from_hyperedges()");
}

}  // namespace

template <typename vertex_t, typename weight_t>
triad_count::result triad_count_compute(const resources& res,
                                        const hypergraph<vertex_t, weight_t>& h,
                                        const triad_count::options& opt) try {
  constexpr const char* what = "triad_count::compute";
  expect_backend(res, what);
  expect_pattern(opt.kind, what);
  expect_engine_value(opt, what);
  if (res.get_backend() == backend::cuda) {
    expect_cuda_engine(opt, what);
  }
  hypergraph_access::expect_placement(res, h, what);
  expect_not_poisoned_hypergraph(h, what);
  const local_patterns::kernels& k = triad_kernels(opt.kind);
  expect_requirements(h, k, concat_message("pattern ", triad_count::to_string(opt.kind)), what);
  auto st = std::make_unique<triad_count_state>();
  st->opt = opt;
  st->kernels = k;
  triad_op_value(opt.kind, st->op);
  st->num_bins = k.num_bins;
  st->counts.assign(static_cast<std::size_t>(k.num_bins), 0);
  run_compute<vertex_t, weight_t, triad_flavor>(res, h, *st, what);
  return triad_count_access::make(std::move(st));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("triad_count::compute (", h.hyperedge_id_bound(),
                                  " hyperedge ids, pattern ", triad_count::to_string(opt.kind), ")")

template <typename vertex_t, typename weight_t>
triad_count::stats triad_count_update(const resources& res, hypergraph<vertex_t, weight_t>& h,
                                      const hyperedge_batch_view<vertex_t, weight_t>& batch,
                                      triad_count::result& r) try {
  using problem_t = triad_count_problem<vertex_t, weight_t>;
  return framework::update_one<problem_t>(res, h, batch, triad_flavor::handle{&r});
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("triad_count::update (", h.hyperedge_id_bound(),
                                  " hyperedge ids; batch of ", batch.num_insertions(),
                                  " insertions, ", batch.num_deletions(), " deletions)")

template <typename vertex_t, typename weight_t>
std::unique_ptr<hyper_update_participant<vertex_t, weight_t>> make_triad_count_participant(
    triad_count::result& r, triad_count::stats& out) {
  return framework::make_participant<triad_count_problem<vertex_t, weight_t>>(
      out, triad_flavor::handle{&r});
}

// ------------------------------------------------------------------------------------------------
// experimental::count_local_patterns (M9b, ADR 0058): the same driver, the experimental flavor
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename weight_t>
local_patterns_state_ptr local_patterns_compute(const resources& res,
                                                const hypergraph<vertex_t, weight_t>& h,
                                                const local_patterns::kernels& k,
                                                const void* op) try {
  constexpr const char* what = "count_local_patterns::compute";
  expect_backend(res, what);
  if (k.abi != local_patterns::abi_version) {
    throw not_supported_error(concat_message(
        "dyng: ", what, ": the header <dyng/experimental/local_patterns.cuh> (hand-off version ",
        k.abi, ") and the library (version ", local_patterns::abi_version,
        ") differ; compile against the headers of the installed library"));
  }
  DYNG_EXPECTS(k.num_bins >= 1 && k.num_bins <= experimental::max_pattern_bins &&
                   k.source != nullptr && k.launch != nullptr && op != nullptr && k.op_size > 0,
               what, ": the Op's table is incomplete (num_bins ", k.num_bins,
               "); build it with count_local_patterns<op_t>");
  hypergraph_access::expect_placement(res, h, what);
  expect_not_poisoned_hypergraph(h, what);
  expect_requirements(
      h, k, concat_message("an Op of shape ", experimental::to_string(k.pattern_shape)), what);
  auto st = std::make_unique<local_patterns_state>();
  st->kernels = k;
  const auto* bytes = static_cast<const unsigned char*>(op);
  st->op.assign(bytes, bytes + k.op_size);
  st->num_bins = k.num_bins;
  st->counts.assign(static_cast<std::size_t>(k.num_bins), 0);
  run_compute<vertex_t, weight_t, local_patterns_flavor>(res, h, *st, what);
  return local_patterns_state_ptr(st.release());
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("count_local_patterns::compute (", h.hyperedge_id_bound(),
                                  " hyperedge ids, ", k.num_bins, " bins)")

template <typename vertex_t, typename weight_t>
experimental::local_patterns_stats local_patterns_update(
    const resources& res, hypergraph<vertex_t, weight_t>& h,
    const hyperedge_batch_view<vertex_t, weight_t>& batch, local_patterns_state_ptr& st,
    const void* target) try {
  using problem_t = local_pattern_problem<vertex_t, weight_t, local_patterns_flavor>;
  return framework::update_one<problem_t>(res, h, batch,
                                          local_patterns_flavor::handle{&st, target});
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("count_local_patterns::update (", h.hyperedge_id_bound(),
                                  " hyperedge ids; batch of ", batch.num_insertions(),
                                  " insertions, ", batch.num_deletions(), " deletions)")

template <typename vertex_t, typename weight_t>
std::unique_ptr<hyper_update_participant<vertex_t, weight_t>> make_local_patterns_participant(
    local_patterns_state_ptr& st, const void* target, experimental::local_patterns_stats& out) {
  return framework::make_participant<
      local_pattern_problem<vertex_t, weight_t, local_patterns_flavor>>(
      out, local_patterns_flavor::handle{&st, target});
}

local_pattern_core* local_patterns_flavor::find(handle h) noexcept {
  return h.owner == nullptr ? nullptr : h.owner->get();
}

local_pattern_core& local_patterns_flavor::state(handle h) {
  local_pattern_core* st = find(h);
  DYNG_EXPECTS(st != nullptr, "count_local_patterns: the result was moved from");
  return *st;
}

void triad_flavor::check_update(const local_pattern_core& st, bool on_cuda, const char* what) {
  if (on_cuda) {
    expect_cuda_engine(static_cast<const triad_count_state&>(st).opt, what);
  }
}

// Explicit instantiations (PLAN Section 4.4.3: the hypergraph types).
#define DYNG_INSTANTIATE_TRIAD_COUNT(V, W)                                                        \
  template class local_pattern_problem<V, W, triad_flavor>;                                       \
  template class local_pattern_problem<V, W, local_patterns_flavor>;                              \
  template std::unique_ptr<hyper_update_participant<V, W>> make_triad_count_participant<V, W>(    \
      triad_count::result&, triad_count::stats&);                                                 \
  template triad_count::result triad_count_compute<V, W>(                                         \
      const resources&, const hypergraph<V, W>&, const triad_count::options&);                    \
  template triad_count::stats triad_count_update<V, W>(const resources&, hypergraph<V, W>&,       \
                                                       const hyperedge_batch_view<V, W>&,         \
                                                       triad_count::result&);                     \
  template local_patterns_state_ptr local_patterns_compute<V, W>(                                 \
      const resources&, const hypergraph<V, W>&, const local_patterns::kernels&, const void*);    \
  template experimental::local_patterns_stats local_patterns_update<V, W>(                        \
      const resources&, hypergraph<V, W>&, const hyperedge_batch_view<V, W>&,                     \
      local_patterns_state_ptr&, const void*);                                                    \
  template std::unique_ptr<hyper_update_participant<V, W>> make_local_patterns_participant<V, W>( \
      local_patterns_state_ptr&, const void*, experimental::local_patterns_stats&);
DYNG_FOR_EACH_HYPERGRAPH_TYPE(DYNG_INSTANTIATE_TRIAD_COUNT)
#undef DYNG_INSTANTIATE_TRIAD_COUNT

// ------------------------------------------------------------------------------------------------
// The experimental result's accessors
// ------------------------------------------------------------------------------------------------

void local_patterns_state_deleter::operator()(local_patterns_state* st) const noexcept {
  delete st;
}

array_view<const std::uint64_t> local_patterns_counts(const local_patterns_state* st) {
  const auto& s = readable(st, "count_local_patterns::result::counts");
  return array_view<const std::uint64_t>(s.counts.data(), static_cast<std::size_t>(s.num_bins),
                                         memory_space::host);
}

std::uint64_t local_patterns_count(const local_patterns_state* st, int bin) {
  const auto& s = readable(st, "count_local_patterns::result::count");
  DYNG_EXPECTS(bin >= 0 && bin < s.num_bins, "count_local_patterns::result::count: bin ", bin,
               " is outside [0, ", s.num_bins, ")");
  return s.counts[static_cast<std::size_t>(bin)];
}

std::uint64_t local_patterns_total(const local_patterns_state* st) {
  const auto& s = readable(st, "count_local_patterns::result::total");
  return checked_total(s.counts.data(), s.num_bins, "count_local_patterns::result::total");
}

int local_patterns_num_bins(const local_patterns_state* st) {
  return readable(st, "count_local_patterns::result::num_bins").num_bins;
}

std::uint64_t local_patterns_graph_version(const local_patterns_state* st) noexcept {
  return st == nullptr ? 0 : st->version;
}

local_patterns_state_ptr local_patterns_clone(const resources& res,
                                              const local_patterns_state* st) try {
  const auto& s = readable(st, "count_local_patterns::result::clone");
  expect_backend(res, "count_local_patterns::result::clone");
  return local_patterns_state_ptr(new local_patterns_state(s));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("count_local_patterns::result::clone")

void local_patterns_op(const local_patterns_state* st, void* out, std::size_t size) {
  const auto& s = readable(st, "count_local_patterns::result::op");
  DYNG_EXPECTS(size == s.op.size(), "count_local_patterns::result::op: the Op has ", s.op.size(),
               " bytes, not ", size);
  std::memcpy(out, s.op.data(), size);
}

}  // namespace dyng::detail

// ------------------------------------------------------------------------------------------------
// The result
// ------------------------------------------------------------------------------------------------

namespace dyng::triad_count {

result::result(std::unique_ptr<detail::triad_count_state> state) noexcept
    : impl_(std::move(state)) {}
result::result(result&& other) noexcept = default;
result& result::operator=(result&& other) noexcept = default;
result::~result() = default;

pattern result::kind() const {
  return detail::readable(impl_.get(), "triad_count::result::kind").opt.kind;
}

int result::num_bins() const {
  return detail::readable(impl_.get(), "triad_count::result::num_bins").num_bins;
}

array_view<const std::uint64_t> result::counts() const {
  const auto& st = detail::readable(impl_.get(), "triad_count::result::counts");
  return array_view<const std::uint64_t>(st.counts.data(), static_cast<std::size_t>(st.num_bins),
                                         memory_space::host);
}

std::uint64_t result::count(int bin) const {
  const auto& st = detail::readable(impl_.get(), "triad_count::result::count");
  DYNG_EXPECTS(bin >= 0 && bin < st.num_bins, "triad_count::result::count: bin ", bin,
               " is outside [0, ", st.num_bins, ")");
  return st.counts[static_cast<std::size_t>(bin)];
}

std::uint64_t result::total() const {
  const auto& st = detail::readable(impl_.get(), "triad_count::result::total");
  return detail::checked_total(st.counts.data(), st.num_bins, "triad_count::result::total");
}

const options& result::get_options() const {
  return detail::readable(impl_.get(), "triad_count::result::get_options").opt;
}

void result::set_options(const options& opt) {
  constexpr const char* what = "triad_count::result::set_options";
  const auto& st = detail::readable(impl_.get(), what);
  detail::expect_pattern(opt.kind, what);
  DYNG_EXPECTS(opt.kind == st.opt.kind, what, ": options::kind is fixed at compute() (",
               to_string(st.opt.kind), "); compute a new result for ", to_string(opt.kind));
  detail::expect_engine_value(opt, what);
  impl_->opt = opt;
}

std::uint64_t result::graph_version() const noexcept {
  return impl_ == nullptr ? 0 : impl_->version;
}

memory_space result::space() const noexcept {
  return memory_space::host;
}

result result::clone(const resources& res) const try {
  const auto& st = detail::readable(impl_.get(), "triad_count::result::clone");
  detail::expect_backend(res, "triad_count::result::clone");
  return result(std::make_unique<detail::triad_count_state>(st));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("triad_count::result::clone")

}  // namespace dyng::triad_count
