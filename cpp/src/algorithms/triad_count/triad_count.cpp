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
void expect_not_poisoned(const triad_count_state& st, const char* what) {
  if (st.poisoned) {
    throw stale_result_error(concat_message(
        "dyng: ", what, ": a failed update left this result unusable; recompute it"));
  }
}

/// The state of a result, or invalid_argument_error for a moved-from one; stale_result_error for
/// a poisoned one.
const triad_count_state& readable(const triad_count_state* st, const char* what) {
  DYNG_EXPECTS(st != nullptr, what, ": the result was moved from");
  expect_not_poisoned(*st, what);
  return *st;
}

/// The largest vertex count the classifying patterns accept: the rows hold vertex + 1 and the
/// three-row merge of the classification (ESCHER-GPU's venn3) ends a row at INT32_MAX, so the
/// largest vertex id must stay below 2^31 - 2 (2^31 - 3 at most).
constexpr std::int64_t max_classified_vertices = std::numeric_limits<std::int32_t>::max() - 1;

/// What a pattern needs of the hypergraph (checked by compute(), naming the property).
template <typename vertex_t, typename weight_t>
void expect_requirements(const hypergraph<vertex_t, weight_t>& h, triad_count::pattern kind,
                         const char* what) {
  const char* name = triad_count::to_string(kind).data();
  if (kind == triad_count::pattern::h_motif || kind == triad_count::pattern::stathyper_type3) {
    DYNG_EXPECTS(static_cast<std::int64_t>(h.num_vertices()) <= max_classified_vertices, what,
                 ": pattern ", name,
                 " classifies triads by a merge of their member rows, which "
                 "supports vertex ids up to 2^31 - 3 (num_vertices() <= 2,147,483,646); the "
                 "hypergraph has ",
                 static_cast<std::int64_t>(h.num_vertices()), " vertices");
  }
  if (!triad_reads_line_graph(kind)) {
    return;
  }
  const hypergraph_properties& props = h.properties();
  DYNG_EXPECTS(props.line_graph != line_graph_policy::none, what, ": pattern ", name,
               " reads the line graph, but the hypergraph was built with "
               "line_graph_policy::none (hypergraph_properties::line_graph); build it with "
               "line_graph_policy::automatic or sorted_merge");
  DYNG_EXPECTS(props.store_v2h, what, ": pattern ", name,
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
std::uint64_t checked_total(const triad_bins& bins, int n, const char* what) {
  std::uint64_t total = 0;
  for (int k = 0; k < n; ++k) {
    total = checked_add(total, bins[static_cast<std::size_t>(k)], what);
  }
  return total;
}

}  // namespace

int triad_num_bins(triad_count::pattern kind) noexcept {
  return kind == triad_count::pattern::h_motif ? triad_count::max_bins : 1;
}

bool triad_reads_line_graph(triad_count::pattern kind) noexcept {
  return kind != triad_count::pattern::stathyper_type1;
}

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

void triad_cuda_open(const resources& /*res*/, triad_count::pattern /*kind*/,
                     triad_cuda_engine& /*e*/) {
  no_cuda();
}
void triad_cuda_compute(const resources& /*res*/, triad_cuda_engine& /*e*/,
                        const hyper::row_views& /*views*/, triad_bins& /*counts*/) {
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
                         triad_phase_bins& /*minus*/, triad_phase_bins& /*plus*/) {
  no_cuda();
}
void triad_cuda_take_launches(const triad_cuda_engine& /*e*/,
                              triad_launch_counters& /*into*/) noexcept {}
std::int64_t triad_cuda_host_syncs(triad_count::pattern /*kind*/) noexcept {
  return 0;
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

template <typename vertex_t, typename weight_t>
triad_count_problem<vertex_t, weight_t>::triad_count_problem(triad_count::result& r) : result_(&r) {
  // The result's pattern decides whether the commit keeps the line graph (asked before
  // begin_update); a moved-from result fails in begin_update.
  const triad_count_state* st = triad_count_access::find(r);
  reads_line_graph_ = st == nullptr || triad_reads_line_graph(st->opt.kind);
}

template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::begin_update(framework::context& ctx, old_graph g,
                                                           const requested& batch) {
  const resources& res = ctx.res();
  constexpr const char* what = "triad_count::update";
  expect_backend(res, what);
  state_ = &triad_count_access::state(*result_);
  expect_not_poisoned(*state_, what);
  framework::expect_current_result(what, state_->version, state_->graph_state, g.get());
  hypergraph_access::expect_placement(res, g.get(), what);
  if (ctx.on_cuda()) {
    expect_cuda_engine(state_->opt, what);
  }
  DYNG_EXPECTS(batch.normalized != nullptr, what, ": Step 0 of the hypergraph is missing");
  normalized_ = batch.normalized;
  deleted_ = static_cast<std::int64_t>(normalized_->deletions.size());
  inserted_ = normalized_->rows.size();
}

template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::resume(framework::context& /*ctx*/, new_graph g,
                                                     const applied& /*applied*/) {
  graph_ = &g.get();
}

template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::end_update(framework::context& /*ctx*/, new_graph g,
                                                         const stats_type& /*stats*/) {
  framework::stamp_result(state_->version, state_->graph_state, g.get());
  cuda_.reset();
  host_.reset();
}

template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::poison() noexcept {
  if (state_ != nullptr) {
    state_->poisoned = true;
  }
}

template <typename vertex_t, typename weight_t>
engine triad_count_problem<vertex_t, weight_t>::select_engine(
    framework::context& ctx) const noexcept {
  return ctx.on_cuda() ? engine::fused : engine::operators;
}

template <typename vertex_t, typename weight_t>
framework::budget triad_count_problem<vertex_t, weight_t>::algorithm_budget(
    framework::context& ctx) const noexcept {
  if (!ctx.on_cuda() || state_ == nullptr) {
    return framework::budget::steady_state(0);
  }
  return framework::budget::steady_state(triad_cuda_host_syncs(state_->opt.kind));
}

namespace {

/// Record one count's outcome (mode 0: the full count, 1: a frontier phase) in `c`.
void note_launch(triad_launch_counters& c, triad_count::pattern kind, int mode,
                 triad::wedge::launch outcome) noexcept {
  const int shape = kind == triad_count::pattern::stathyper_type1   ? 2
                    : kind == triad_count::pattern::stathyper_type2 ? 1
                                                                    : 0;
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
triad::wedge::launch host_phase(const resources& res, triad_count::pattern kind,
                                const hyper::row_views& views, const int* keys, int n,
                                const unsigned char* flags, framework::sign s,
                                framework::ownership::min_member rule, std::uint64_t* scratch,
                                triad_phase_bins& out) {
  if (res.get_backend() == backend::openmp) {
    return triad_openmp_phase(res, kind, views, keys, n, flags, s, rule, scratch, out);
  }
  return triad_sequential_phase(res, kind, views, keys, n, flags, s, rule, scratch, out);
}

}  // namespace

template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::open_host(framework::context& ctx) {
  const resources& res = ctx.res();
  triad_host_engine& e = host_.emplace();
  e.kind = state_->opt.kind;
  e.ws.emplace(ctx.lease_workspace<triad_host_workspace>());
  e.ws->get().bins.reserve(res, triad_host_scratch_words(res));
}

template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::host_frontier(const hyper::row_views& views,
                                                            const std::int64_t* ids, int n) {
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
  if (!triad_reads_line_graph(e.kind)) {
    return;  // type 1 counts keys: no flags
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
template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::count(framework::context& ctx, old_graph g,
                                                    frontier_type& /*f*/, framework::sign s,
                                                    ownership_type rule) {
  const resources& res = ctx.res();
  const hyper::row_views views = hypergraph_access::views(g.get());
  if (ctx.chosen_engine() == engine::fused) {
    triad_cuda_open(res, state_->opt.kind, cuda_.emplace());
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
  e.max_keys = std::max(deleted, inserted);
  ws.keys.reserve(res, static_cast<std::size_t>(std::max(e.max_keys, 1)));
  if (triad_reads_line_graph(e.kind)) {
    e.max_flags = static_cast<std::int64_t>(views.h2h.max_key) + inserted_ + 1;
    ws.flags.reserve(res, static_cast<std::size_t>(e.max_flags));
  }
  host_frontier(views, normalized_->deletions.data(), deleted);
  note_launch(e.launches, e.kind, 1,
              host_phase(res, e.kind, views, ws.keys.data(), e.num_keys, ws.flags.data(),
                         triad::phase_sign(s), rule, ws.bins.data(), e.minus));
}

/// identify_affected (host backends): the add phase's frontier, the ids the commit ASSIGNED
/// (ESCHER-GPU's add() reads the keys the CBST insert chose).
template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::identify_affected(framework::context& /*ctx*/,
                                                                new_graph g, const applied& applied,
                                                                frontier_type& /*f*/) {
  const triad_inserted_frontier ids =
      triad_inserted_ids(applied.record, normalized_->id_bound, requested_);
  host_frontier(hypergraph_access::views(g.get()), ids.host, ids.size);
}

/// update(): count(+), the add phase on G_{t+1} (host backends); compute(): the full count.
template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::count(framework::context& ctx, new_graph g,
                                                    frontier_type& /*f*/, framework::sign s,
                                                    ownership_type rule) {
  const resources& res = ctx.res();
  graph_ = &g.get();
  const hyper::row_views views = hypergraph_access::views(g.get());
  if (computing_) {
    open_host(ctx);
    triad_host_engine& e = *host_;
    std::uint64_t* scratch = e.ws->get().bins.data();
    const triad::wedge::launch outcome =
        ctx.get_backend() == backend::openmp
            ? triad_openmp_full(res, e.kind, views, rule, scratch, state_->counts)
            : triad_sequential_full(res, e.kind, views, rule, scratch, state_->counts);
    note_launch(state_->launches, e.kind, 0, outcome);
    host_.reset();
    return;
  }
  triad_host_engine& e = *host_;
  triad_host_workspace& ws = e.ws->get();
  note_launch(e.launches, e.kind, 1,
              host_phase(res, e.kind, views, ws.keys.data(), e.num_keys, ws.flags.data(),
                         triad::phase_sign(s), rule, ws.bins.data(), e.plus));
}

template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::finalize(framework::context& /*ctx*/,
                                                       stats_type& stats) {
  if (computing_) {
    framework::stamp_result(state_->version, state_->graph_state, *graph_);
    return;
  }
  // As on cuda: the subtract and add phases (magnitudes and signs), applied once.
  add_launches(host_->launches, state_->launches);
  apply_delta(host_->minus, host_->plus, stats);
}

template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::fill_counts(stats_type& stats) const {
  stats.iterations = 0;
  stats.deleted_frontier = deleted_;
  stats.inserted_frontier = inserted_;
  stats.frontier_visits = deleted_ + inserted_;
}

template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::apply_delta(const triad_phase_bins& minus,
                                                          const triad_phase_bins& plus,
                                                          stats_type& stats) {
  constexpr const char* what = "triad_count::update";
  constexpr auto largest = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  const int n = state_->num_bins;
  // ADR 0057: each phase holds unsigned magnitudes (exact up to 2^64 - 1 per bin) and the sign
  // it counted with; the subtract phase must have counted with minus, the add phase with plus.
  // Check every bin first, so a failure writes nothing.
  std::array<std::int64_t, triad_count::max_bins> delta{};
  triad_bins after{};
  for (int k = 0; k < triad_count::max_bins; ++k) {
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
    const std::uint64_t before = state_->counts[i];
    if (removed > before && added < removed - before) {
      DYNG_FAIL(what, " would make the count of bin ", k,
                " negative (the result does not match the hypergraph: ", before, " + ", added,
                " - ", removed, ")");
    }
    after[i] =
        removed <= before ? checked_add(before - removed, added, what) : added - (removed - before);
    // stats::delta is signed 64-bit (PLAN I7): a net change of 2^63 or more cannot be reported.
    if (added >= removed ? added - removed > largest : removed - added > largest + 1) {
      throw capacity_error(concat_message(
          "dyng: ", what, ": the change of bin ", k, " (", added, " triads added, ", removed,
          " removed) is outside the signed 64-bit range of triad_count::stats::delta"));
    }
    delta[i] = added >= removed ? static_cast<std::int64_t>(added - removed)
                                : static_cast<std::int64_t>(std::uint64_t{0} - (removed - added));
  }
  stats.triads_removed = checked_total(minus.counts, n, what);
  stats.triads_added = checked_total(plus.counts, n, what);
  std::int64_t affected = 0;
  for (int k = 0; k < n; ++k) {
    const auto i = static_cast<std::size_t>(k);
    state_->counts[i] = after[i];
    stats.delta[i] = delta[i];
    affected += delta[i] != 0 ? 1 : 0;
  }
  stats.affected = affected;
  fill_counts(stats);
}

/// cuda, Tier B: identify_affected, count_plus and finalize on G_{t+1} (ESCHER-GPU's add() on the
/// snapshot after graph.update(), with the keys the CBST insert assigned).
template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::enact_fused(framework::context& ctx, new_graph g,
                                                          const applied& applied,
                                                          stats_type& stats) {
  const resources& res = ctx.res();
  const framework::hook_stages& names = framework::stages_of<triad_count_problem>();
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
  triad_cuda_finalize(res, *cuda_, minus, plus);
  triad_cuda_take_launches(*cuda_, state_->launches);
  apply_delta(minus, plus, stats);
}

template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::reset(framework::context& /*ctx*/) {
  state_->counts.fill(0);
}

template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::compute_fused(framework::context& ctx, new_graph g,
                                                            stats_type& /*stats*/) {
  const resources& res = ctx.res();
  const framework::hook_stages& names = framework::stages_of<triad_count_problem>();
  graph_ = &g.get();
  {
    scoped_stage stage(res, names.reset);
    state_->counts.fill(0);
    triad_cuda_open(res, state_->opt.kind, cuda_.emplace());
  }
  // triad_count.count (the full count) and triad_count.finalize (the read of the bins) inside.
  triad_cuda_compute(res, *cuda_, hypergraph_access::views(g.get()), state_->counts);
  triad_cuda_take_launches(*cuda_, state_->launches);
  cuda_.reset();
  framework::stamp_result(state_->version, state_->graph_state, *graph_);
}

// ------------------------------------------------------------------------------------------------
// compute(), update(), the participant
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename weight_t>
triad_count::result triad_count_compute(const resources& res,
                                        const hypergraph<vertex_t, weight_t>& h,
                                        const triad_count::options& opt) {
  using problem_t = triad_count_problem<vertex_t, weight_t>;
  constexpr const char* what = "triad_count::compute";
  expect_backend(res, what);
  expect_pattern(opt.kind, what);
  expect_engine_value(opt, what);
  if (res.get_backend() == backend::cuda) {
    expect_cuda_engine(opt, what);
  }
  hypergraph_access::expect_placement(res, h, what);
  DYNG_EXPECTS(!hypergraph_access::impl(h).poisoned, what,
               ": the hypergraph was poisoned by a failed apply(); rebuild it with "
               "from_hyperedges()");
  expect_requirements(h, opt.kind, what);
  const framework::hook_stages& names = framework::stages_of<problem_t>();
  scoped_stage stage(res, names.compute);
  auto st = std::make_unique<triad_count_state>();
  st->opt = opt;
  st->num_bins = triad_num_bins(opt.kind);
  problem_t p(*st);
  framework::context ctx(res, problem_t::name);
  framework::static_enactor<problem_t> enactor(p);
  (void)enactor.run(ctx, framework::new_view<hypergraph<vertex_t, weight_t>>(h));
  (void)checked_total(st->counts, st->num_bins, what);
  return triad_count_access::make(std::move(st));
}

template <typename vertex_t, typename weight_t>
triad_count::stats triad_count_update(const resources& res, hypergraph<vertex_t, weight_t>& h,
                                      const hyperedge_batch_view<vertex_t, weight_t>& batch,
                                      triad_count::result& r) {
  using problem_t = triad_count_problem<vertex_t, weight_t>;
  return framework::update_one<problem_t>(res, h, batch, r);
}

template <typename vertex_t, typename weight_t>
std::unique_ptr<hyper_update_participant<vertex_t, weight_t>> make_triad_count_participant(
    triad_count::result& r, triad_count::stats& out) {
  return framework::make_participant<triad_count_problem<vertex_t, weight_t>>(out, r);
}

// Explicit instantiations (PLAN Section 4.4.3: the hypergraph types).
#define DYNG_INSTANTIATE_TRIAD_COUNT(V, W)                                                     \
  template class triad_count_problem<V, W>;                                                    \
  template std::unique_ptr<hyper_update_participant<V, W>> make_triad_count_participant<V, W>( \
      triad_count::result&, triad_count::stats&);                                              \
  template triad_count::result triad_count_compute<V, W>(                                      \
      const resources&, const hypergraph<V, W>&, const triad_count::options&);                 \
  template triad_count::stats triad_count_update<V, W>(const resources&, hypergraph<V, W>&,    \
                                                       const hyperedge_batch_view<V, W>&,      \
                                                       triad_count::result&);
DYNG_FOR_EACH_HYPERGRAPH_TYPE(DYNG_INSTANTIATE_TRIAD_COUNT)
#undef DYNG_INSTANTIATE_TRIAD_COUNT

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
  return detail::checked_total(st.counts, st.num_bins, "triad_count::result::total");
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

result result::clone(const resources& res) const {
  const auto& st = detail::readable(impl_.get(), "triad_count::result::clone");
  detail::expect_backend(res, "triad_count::result::clone");
  return result(std::make_unique<detail::triad_count_state>(st));
}

}  // namespace dyng::triad_count
