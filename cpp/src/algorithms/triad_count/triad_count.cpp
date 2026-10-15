// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
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
 * engine (cuda.cu, engine/), Tier B; the host backends recount in this release (fallback_used).
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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

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
                            std::int64_t /*insertions*/,
                            framework::ownership::min_member /*rule*/) {
  no_cuda();
}
void triad_cuda_identify_affected(const resources& /*res*/, triad_cuda_engine& /*e*/,
                                  const hyper::row_views& /*views*/,
                                  const hyper_applied& /*record*/) {
  no_cuda();
}
void triad_cuda_count_plus(const resources& /*res*/, triad_cuda_engine& /*e*/,
                           const hyper::row_views& /*views*/,
                           framework::ownership::min_member /*rule*/) {
  no_cuda();
}
void triad_cuda_finalize(const resources& /*res*/, triad_cuda_engine& /*e*/,
                         triad_signed_bins& /*minus*/, triad_signed_bins& /*plus*/) {
  no_cuda();
}
void triad_cuda_take_launches(const triad_cuda_engine& /*e*/,
                              triad_launch_counters& /*into*/) noexcept {}
std::int64_t triad_cuda_host_syncs(triad_count::pattern /*kind*/) noexcept {
  return 0;
}
#endif

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
  before_ = state_->counts;
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

/// count(-) on G_t: cuda, the subtract phase with the frontier of Step 0's deletions (ESCHER-GPU's
/// subtract() on the snapshot before graph.update()); host, nothing (count(+) recounts).
template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::count(framework::context& ctx, old_graph g,
                                                    frontier_type& /*f*/, framework::sign s,
                                                    ownership_type rule) {
  (void)s;  // sign::minus: the engine subtracts on G_t
  if (ctx.chosen_engine() != engine::fused) {
    return;  // the host backends recount G_{t+1} in count(+) (M9a step 4: incremental)
  }
  const resources& res = ctx.res();
  triad_cuda_open(res, state_->opt.kind, cuda_.emplace());
  triad_cuda_count_minus(res, *cuda_, hypergraph_access::views(g.get()), normalized_->deletions,
                         inserted_, rule);
}

/// update(): count(+), the recount of G_{t+1} on the host backends; compute(): the full count.
template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::count(framework::context& ctx, new_graph g,
                                                    frontier_type& /*f*/, framework::sign s,
                                                    ownership_type /*rule*/) {
  (void)s;  // sign::plus
  graph_ = &g.get();
  const hyper::row_views views = hypergraph_access::views(g.get());
  if (ctx.get_backend() == backend::openmp) {
    triad_count_openmp(ctx.res(), state_->opt.kind, views, state_->counts);
  } else {
    triad_count_sequential(state_->opt.kind, views, state_->counts);
  }
}

template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::finalize(framework::context& /*ctx*/,
                                                       stats_type& stats) {
  if (computing_) {
    framework::stamp_result(state_->version, state_->graph_state, *graph_);
    return;
  }
  // The host backends recounted: the update removed every triad of G_t and added every triad of
  // G_{t+1} (stats::triads_removed / triads_added), and changed each bin by the difference.
  constexpr const char* what = "triad_count::update";
  const int n = state_->num_bins;
  stats.triads_removed = checked_total(before_, n, what);
  stats.triads_added = checked_total(state_->counts, n, what);
  std::int64_t affected = 0;
  for (int k = 0; k < n; ++k) {
    const auto i = static_cast<std::size_t>(k);
    stats.delta[i] = static_cast<std::int64_t>(state_->counts[i] - before_[i]);
    affected += state_->counts[i] != before_[i] ? 1 : 0;
  }
  stats.affected = affected;
  stats.fallback_used = true;  // a recount (PLAN Section 4.8, "start green"; M9a step 4)
  fill_counts(stats);
}

template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::fill_counts(stats_type& stats) const {
  stats.iterations = 0;
  stats.deleted_frontier = deleted_;
  stats.inserted_frontier = inserted_;
  stats.frontier_visits = deleted_ + inserted_;
}

template <typename vertex_t, typename weight_t>
void triad_count_problem<vertex_t, weight_t>::apply_delta(const triad_signed_bins& minus,
                                                          const triad_signed_bins& plus,
                                                          stats_type& stats) {
  constexpr const char* what = "triad_count::update";
  const int n = state_->num_bins;
  triad_bins removed{};
  triad_bins added{};
  // Check every bin first, so a failure writes nothing.
  for (int k = 0; k < triad_count::max_bins; ++k) {
    const auto i = static_cast<std::size_t>(k);
    if (minus[i] > 0 || plus[i] < 0 || (k >= n && (minus[i] != 0 || plus[i] != 0))) {
      DYNG_FAIL(what, ": bin ", k, " of the device engine holds ", minus[i], " (subtract) and ",
                plus[i], " (add), outside the pattern's signs or bins");
    }
    removed[i] = std::uint64_t{0} - static_cast<std::uint64_t>(minus[i]);  // -minus (<= 2^63)
    added[i] = static_cast<std::uint64_t>(plus[i]);
    const std::uint64_t grown = checked_add(state_->counts[i], added[i], what);
    if (grown < removed[i]) {
      DYNG_FAIL(what, " would make the count of bin ", k,
                " negative (the result does not match the hypergraph: ", state_->counts[i], " + ",
                added[i], " - ", removed[i], ")");
    }
  }
  stats.triads_removed = checked_total(removed, n, what);
  stats.triads_added = checked_total(added, n, what);
  std::int64_t affected = 0;
  for (int k = 0; k < n; ++k) {
    const auto i = static_cast<std::size_t>(k);
    state_->counts[i] = state_->counts[i] + added[i] - removed[i];
    stats.delta[i] = plus[i] + minus[i];
    affected += plus[i] + minus[i] != 0 ? 1 : 0;
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
    triad_cuda_identify_affected(res, *cuda_, views, applied.record);
  }
  {
    scoped_stage stage(res, names.count_plus);
    triad_cuda_count_plus(res, *cuda_, views, ownership_type{});
  }
  scoped_stage stage(res, names.finalize);
  triad_signed_bins minus{};
  triad_signed_bins plus{};
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
