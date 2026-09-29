// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fake_problems.hpp
 * @brief Two small problems for the tests of the framework itself, one per family:
 *
 *   - levels_problem (family::fixed_point): hop levels from a source (BFS). Its update invalidates
 *     every level in identify_affected and re-seeds the source (a correct, "start green" update),
 *     then Step 2 runs round by round through the enactor with a real frontier type, so the
 *     convergence rule, the iteration cap and the three on_limit policies are exercised.
 *   - pairs_problem (family::aggregate_delta): the number of reciprocal pairs {u -> v, v -> u}
 *     (2-cycles). P_{t+1} = P_t - pairs through deleted edges (counted on G_t) + pairs through
 *     inserted edges (counted on G_{t+1}), each pair attributed to its smallest-id changed edge
 *     (ownership::min_member), so a pair whose two edges change together is counted once.
 *
 * Both record every hook call with the graph version it saw (hook_log), and a few switches make
 * them misbehave on purpose (device errors, over-budget allocations, a view kept past the
 * commit).
 */
#pragma once

#include "core/budget_counters.hpp"
#include "framework/budgets.hpp"
#include "framework/composition.hpp"
#include "framework/context.hpp"
#include "framework/enactor.hpp"
#include "framework/policies.hpp"
#include "framework/problem.hpp"
#include "framework/views.hpp"
#include "framework/workspace.hpp"
#include "graph/graph_impl.hpp"
#include "util/device_error_flags.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stats.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dyng::test::framework_fakes {

namespace fw = ::dyng::detail::framework;

/// The graph type of the fakes: sorted rows without parallel edges, set semantics.
using graph_type = graph<std::int32_t, std::int64_t, unweighted>;
/// A directed edge.
using edge = std::pair<std::int32_t, std::int32_t>;

/// A graph on n vertices (graph_properties::cycle_enum_compatible(): sorted rows, as_sets).
inline graph_type make_graph(const resources& res, std::int32_t n, const std::vector<edge>& edges) {
  edge_list<std::int32_t, unweighted> list;
  list.num_vertices = n;
  for (const auto& [u, v] : edges) {
    list.add_edge(u, v);
  }
  return graph_type::from_edges(res, list.view(), graph_properties::cycle_enum_compatible());
}

/// A batch of deletions and insertions.
inline edge_batch<std::int32_t, unweighted> make_batch(const std::vector<edge>& deletions,
                                                       const std::vector<edge>& insertions) {
  edge_batch<std::int32_t, unweighted> b;
  for (const auto& [u, v] : deletions) {
    b.delete_edge(u, v);
  }
  for (const auto& [u, v] : insertions) {
    b.insert_edge(u, v);
  }
  return b;
}

/// Whether u -> v is an edge of g (sorted rows).
inline bool has_edge(const graph_type& g, std::int32_t u, std::int32_t v) {
  const auto out = detail::graph_access::out_view(g);
  if (u < 0 || u >= out.num_vertices()) {
    return false;
  }
  const std::int32_t* first = out.col_ind.data() + out.row_ptr[static_cast<std::size_t>(u)];
  const std::int32_t* last = out.col_ind.data() + out.row_ptr[static_cast<std::size_t>(u) + 1];
  return std::binary_search(first, last, v);
}

/// The hook calls of a run, as "hook@version".
struct hook_log {
  std::vector<std::string> calls;  ///< in call order

  /// Record one call.
  void add(std::string_view hook, std::uint64_t version) {
    calls.push_back(std::string(hook) + "@" + std::to_string(version));
  }
};

/// What a fake does wrong on purpose.
struct misbehaviour {
  std::uint32_t device_error_before = 0;  ///< raised in prepare / count(-) (before the commit)
  std::uint32_t device_error_after = 0;   ///< raised in seed / count(+) (after the commit)
  bool allocate_after_commit = false;     ///< finalize allocates through the handle's memory
  bool reserve_after_commit = false;      ///< ... and notes it as a reservation (a growth)
  bool keep_old_view = false;             ///< finalize reads the old_view of before_apply
};

/// Scratch of the fakes (pooled, as a real workspace).
struct fake_workspace final : detail::pooled_workspace {
  std::vector<std::int32_t> a;  ///< a frontier list
  std::vector<std::int32_t> b;  ///< the other frontier list

  [[nodiscard]] std::size_t bytes() const noexcept override {
    return (a.capacity() + b.capacity()) * sizeof(std::int32_t);
  }
};

// ------------------------------------------------------------------------------------------------
// levels_problem: family::fixed_point
// ------------------------------------------------------------------------------------------------

/// The result of the levels fake.
struct levels_result {
  std::int32_t source = 0;          ///< the source
  std::vector<std::int32_t> level;  ///< hops from the source, -1 if unreachable
  std::uint64_t version = 0;        ///< graph version matched
  std::uint64_t graph_state = 0;    ///< graph state matched
  bool poisoned = false;            ///< a failed update left it unusable
};

/// The stats of the levels fake.
struct levels_stats : update_stats {
  apply_summary batch;  ///< what the commit did (set by the participant adapter)
};

/// A frontier of vertex ids that lives in the workspace (the frontier object is a reference).
struct list_frontier {
  std::vector<std::int32_t>* items = nullptr;  ///< the list, or nullptr before the first bind

  /// Whether the frontier holds no vertex.
  [[nodiscard]] bool empty() const noexcept {
    return items == nullptr || items->empty();
  }
};

/// Options of the levels fake.
struct levels_options {
  fw::convergence policy;  ///< the convergence policy of Step 2
  bool fused = false;      ///< select the fused engine (enact_fused / compute_fused)
  fw::budget limit = fw::budget::unchecked();  ///< the budget of the algorithm phase
  misbehaviour bad;                            ///< deliberate faults
};

/// Hop levels from a source; see the file comment.
class levels_problem : public fw::problem_base<levels_problem, fw::family::fixed_point> {
 public:
  static constexpr std::string_view name = "test_levels";  ///< stage prefix
  using container_type = graph_type;                       ///< the container
  using stats_type = levels_stats;                         ///< the stats
  using frontier_type = list_frontier;                     ///< the frontier

  /// A problem for `r`.
  levels_problem(levels_result& r, hook_log& log, levels_options opt = {})
      : r_(r), log_(log), opt_(opt) {}

  // ---- lifecycle ----
  [[nodiscard]] const void* target() const noexcept {
    return &r_;
  }
  void begin_update(fw::context& /*ctx*/, fw::old_view<graph_type> g,
                    const fw::requested_batch<std::int32_t, unweighted>& /*batch*/) {
    log_.add("begin_update", g.version());
    DYNG_EXPECTS(!r_.poisoned, "test_levels: poisoned");
    fw::expect_current_result("test_levels::update", r_.version, r_.graph_state, g.get());
  }
  void resume(fw::context& ctx, fw::new_view<graph_type> g,
              const fw::applied_batch<std::int32_t>& /*applied*/) {
    log_.add("resume", g.version());
    ws_.emplace(ctx.lease_workspace<fake_workspace>());
    r_.level.resize(static_cast<std::size_t>(g->num_vertices()), -1);
    before_ = r_.level;
  }
  void end_update(fw::context& /*ctx*/, fw::new_view<graph_type> g, const levels_stats& /*s*/) {
    log_.add("end_update", g.version());
    fw::stamp_result(r_.version, r_.graph_state, g.get());
    ws_.reset();
  }
  void poison() noexcept {
    r_.poisoned = true;
  }
  [[nodiscard]] bool reads_prepared_graph() const noexcept {
    return false;  // out-edges only
  }

  // ---- policies ----
  [[nodiscard]] engine select_engine(fw::context& /*ctx*/) const noexcept {
    return opt_.fused ? engine::fused : engine::operators;
  }
  [[nodiscard]] fw::convergence convergence_policy(fw::context& /*ctx*/) const noexcept {
    return opt_.policy;
  }
  [[nodiscard]] fw::budget algorithm_budget(fw::context& /*ctx*/) const noexcept {
    return opt_.limit;
  }

  // ---- Step 0 / 1a ----
  void prepare(fw::context& ctx, fw::old_view<graph_type> g,
               const fw::requested_batch<std::int32_t, unweighted>& /*batch*/) {
    log_.add("prepare", g.version());
    ctx.raise_device_error(opt_.bad.device_error_before, "prepare misbehaved");
  }
  void before_apply(fw::context& /*ctx*/, fw::old_view<graph_type> g,
                    const fw::requested_batch<std::int32_t, unweighted>& /*batch*/,
                    list_frontier& /*f*/) {
    log_.add("before_apply", g.version());
    old_.emplace(g);
  }

  // ---- Step 1b ----
  void identify_affected(fw::context& /*ctx*/, fw::new_view<graph_type> g,
                         const fw::applied_batch<std::int32_t>& /*applied*/, list_frontier& f) {
    log_.add("identify_affected", g.version());
    // Invalidate everything (correct for any batch); seed re-roots the search.
    std::fill(r_.level.begin(), r_.level.end(), -1);
    f.items = &(*ws_)->a;
    f.items->clear();
  }
  void seed(fw::context& ctx, fw::new_view<graph_type> g, list_frontier& f) {
    log_.add("seed", g.version());
    ctx.raise_device_error(opt_.bad.device_error_after, "seed misbehaved");
    r_.level[static_cast<std::size_t>(r_.source)] = 0;
    f.items->push_back(r_.source);
  }

  // ---- Step 2 ----
  void loop(fw::context& /*ctx*/, fw::new_view<graph_type> g, list_frontier& in,
            list_frontier& out) {
    log_.add("loop", g.version());
    ++rounds_;
    fake_workspace& ws = scratch();
    out.items = in.items == &ws.a ? &ws.b : &ws.a;
    out.items->clear();
    const auto csr = detail::graph_access::out_view(g.get());
    for (const std::int32_t u : *in.items) {
      for (auto e = csr.row_ptr[static_cast<std::size_t>(u)];
           e < csr.row_ptr[static_cast<std::size_t>(u) + 1]; ++e) {
        const std::int32_t v = csr.col_ind[static_cast<std::size_t>(e)];
        if (r_.level[static_cast<std::size_t>(v)] < 0) {
          r_.level[static_cast<std::size_t>(v)] = r_.level[static_cast<std::size_t>(u)] + 1;
          out.items->push_back(v);
        }
      }
    }
  }
  void recompute(fw::context& ctx, fw::new_view<graph_type> g, levels_stats& s) {
    log_.add("recompute", g.version());
    list_frontier in;
    list_frontier out;
    identify_affected(ctx, g,
                      fw::applied_batch<std::int32_t>{dummy_summary_, dummy_delta_, nullptr}, in);
    seed(ctx, g, in);
    while (!in.empty()) {
      loop(ctx, g, in, out);
      std::swap(in, out);
    }
    finalize(ctx, s);
  }

  // ---- Tier B ----
  void enact_fused(fw::context& ctx, fw::new_view<graph_type> g,
                   const fw::applied_batch<std::int32_t>& /*applied*/, levels_stats& s) {
    log_.add("enact_fused", g.version());
    full_search(ctx, g);
    s.affected = changed();
  }
  void compute_fused(fw::context& ctx, fw::new_view<graph_type> g, levels_stats& /*s*/) {
    log_.add("compute_fused", g.version());
    full_search(ctx, g);
  }

  // ---- finish ----
  void finalize(fw::context& ctx, levels_stats& s) {
    log_.add("finalize", 0);
    if (opt_.bad.reserve_after_commit) {
      detail::note_reservation();  // a deliberate growth: the run is a reserving one
    }
    if (opt_.bad.allocate_after_commit) {
      const buffer<int> scratch(ctx.res(), 16);  // an allocation in the algorithm phase
      (void)scratch;
    }
    if (opt_.bad.keep_old_view && old_) {
      (void)old_->get().num_vertices();  // I1: G_t after the commit
    }
    s.iterations = rounds_;
    s.affected = changed();
  }

  // ---- compute() ----
  void reset(fw::context& ctx) {
    log_.add("reset", 0);
    compute_ws_.emplace(ctx.lease_workspace<fake_workspace>());
    std::fill(r_.level.begin(), r_.level.end(), -1);
    before_ = r_.level;
  }
  void seed_static(fw::context& /*ctx*/, fw::new_view<graph_type> g, list_frontier& f) {
    log_.add("seed_static", g.version());
    f.items = &(*compute_ws_)->a;
    f.items->assign(1, r_.source);
    r_.level[static_cast<std::size_t>(r_.source)] = 0;
  }

  /// Loop calls so far.
  [[nodiscard]] std::int64_t rounds() const noexcept {
    return rounds_;
  }

 private:
  fake_workspace& scratch() {
    return ws_ ? ws_->get() : compute_ws_->get();
  }

  void full_search(fw::context& ctx, fw::new_view<graph_type> g) {
    if (!ws_ && !compute_ws_) {
      compute_ws_.emplace(ctx.lease_workspace<fake_workspace>());
    }
    fake_workspace& ws = scratch();
    std::fill(r_.level.begin(), r_.level.end(), -1);
    list_frontier in{&ws.a};
    list_frontier out;
    ws.a.assign(1, r_.source);
    r_.level[static_cast<std::size_t>(r_.source)] = 0;
    while (!in.empty()) {
      loop(ctx, g, in, out);
      std::swap(in, out);
    }
  }

  [[nodiscard]] std::int64_t changed() const {
    std::int64_t n = 0;
    for (std::size_t v = 0; v < r_.level.size(); ++v) {
      n += (v >= before_.size() || before_[v] != r_.level[v]) ? 1 : 0;
    }
    return n;
  }

  levels_result& r_;
  hook_log& log_;
  levels_options opt_;
  std::vector<std::int32_t> before_;
  std::optional<detail::workspace_pool::lease<fake_workspace>> ws_;
  std::optional<detail::workspace_pool::lease<fake_workspace>> compute_ws_;
  std::optional<fw::old_view<graph_type>> old_;
  std::int64_t rounds_ = 0;
  apply_summary dummy_summary_;
  detail::apply_delta<std::int32_t> dummy_delta_;
};

/// compute() of the levels fake through the static enactor.
inline levels_result compute_levels(const resources& res, const graph_type& g, std::int32_t source,
                                    hook_log& log, levels_options opt = {}) {
  levels_result r;
  r.source = source;
  r.level.assign(static_cast<std::size_t>(g.num_vertices()), -1);
  levels_problem p(r, log, opt);
  fw::context ctx(res, levels_problem::name);
  fw::static_enactor<levels_problem> enactor(p);
  (void)enactor.run(ctx, fw::new_view<graph_type>(g));
  fw::stamp_result(r.version, r.graph_state, g);
  return r;
}

// ------------------------------------------------------------------------------------------------
// pairs_problem: family::aggregate_delta
// ------------------------------------------------------------------------------------------------

/// The result of the pairs fake.
struct pairs_result {
  std::int64_t pairs = 0;         ///< reciprocal pairs {u -> v, v -> u}
  std::uint64_t version = 0;      ///< graph version matched
  std::uint64_t graph_state = 0;  ///< graph state matched
  bool poisoned = false;          ///< a failed update left it unusable
};

/// The stats of the pairs fake.
struct pairs_stats : update_stats {
  apply_summary batch;       ///< what the commit did
  std::int64_t removed = 0;  ///< pairs through deleted edges (on G_t)
  std::int64_t added = 0;    ///< pairs through inserted edges (on G_{t+1})
};

/// Reciprocal pairs; see the file comment.
class pairs_problem : public fw::problem_base<pairs_problem, fw::family::aggregate_delta> {
 public:
  static constexpr std::string_view name = "test_pairs";  ///< stage prefix
  using container_type = graph_type;                      ///< the container
  using stats_type = pairs_stats;                         ///< the stats
  using ownership_type = fw::ownership::min_member;       ///< exactly-once rule (I2)

  /// A problem for `r` (update or compute).
  pairs_problem(pairs_result& r, hook_log& log, misbehaviour bad = {})
      : r_(r), log_(log), bad_(bad) {}

  // ---- lifecycle ----
  [[nodiscard]] const void* target() const noexcept {
    return &r_;
  }
  void begin_update(fw::context& /*ctx*/, fw::old_view<graph_type> g,
                    const fw::requested_batch<std::int32_t, unweighted>& /*batch*/) {
    log_.add("begin_update", g.version());
    DYNG_EXPECTS(!r_.poisoned, "test_pairs: poisoned");
    fw::expect_current_result("test_pairs::update", r_.version, r_.graph_state, g.get());
  }
  void end_update(fw::context& /*ctx*/, fw::new_view<graph_type> g, const pairs_stats& /*s*/) {
    log_.add("end_update", g.version());
    fw::stamp_result(r_.version, r_.graph_state, g.get());
  }
  void poison() noexcept {
    r_.poisoned = true;
  }
  [[nodiscard]] bool reads_prepared_graph() const noexcept {
    return false;
  }

  // ---- Step 0: the change lists (the framework's normalized batch under as_sets) ----
  void normalize(fw::context& /*ctx*/, fw::old_view<graph_type> g,
                 const fw::requested_batch<std::int32_t, unweighted>& batch) {
    log_.add("normalize", g.version());
    DYNG_EXPECTS(batch.normalized != nullptr, "test_pairs needs batch_semantics::as_sets");
    deletions_.clear();
    insertions_.clear();
    for (const auto& c : batch.normalized->deletions) {
      deletions_.emplace_back(c.source, c.target);
    }
    for (const auto& c : batch.normalized->insertions) {
      insertions_.emplace_back(c.source, c.target);
    }
  }

  // ---- count(-) on G_t, count(+) on G_{t+1} (and the full count of compute()) ----
  void count(fw::context& ctx, fw::old_view<graph_type> g, fw::internal_frontier& /*f*/, fw::sign s,
             fw::ownership::min_member /*rule*/) {
    log_.add("count_minus", g.version());
    DYNG_EXPECTS(s == fw::sign::minus, "test_pairs: count on G_t subtracts");
    ctx.raise_device_error(bad_.device_error_before, "count(-) misbehaved");
    removed_ = owned_pairs(g.get(), deletions_);
  }
  void count(fw::context& ctx, fw::new_view<graph_type> g, fw::internal_frontier& /*f*/, fw::sign s,
             fw::ownership::min_member /*rule*/) {
    DYNG_EXPECTS(s == fw::sign::plus, "test_pairs: count on G_{t+1} adds");
    if (computing_) {
      log_.add("count", g.version());
      r_.pairs = all_pairs(g.get());
      return;
    }
    log_.add("count_plus", g.version());
    ctx.raise_device_error(bad_.device_error_after, "count(+) misbehaved");
    added_ = owned_pairs(g.get(), insertions_);
  }

  // ---- Step 1b and finish ----
  void identify_affected(fw::context& /*ctx*/, fw::new_view<graph_type> g,
                         const fw::applied_batch<std::int32_t>& applied,
                         fw::internal_frontier& /*f*/) {
    log_.add("identify_affected", g.version());
    DYNG_EXPECTS(applied.delta.insert_src.size() == insertions_.size(),
                 "test_pairs: the commit's insertions differ from Step 0");
  }
  void finalize(fw::context& /*ctx*/, pairs_stats& s) {
    log_.add("finalize", 0);
    if (computing_) {
      return;
    }
    r_.pairs = r_.pairs - removed_ + added_;
    s.removed = removed_;
    s.added = added_;
    s.affected = removed_ != added_ ? 1 : 0;
  }

  // ---- compute() ----
  void reset(fw::context& /*ctx*/) {
    log_.add("reset", 0);
    computing_ = true;
    r_.pairs = 0;
  }

 private:
  /// Pairs of `g` through the listed edges, each attributed to its smallest-id listed edge.
  static std::int64_t owned_pairs(const graph_type& g, const std::vector<edge>& changes) {
    std::int64_t n = 0;
    for (std::size_t id = 0; id < changes.size(); ++id) {
      const auto [u, v] = changes[id];
      if (u == v || !has_edge(g, v, u)) {
        continue;
      }
      // The reverse edge is also a change of this phase: the smaller id owns the pair.
      const auto reverse = std::lower_bound(changes.begin(), changes.end(), edge{v, u});
      const bool reverse_listed = reverse != changes.end() && *reverse == edge{v, u};
      if (reverse_listed && static_cast<std::size_t>(reverse - changes.begin()) < id) {
        continue;
      }
      ++n;
    }
    return n;
  }

  /// Every pair of `g` (the oracle of compute()).
  static std::int64_t all_pairs(const graph_type& g) {
    const auto csr = detail::graph_access::out_view(g);
    std::int64_t n = 0;
    for (std::int32_t u = 0; u < csr.num_vertices(); ++u) {
      for (auto e = csr.row_ptr[static_cast<std::size_t>(u)];
           e < csr.row_ptr[static_cast<std::size_t>(u) + 1]; ++e) {
        const std::int32_t v = csr.col_ind[static_cast<std::size_t>(e)];
        n += (u < v && has_edge(g, v, u)) ? 1 : 0;
      }
    }
    return n;
  }

  pairs_result& r_;
  hook_log& log_;
  misbehaviour bad_;
  bool computing_ = false;
  std::vector<edge> deletions_;
  std::vector<edge> insertions_;
  std::int64_t removed_ = 0;
  std::int64_t added_ = 0;
};

/// compute() of the pairs fake through the static enactor.
inline pairs_result compute_pairs(const resources& res, const graph_type& g, hook_log& log) {
  pairs_result r;
  pairs_problem p(r, log);
  fw::context ctx(res, pairs_problem::name);
  fw::static_enactor<pairs_problem> enactor(p);
  (void)enactor.run(ctx, fw::new_view<graph_type>(g));
  fw::stamp_result(r.version, r.graph_state, g);
  return r;
}

}  // namespace dyng::test::framework_fakes
