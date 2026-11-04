// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fake_hyper_fixed_point.hpp
 * @brief A small fixed-point problem on the hypergraph (PLAN risk R3: a fixed-point hypergraph
 *        problem runs through update_enactor before hyper_sssp does; M11a step 4, ADR 0092).
 *
 * hyper_hops_problem keeps, for one source hyperedge, the hop count of every live hyperedge in the
 * line graph h2h (two hyperedges are adjacent when they share a vertex) and its canonical parent:
 * the lowest-id neighbour one hop closer (MOSP_ESCHER's tie rule). It is the shape of hyper_sssp
 * with unit node weights, written as Tier A hooks on the host (every backend reads the public h2h()
 * view and the applied record's pair delta, downloaded on cuda):
 *
 *   - identify_affected: the roots (the endpoint of a deleted pair whose tree edge it was; every
 *     new, recreated and dead id) and their subtrees in the old tree are invalidated;
 *   - seed: every invalidated live hyperedge pulls its best (hops, id) neighbour, and both
 *     directions of every inserted pair are relaxed; the improved ones form the frontier;
 *   - loop (one call per push round, until the frontier is empty): each frontier hyperedge offers
 *     (hops + 1, itself) to its neighbours, which adopt a smaller pair;
 *   - finalize: the counters.
 *
 * compute(): reset -> seed_static (the source at 0) -> the same loop. Knobs for the tests: a
 * convergence policy (the cap and on_limit), and an own work budget (over_budget, the hook
 * hyper_sssp's host engine uses) that is spent after a number of loop calls, or throws
 * convergence_error instead (an "error" budget action). The hypergraph must have slack_csr storage
 * (its applied record holds the pair delta; ADR 0086).
 */
#pragma once

#include "framework/composition.hpp"
#include "framework/context.hpp"
#include "framework/enactor.hpp"
#include "framework/policies.hpp"
#include "framework/problem.hpp"
#include "framework/views.hpp"
#include "hypergraph/hypergraph_impl.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stats.hpp>
#include <dyng/hypergraph/hyper_apply_summary.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/update.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dyng::test::framework_fakes {

namespace hfp = ::dyng::detail::framework;

/// "Unreachable" in a hops result.
inline constexpr std::int64_t hops_infinity = std::numeric_limits<std::int64_t>::max() / 4;

/// The result of the fake: hops and canonical parents by hyperedge id.
struct hyper_hops_result {
  std::int64_t source = 0;           ///< the source hyperedge
  std::vector<std::int64_t> hops;    ///< by id; hops_infinity for dead or unreachable ids
  std::vector<std::int64_t> parent;  ///< by id; -1 for none (and the source)
  std::uint64_t version = 0;         ///< hypergraph version matched
  std::uint64_t graph_state = 0;     ///< hypergraph state matched
  bool poisoned = false;             ///< a failed update left it unusable

  /// The hypergraph version this result matches.
  [[nodiscard]] std::uint64_t graph_version() const noexcept {
    return version;
  }
};

/// The stats of the fake.
struct hyper_hops_stats : update_stats {
  hyper_apply_summary batch;     ///< what the commit did (set by the participant adapter)
  std::int64_t invalidated = 0;  ///< hyperedges in invalidated subtrees
  std::int64_t pushes = 0;       ///< frontier hyperedges expanded by the loop
};

/// The knobs of the fake.
struct hyper_hops_knobs {
  hfp::convergence policy;      ///< the convergence policy of Step 2 (the cap and on_limit)
  int budget_after_loops = -1;  ///< over_budget() is true after this many loop calls (-1: never)
  bool budget_throws = false;   ///< over_budget() throws convergence_error instead
};

/// The frontier: a list of hyperedge ids in the problem (no allocation at construction).
struct hops_frontier {
  std::vector<std::int64_t>* ids = nullptr;  ///< the list (owned by the problem)
  /// @brief Converged when empty. @return Whether the list is empty.
  [[nodiscard]] bool empty() const noexcept {
    return ids == nullptr || ids->empty();
  }
};

/**
 * @brief The fixed-point fake (see the file comment).
 * @tparam container_t The hypergraph type.
 */
template <typename container_t>
class hyper_hops_problem final
    : public hfp::problem_base<hyper_hops_problem<container_t>, hfp::family::fixed_point> {
 public:
  static constexpr std::string_view name = "test_hyper_hops";              ///< stage prefix
  using container_type = container_t;                                      ///< the container
  using stats_type = hyper_hops_stats;                                     ///< the stats
  using frontier_type = hops_frontier;                                     ///< the frontier
  using vertex_type = typename container_t::vertex_type;                   ///< vertex ids
  using weight_type = typename container_t::weight_type;                   ///< weights
  using requested = hfp::hyper_requested_batch<vertex_type, weight_type>;  ///< before the commit
  using applied = hfp::hyper_applied_batch<vertex_type, weight_type>;      ///< after the commit

  /// A problem for `r` with the test's knobs.
  explicit hyper_hops_problem(hyper_hops_result& r, hyper_hops_knobs knobs = {})
      : r_(r), knobs_(knobs) {}

  // ---- lifecycle ----
  [[nodiscard]] const void* target() const noexcept {
    return &r_;
  }
  /// It reads the line graph: the commit keeps it.
  [[nodiscard]] bool reads_prepared_graph() const noexcept {
    return true;
  }
  void begin_update(hfp::context& /*ctx*/, hfp::old_view<container_t> g, const requested& batch) {
    DYNG_EXPECTS(!r_.poisoned, name, ": poisoned");
    DYNG_EXPECTS(batch.normalized != nullptr, name, ": no staged batch");
    DYNG_EXPECTS(g.get().properties().storage == hypergraph_storage::slack_csr, name,
                 ": the fake reads the pair delta of slack_csr storage");
    hfp::expect_current_result(name, r_.version, r_.graph_state, g.get());
  }
  void resume(hfp::context& ctx, hfp::new_view<container_t> g, const applied& /*a*/) {
    read_line_graph(ctx, g.get());
  }
  void end_update(hfp::context& /*ctx*/, hfp::new_view<container_t> g,
                  const hyper_hops_stats& /*s*/) {
    hfp::stamp_result(r_.version, r_.graph_state, g.get());
  }
  void poison() noexcept {
    r_.poisoned = true;
  }

  // ---- policies ----
  [[nodiscard]] hfp::convergence convergence_policy(hfp::context& /*ctx*/) const noexcept {
    return knobs_.policy;
  }
  /// The fake's own budget: spent after knobs.budget_after_loops loop calls.
  bool over_budget(hfp::context& /*ctx*/) {
    if (knobs_.budget_after_loops < 0 || loops_ < knobs_.budget_after_loops) return false;
    if (knobs_.budget_throws) {
      throw convergence_error("test_hyper_hops: the work budget is spent");
    }
    return true;
  }

  // ---- Step 1b ----
  void identify_affected(hfp::context& /*ctx*/, hfp::new_view<container_t> g, const applied& a,
                         hops_frontier& f) {
    const detail::hyper_applied& record = a.record;
    DYNG_EXPECTS(record.has_slack_record && record.has_line_graph_delta, name,
                 ": the applied record has no pair delta");
    const detail::hyper::line_graph_delta& pairs = detail::hypergraph_access::pairs(g.get());
    const std::size_t n = row_ptr_.size() - 1;
    grow(n);
    std::vector<std::uint8_t> inv(n, 0);
    for (std::size_t i = 0; i < pairs.deleted_a.size(); ++i) {
      const std::int64_t x = pairs.deleted_a[i];
      const std::int64_t y = pairs.deleted_b[i];
      if (r_.parent[static_cast<std::size_t>(y)] == x) inv[static_cast<std::size_t>(y)] = 1;
      if (r_.parent[static_cast<std::size_t>(x)] == y) inv[static_cast<std::size_t>(x)] = 1;
    }
    for (const auto* ids : {&record.new_ids, &record.recreated_ids, &record.dead_ids}) {
      for (const std::int64_t id : *ids) inv[static_cast<std::size_t>(id)] = 1;
    }
    // The subtrees of the roots in the tree before the batch.
    std::vector<std::vector<std::int64_t>> children(n);
    for (std::size_t v = 0; v < n; ++v) {
      if (r_.parent[v] >= 0)
        children[static_cast<std::size_t>(r_.parent[v])].push_back(static_cast<std::int64_t>(v));
    }
    std::vector<std::int64_t> stack;
    for (std::size_t v = 0; v < n; ++v) {
      if (inv[v]) stack.push_back(static_cast<std::int64_t>(v));
    }
    while (!stack.empty()) {
      const std::int64_t u = stack.back();
      stack.pop_back();
      for (const std::int64_t c : children[static_cast<std::size_t>(u)]) {
        if (!inv[static_cast<std::size_t>(c)]) {
          inv[static_cast<std::size_t>(c)] = 1;
          stack.push_back(c);
        }
      }
    }
    list_a_.clear();
    invalidated_ = 0;
    for (std::size_t v = 0; v < n; ++v) {
      if (!inv[v] || static_cast<std::int64_t>(v) == r_.source) continue;
      r_.hops[v] = hops_infinity;
      r_.parent[v] = -1;
      ++invalidated_;
      list_a_.push_back(static_cast<std::int64_t>(v));
    }
    for (std::size_t i = 0; i < pairs.inserted_a.size(); ++i) {
      inserted_.emplace_back(pairs.inserted_a[i], pairs.inserted_b[i]);
    }
    f.ids = &list_a_;
  }

  void seed(hfp::context& /*ctx*/, hfp::new_view<container_t> /*g*/, hops_frontier& f) {
    list_b_.clear();
    ++stamp_;
    for (const std::int64_t v : *f.ids) {
      std::int64_t best = hops_infinity;
      std::int64_t best_p = -1;
      for (std::int64_t e = row_ptr_[static_cast<std::size_t>(v)];
           e < row_ptr_[static_cast<std::size_t>(v) + 1]; ++e) {
        const std::int64_t p = col_ind_[static_cast<std::size_t>(e)];
        const std::int64_t d = r_.hops[static_cast<std::size_t>(p)];
        if (d >= hops_infinity) continue;
        if (d + 1 < best || (d + 1 == best && p < best_p)) {
          best = d + 1;
          best_p = p;
        }
      }
      if (best_p >= 0) offer(v, best, best_p);
    }
    for (const auto& [x, y] : inserted_) {
      if (r_.hops[static_cast<std::size_t>(x)] < hops_infinity)
        offer(y, r_.hops[static_cast<std::size_t>(x)] + 1, x);
      if (r_.hops[static_cast<std::size_t>(y)] < hops_infinity)
        offer(x, r_.hops[static_cast<std::size_t>(y)] + 1, y);
    }
    inserted_.clear();
    std::swap(list_a_, list_b_);
    f.ids = &list_a_;
  }

  // ---- Step 2: one push round per call ----
  void loop(hfp::context& /*ctx*/, hfp::new_view<container_t> /*g*/, hops_frontier& in,
            hops_frontier& out) {
    ++loops_;
    std::vector<std::int64_t>& next = in.ids == &list_a_ ? list_b_ : list_a_;
    next.clear();
    ++stamp_;
    out_list_ = &next;
    for (const std::int64_t u : *in.ids) {
      ++pushes_;
      const std::int64_t d = r_.hops[static_cast<std::size_t>(u)];
      for (std::int64_t e = row_ptr_[static_cast<std::size_t>(u)];
           e < row_ptr_[static_cast<std::size_t>(u) + 1]; ++e) {
        offer(col_ind_[static_cast<std::size_t>(e)], d + 1, u);
      }
    }
    out.ids = &next;
  }

  void recompute(hfp::context& ctx, hfp::new_view<container_t> g, hyper_hops_stats& s) {
    full_search(ctx, g.get());
    s.invalidated = invalidated_;
    s.iterations = loops_;
  }

  // ---- finish ----
  void finalize(hfp::context& /*ctx*/, hyper_hops_stats& s) {
    s.invalidated = invalidated_;
    s.iterations = loops_;
    s.pushes = pushes_;
    s.frontier_visits = pushes_;
    s.affected = invalidated_;
  }

  // ---- compute() ----
  void reset(hfp::context& /*ctx*/) {
    computing_ = true;
  }
  void seed_static(hfp::context& ctx, hfp::new_view<container_t> g, hops_frontier& f) {
    read_line_graph(ctx, g.get());
    const std::size_t n = row_ptr_.size() - 1;
    r_.hops.assign(n, hops_infinity);
    r_.parent.assign(n, -1);
    list_a_.clear();
    // The source must be live (its h2v row is not empty); the tests never delete it.
    const std::vector<std::int64_t> h2v_ptr = to_vector(ctx.res(), g.get().h2v().row_ptr);
    const auto s = static_cast<std::size_t>(r_.source);
    if (r_.source >= 0 && s < n && s + 1 < h2v_ptr.size() && h2v_ptr[s + 1] > h2v_ptr[s]) {
      r_.hops[static_cast<std::size_t>(r_.source)] = 0;
      list_a_.push_back(r_.source);
    }
    f.ids = &list_a_;
  }

 private:
  void read_line_graph(hfp::context& ctx, const container_t& h) {
    const auto view = h.h2h();
    row_ptr_ = to_vector(ctx.res(), view.row_ptr);
    col_ind_ = to_vector(ctx.res(), view.col_ind);
    if (row_ptr_.empty()) row_ptr_.assign(1, 0);
  }
  void grow(std::size_t n) {
    if (r_.hops.size() < n) r_.hops.resize(n, hops_infinity);
    if (r_.parent.size() < n) r_.parent.resize(n, -1);
    if (stamps_.size() < n) stamps_.resize(n, 0);
  }
  /// v adopts (d, p) if it is smaller than its own pair; then it joins the next frontier.
  void offer(std::int64_t v, std::int64_t d, std::int64_t p) {
    if (v == r_.source) return;
    const auto i = static_cast<std::size_t>(v);
    grow(i + 1);
    const std::int64_t cur_p =
        r_.parent[i] < 0 ? std::numeric_limits<std::int64_t>::max() : r_.parent[i];
    if (d < r_.hops[i] || (d == r_.hops[i] && p < cur_p)) {
      r_.hops[i] = d;
      r_.parent[i] = p;
      if (stamps_[i] != stamp_) {
        stamps_[i] = stamp_;
        (out_list_ != nullptr ? *out_list_ : list_b_).push_back(v);
      }
    }
  }
  void full_search(hfp::context& ctx, const container_t& h) {
    hops_frontier f;
    seed_static(ctx, hfp::new_view<container_t>(h), f);
    while (!f.empty()) {
      hops_frontier out;
      loop(ctx, hfp::new_view<container_t>(h), f, out);
      f = out;
    }
    out_list_ = nullptr;
  }

  hyper_hops_result& r_;
  hyper_hops_knobs knobs_;
  std::vector<std::int64_t> row_ptr_, col_ind_;
  std::vector<std::int64_t> list_a_, list_b_;
  std::vector<std::int64_t>* out_list_ = nullptr;
  std::vector<std::pair<std::int64_t, std::int64_t>> inserted_;
  std::vector<std::uint64_t> stamps_;
  std::uint64_t stamp_ = 0;
  std::int64_t invalidated_ = 0;
  std::int64_t pushes_ = 0;
  int loops_ = 0;
  bool computing_ = false;
};

/// compute() of the fake through the static enactor.
template <typename container_t>
hyper_hops_result compute_hops(const resources& res, const container_t& h, std::int64_t source) {
  hyper_hops_result r;
  r.source = source;
  hyper_hops_problem<container_t> p(r);
  hfp::context ctx(res, hyper_hops_problem<container_t>::name);
  hfp::static_enactor<hyper_hops_problem<container_t>> enactor(p);
  (void)enactor.run(ctx, hfp::new_view<container_t>(h));
  hfp::stamp_result(r.version, r.graph_state, h);
  return r;
}

/// update() of the fake (one result through run_update()).
template <typename container_t>
hyper_hops_stats update_hops(const resources& res, container_t& h,
                             const typename container_t::batch_view_type& batch,
                             hyper_hops_result& r, hyper_hops_knobs knobs = {}) {
  return hfp::update_one<hyper_hops_problem<container_t>>(res, h, batch, r, knobs);
}

}  // namespace dyng::test::framework_fakes

namespace dyng::detail {

/// dyng::update() support of the fixed-point fake.
template <>
struct update_traits<test::framework_fakes::hyper_hops_result> {
  using stats_type = test::framework_fakes::hyper_hops_stats;  ///< the stats type

  /// The participant (a hypergraph algorithm).
  template <typename container_t>
  static std::unique_ptr<typename participant_of<container_t>::type> make_participant(
      test::framework_fakes::hyper_hops_result& r, stats_type& out) {
    return framework::make_participant<test::framework_fakes::hyper_hops_problem<container_t>>(out,
                                                                                               r);
  }
};

}  // namespace dyng::detail
