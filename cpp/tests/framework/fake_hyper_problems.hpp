// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fake_hyper_problems.hpp
 * @brief Two small aggregate-delta problems on the hypergraph, for the tests of the hypergraph
 *        update path and of the conformance kit's hypergraph axis (ADR 0043), in the style of
 *        fake_problems.hpp:
 *
 *   - hyper_pairs_problem: the number of overlapping pairs {a, b} of live hyperedges (the edges of
 *     the line graph h2h). P_{t+1} = P_t - the pairs through the deleted ids, counted on the OLD
 *     views (G_t) with frontier = the deleted ids in before_apply, + the pairs through the
 *     inserted ids, counted on the NEW views (G_{t+1}) with frontier = hyper_apply_summary's
 *     inserted_ids (read on the device on CUDA: the applied record's device copy). A pair whose two
 *     hyperedges change in the same phase is counted by its smaller id (ownership::min_member).
 *   - hyper_squares_problem: the sum of |e|^2 over the live hyperedges, the same way on h2v (the
 *     partner of check C10: two problems on one container).
 *
 * Both count on the internal row views (detail::hypergraph_access::views(): ESCHER-GPU's
 * key-indexed rows, host memory on the host backends and device memory on CUDA, where the counts
 * run as kernels: fake_hyper_kernels.cu). Their compute() counts on the public views (h2h(), h2v())
 * and the kit's independent oracle rebuilds the line graph (testing::rebuild_line_graph). A few
 * switches make them misbehave on purpose (an old view used after the commit, a failing count(+)).
 *
 * The planted bug of the kit (M8b criterion 11; parity/mutate.py, suite `kit`): with
 * DYNG_MUTATION_KIT_FRONTIER_REQUESTED_IDS the frontier of count(+) is the ids the request would
 * get without reuse (id_bound(G_t) + i) instead of inserted_ids, and the kit must fail.
 */
#pragma once

#include "core/budget_counters.hpp"
#include "framework/budgets.hpp"
#include "framework/composition.hpp"
#include "framework/context.hpp"
#include "framework/enactor.hpp"
#include "framework/fake_hyper_counts.hpp"
#include "framework/policies.hpp"
#include "framework/problem.hpp"
#include "framework/scratch_buffer.hpp"
#include "framework/views.hpp"
#include "framework/workspace.hpp"
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/row_views.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stats.hpp>
#include <dyng/core/types.hpp>
#include <dyng/hypergraph/hyper_apply_summary.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/update.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace dyng::test::framework_fakes {

namespace hfw = ::dyng::detail::framework;

/// The result of a fake: one aggregate.
struct hyper_count_result {
  std::int64_t value = 0;         ///< the aggregate
  std::uint64_t version = 0;      ///< hypergraph version matched
  std::uint64_t graph_state = 0;  ///< hypergraph state matched
  bool poisoned = false;          ///< a failed update left it unusable

  /// The hypergraph version this result matches.
  [[nodiscard]] std::uint64_t graph_version() const noexcept {
    return version;
  }
};

/// The result of hyper_pairs_problem.
struct hyper_pairs_result : hyper_count_result {};
/// The result of hyper_squares_problem.
struct hyper_squares_result : hyper_count_result {};

/// The stats of a fake.
struct hyper_count_stats : update_stats {
  hyper_apply_summary batch;  ///< what the commit did (set by the participant adapter)
  std::int64_t removed = 0;   ///< counted on G_t through the deleted ids
  std::int64_t added = 0;     ///< counted on G_{t+1} through the inserted ids
};

/// What a fake does wrong on purpose.
struct hyper_misbehaviour {
  bool keep_old_view = false;  ///< finalize reads the old_view of before_apply (I1)
  bool fail_after = false;     ///< count(+) throws (the participant is poisoned)
};

/**
 * @brief The shared body of the two fakes (see the file comment).
 * @tparam derived_t     The problem.
 * @tparam container_t   The hypergraph type.
 * @tparam result_t      Its result type.
 */
template <typename derived_t, typename container_t, typename result_t>
class hyper_count_problem : public hfw::problem_base<derived_t, hfw::family::aggregate_delta> {
 public:
  using container_type = container_t;                     ///< the container
  using stats_type = hyper_count_stats;                   ///< the stats
  using ownership_type = hfw::ownership::min_member;      ///< exactly-once rule (I2)
  using frontier_type = id_frontier;                      ///< the frontier
  using vertex_type = typename container_t::vertex_type;  ///< vertex ids
  using weight_type = typename container_t::weight_type;  ///< weights
  using requested = hfw::hyper_requested_batch<vertex_type, weight_type>;  ///< before the commit
  using applied = hfw::hyper_applied_batch<vertex_type, weight_type>;      ///< after the commit

  /// A problem for `r`.
  explicit hyper_count_problem(result_t& r, hyper_misbehaviour bad = {}) : r_(r), bad_(bad) {}

  // ---- lifecycle ----
  [[nodiscard]] const void* target() const noexcept {
    return &r_;
  }
  void begin_update(hfw::context& ctx, hfw::old_view<container_t> g, const requested& batch) {
    DYNG_EXPECTS(!r_.poisoned, derived_t::name, ": poisoned");
    DYNG_EXPECTS(batch.normalized != nullptr, derived_t::name, ": no staged batch");
    hfw::expect_current_result(derived_t::name, r_.version, r_.graph_state, g.get());
    ws_.emplace(ctx.lease_workspace<fake_hyper_workspace>());
  }
  void end_update(hfw::context& /*ctx*/, hfw::new_view<container_t> g,
                  const hyper_count_stats& /*s*/) {
    hfw::stamp_result(r_.version, r_.graph_state, g.get());
    ws_.reset();
  }
  void poison() noexcept {
    r_.poisoned = true;
  }

  // ---- policies ----
  [[nodiscard]] hfw::budget algorithm_budget(hfw::context& ctx) const noexcept {
    return hfw::budget::steady_state(ctx.on_cuda() ? 2 : 0);  // one read-back per count
  }

  // ---- Step 1a on G_t: frontier = the deleted ids; count(-) on the old views ----
  void before_apply(hfw::context& /*ctx*/, hfw::old_view<container_t> g, const requested& batch,
                    id_frontier& f) {
    f.ids = batch.normalized->deletions.data();
    f.size = static_cast<std::int64_t>(batch.normalized->deletions.size());
    f.on_device = false;
    old_.emplace(g);
  }
  void count(hfw::context& ctx, hfw::old_view<container_t> g, id_frontier& f, hfw::sign s,
             hfw::ownership::min_member /*rule*/) {
    DYNG_EXPECTS(s == hfw::sign::minus, derived_t::name, ": count on G_t subtracts");
    removed_ = self().count_rows(ctx, detail::hypergraph_access::views(g.get()), f);
  }

  // ---- Step 1b on G_{t+1}: frontier = the inserted ids; count(+) on the new views ----
  void identify_affected(hfw::context& ctx, hfw::new_view<container_t> g, const applied& a,
                         id_frontier& f) {
    (void)g;
    const detail::hyper_applied& record = a.record;
    DYNG_EXPECTS(record.inserted_ids == a.summary.inserted_ids, derived_t::name,
                 ": the applied record and the summary disagree");
    f.size = static_cast<std::int64_t>(record.inserted_ids.size());
    if (ctx.on_cuda() && f.size > 0) {
      f.ids = record.inserted_ids_device.data();  // read on the device
      f.on_device = true;
    } else {
      f.ids = record.inserted_ids.data();
      f.on_device = false;
    }
#if defined(DYNG_MUTATION_KIT_FRONTIER_REQUESTED_IDS)
    // MUTATION (M8b criterion 11): the frontier is the ids the request would get without reuse
    // (after the bound of G_t), not the ids the store chose (inserted_ids).
    fake_hyper_workspace& ws = ws_->get();
    ws.host_ids.clear();
    for (std::int64_t i = 0; i < f.size; ++i) {
      ws.host_ids.push_back(a.normalized->id_bound + i);
    }
    f.ids = ws.host_ids.data();
    f.on_device = false;
#endif
  }
  void count(hfw::context& ctx, hfw::new_view<container_t> g, id_frontier& f, hfw::sign s,
             hfw::ownership::min_member /*rule*/) {
    DYNG_EXPECTS(s == hfw::sign::plus, derived_t::name, ": count on G_{t+1} adds");
    if (computing_) {
      r_.value = self().full_count(ctx, g.get());
      return;
    }
    if (bad_.fail_after) {
      throw internal_error("test: count(+) misbehaved");
    }
    added_ = self().count_rows(ctx, detail::hypergraph_access::views(g.get()), f);
  }

  // ---- finish ----
  void finalize(hfw::context& /*ctx*/, hyper_count_stats& s) {
    if (computing_) return;
    if (bad_.keep_old_view && old_) {
      (void)old_->get().num_hyperedges();  // I1: G_t after the commit
    }
    r_.value = r_.value - removed_ + added_;
    s.removed = removed_;
    s.added = added_;
    s.affected = s.batch.deleted_hyperedges + s.batch.inserted_hyperedges;
  }

  // ---- compute() ----
  void reset(hfw::context& /*ctx*/) {
    computing_ = true;
    r_.value = 0;
  }

 protected:
  /// The count over the frontier's rows on this backend.
  template <typename count_host_t, typename count_device_t>
  std::int64_t dispatch(hfw::context& ctx, const id_frontier& f, count_host_t&& on_host,
                        count_device_t&& on_device) {
    if (f.empty()) return 0;
    if (ctx.on_cuda()) {
      return on_device(ws_->get());
    }
    return on_host(ws_->get());
  }

  result_t& r_;

 private:
  derived_t& self() noexcept {
    return static_cast<derived_t&>(*this);
  }

  hyper_misbehaviour bad_;
  bool computing_ = false;
  std::int64_t removed_ = 0;
  std::int64_t added_ = 0;
  std::optional<detail::workspace_pool::lease<fake_hyper_workspace>> ws_;
  std::optional<hfw::old_view<container_t>> old_;
};

/// The overlapping pairs of live hyperedges; see the file comment.
template <typename container_t>
class hyper_pairs_problem final : public hyper_count_problem<hyper_pairs_problem<container_t>,
                                                             container_t, hyper_pairs_result> {
  using base =
      hyper_count_problem<hyper_pairs_problem<container_t>, container_t, hyper_pairs_result>;

 public:
  static constexpr std::string_view name = "test_hyper_pairs";  ///< stage prefix
  using base::base;

  /// It reads the line graph: run_update() keeps it (line_graph_policy::automatic builds it).
  [[nodiscard]] bool reads_prepared_graph() const noexcept {
    return true;
  }

  /// The owned pairs through the frontier's rows of h2h.
  std::int64_t count_rows(hfw::context& ctx, const detail::hyper::row_views& v,
                          const id_frontier& f) {
    DYNG_EXPECTS(v.has_h2h, name, ": the line graph is not maintained");
    return this->dispatch(
        ctx, f, [&](fake_hyper_workspace& ws) { return host_owned_pairs(v.h2h, f, ws); },
        [&](fake_hyper_workspace& ws) -> std::int64_t {
#if defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA
          return device_owned_pairs(ctx.res(), v.h2h, f, ws);
#else
          (void)ws;
          throw not_supported_error("test_hyper_pairs: cuda is not in this test executable");
#endif
        });
  }

  /// compute(): every pair, from the public line graph (half its entries).
  std::int64_t full_count(hfw::context& ctx, const container_t& h) {
    const auto view = h.h2h();
    const std::vector<std::int64_t> row_ptr = to_vector(ctx.res(), view.row_ptr);
    return row_ptr.empty() ? 0 : row_ptr.back() / 2;
  }
};

/// The sum of |e|^2 over the live hyperedges; see the file comment.
template <typename container_t>
class hyper_squares_problem final : public hyper_count_problem<hyper_squares_problem<container_t>,
                                                               container_t, hyper_squares_result> {
  using base =
      hyper_count_problem<hyper_squares_problem<container_t>, container_t, hyper_squares_result>;

 public:
  static constexpr std::string_view name = "test_hyper_squares";  ///< stage prefix
  using base::base;

  /// The squared lengths of the frontier's rows of h2v.
  std::int64_t count_rows(hfw::context& ctx, const detail::hyper::row_views& v,
                          const id_frontier& f) {
    return this->dispatch(
        ctx, f, [&](fake_hyper_workspace& /*ws*/) { return host_squares(v.h2v, f); },
        [&](fake_hyper_workspace& ws) -> std::int64_t {
#if defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA
          return device_squares(ctx.res(), v.h2v, f, ws);
#else
          (void)ws;
          throw not_supported_error("test_hyper_squares: cuda is not in this test executable");
#endif
        });
  }

  /// compute(): from the public h2v view.
  std::int64_t full_count(hfw::context& ctx, const container_t& h) {
    const auto view = h.h2v();
    const std::vector<std::int64_t> row_ptr = to_vector(ctx.res(), view.row_ptr);
    std::int64_t sum = 0;
    for (std::size_t i = 0; i + 1 < row_ptr.size(); ++i) {
      const std::int64_t len = row_ptr[i + 1] - row_ptr[i];
      sum += len * len;
    }
    return sum;
  }
};

/// compute() of a fake through the static enactor.
template <typename problem_t, typename result_t>
result_t compute_hyper(const resources& res, const typename problem_t::container_type& h) {
  result_t r;
  problem_t p(r);
  hfw::context ctx(res, problem_t::name);
  hfw::static_enactor<problem_t> enactor(p);
  (void)enactor.run(ctx, hfw::new_view<typename problem_t::container_type>(h));
  hfw::stamp_result(r.version, r.graph_state, h);
  return r;
}

/// The overlapping pairs of `h`, computed.
template <typename container_t>
hyper_pairs_result compute_pairs(const resources& res, const container_t& h) {
  return compute_hyper<hyper_pairs_problem<container_t>, hyper_pairs_result>(res, h);
}

/// The sum of |e|^2 of `h`, computed.
template <typename container_t>
hyper_squares_result compute_squares(const resources& res, const container_t& h) {
  return compute_hyper<hyper_squares_problem<container_t>, hyper_squares_result>(res, h);
}

/// update() of the pairs fake (one result through run_update()).
template <typename container_t>
hyper_count_stats update_pairs(const resources& res, container_t& h,
                               const typename container_t::batch_view_type& batch,
                               hyper_pairs_result& r, hyper_misbehaviour bad = {}) {
  return hfw::update_one<hyper_pairs_problem<container_t>>(res, h, batch, r, bad);
}

/// update() of the squares fake.
template <typename container_t>
hyper_count_stats update_squares(const resources& res, container_t& h,
                                 const typename container_t::batch_view_type& batch,
                                 hyper_squares_result& r, hyper_misbehaviour bad = {}) {
  return hfw::update_one<hyper_squares_problem<container_t>>(res, h, batch, r, bad);
}

}  // namespace dyng::test::framework_fakes

namespace dyng::detail {

/// dyng::update() support of the pairs fake.
template <>
struct update_traits<test::framework_fakes::hyper_pairs_result> {
  using stats_type = test::framework_fakes::hyper_count_stats;  ///< the stats type

  /// The participant (a hypergraph algorithm: the container must be a hypergraph).
  template <typename container_t>
  static std::unique_ptr<typename participant_of<container_t>::type> make_participant(
      test::framework_fakes::hyper_pairs_result& r, stats_type& out) {
    return framework::make_participant<test::framework_fakes::hyper_pairs_problem<container_t>>(out,
                                                                                                r);
  }
};

/// dyng::update() support of the squares fake.
template <>
struct update_traits<test::framework_fakes::hyper_squares_result> {
  using stats_type = test::framework_fakes::hyper_count_stats;  ///< the stats type

  /// The participant.
  template <typename container_t>
  static std::unique_ptr<typename participant_of<container_t>::type> make_participant(
      test::framework_fakes::hyper_squares_result& r, stats_type& out) {
    return framework::make_participant<test::framework_fakes::hyper_squares_problem<container_t>>(
        out, r);
  }
};

}  // namespace dyng::detail
