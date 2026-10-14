// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_update_test.cpp
 * @brief The hypergraph's update path (ADR 0043; M8b criterion 10): the two-phase transaction
 *        (begin, before() with the old view valid until the commit, commit, after() with the new
 *        view and the applied record; the Debug check of an old view used after the commit), and
 *        dyng::update() over hypergraphs through detail::participant_of<hypergraph<V,W>>: one
 *        Step 0 and one commit for every result, the order before_apply -> commit -> after_apply,
 *        duplicate targets, the strong guarantee before the commit, a failing result poisoned
 *        while the others complete, stale results, the line graph prepared for its readers, and
 *        the framework's aggregate-delta fakes kept equal to compute(). Compiled into the host
 *        test executable (sequential, openmp) and, with DYNG_TEST_CUDA=1, into the CUDA one.
 */
#include "framework/fake_hyper_problems.hpp"
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/hypergraph_support.hpp"
#include "hypergraph/transaction.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace record {

/// What the recording participant saw.
struct log {
  std::vector<std::string> calls;  ///< "<hook>@<version>" in call order
};

/// A result whose participant implements the hypergraph participant interface directly.
struct result {
  log* calls = nullptr;                    ///< where the calls go
  const void* staged = nullptr;            ///< the Step 0 object it received
  std::vector<std::int64_t> inserted_ids;  ///< from the applied record
  std::vector<std::int64_t> device_ids;    ///< the record's device copy (cuda), downloaded
  std::vector<std::int64_t> deleted_ids;   ///< from the applied record
  bool line_graph_before = false;          ///< whether G_t had its line graph in before_apply
  bool reads_line_graph = false;           ///< reads_prepared_graph()
  bool fail_before = false;                ///< before_apply throws
  bool fail_after = false;                 ///< after_apply throws
  bool poisoned = false;                   ///< poison() was called
};

/// Its stats.
struct stats : dyng::update_stats {};

/// The participant.
template <typename container_t>
class participant final : public dyng::detail::participant_of<container_t>::type {
  using base = typename dyng::detail::participant_of<container_t>::type;

 public:
  participant(result& r, stats& s) : r_(r), s_(s) {}
  [[nodiscard]] const void* target() const noexcept override {
    return &r_;
  }
  void use_normalized(const dyng::detail::hyper_staged_batch<typename container_t::weight_type>*
                          staged) noexcept override {
    r_.staged = staged;
    r_.calls->calls.push_back("use_normalized");
  }
  void before_apply(const dyng::resources& /*res*/, const container_t& h,
                    const typename base::batch_type& /*batch*/) override {
    r_.calls->calls.push_back("before@" + std::to_string(h.version()));
    r_.line_graph_before = dyng::detail::hypergraph_access::views(h).has_h2h;
    if (r_.fail_before) throw dyng::invalid_argument_error("test: before_apply rejects");
  }
  void after_apply(const dyng::resources& res, const container_t& h,
                   const dyng::hyper_apply_summary& summary,
                   const dyng::detail::hyper_applied& applied) override {
    r_.calls->calls.push_back("after@" + std::to_string(h.version()));
    if (r_.fail_after) throw dyng::internal_error("test: after_apply fails");
    EXPECT_EQ(applied.inserted_ids, summary.inserted_ids);
    r_.inserted_ids = applied.inserted_ids;
    r_.deleted_ids = applied.deleted_ids;
    if (h.space() == dyng::memory_space::device && !applied.inserted_ids.empty()) {
      r_.device_ids = dyng::to_vector(res, applied.inserted_ids_device.view());
    }
    s_.affected = summary.inserted_hyperedges;
  }
  void poison() noexcept override {
    r_.poisoned = true;
  }
  [[nodiscard]] bool reads_prepared_graph() const noexcept override {
    return r_.reads_line_graph;
  }

 private:
  result& r_;
  stats& s_;
};

}  // namespace record

namespace dyng::detail {

template <>
struct update_traits<record::result> {
  using stats_type = record::stats;
  template <typename container_t>
  static std::unique_ptr<typename participant_of<container_t>::type> make_participant(
      record::result& r, record::stats& s) {
    return std::make_unique<record::participant<container_t>>(r, s);
  }
};

}  // namespace dyng::detail

namespace {

using dyng::backend;
using dyng::hypergraph_properties;
using dyng::id_reuse;
using dyng::resources;
using dyng::unweighted;
using dyng::test::hyper::h2v_rows;
using dyng::test::hyper::list_of;
using dyng::test::hyper::rows;
using hgraph = dyng::hypergraph<std::int32_t, unweighted>;
using whgraph = dyng::hypergraph<std::int32_t, std::int32_t>;
using batch_t = dyng::hyperedge_batch<std::int32_t, unweighted>;
namespace fakes = ::dyng::test::framework_fakes;

class HypergraphUpdate : public ::testing::TestWithParam<backend> {
 protected:
  resources res() const {
    return dyng::test::hyper::resources_of(GetParam());
  }
  /// {0,1,2,3} (id 0), {4} (id 1), {5,6} (id 2), {0,6} (id 3) on 10 vertices (ADR 0041's
  /// example plus one).
  hgraph make(dyng::line_graph_policy lg = dyng::line_graph_policy::automatic,
              id_reuse reuse = id_reuse::erase_first) const {
    hypergraph_properties p;
    p.line_graph = lg;
    p.reuse = reuse;
    p.capacity = dyng::capacity_policy::escher_gpu();
    return hgraph::from_hyperedges(res(),
                                   list_of(10, rows{{0, 1, 2, 3}, {4}, {5, 6}, {0, 6}}).view(), p);
  }
};

std::string backend_name(const ::testing::TestParamInfo<backend>& info) {
  return std::string(dyng::to_string(info.param));
}

batch_t reuse_batch() {
  batch_t b;
  b.delete_hyperedge(1);
  b.insert_hyperedge({7, 8});
  return b;
}

// ---- the transaction ---------------------------------------------------------------------------

TEST_P(HypergraphUpdate, TheTransactionRunsBeginBeforeCommitAfter) {
  hgraph h = make();
  const batch_t b = reuse_batch();
  dyng::detail::hyper_transaction<std::int32_t, unweighted> txn(res(), h, b.view(), "test");
  EXPECT_EQ(txn.staged().deletions, std::vector<std::int64_t>{1});
  EXPECT_EQ(txn.staged().rows.size(), 1);
  EXPECT_EQ(txn.staged().version, 0U);
  EXPECT_THROW((void)txn.after(), dyng::internal_error);
  EXPECT_THROW((void)txn.applied(), dyng::internal_error);
  const auto old = txn.before();
  EXPECT_EQ(old.version(), 0U);
  EXPECT_EQ(old->num_hyperedges(), 4);
  EXPECT_EQ(h.version(), 0U) << "begin changes nothing";
  const dyng::hyper_apply_summary s = txn.commit();
  EXPECT_TRUE(txn.committed());
  EXPECT_EQ(s.inserted_ids, std::vector<std::int64_t>{1}) << "erase_first reuses the freed slot";
  const auto now = txn.after();
  EXPECT_EQ(now.version(), 1U);
  EXPECT_EQ(txn.applied().inserted_ids, s.inserted_ids);
  EXPECT_EQ(txn.applied().deleted_ids, std::vector<std::int64_t>{1});
  ASSERT_EQ(txn.applied().deleted_members.size(), 1);
  EXPECT_EQ(txn.applied().deleted_members.members, std::vector<std::int32_t>{4});
  EXPECT_THROW((void)txn.before(), dyng::internal_error) << "G_t is gone after the commit";
  EXPECT_THROW((void)txn.commit(), dyng::internal_error) << "a transaction commits once";
  EXPECT_FALSE(old.is_current());
}

TEST_P(HypergraphUpdate, AnOldViewUsedAfterTheCommitThrowsInDebugBuilds) {
  hgraph h = make();
  const batch_t b = reuse_batch();
  dyng::detail::hyper_transaction<std::int32_t, unweighted> txn(res(), h, b.view(), "test");
  const auto old = txn.before();
  EXPECT_NO_THROW((void)old.get());
  (void)txn.commit();
#ifndef NDEBUG
  EXPECT_THROW((void)old.get(), dyng::internal_error) << "invariant I1";
  EXPECT_THROW((void)old->num_hyperedges(), dyng::internal_error);
#else
  EXPECT_FALSE(old.is_current());
  GTEST_SKIP() << "the old view's version check is a Debug check (NDEBUG is defined)";
#endif
}

TEST_P(HypergraphUpdate, TheBeginRejectsAnInvalidBatchBeforeAnythingChanges) {
  hgraph h = make();
  const std::uint64_t state = dyng::detail::hypergraph_access::state_id(h);
  batch_t b;
  b.delete_hyperedge(0);
  b.insert_hyperedge({3, 10});  // vertex 10 of 10
  EXPECT_THROW(
      (dyng::detail::hyper_transaction<std::int32_t, unweighted>(res(), h, b.view(), "test")),
      dyng::invalid_argument_error);
  batch_t edits;
  edits.insert_incidence(0, 5);
  EXPECT_THROW(
      (dyng::detail::hyper_transaction<std::int32_t, unweighted>(res(), h, edits.view(), "test")),
      dyng::not_supported_error);
  EXPECT_EQ(h.version(), 0U);
  EXPECT_EQ(dyng::detail::hypergraph_access::state_id(h), state);
  EXPECT_EQ(h2v_rows(h), (rows{{0, 1, 2, 3}, {4}, {5, 6}, {0, 6}}));
}

TEST_P(HypergraphUpdate, ACommitAfterAnotherApplyIsRefused) {
  hgraph h = make();
  const batch_t b = reuse_batch();
  dyng::detail::hyper_transaction<std::int32_t, unweighted> txn(res(), h, b.view(), "test");
  batch_t other;
  other.insert_hyperedge({9});
  (void)h.apply(res(), other.view());
  EXPECT_THROW((void)txn.commit(), dyng::invalid_argument_error);
  EXPECT_EQ(h.version(), 1U);
  EXPECT_NO_THROW(h.check_integrity(res())) << "the refused commit changed nothing";
}

// ---- dyng::update() over hypergraphs ------------------------------------------------------------

TEST_P(HypergraphUpdate, OneStepZeroAndOneCommitForEveryResult) {
  hgraph h = make();
  record::log calls;
  record::result a;
  record::result b;
  a.calls = &calls;
  b.calls = &calls;
  const batch_t batch = reuse_batch();
  const auto [sa, sb] = dyng::update(res(), h, batch.view(), a, b);
  EXPECT_EQ(h.version(), 1U) << "one commit";
  EXPECT_EQ(calls.calls, (std::vector<std::string>{"use_normalized", "use_normalized", "before@0",
                                                   "before@0", "after@1", "after@1"}));
  EXPECT_NE(a.staged, nullptr);
  EXPECT_EQ(a.staged, b.staged) << "Step 0 runs once for every participant";
  EXPECT_EQ(a.inserted_ids, std::vector<std::int64_t>{1});
  EXPECT_EQ(b.inserted_ids, a.inserted_ids);
  EXPECT_EQ(a.deleted_ids, std::vector<std::int64_t>{1});
  if (GetParam() == backend::cuda) {
    EXPECT_EQ(a.device_ids, a.inserted_ids) << "the applied record's device copy";
  }
  EXPECT_EQ(sa.affected, 1);
  EXPECT_EQ(sb.affected, 1);
  // update_each, a run-time list of one type.
  std::vector<record::result*> list{&a, &b};
  calls.calls.clear();
  const auto each = dyng::update_each(res(), h, batch_t{}.view(), dyng::host_view(list));
  EXPECT_EQ(each.size(), 2U);
  EXPECT_EQ(h.version(), 2U);
}

TEST_P(HypergraphUpdate, AResultPassedTwiceIsRejectedBeforeAnythingChanges) {
  hgraph h = make();
  record::log calls;
  record::result a;
  a.calls = &calls;
  const batch_t batch = reuse_batch();
  EXPECT_THROW((void)dyng::update(res(), h, batch.view(), a, a), dyng::invalid_argument_error);
  EXPECT_EQ(h.version(), 0U);
  EXPECT_TRUE(calls.calls.empty());
}

TEST_P(HypergraphUpdate, ARejectionBeforeTheCommitChangesNothing) {
  hgraph h = make();
  record::log calls;
  record::result a;
  record::result b;
  a.calls = &calls;
  b.calls = &calls;
  b.fail_before = true;
  const batch_t batch = reuse_batch();
  EXPECT_THROW((void)dyng::update(res(), h, batch.view(), a, b), dyng::invalid_argument_error);
  EXPECT_EQ(h.version(), 0U);
  EXPECT_FALSE(a.poisoned);
  EXPECT_FALSE(b.poisoned);
  EXPECT_EQ(h2v_rows(h), (rows{{0, 1, 2, 3}, {4}, {5, 6}, {0, 6}}));
  // An invalid batch is rejected by the transaction's begin, before any participant runs.
  calls.calls.clear();
  b.fail_before = false;
  batch_t bad;
  bad.insert_hyperedge({11});
  EXPECT_THROW((void)dyng::update(res(), h, bad.view(), a, b), dyng::invalid_argument_error);
  EXPECT_TRUE(calls.calls.empty());
  EXPECT_EQ(h.version(), 0U);
}

TEST_P(HypergraphUpdate, AFailingResultIsPoisonedAndTheOthersComplete) {
  hgraph h = make();
  record::log calls;
  record::result a;
  record::result b;
  record::result c;
  a.calls = &calls;
  b.calls = &calls;
  c.calls = &calls;
  a.fail_after = true;
  c.fail_after = true;
  const batch_t batch = reuse_batch();
  try {
    (void)dyng::update(res(), h, batch.view(), a, b, c);
    ADD_FAILURE() << "no exception";
  } catch (const dyng::internal_error& e) {
    EXPECT_NE(std::string(e.what()).find("after_apply fails"), std::string::npos);
  }
  EXPECT_EQ(h.version(), 1U) << "the commit stays (basic guarantee)";
  EXPECT_TRUE(a.poisoned);
  EXPECT_FALSE(b.poisoned);
  EXPECT_TRUE(c.poisoned);
  EXPECT_EQ(b.inserted_ids, std::vector<std::int64_t>{1}) << "the others still run";
}

TEST_P(HypergraphUpdate, TheLineGraphIsPreparedForItsReaders) {
  {
    hgraph h = make(dyng::line_graph_policy::automatic);
    record::log calls;
    record::result reader;
    reader.calls = &calls;
    reader.reads_line_graph = true;
    (void)dyng::update(res(), h, reuse_batch().view(), reader);
    EXPECT_TRUE(reader.line_graph_before) << "automatic builds it on G_t, before before_apply";
  }
  {
    hgraph h = make(dyng::line_graph_policy::automatic);
    record::log calls;
    record::result other;
    other.calls = &calls;
    (void)dyng::update(res(), h, reuse_batch().view(), other);
    EXPECT_FALSE(other.line_graph_before) << "nobody reads it: not built";
  }
  {
    hgraph h = make(dyng::line_graph_policy::none);
    record::log calls;
    record::result reader;
    reader.calls = &calls;
    reader.reads_line_graph = true;
    EXPECT_THROW((void)dyng::update(res(), h, reuse_batch().view(), reader),
                 dyng::invalid_argument_error);
    EXPECT_EQ(h.version(), 0U);
  }
}

TEST_P(HypergraphUpdate, TheStagesOfAnUpdate) {
  dyng::profiler prof;
  resources r = res();
  r.attach_profiler(&prof);
  hgraph h = make();
  record::log calls;
  record::result a;
  a.calls = &calls;
  (void)dyng::update(r, h, reuse_batch().view(), a);
  r.attach_profiler(nullptr);
  std::vector<std::string> names;
  for (const auto& st : prof.stages()) names.push_back(st.name);
  for (const char* n : {"update.normalize", "update.commit", "hypergraph.apply",
                        "hypergraph.apply.delete", "hypergraph.apply.insert"}) {
    EXPECT_NE(std::find(names.begin(), names.end(), n), names.end()) << n;
  }
}

// ---- the framework's aggregate-delta fakes through dyng::update() -----------------------------

TEST_P(HypergraphUpdate, TheFakesStayEqualToComputeThroughDyngUpdate) {
  for (const id_reuse reuse : {id_reuse::erase_first, id_reuse::insert_first}) {
    SCOPED_TRACE(std::string(dyng::to_string(reuse)));
    hgraph h = make(dyng::line_graph_policy::automatic, reuse);
    fakes::hyper_pairs_result pairs = fakes::compute_pairs(res(), h);
    fakes::hyper_squares_result squares = fakes::compute_squares(res(), h);
    EXPECT_EQ(pairs.value, 2);  // {0,3}: {0,1,2,3}-{0,6}, {2,3}: {5,6}-{0,6}
    EXPECT_EQ(squares.value, 16 + 1 + 4 + 4);
    std::vector<batch_t> batches(4);
    batches[0] = reuse_batch();
    batches[1].delete_hyperedge(0);
    batches[1].delete_hyperedge(3);
    batches[1].insert_hyperedge({0, 1, 2});
    batches[1].insert_hyperedge({2, 8});
    batches[1].insert_hyperedge({8, 9});
    batches[2].delete_hyperedge(2);
    batches[2].delete_hyperedge(2);   // repeated: skipped
    batches[2].delete_hyperedge(40);  // out of range: skipped
    batches[2].insert_hyperedge({5, 5, 6});
    for (const batch_t& b : batches) {
      const auto [sp, ss] = dyng::update(res(), h, b.view(), pairs, squares);
      EXPECT_EQ(pairs.value, fakes::compute_pairs(res(), h).value);
      EXPECT_EQ(squares.value, fakes::compute_squares(res(), h).value);
      EXPECT_EQ(pairs.graph_version(), h.version());
      EXPECT_EQ(sp.batch.inserted_ids, ss.batch.inserted_ids);
    }
  }
}

TEST_P(HypergraphUpdate, AStaleResultIsDetected) {
  hgraph h = make();
  fakes::hyper_pairs_result pairs = fakes::compute_pairs(res(), h);
  (void)h.apply(res(), reuse_batch().view());
  try {
    (void)dyng::update(res(), h, batch_t{}.view(), pairs);
    ADD_FAILURE() << "no exception";
  } catch (const dyng::stale_result_error& e) {
    EXPECT_NE(std::string(e.what()).find("hypergraph version 0"), std::string::npos) << e.what();
  }
  // Another hypergraph with the same hyperedges at the same version: another state.
  hgraph other = make();
  (void)other.apply(res(), reuse_batch().view());
  EXPECT_THROW((void)dyng::update(res(), other, batch_t{}.view(), pairs), dyng::stale_result_error);
  // A clone is the same state until the two diverge.
  pairs = fakes::compute_pairs(res(), h);
  hgraph copy = h.clone(res());
  EXPECT_NO_THROW((void)dyng::update(res(), copy, batch_t{}.view(), pairs));
  EXPECT_THROW((void)dyng::update(res(), h, batch_t{}.view(), pairs), dyng::stale_result_error);
}

TEST_P(HypergraphUpdate, AProblemThatKeepsItsOldViewPastTheCommitFailsInDebugBuilds) {
  hgraph h = make();
  fakes::hyper_pairs_result pairs = fakes::compute_pairs(res(), h);
  fakes::hyper_misbehaviour bad;
  bad.keep_old_view = true;
#ifndef NDEBUG
  EXPECT_THROW((void)fakes::update_pairs(res(), h, reuse_batch().view(), pairs, bad),
               dyng::internal_error);
  EXPECT_TRUE(pairs.poisoned) << "the failing result is poisoned";
  EXPECT_EQ(h.version(), 1U);
#else
  EXPECT_NO_THROW((void)fakes::update_pairs(res(), h, reuse_batch().view(), pairs, bad));
  GTEST_SKIP() << "the old view's version check is a Debug check (NDEBUG is defined)";
#endif
}

TEST_P(HypergraphUpdate, AFailingProblemIsPoisonedWhileItsPartnerCompletes) {
  hgraph h = make();
  fakes::hyper_pairs_result pairs = fakes::compute_pairs(res(), h);
  fakes::hyper_squares_result squares = fakes::compute_squares(res(), h);
  // The pairs participant fails in count(+); dyng::update has no misbehaviour switch, so run the
  // two participants through run_update() by hand.
  namespace fw = ::dyng::detail::framework;
  fakes::hyper_count_stats sp;
  fakes::hyper_count_stats ss;
  fakes::hyper_misbehaviour bad;
  bad.fail_after = true;
  fw::problem_participant<fakes::hyper_pairs_problem<hgraph>> p(sp, pairs, bad);
  fw::problem_participant<fakes::hyper_squares_problem<hgraph>> q(ss, squares);
  dyng::detail::hyper_update_participant<std::int32_t, unweighted>* list[] = {&p, &q};
  EXPECT_THROW(
      (void)dyng::detail::run_update(res(), h, reuse_batch().view(), list, 2, "update.commit"),
      dyng::internal_error);
  EXPECT_TRUE(pairs.poisoned);
  EXPECT_FALSE(squares.poisoned);
  EXPECT_EQ(squares.value, fakes::compute_squares(res(), h).value);
  EXPECT_EQ(squares.graph_version(), h.version());
}

TEST_P(HypergraphUpdate, WeightedHypergraphsTakePartToo) {
  hypergraph_properties p;
  whgraph h = whgraph::from_hyperedges(
      res(), list_of<std::int32_t>(6, rows{{0, 1}, {1, 2}, {3}}, {5, 6, 7}).view(), p);
  fakes::hyper_pairs_result pairs = fakes::compute_pairs(res(), h);
  dyng::hyperedge_batch<std::int32_t, std::int32_t> b;
  b.delete_hyperedge(2);
  b.insert_hyperedge({2, 3}, 9);
  (void)dyng::update(res(), h, b.view(), pairs);
  EXPECT_EQ(pairs.value, fakes::compute_pairs(res(), h).value);
  EXPECT_EQ(pairs.value, 2);
}

INSTANTIATE_TEST_SUITE_P(Backends, HypergraphUpdate,
                         ::testing::ValuesIn(dyng::test::suite_backends()), backend_name);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HypergraphUpdate);

}  // namespace
