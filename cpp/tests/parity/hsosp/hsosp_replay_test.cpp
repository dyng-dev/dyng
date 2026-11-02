// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hsosp_replay_test.cpp
 * @brief hyper_sssp's container-side replay (M11a criteria 8 and 10; ADRs 0085 and 0086) on the
 *        committed subset of the golden set hyper_sssp_mosp_escher: every session replayed whole
 *        on slack_csr storage matches MOSP_ESCHER by id on sequential and openmp (1 and N
 *        threads), and the comparator catches a planted wrong id, skip count, dead id, pair, pair
 *        count, row or weight in MOSP_ESCHER's records.
 */
#include "parity/hsosp/hsosp_replay.hpp"

#include "support/test_threads.hpp"

#include <dyng/core/resources.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace {

namespace hd = dyng::test::hsosp_dump;
namespace hr = dyng::test::hsosp_replay;

const std::string subset =
    std::string(DYNG_TEST_DATA_DIR) + "/goldens/hyper_sssp/test_hsosp_matches_dijkstra.hsd";

std::vector<hd::record> load() {
  return hd::read_dump(subset);
}

hr::report replay(const std::vector<hd::record>& recs,
                  const dyng::resources& res = dyng::resources::sequential()) {
  return hr::replay(recs, res, hr::options{}, "subset");
}

TEST(HsospReplay, TheSubsetReplaysWholeOnSlackCsr) {
  std::vector<dyng::resources> backends{dyng::resources::sequential()};
#if defined(DYNG_HAS_OPENMP)
  backends.push_back(dyng::resources::openmp(1));
  backends.push_back(dyng::resources::openmp(dyng::test::openmp_test_threads(8)));
#endif
  for (const auto& res : backends) {
    const auto rep = replay(load(), res);
    SCOPED_TRACE(rep.backend + " " + std::to_string(rep.threads));
    EXPECT_TRUE(rep.ok()) << (rep.mismatches.empty() ? "" : rep.mismatches.front());
    EXPECT_EQ(rep.stopped_sessions, 0);
    EXPECT_GT(rep.sessions, 0);
    EXPECT_GT(rep.vertex_batches, 0);  // the vertex-kind batches are replayed, not stopped at
    EXPECT_GT(rep.removals + rep.additions, 0);
    EXPECT_GT(rep.pair_batches, 0);
    EXPECT_EQ(rep.oracle_states, rep.batches);
  }
}

// The first record with `tag` (and `phase`, if given) for which `pick` holds.
hd::record* find(std::vector<hd::record>& recs, const std::string& tag, const std::string& phase,
                 const std::function<bool(const hd::record&)>& pick) {
  for (auto& r : recs) {
    if (r.tag != tag || (!phase.empty() && r.str("phase") != phase)) continue;
    if (r.has("exception") && !r.str("exception").empty()) continue;
    if (pick(r)) return &r;
  }
  return nullptr;
}

void expect_caught(const std::vector<hd::record>& recs, const std::string& what) {
  const auto rep = replay(recs);
  EXPECT_FALSE(rep.ok()) << what;
  EXPECT_GE(rep.mismatch_count, 1) << what;
}

TEST(HsospReplay, APlantedIdSkipCountOrDeadIdIsCaught) {
  auto recs = load();
  auto* b = find(recs, "begin_batch", "exit",
                 [](const hd::record& r) { return !r.ints("final_ids").empty(); });
  ASSERT_NE(b, nullptr);
  b->fields.at("final_ids").ints.front() += 1;
  expect_caught(recs, "a wrong final id");

  recs = load();
  // A vertex-kind batch's skip count (its skipped incidence edits).
  std::size_t vertex_exit = recs.size();
  for (std::size_t k = 0; k + 1 < recs.size(); ++k) {
    const auto& r = recs[k];
    if (r.tag == "begin_batch" && r.str("phase") == "enter" &&
        (!r.ints("vtx_delete.he").empty() || !r.ints("vtx_insert.he").empty())) {
      for (std::size_t j = k + 1; j < recs.size(); ++j) {
        if (recs[j].tag == "begin_batch" && recs[j].str("phase") == "exit") {
          vertex_exit = j;
          break;
        }
      }
      if (vertex_exit != recs.size() && recs[vertex_exit].str("exception").empty()) break;
      vertex_exit = recs.size();
    }
  }
  ASSERT_LT(vertex_exit, recs.size());
  recs[vertex_exit].fields.at("skipped_ops").value += 1;
  expect_caught(recs, "a wrong skip count of a vertex-kind batch");

  recs = load();
  b = find(recs, "begin_batch", "exit", [](const hd::record&) { return true; });
  ASSERT_NE(b, nullptr);
  auto& dead = b->fields.at("dead_he").ints;
  if (dead.empty()) {
    dead.push_back(1);
  } else {
    dead.pop_back();
  }
  expect_caught(recs, "a wrong dead id");
}

TEST(HsospReplay, APlantedPairPairCountRowOrWeightIsCaught) {
  auto recs = load();
  auto* a = find(recs, "apply_batch", "",
                 [](const hd::record& r) { return !r.ints("pairs.inserted.a").empty(); });
  ASSERT_NE(a, nullptr);
  a->fields.at("pairs.inserted.a").ints.pop_back();
  a->fields.at("pairs.inserted.b").ints.pop_back();
  expect_caught(recs, "a missing inserted pair");

  recs = load();
  a = find(recs, "apply_batch", "",
           [](const hd::record& r) { return !r.ints("pairs.deleted.a").empty(); });
  ASSERT_NE(a, nullptr);
  std::swap(a->fields.at("pairs.deleted.a").ints.front(),
            a->fields.at("pairs.deleted.b").ints.front());  // (b, a): not B's canonical pair
  expect_caught(recs, "a deleted pair out of order");

  recs = load();
  a = find(recs, "apply_batch", "", [](const hd::record&) { return true; });
  ASSERT_NE(a, nullptr);
  a->fields.at("h2h_pair_count").value += 1;
  expect_caught(recs, "a wrong pair count");

  recs = load();
  a = find(recs, "apply_batch", "", [](const hd::record& r) {
    const auto& alive = r.ints("rows.alive");
    for (const auto x : alive)
      if (x != 0) return true;
    return false;
  });
  ASSERT_NE(a, nullptr);
  {
    auto& alive = a->fields.at("rows.alive").ints;
    std::size_t k = 0;
    while (alive[k] == 0) ++k;
    const auto& off = a->fields.at("rows.offsets").longs;
    auto& values = a->fields.at("rows.values").ints;
    ASSERT_LT(off[k], off[k + 1]);
    values[static_cast<std::size_t>(off[k])] += 1;  // a member of a live row
    expect_caught(recs, "a wrong member");

    // A weight of a live row that is not one of B's virtual rows (those weigh 1 in dynG
    // whatever B records): the first batch of the first session.
    recs = load();
    const hd::record* l = find(recs, "load", "enter", [](const hd::record&) { return true; });
    ASSERT_NE(l, nullptr);
    const std::int64_t source = l->i64("source_he"), target = l->i64("target_he");
    a = find(recs, "apply_batch", "", [](const hd::record&) { return true; });
    ASSERT_NE(a, nullptr);
    const auto& alive2 = a->fields.at("rows.alive").ints;
    const auto& ids = a->fields.at("rows.ids").ints;
    std::size_t j = alive2.size();
    for (std::size_t i = 0; i < alive2.size() && j == alive2.size(); ++i)
      if (alive2[i] != 0 && ids[i] != source && ids[i] != target) j = i;
    ASSERT_LT(j, alive2.size());
    a->fields.at("rows.weight").longs[j] += 1;
    expect_caught(recs, "a wrong weight");
  }
}

// An apply_batch whose begin_batch records are lost (a dump that lost them, or a reordered one):
// the session stops as a mismatch, never a silent skip (M11a review).
TEST(HsospReplay, AnApplyBatchWithoutItsBeginBatchIsCaught) {
  const auto recs = load();
  std::vector<hd::record> cut;
  std::size_t removed = 0;
  std::size_t seen = 0;
  for (const auto& r : recs) {
    // The second batch's begin_batch enter and exit.
    if (r.tag == "begin_batch") {
      const std::size_t k = seen++;
      if (k == 2 || k == 3) {
        ++removed;
        continue;
      }
    }
    cut.push_back(r);
  }
  ASSERT_EQ(removed, 2U);
  const auto rep = replay(cut);
  EXPECT_FALSE(rep.ok());
  EXPECT_GT(rep.stopped_sessions, 0);
}

}  // namespace
