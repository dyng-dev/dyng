// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_replay_test.cpp
 * @brief The triad replay library (M9a step 2, ADR 0050) on a small case it writes itself (a
 *        `.hg` and `.dgb` files with A's annotations, the expected ids from a container that
 *        applied the batches): the oracle counts after every batch equal the oracle on the
 *        state, both modes agree, and the defects the harness must see are reported (a wrong
 *        expected id, a perturbed deleted row, the algorithm counter before it exists).
 */
#include "parity/escher/triad_replay.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/hypergraph_io.hpp>
#include <dyng/testing/triad_reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

namespace trp = dyng::test::triad_replay;
using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;
using rows = std::vector<std::vector<std::int32_t>>;

struct case_files {
  std::string hg;
  std::vector<std::string> batches;
  std::vector<std::vector<std::int64_t>> expected;  // inserted ids per batch
  std::vector<std::vector<std::int64_t>> h_motif;   // oracle after the build and every batch
};

std::vector<std::int64_t> h_motif_of(const dyng::resources& res, const hgraph& h) {
  const auto list = h.to_hyperedge_list(res);
  std::vector<std::int64_t> out;
  for (const auto x : dyng::testing::triad_reference(list.offsets, list.members, list.num_vertices,
                                                     dyng::testing::triad_pattern::h_motif))
    out.push_back(static_cast<std::int64_t>(x));
  return out;
}

// A random hypergraph and three batches (deletions of live ids, sampled and fresh insertions),
// written as A's dumps are: hyperedge i = id i, the batches with their deleted member sets and
// the ids a container under {erase_first, escher_gpu()} assigned.
case_files write_case(const std::filesystem::path& dir, std::uint64_t seed,
                      const std::string& corrupt = "") {
  std::filesystem::create_directories(dir);
  const dyng::resources res = dyng::resources::sequential();
  std::mt19937_64 rng(seed);
  const std::int32_t n = 40;
  dyng::hyperedge_list<std::int32_t> list;
  list.num_vertices = n;
  std::uniform_int_distribution<int> card(1, 5), vertex(0, n - 1);
  for (int e = 0; e < 60; ++e) {
    std::vector<std::int32_t> r;
    const int k = card(rng);
    for (int t = 0; t < k; ++t) r.push_back(vertex(rng));
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
    list.add_hyperedge(dyng::host_view(r));
  }
  case_files c;
  c.hg = (dir / "case.hg").string();
  dyng::io::write_hypergraph(c.hg, list.view(), 1);
  dyng::hypergraph_properties props;
  props.reuse = dyng::id_reuse::erase_first;
  props.capacity = dyng::capacity_policy::escher_gpu();
  hgraph h = hgraph::from_hyperedges(res, list.view(), props);
  c.h_motif.push_back(h_motif_of(res, h));
  for (int b = 0; b < 3; ++b) {
    const auto view = h.h2v();
    const auto ptr = dyng::to_vector(res, view.row_ptr);
    const auto col = dyng::to_vector(res, view.col_ind);
    std::vector<std::int64_t> live;
    for (std::size_t id = 0; id + 1 < ptr.size(); ++id)
      if (ptr[id + 1] > ptr[id]) live.push_back(static_cast<std::int64_t>(id));
    std::shuffle(live.begin(), live.end(), rng);
    std::vector<std::int64_t> del(live.begin(), live.begin() + 8);
    std::sort(del.begin(), del.end());
    dyng::hyperedge_batch<std::int32_t, dyng::unweighted> batch;
    std::vector<std::int64_t> doff{0};
    std::vector<std::int32_t> dmem;
    for (const auto id : del) {
      batch.delete_hyperedge(id);
      dmem.insert(dmem.end(), col.begin() + ptr[static_cast<std::size_t>(id)],
                  col.begin() + ptr[static_cast<std::size_t>(id) + 1]);
      doff.push_back(static_cast<std::int64_t>(dmem.size()));
    }
    for (int i = 0; i < 9; ++i) {
      std::vector<std::int32_t> r;
      if (i % 2 == 0) {  // a copy of a live hyperedge (a duplicate member set)
        const auto id = static_cast<std::size_t>(live[static_cast<std::size_t>(8 + i)]);
        r.assign(col.begin() + ptr[id], col.begin() + ptr[id + 1]);
      } else {
        for (int t = 0; t < 3; ++t) r.push_back(vertex(rng));
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
      }
      batch.insert_hyperedge(dyng::host_view(r));
    }
    const auto s = h.apply(res, batch.view());
    c.expected.push_back(s.inserted_ids);
    c.h_motif.push_back(h_motif_of(res, h));
    std::vector<std::int64_t> expected = s.inserted_ids;
    if (corrupt == "id" && b == 1) expected[2] += 1;
    if (corrupt == "row" && b == 2) {
      // The first deleted set becomes {0, ..., 5}: six members, more than any hyperedge has.
      std::vector<std::int32_t> perturbed = {0, 1, 2, 3, 4, 5};
      perturbed.insert(perturbed.end(), dmem.begin() + doff[1], dmem.end());
      dmem = perturbed;
      const std::int64_t first = doff[1];
      for (std::size_t j = 1; j < doff.size(); ++j) doff[j] += 6 - first;
    }
    const std::string path = (dir / ("batch" + std::to_string(b + 1) + ".dgb")).string();
    dyng::io::batch_writer out(
        path, dyng::io::hyperedge_batch_file_header<std::int32_t, dyng::unweighted>(
                  dyng::io::batch_file_flags::deleted_members |
                  dyng::io::batch_file_flags::expected_ids));
    dyng::io::hyperedge_batch_annotations<std::int32_t> a;
    a.deleted_offsets = dyng::host_view(doff);
    a.deleted_members = dyng::host_view(dmem);
    a.expected_ids = dyng::host_view(expected);
    out.write(batch.view(), a);
    out.close();
    c.batches.push_back(path);
  }
  return c;
}

std::filesystem::path temp_dir(const std::string& name) {
  return std::filesystem::temp_directory_path() /
         ("dyng_triad_replay_test_" + name + "_" +
          std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
}

TEST(TriadReplay, OracleCountsFollowTheStateInBothModes) {
  const auto dir = temp_dir("ok");
  const case_files c = write_case(dir, 11);
  for (const auto& res : {dyng::resources::sequential(), dyng::resources::openmp(3)}) {
    trp::options opt;
    opt.num_vertices = 40;
    const trp::report rep = trp::replay(res, c.hg, c.batches, opt);
    ASSERT_EQ(rep.modes.size(), 2u);
    for (const auto& m : rep.modes) {
      EXPECT_EQ(m.full.at("h_motif"), c.h_motif[0]) << m.mode;
      ASSERT_EQ(m.batches.size(), 3u);
      std::vector<std::int64_t> sum = m.full.at("h_motif");
      for (std::size_t b = 0; b < 3; ++b) {
        const auto& br = m.batches[b];
        EXPECT_EQ(br.counts_after.at("h_motif"), c.h_motif[b + 1]) << m.mode << " batch " << b;
        for (std::size_t k = 0; k < sum.size(); ++k) sum[k] += br.delta.at("h_motif")[k];
        EXPECT_EQ(sum, br.counts_after.at("h_motif"));
        EXPECT_TRUE(br.ids_as_expected);
        EXPECT_EQ(br.unresolved, 0);
        EXPECT_TRUE(br.integrity_ok);
        EXPECT_EQ(br.deletions, 8);
        EXPECT_EQ(br.insertions, 9);
        EXPECT_EQ(br.counts_after.count("stathyper_type3"), 1u);
      }
    }
    EXPECT_EQ(rep.modes[0].batches.back().counts_after, rep.modes[1].batches.back().counts_after);
    EXPECT_NE(rep.to_json().find("\"by_content\""), std::string::npos);
  }
  std::filesystem::remove_all(dir);
}

TEST(TriadReplay, AWrongExpectedIdIsReportedByKey) {
  const auto dir = temp_dir("id");
  const case_files c = write_case(dir, 12, "id");
  trp::options opt;
  opt.mode = "by_key";
  opt.patterns = {dyng::testing::triad_pattern::stathyper_type1};
  const auto rep = trp::replay(dyng::resources::sequential(), c.hg, c.batches, opt);
  ASSERT_EQ(rep.modes.size(), 1u);
  EXPECT_TRUE(rep.modes[0].batches[0].ids_as_expected);
  EXPECT_FALSE(rep.modes[0].batches[1].ids_as_expected);
  EXPECT_EQ(rep.modes[0].batches[1].first_id_mismatch, 2);
  std::filesystem::remove_all(dir);
}

TEST(TriadReplay, APerturbedDeletedRowIsUnresolvedByContent) {
  const auto dir = temp_dir("row");
  const case_files c = write_case(dir, 13, "row");
  trp::options opt;
  opt.mode = "by_content";
  opt.patterns = {dyng::testing::triad_pattern::stathyper_type1};
  const auto rep = trp::replay(dyng::resources::sequential(), c.hg, c.batches, opt);
  // The perturbed set matches no live hyperedge: one deletion is not made, so the live sets
  // (and the counts) differ from those of the by-key replay.
  EXPECT_EQ(rep.modes[0].batches[0].unresolved, 0);
  EXPECT_EQ(rep.modes[0].batches[2].unresolved, 1);
  opt.mode = "by_key";
  const auto by_key = trp::replay(dyng::resources::sequential(), c.hg, c.batches, opt);
  EXPECT_EQ(by_key.modes[0].batches[2].unresolved, 0);
  EXPECT_EQ(rep.modes[0].batches[2].deleted + 1, by_key.modes[0].batches[2].deleted);
  std::filesystem::remove_all(dir);
}

TEST(TriadReplay, TheAlgorithmCounterIsNotAvailableYetAndBadOptionsAreRefused) {
  const auto dir = temp_dir("na");
  const case_files c = write_case(dir, 14);
  trp::options opt;
  opt.counter = "algorithm";
  EXPECT_THROW((void)trp::replay(dyng::resources::sequential(), c.hg, c.batches, opt),
               trp::not_available);
  opt.counter = "oracle";
  opt.mode = "by_id";
  EXPECT_THROW((void)trp::replay(dyng::resources::sequential(), c.hg, c.batches, opt),
               std::invalid_argument);
  EXPECT_EQ(trp::pattern_of("type2"), dyng::testing::triad_pattern::stathyper_type2);
  EXPECT_THROW((void)trp::pattern_of("type4"), std::invalid_argument);
  std::filesystem::remove_all(dir);
}

}  // namespace
