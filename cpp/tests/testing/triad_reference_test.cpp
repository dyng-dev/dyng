// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_reference_test.cpp
 * @brief dyng::testing::triad_reference (M9a criterion 4): the class map it derives equals
 *        ESCHER-GPU@abcf9b2's table on all 128 Venn ids (read from a test-only fixture), one
 *        triple per reachable bin lands in that bin alone, hand cases of the three StatHyper
 *        types and of duplicate hyperedges, a brute force over all triples, removed() / added(),
 *        thread counts, and the refusal of malformed input.
 */
#include "testing/escher_motif_table.hpp"

#include <dyng/testing/triad_reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace tr = dyng::testing;
using rows = std::vector<std::vector<std::int32_t>>;

struct csr {
  std::vector<std::int64_t> offsets{0};
  std::vector<std::int32_t> members;
  std::int32_t num_vertices = 0;
};

csr to_csr(const rows& r, std::int32_t num_vertices) {
  csr c;
  c.num_vertices = num_vertices;
  for (const auto& row : r) {
    c.members.insert(c.members.end(), row.begin(), row.end());
    c.offsets.push_back(static_cast<std::int64_t>(c.members.size()));
  }
  return c;
}

std::vector<std::uint64_t> count(const rows& r, std::int32_t n, tr::triad_pattern p,
                                 int threads = 0) {
  const csr c = to_csr(r, n);
  return tr::triad_reference(c.offsets, c.members, c.num_vertices, p, {threads});
}

// Three hyperedges whose non-empty Venn regions are exactly the bits of `venn_id`: one fresh
// vertex per region (A only = 0, B only = 1, C only = 2, A and B only = 3, B and C only = 4,
// C and A only = 5, all three = 6).
rows triple_of(int venn_id) {
  rows t(3);
  const auto add = [&](int region, std::initializer_list<int> to) {
    if ((venn_id >> region) & 1)
      for (const int h : to) t[static_cast<std::size_t>(h)].push_back(region);
  };
  add(0, {0});
  add(1, {1});
  add(2, {2});
  add(3, {0, 1});
  add(4, {1, 2});
  add(5, {2, 0});
  add(6, {0, 1, 2});
  for (auto& row : t) std::sort(row.begin(), row.end());
  return t;
}

TEST(TriadReference, DerivedClassMapEqualsEschersTableOnAll128Ids) {
  const auto map = tr::triad_motif_class_map();
  for (int id = 0; id < 128; ++id) {
    EXPECT_EQ(map[static_cast<std::size_t>(id)],
              dyng::test::escher_fixture::motif_id_to_index[static_cast<std::size_t>(id)])
        << "Venn id " << id;
  }
}

TEST(TriadReference, ClassesAreTheOrbitsOfThePermutations) {
  const auto& reps = tr::triad_class_representatives();
  std::set<int> distinct(reps.begin(), reps.end());
  EXPECT_EQ(distinct.size(), 30u);
  std::array<int, 31> members{};
  for (int id = 0; id < 128; ++id) {
    const int cls = tr::triad_motif_class(id);
    ++members[static_cast<std::size_t>(cls)];
    // A class is invariant under the six permutations of (A, B, C).
    std::array<int, 3> image = {0, 1, 2};
    do {
      EXPECT_EQ(tr::triad_motif_class(tr::triad_venn_permuted(id, image)), cls) << id;
    } while (std::next_permutation(image.begin(), image.end()));
    if (cls > 0) {
      EXPECT_EQ(tr::triad_venn_canonical(id), reps[static_cast<std::size_t>(cls - 1)]) << id;
    }
  }
  // 96 ids in classes (64 with the triple region in 20 orbits, 24 open triads in 6, 8 with the
  // three pairwise-only regions in 4), 32 ids with fewer than two intersecting pairs.
  EXPECT_EQ(members[0], 32);
  int in_classes = 0;
  for (int k = 1; k <= 30; ++k) {
    EXPECT_GT(members[static_cast<std::size_t>(k)], 0) << "class " << k;
    in_classes += members[static_cast<std::size_t>(k)];
    // Each representative is the smallest id of its orbit.
    EXPECT_EQ(tr::triad_venn_canonical(reps[static_cast<std::size_t>(k - 1)]),
              reps[static_cast<std::size_t>(k - 1)]);
  }
  EXPECT_EQ(in_classes, 96);
  EXPECT_THROW((void)tr::triad_motif_class(128), std::invalid_argument);
  EXPECT_THROW((void)tr::triad_venn_permuted(3, {0, 0, 1}), std::invalid_argument);
}

TEST(TriadReference, OneTriplePerReachableBinLandsInThatBinAlone) {
  const auto& reps = tr::triad_class_representatives();
  int reachable = 0;
  for (int k = 1; k <= 30; ++k) {
    const int rep = reps[static_cast<std::size_t>(k - 1)];
    const rows t = triple_of(rep);
    ASSERT_EQ(tr::triad_venn_id(t[0], t[1], t[2]), rep);
    EXPECT_EQ(dyng::test::escher_fixture::motif_id_to_index[static_cast<std::size_t>(rep)], k);
    const auto bins = count(t, 7, tr::triad_pattern::h_motif);
    ASSERT_EQ(bins.size(), 30u);
    std::uint64_t total = 0;
    for (const auto b : bins) total += b;
    if (k >= 21 && k <= 26) {
      // An open triad (one pair does not intersect) is no triangle of the line graph.
      EXPECT_EQ(total, 0u) << "class " << k;
    } else {
      ++reachable;
      EXPECT_EQ(total, 1u) << "class " << k;
      EXPECT_EQ(bins[static_cast<std::size_t>(k - 1)], 1u) << "class " << k;
    }
  }
  EXPECT_EQ(reachable, 24);
}

TEST(TriadReference, StatHyperHandCases) {
  // type1: C(4, 3) + C(3, 3) + C(2, 3) = 4 + 1 + 0.
  EXPECT_EQ(count({{0, 1, 2, 3}, {4, 5, 6}, {7, 8}}, 9, tr::triad_pattern::stathyper_type1),
            (std::vector<std::uint64_t>{5}));
  // type2: e = {0,1,2,3}, f = {0,1,4}: a = 2, b = 2, c = 1 -> C(2,2) * 2 * 1 = 2; the pair
  // ({0,1,4}, {4,9}) has a = 1 and counts nothing.
  EXPECT_EQ(count({{0, 1, 2, 3}, {0, 1, 4}, {4, 9}}, 10, tr::triad_pattern::stathyper_type2),
            (std::vector<std::uint64_t>{2}));
  // type2 with a = 3: C(3, 2) * 1 * 2 = 6.
  EXPECT_EQ(count({{0, 1, 2, 5}, {0, 1, 2, 6, 7}}, 8, tr::triad_pattern::stathyper_type2),
            (std::vector<std::uint64_t>{6}));
  // type3: A = {1,2,5,6}, B = {1,2,3}, C = {3,4,5,6}: pairwise-only overlaps 2, 1, 2 -> 4.
  const rows t3 = {{1, 2, 5, 6}, {1, 2, 3}, {3, 4, 5, 6}};
  EXPECT_EQ(count(t3, 7, tr::triad_pattern::stathyper_type3), (std::vector<std::uint64_t>{4}));
  // Its h-motif: regions C only, AB, BC, CA -> Venn id 60, ESCHER's class 28.
  EXPECT_EQ(tr::triad_venn_id(t3[0], t3[1], t3[2]), 60);
  const auto bins = count(t3, 7, tr::triad_pattern::h_motif);
  EXPECT_EQ(bins[27], 1u);
  // A triple vertex removes a triangle from type3 (x * y * z needs the pairwise-only parts).
  EXPECT_EQ(count({{1, 2, 9}, {2, 3, 9}, {3, 1, 9}}, 10, tr::triad_pattern::stathyper_type3),
            (std::vector<std::uint64_t>{1}));
  EXPECT_EQ(count({{1, 9}, {2, 9}, {3, 9}}, 10, tr::triad_pattern::stathyper_type3),
            (std::vector<std::uint64_t>{0}));
}

TEST(TriadReference, DuplicateHyperedgesDeadRowsAndRepeatedMembers) {
  // Three identical hyperedges: one triangle of class 1 (only the triple region), type1 3,
  // type2 0 (b = c = 0), type3 0.
  const rows same = {{0, 1, 2}, {0, 1, 2}, {2, 1, 0}};
  auto bins = count(same, 3, tr::triad_pattern::h_motif);
  EXPECT_EQ(bins[0], 1u);
  EXPECT_EQ(count(same, 3, tr::triad_pattern::stathyper_type1), (std::vector<std::uint64_t>{3}));
  EXPECT_EQ(count(same, 3, tr::triad_pattern::stathyper_type2), (std::vector<std::uint64_t>{0}));
  EXPECT_EQ(count(same, 3, tr::triad_pattern::stathyper_type3), (std::vector<std::uint64_t>{0}));
  // Four copies: C(4, 3) = 4 triangles, all class 1.
  bins = count({{5}, {5}, {5}, {5}}, 6, tr::triad_pattern::h_motif);
  EXPECT_EQ(bins[0], 4u);
  // Dead rows (empty) take part in nothing; repeated members count once.
  const rows with_dead = {{}, {0, 1, 1, 2}, {}, {0, 1, 2}, {2, 0, 1}, {}};
  std::vector<std::uint64_t> one_class1(30, 0);
  one_class1[0] = 1;
  EXPECT_EQ(count(with_dead, 3, tr::triad_pattern::h_motif), one_class1);
  EXPECT_EQ(count(with_dead, 3, tr::triad_pattern::stathyper_type1),
            (std::vector<std::uint64_t>{3}));
  // A hyperedge of 1,024 members: C(1024, 3) in 64 bits.
  std::vector<std::int32_t> big(1024);
  for (std::int32_t v = 0; v < 1024; ++v) big[static_cast<std::size_t>(v)] = v;
  EXPECT_EQ(count({big, {0}}, 1024, tr::triad_pattern::stathyper_type1),
            (std::vector<std::uint64_t>{1024ull * 1023 * 1022 / 6}));
}

// An independent brute force: every triple of live hyperedges, classified with ESCHER's table
// (the fixture), plus the StatHyper formulas over all pairs and triples.
struct brute {
  std::vector<std::uint64_t> motif = std::vector<std::uint64_t>(30, 0);
  std::uint64_t type1 = 0, type2 = 0, type3 = 0;
};

std::size_t common(const std::set<std::int32_t>& a, const std::set<std::int32_t>& b) {
  std::size_t n = 0;
  for (const auto v : a) n += b.count(v);
  return n;
}

brute brute_force(const rows& r) {
  std::vector<std::set<std::int32_t>> s;
  for (const auto& row : r) s.emplace_back(row.begin(), row.end());
  brute out;
  const std::size_t m = s.size();
  for (std::size_t i = 0; i < m; ++i) {
    const std::uint64_t n = s[i].size();
    if (n >= 3) out.type1 += n * (n - 1) * (n - 2) / 6;
    for (std::size_t j = i + 1; j < m; ++j) {
      const std::uint64_t a = common(s[i], s[j]);
      const std::uint64_t b = s[i].size() - a, c = s[j].size() - a;
      if (a >= 2 && b > 0 && c > 0) out.type2 += a * (a - 1) / 2 * b * c;
      for (std::size_t k = j + 1; k < m; ++k) {
        if (common(s[i], s[j]) == 0 || common(s[j], s[k]) == 0 || common(s[i], s[k]) == 0) continue;
        std::size_t g = 0;
        for (const auto v : s[i]) g += s[j].count(v) && s[k].count(v);
        const std::uint64_t x = common(s[i], s[j]) - g, y = common(s[j], s[k]) - g,
                            z = common(s[i], s[k]) - g;
        if (x > 0 && y > 0 && z > 0) out.type3 += x * y * z;
        const int id = tr::triad_venn_id(r[i], r[j], r[k]);
        const int cls = dyng::test::escher_fixture::motif_id_to_index[static_cast<std::size_t>(id)];
        EXPECT_GT(cls, 0);
        if (cls > 0) ++out.motif[static_cast<std::size_t>(cls - 1)];
      }
    }
  }
  return out;
}

rows random_rows(std::mt19937_64& rng, int m, int n, int max_card) {
  std::uniform_int_distribution<int> card(0, max_card), vertex(0, n - 1);
  rows r(static_cast<std::size_t>(m));
  for (auto& row : r) {
    const int k = card(rng);  // 0: a dead id
    for (int t = 0; t < k; ++t) row.push_back(vertex(rng));
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
  }
  return r;
}

TEST(TriadReference, EqualsABruteForceOverAllTriplesOnEveryThreadCount) {
  std::mt19937_64 rng(20261015);
  for (int round = 0; round < 40; ++round) {
    const int n = 4 + round % 9;
    const rows r = random_rows(rng, 6 + round % 25, n, 1 + round % 6);
    const brute want = brute_force(r);
    for (const int threads : {1, 3, 0}) {
      EXPECT_EQ(count(r, n, tr::triad_pattern::h_motif, threads), want.motif) << round;
      EXPECT_EQ(count(r, n, tr::triad_pattern::stathyper_type1, threads)[0], want.type1);
      EXPECT_EQ(count(r, n, tr::triad_pattern::stathyper_type2, threads)[0], want.type2);
      EXPECT_EQ(count(r, n, tr::triad_pattern::stathyper_type3, threads)[0], want.type3);
    }
    const csr c = to_csr(r, n);
    const auto all = tr::triad_reference_all(
        c.offsets, c.members, n,
        {tr::triad_pattern::h_motif, tr::triad_pattern::stathyper_type1,
         tr::triad_pattern::stathyper_type2, tr::triad_pattern::stathyper_type3});
    EXPECT_EQ(all.of(tr::triad_pattern::h_motif), want.motif);
    EXPECT_EQ(all.type1, want.type1);
    EXPECT_EQ(all.type2, want.type2);
    EXPECT_EQ(all.type3, want.type3);
    // Open triads (bins 20-25) are never counted.
    for (int b = 20; b < 26; ++b) EXPECT_EQ(all.h_motif[static_cast<std::size_t>(b)], 0u);
  }
}

TEST(TriadReference, RemovedAndAddedCountTheTriadsTouchingTheIds) {
  std::mt19937_64 rng(7);
  for (int round = 0; round < 20; ++round) {
    const int n = 6;
    const rows r = random_rows(rng, 12, n, 4);
    std::vector<std::int64_t> ids = {1, 4, 4, 7, -1, 99};  // repeats and out-of-range ignored
    rows without = r;
    for (const auto id : ids)
      if (id >= 0 && id < static_cast<std::int64_t>(without.size()))
        without[static_cast<std::size_t>(id)].clear();
    const brute all = brute_force(r), rest = brute_force(without);
    const csr c = to_csr(r, n);
    const auto removed =
        tr::triad_reference_removed(c.offsets, c.members, n, ids, tr::triad_pattern::h_motif);
    for (std::size_t b = 0; b < 30; ++b) EXPECT_EQ(removed[b], all.motif[b] - rest.motif[b]);
    const auto added =
        tr::triad_reference_added(c.offsets, c.members, n, ids, tr::triad_pattern::stathyper_type3);
    EXPECT_EQ(added, (std::vector<std::uint64_t>{all.type3 - rest.type3}));
    EXPECT_EQ(tr::triad_reference_removed(c.offsets, c.members, n, ids,
                                          tr::triad_pattern::stathyper_type1),
              (std::vector<std::uint64_t>{all.type1 - rest.type1}));
    EXPECT_EQ(
        tr::triad_reference_added(c.offsets, c.members, n, ids, tr::triad_pattern::stathyper_type2),
        (std::vector<std::uint64_t>{all.type2 - rest.type2}));
  }
}

TEST(TriadReference, RefusesMalformedInput) {
  const std::vector<std::int32_t> members = {0, 1, 2};
  EXPECT_THROW((void)tr::triad_reference({0, 3}, members, 2, tr::triad_pattern::h_motif),
               std::invalid_argument);  // member 2 out of range
  EXPECT_THROW((void)tr::triad_reference({0, 2}, members, 3, tr::triad_pattern::h_motif),
               std::invalid_argument);  // offsets do not end at the member count
  EXPECT_THROW((void)tr::triad_reference({1, 3}, members, 3, tr::triad_pattern::h_motif),
               std::invalid_argument);  // offsets do not start at 0
  EXPECT_THROW((void)tr::triad_reference({0, 2, 1, 3}, members, 3, tr::triad_pattern::h_motif),
               std::invalid_argument);  // decreasing offsets
  EXPECT_THROW((void)tr::triad_reference({}, {}, 3, tr::triad_pattern::h_motif),
               std::invalid_argument);
  EXPECT_THROW((void)tr::triad_reference({0, 3}, members, -1, tr::triad_pattern::h_motif),
               std::invalid_argument);
  EXPECT_THROW((void)tr::triad_reference({0, 3}, members, 3, tr::triad_pattern::h_motif, {-2}),
               std::invalid_argument);
  // An empty hypergraph counts nothing; vertex ids up to 2^31 - 2 are accepted.
  EXPECT_EQ(tr::triad_reference({0}, {}, 0, tr::triad_pattern::stathyper_type1),
            (std::vector<std::uint64_t>{0}));
  const std::int32_t top = INT32_MAX - 1;
  const rows high = {{0, top}, {1, top}, {0, 1, top}};
  const int cls = dyng::test::escher_fixture::motif_id_to_index[static_cast<std::size_t>(
      tr::triad_venn_id(high[0], high[1], high[2]))];
  ASSERT_GT(cls, 0);
  EXPECT_EQ(count(high, INT32_MAX, tr::triad_pattern::h_motif)[static_cast<std::size_t>(cls - 1)],
            1u);
  EXPECT_EQ(tr::triad_pattern_name(tr::triad_pattern::stathyper_type2),
            std::string("stathyper_type2"));
  EXPECT_EQ(tr::triad_num_bins(tr::triad_pattern::h_motif), 30);
  EXPECT_EQ(tr::triad_num_bins(tr::triad_pattern::stathyper_type3), 1);
}

}  // namespace
