// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_count_docs_test.cpp
 * @brief The 30-bin table of docs/algorithms/triad_count.md against the code's tables (M9b,
 *        acceptance criterion 8): the counting engine's Venn-id map (engine/motif_table.hpp) and
 *        the oracle's representatives (testing::triad_class_representatives()).
 *
 * Every row of the page's table (`| bin | h-motif | canonical Venn id | regions | counted |`) must
 * name bin k as ESCHER-GPU's h-motif k + 1, its canonical id must be the bitmask of its listed
 * regions and the smallest id that the engine's table maps to h-motif k + 1, and the open rows
 * (bins 20-25) must say that they are always 0. The page lists exactly the 30 bins in order.
 */
#include "algorithms/triad_count/engine/motif_table.hpp"

#include <dyng/testing/triad_reference.hpp>

#include <gtest/gtest.h>

#include <array>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::array<int, 128> engine_table = DYNG_TRIAD_MOTIF_ID_TO_INDEX;

struct row {
  int bin = 0;
  int motif = 0;
  int canonical = 0;
  std::string regions;
  std::string counted;
};

std::vector<row> page_rows() {
  std::ifstream in(DYNG_DOCS_DIR "/algorithms/triad_count.md");
  EXPECT_TRUE(in.good()) << "cannot read " << DYNG_DOCS_DIR "/algorithms/triad_count.md";
  const std::regex line(R"(^\| (\d+) \| (\d+) \| (\d+) \| ([A-C, ]+) \| (.+) \|$)");
  std::vector<row> rows;
  std::string text;
  while (std::getline(in, text)) {
    std::smatch m;
    if (std::regex_match(text, m, line)) {
      rows.push_back({std::stoi(m[1]), std::stoi(m[2]), std::stoi(m[3]), m[4], m[5]});
    }
  }
  return rows;
}

int mask_of(const std::string& regions) {
  // Bits as in motif_table.hpp: 0..2 = A, B, C only; 3 = AB, 4 = BC, 5 = CA (only); 6 = ABC.
  const std::map<std::string, int> bit{{"A", 0},  {"B", 1},  {"C", 2},  {"AB", 3},
                                       {"BC", 4}, {"CA", 5}, {"ABC", 6}};
  int mask = 0;
  std::stringstream ss(regions);
  std::string name;
  while (std::getline(ss, name, ',')) {
    name.erase(0, name.find_first_not_of(' '));
    name.erase(name.find_last_not_of(' ') + 1);
    const auto it = bit.find(name);
    EXPECT_NE(it, bit.end()) << "unknown region '" << name << "'";
    if (it != bit.end()) mask |= 1 << it->second;
  }
  return mask;
}

TEST(TriadCountDocs, TheThirtyBinTableIsTheCodesTable) {
  const std::vector<row> rows = page_rows();
  ASSERT_EQ(rows.size(), 30U) << "the page must list the 30 bins, one table row each";
  const auto& reps = dyng::testing::triad_class_representatives();
  for (int k = 0; k < 30; ++k) {
    const row& r = rows[static_cast<std::size_t>(k)];
    SCOPED_TRACE("bin " + std::to_string(k));
    EXPECT_EQ(r.bin, k);
    EXPECT_EQ(r.motif, k + 1);  // 0-based bins, ESCHER-GPU's ids 1-30
    EXPECT_EQ(r.canonical, mask_of(r.regions));
    EXPECT_EQ(r.canonical, reps[static_cast<std::size_t>(k)]);
    EXPECT_EQ(engine_table[static_cast<std::size_t>(r.canonical)], k + 1);
    for (int id = 0; id < r.canonical; ++id) {  // the smallest id of its class
      EXPECT_NE(engine_table[static_cast<std::size_t>(id)], k + 1) << "id " << id;
    }
    const bool open = k >= 20 && k <= 25;
    EXPECT_EQ(r.counted.rfind("always 0", 0) == 0, open) << r.counted;
  }
}

TEST(TriadCountDocs, EveryClassifiedIdHasItsRowsClass) {
  // Every Venn id the engine classifies falls in one of the 30 classes the page lists, and the
  // open classes (21-26) are exactly the ids where one pair of hyperedges shares no vertex.
  for (int id = 0; id < 128; ++id) {
    const int motif = engine_table[static_cast<std::size_t>(id)];
    if (motif == 0) continue;
    ASSERT_GE(motif, 1);
    ASSERT_LE(motif, 30);
    const bool ab = (id >> 3 & 1) || (id >> 6 & 1);
    const bool bc = (id >> 4 & 1) || (id >> 6 & 1);
    const bool ca = (id >> 5 & 1) || (id >> 6 & 1);
    EXPECT_EQ(motif >= 21 && motif <= 26, !(ab && bc && ca)) << "id " << id;
  }
}

}  // namespace
