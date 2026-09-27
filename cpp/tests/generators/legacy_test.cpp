// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file legacy_test.cpp
 * @brief generators::legacy::mosp_changes() reproduces `mospPrep changes` of MOSP-OpenMP@c352151
 *        (and MOSP-CUDA@e220ee2, which agrees) byte for byte for fixed seeds: every mode, local and
 *        safe batches, written with io::write_legacy_batch(), and the report line.
 *
 * The fixtures (cpp/tests/data/mosp_changes) come from the originals' own tool:
 * parity/fixtures/generators/make_generator_fixtures.sh.
 */
#include "support/data_paths.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/csr_triplet.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace {

using dyng::test::data_path;
using dyng::test::read_text;
namespace legacy = dyng::generators::legacy;

struct fixture_case {
  std::string name;
  legacy::mosp_change_options options;
};

/// cases.txt: "<name> <mospPrep changes options>" per line.
std::vector<fixture_case> load_cases() {
  std::vector<fixture_case> out;
  std::istringstream lines(read_text(data_path("mosp_changes/cases.txt")));
  std::string line;
  while (std::getline(lines, line)) {
    std::istringstream words(line);
    fixture_case c;
    if (!(words >> c.name)) {
      continue;
    }
    std::string flag;
    while (words >> flag) {
      if (flag == "--safe") {
        c.options.safe_deletions = true;
        continue;
      }
      std::string value;
      words >> value;
      if (flag == "--changes") {
        c.options.num_changes = std::stoll(value);
      } else if (flag == "--ins") {
        c.options.insertion_percentage = std::stod(value);
      } else if (flag == "--mode") {
        c.options.mode = value == "targeted"   ? legacy::mosp_change_mode::targeted
                         : value == "reweight" ? legacy::mosp_change_mode::reweight
                         : value == "increase" ? legacy::mosp_change_mode::increase
                                               : legacy::mosp_change_mode::uniform;
      } else if (flag == "--local") {
        c.options.local_hops = std::stoll(value);
      } else if (flag == "--seed") {
        c.options.seed = static_cast<std::uint32_t>(std::stoul(value));
      } else if (flag == "--source") {
        c.options.source = std::stoll(value);
      } else if (flag == "--wmin") {
        c.options.weight_min = std::stoi(value);
      } else if (flag == "--wmax") {
        c.options.weight_max = std::stoi(value);
      } else {
        ADD_FAILURE() << "unknown option " << flag << " in cases.txt";
      }
    }
    out.push_back(c);
  }
  return out;
}

template <typename graph_types>
class LegacyMospChanges : public ::testing::Test {};

template <typename vertex_t, typename edge_t>
struct index_types {
  using vertex_type = vertex_t;
  using edge_type = edge_t;
};

using types = ::testing::Types<index_types<std::int32_t, std::int32_t>,
                               index_types<std::int32_t, std::int64_t>,
                               index_types<std::int64_t, std::int64_t>>;
TYPED_TEST_SUITE(LegacyMospChanges, types, dyng::test::type_index_name);

TYPED_TEST(LegacyMospChanges, ReproducesMospPrepChangesByteForByte) {
  using vertex_t = typename TypeParam::vertex_type;
  using edge_t = typename TypeParam::edge_type;
  const auto graph = dyng::io::read_csr_triplet<vertex_t, edge_t, std::int32_t>(
      data_path("mosp_changes/graph/graphCsr"));
  ASSERT_EQ(graph.num_weights, 3);
  const std::vector<fixture_case> cases = load_cases();
  ASSERT_EQ(cases.size(), 15U);
  dyng::test::temp_dir tmp;
  for (const fixture_case& c : cases) {
    SCOPED_TRACE(c.name);
    legacy::mosp_change_report report;
    const auto batch = legacy::mosp_changes(graph.view(), c.options, &report);
    const std::string out = tmp.path(c.name + "/");
    dyng::io::write_legacy_batch(out + "insert.txt", out + "delete.txt", batch.view());
    const std::string expected = data_path("mosp_changes/" + c.name + "/");
    EXPECT_TRUE(read_text(out + "insert.txt") == read_text(expected + "insert.txt"));
    EXPECT_TRUE(read_text(out + "delete.txt") == read_text(expected + "delete.txt"));
    EXPECT_EQ(report.summary() + "\n", read_text(expected + "report.txt"));
  }
}

TEST(LegacyMospChanges, SameSeedSameBatchOtherSeedOtherBatch) {
  const auto graph = dyng::io::read_csr_triplet<std::int32_t, std::int32_t, std::int32_t>(
      data_path("mosp_changes/graph/graphCsr"));
  legacy::mosp_change_options opt;
  opt.num_changes = 50;
  const auto a = legacy::mosp_changes(graph.view(), opt);
  const auto b = legacy::mosp_changes(graph.view(), opt);
  EXPECT_EQ(a.insert_dst(), b.insert_dst());
  EXPECT_EQ(a.insert_weights(), b.insert_weights());
  EXPECT_EQ(a.delete_src(), b.delete_src());
  opt.seed = 2;
  const auto c = legacy::mosp_changes(graph.view(), opt);
  EXPECT_NE(a.insert_dst(), c.insert_dst());
}

TEST(LegacyMospChanges, RejectsMospsInvalidOptions) {
  const auto graph = dyng::io::read_csr_triplet<std::int32_t, std::int32_t, std::int32_t>(
      data_path("mosp_changes/graph/graphCsr"));
  auto expect_invalid = [&](legacy::mosp_change_options opt) {
    EXPECT_THROW((void)legacy::mosp_changes(graph.view(), opt), dyng::invalid_argument_error);
  };
  legacy::mosp_change_options opt;
  opt.num_changes = 10;
  auto bad = opt;
  bad.num_changes = -1;
  expect_invalid(bad);
  bad = opt;
  bad.weight_min = 0;
  expect_invalid(bad);
  bad = opt;
  bad.weight_min = 9;
  bad.weight_max = 8;
  expect_invalid(bad);
  bad = opt;
  bad.source = graph.num_vertices();
  expect_invalid(bad);
  // A graph without edges has nothing to delete (MOSP: "No existing edges available").
  dyng::csr<std::int32_t, std::int32_t, std::int32_t> empty;
  empty.row_ptr.assign(4, 0);
  empty.num_weights = 1;
  EXPECT_NO_THROW((void)legacy::mosp_changes(graph.view(), opt));
  EXPECT_THROW((void)legacy::mosp_changes(empty.view(), opt), dyng::invalid_argument_error);
}

}  // namespace
