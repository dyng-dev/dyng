// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_enum_datasets_test.cpp
 * @brief Parity with CycleEnumeration-GPU@0a976ad on the TUDataset graphs (CTest label parity):
 *        the parser output, the CSR, the generated batches (seed 1; 1K+1K, 25K+25K, 50K+50K and
 *        locality windows) and the normalized batch and CSR after each batch are equal to the
 *        original's, compared through the FNV-1a digests of cpp/tests/data/cycle_enum/datasets.txt
 *        (parity/fixtures/cycle_enum/make_cycle_enum_fixtures.sh --datasets).
 *
 * The datasets are not in the repository: the test reads them from $DYNG_CYCLE_DATASETS, else
 * $DYNG_SCRATCH/datasets/cycle, else ~/Projects/dyng-work/datasets/cycle, and skips a dataset that
 * is missing.
 */
#include "graph/graph_impl.hpp"
#include "support/cycle_enum_text.hpp"
#include "support/data_paths.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/io/edge_list_io.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using dyng::unweighted;
using graph_t = dyng::graph<std::int32_t, std::int64_t, unweighted>;

std::string dataset_root() {
  if (const char* dir = std::getenv("DYNG_CYCLE_DATASETS")) {
    return dir;
  }
  if (const char* scratch = std::getenv("DYNG_SCRATCH")) {
    return std::string(scratch) + "/datasets/cycle";
  }
  const char* home = std::getenv("HOME");
  return std::string(home == nullptr ? "" : home) + "/Projects/dyng-work/datasets/cycle";
}

/// One line of datasets.txt.
struct digest_line {
  std::string what;
  std::string file;
  std::vector<std::int64_t> numbers;  ///< deletions, insertions, seed [, window]
  std::string digest;                 ///< "fnv1a64 <hex> bytes <n>" or "error"
};

std::vector<digest_line> read_digests() {
  std::vector<digest_line> out;
  std::istringstream in(dyng::test::read_text(dyng::test::data_path("cycle_enum/datasets.txt")));
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream fields(line);
    digest_line d;
    fields >> d.what >> d.file;
    std::string token;
    while (fields >> token && token != "fnv1a64" && token != "error") {
      d.numbers.push_back(std::stoll(token));
    }
    if (token == "error") {
      d.digest = "error";
    } else {
      std::string hex;
      std::string bytes;
      std::string size;
      fields >> hex >> bytes >> size;
      d.digest = "fnv1a64 " + hex + " bytes " + size;
    }
    out.push_back(d);
  }
  return out;
}

class CycleEnumDatasets : public ::testing::Test {
 protected:
  /// The cycle_enum_compatible() graph of a dataset (read once per test program).
  static const graph_t* graph_of(const std::string& path) {
    static std::map<std::string, std::unique_ptr<graph_t>> cache;
    auto& slot = cache[path];
    if (!slot) {
      const auto edges = dyng::io::read_edge_list<std::int32_t, unweighted>(path);
      slot = std::make_unique<graph_t>(
          graph_t::from_edges(dyng::resources::openmp(), edges.view(),
                              dyng::graph_properties::cycle_enum_compatible()));
    }
    return slot.get();
  }
};

TEST_F(CycleEnumDatasets, DigestsEqualTheOriginal) {
  const std::vector<digest_line> lines = read_digests();
  ASSERT_GE(lines.size(), 29u);
  const std::string root = dataset_root();
  int checked = 0;
  for (const digest_line& d : lines) {
    const std::string path = root + "/" + d.file;
    if (!std::filesystem::exists(path)) {
      continue;
    }
    SCOPED_TRACE(d.what + " " + d.file);
    dyng::generators::legacy::cycle_enum_batch_options opt;
    if (d.numbers.size() >= 3) {
      opt.num_deletions = d.numbers[0];
      opt.num_insertions = d.numbers[1];
      opt.seed = static_cast<std::uint64_t>(d.numbers[2]);
    }
    if (d.numbers.size() == 4) {
      opt.locality_window = d.numbers[3];
    }
    std::string text;
    if (d.what == "parse") {
      dyng::io::edge_list_options options;
      options.duplicates = dyng::io::duplicate_edges::keep;
      dyng::io::edge_list_info info;
      const auto edges = dyng::io::read_edge_list<std::int32_t, unweighted>(path, options, &info);
      text = dyng::test::parse_text(edges, info);
    } else if (d.what == "csr") {
      text = dyng::test::csr_text(graph_of(path)->view().out);
    } else if (d.what == "generate") {
      if (d.digest == "error") {
        EXPECT_THROW(
            (void)dyng::generators::legacy::cycle_enum_batch(graph_of(path)->view().out, opt),
            dyng::invalid_argument_error);
        ++checked;
        continue;
      }
      const auto b = dyng::generators::legacy::cycle_enum_batch(graph_of(path)->view().out, opt);
      text = dyng::test::batch_text(b.delete_src(), b.delete_dst(), b.insert_src(), b.insert_dst());
    } else if (d.what == "apply-generated") {
      const auto res = dyng::resources::openmp();
      auto g = graph_of(path)->clone(res);
      const auto b = dyng::generators::legacy::cycle_enum_batch(g.view().out, opt);
      dyng::detail::apply_delta<std::int32_t> delta;
      (void)dyng::detail::graph_access::apply(res, g, b.view(), &delta);
      text = "prepared\n" +
             dyng::test::batch_text(delta.delete_src, delta.delete_dst, delta.insert_src,
                                    delta.insert_dst) +
             "after\n" + dyng::test::csr_text(g.view().out);
    } else {
      ADD_FAILURE() << "unknown digest kind " << d.what;
      continue;
    }
    EXPECT_EQ(dyng::test::fnv1a64_digest(text), d.digest);
    ++checked;
  }
  if (checked == 0) {
    GTEST_SKIP() << "no dataset found under " << root;
  }
}

}  // namespace
