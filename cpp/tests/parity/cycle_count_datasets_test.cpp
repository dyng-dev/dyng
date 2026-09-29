// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_count_datasets_test.cpp
 * @brief Parity of cycle_count with CycleEnumeration-GPU@0a976ad on the TUDataset graphs (CTest
 *        label parity): the static histograms (DD k = 3..7, GitHub and Twitch k = 3, 4, COLLAB
 *        k = 3) and the updated histograms of the seed-1 batches (1K+1K, 25K+25K, 50K+50K, k = 4)
 *        on DD, GitHub and Twitch equal the original's, recorded in
 *        cpp/tests/data/cycle_enum/datasets_counts.txt (make_cycle_enum_fixtures.sh --datasets).
 *
 * The OpenMP backend runs every case (threads: DYNG_CYCLE_PARITY_THREADS, default all). The
 * sequential backend runs the cases whose OpenMP count takes about a second at most (DD k <= 6
 * and the updates) unless DYNG_CYCLE_PARITY_FULL=1, which runs it on every case (a sequential
 * COLLAB k = 3 takes about half an hour). The datasets are read from $DYNG_CYCLE_DATASETS, else
 * $DYNG_SCRATCH/datasets/cycle, else ~/Projects/dyng-work/datasets/cycle; missing ones are skipped.
 * The committed digests of the datasets test (cycle_enum_datasets_test.cpp) already tie the graphs
 * and the batches to the original's.
 */
#include "support/cycle_count_support.hpp"
#include "support/data_paths.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/cycle_count.hpp>
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

bool env_flag(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && std::string(value) == "1";
}

int omp_threads() {
  const char* value = std::getenv("DYNG_CYCLE_PARITY_THREADS");
  return value == nullptr ? 0 : std::atoi(value);
}

struct count_line {
  std::string what;                   ///< count | update
  std::string file;                   ///< dataset file
  std::vector<std::int64_t> numbers;  ///< k [deletions insertions seed]
  std::string histogram;              ///< "<len>:<count> ..."
};

std::vector<count_line> read_lines() {
  std::vector<count_line> out;
  std::istringstream in(
      dyng::test::read_text(dyng::test::data_path("cycle_enum/datasets_counts.txt")));
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream fields(line);
    count_line c;
    fields >> c.what >> c.file;
    const std::size_t numbers = c.what == "count" ? 1 : 4;
    for (std::size_t i = 0; i < numbers; ++i) {
      std::int64_t v = 0;
      fields >> v;
      c.numbers.push_back(v);
    }
    std::getline(fields >> std::ws, c.histogram);
    out.push_back(c);
  }
  return out;
}

const graph_t& graph_of(const std::string& path) {
  static std::map<std::string, std::unique_ptr<graph_t>> cache;
  auto& slot = cache[path];
  if (!slot) {
    const auto edges = dyng::io::read_edge_list<std::int32_t, unweighted>(path);
    slot = std::make_unique<graph_t>(graph_t::from_edges(
        dyng::resources::openmp(), edges.view(), dyng::graph_properties::cycle_enum_compatible()));
  }
  return *slot;
}

TEST(CycleCountDatasets, HistogramsEqualTheOriginal) {
  const std::vector<count_line> lines = read_lines();
  ASSERT_EQ(lines.size(), 19U);
  const std::string root = dataset_root();
  const bool full = env_flag("DYNG_CYCLE_PARITY_FULL");
  int checked = 0;
  for (const count_line& c : lines) {
    const std::string path = root + "/" + c.file;
    if (!std::filesystem::exists(path)) {
      continue;
    }
    const int k = static_cast<int>(c.numbers[0]);
    const bool small = c.what == "update" || (c.file.rfind("DD/", 0) == 0 && k <= 6);
    std::vector<dyng::resources> backends = {dyng::resources::openmp(omp_threads())};
    if (full || small) {
      backends.push_back(dyng::resources::sequential());
    }
    dyng::cycle_count::options opt;
    opt.max_length = k;
    for (const dyng::resources& res : backends) {
      SCOPED_TRACE(c.what + " " + c.file + " k=" + std::to_string(k) + " " +
                   std::string(dyng::to_string(res.get_backend())));
      graph_t g = graph_of(path).clone(res);
      dyng::cycle_count::result r = dyng::cycle_count::compute(res, g, opt);
      if (c.what == "update") {
        dyng::generators::legacy::cycle_enum_batch_options bo;
        bo.num_deletions = c.numbers[1];
        bo.num_insertions = c.numbers[2];
        bo.seed = static_cast<std::uint64_t>(c.numbers[3]);
        const auto batch = dyng::generators::legacy::cycle_enum_batch(g.to_csr(res).view(), bo);
        (void)dyng::cycle_count::update(res, g, batch.view(), r);
      }
      EXPECT_EQ(dyng::test::cc_text(dyng::test::cc_counts(r)), c.histogram);
      ++checked;
    }
  }
  if (checked == 0) {
    GTEST_SKIP() << "no dataset found under " << root;
  }
}

}  // namespace
