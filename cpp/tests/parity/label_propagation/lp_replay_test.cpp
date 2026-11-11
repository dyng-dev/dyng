// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file lp_replay_test.cpp
 * @brief The committed digests of LabelPropagation-CUDA@a276a3a's recordings (M10a step 2; AD8;
 *        cpp/tests/data/goldens/label_propagation, written by parity/label_propagation_goldens.py
 *        subset from the golden sets, i.e. from O's own arrays and files): the legacy generator
 *        reproduces the original's arrays and text files at 5k5 and 50k (seed 42), and
 *        lp_host_model fed the generated batches predicts every deterministic field of O's 5k5
 *        stage trace. These run in the CPU suites, so the hosted gcc-13 and clang-18 jobs check
 *        them as well; the whole corpus is replayed by parity/compare.py label_propagation.
 */
#include "parity/label_propagation/lp_replay.hpp"

#include "generators/dynlp_stream.hpp"
#include "io/dynlp_text.hpp"
#include "support/data_paths.hpp"
#include "support/sha256.hpp"

#include <dyng/testing/lp_host_model.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace lp = dyng::test::lp;

/// "<sha256>  <name>" lines ('#' comments).
std::map<std::string, std::string> read_digests(const std::string& file) {
  std::map<std::string, std::string> out;
  std::istringstream lines(dyng::test::read_text(dyng::test::data_path(file)));
  std::string line;
  while (std::getline(lines, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream words(line);
    std::string sha;
    std::string name;
    words >> sha >> name;
    out[name] = sha;
  }
  return out;
}

dyng::generators::vertex_arrival_options options(std::int64_t n, std::int32_t batches) {
  dyng::generators::vertex_arrival_options o;
  o.total_vertices = n;
  o.num_batches = batches;
  o.average_degree = 5;
  o.seed = 42;
  return o;
}

void expect_generator_digests(const std::string& file, std::int64_t n, std::int32_t batches) {
  const auto want = read_digests(file);
  ASSERT_GE(want.size(), 8U);
  const auto out = dyng::detail::dynlp_legacy_generate(options(n, batches), 2);
  const auto arrays = lp::generator_arrays(out);
  dyng::test::temp_dir tmp;
  std::vector<std::int8_t> labels(out.true_labels.size());
  for (std::size_t i = 0; i < labels.size(); ++i) {
    labels[i] = out.true_labels[i] >= 0.5F ? 1 : 0;
  }
  dyng::detail::write_dynlp_graph_files<std::int32_t, std::int32_t>(
      tmp.path("graph/csr"), out.row_ptr, out.col_ind, out.values, labels);
  std::size_t arrays_checked = 0;
  for (const auto& [name, sha] : want) {
    SCOPED_TRACE(name);
    if (name.size() > 4 && name.substr(name.size() - 4) == ".txt") {
      EXPECT_EQ(dyng::test::sha256_hex(dyng::test::read_text(tmp.path("graph/" + name))), sha);
      continue;
    }
    const auto it = arrays.find(name);
    ASSERT_NE(it, arrays.end());
    EXPECT_EQ(lp::digest(it->second), sha);
    ++arrays_checked;
  }
  EXPECT_EQ(arrays_checked, arrays.size());  // every generated array has its digest
}

TEST(LpReplay, GeneratorReproducesTheOriginal5k5) {
  expect_generator_digests("goldens/label_propagation/generator_5k5_d1e-4_s42.sha256", 5000, 5);
}

TEST(LpReplay, GeneratorReproducesTheOriginal50k) {
  expect_generator_digests("goldens/label_propagation/generator_50k_d1e-4_s42.sha256", 50000, 10);
}

TEST(LpReplay, HostModelReproducesTheOriginalsTrace5k5) {
  const auto want = read_digests("goldens/label_propagation/model_5k5_d1e-4_s42.sha256");
  const auto in = dyng::detail::dynlp_legacy_generate(options(5000, 5), 1);
  dyng::testing::lp_host_model model(lp::row_capacities(in.row_ptr));
  const float tau = dyng::testing::lp_host_model::average_weight(in.col_ind, in.values);
  EXPECT_EQ(tau, in.tau);
  std::size_t checked = 0;
  for (std::size_t b = 0; b < in.batches.size(); ++b) {
    const auto report = model.apply(lp::model_batch(in.batches[b], in.num_nodes), tau);
    for (const auto& [key, ours] :
         lp::model_fields(model, report, tau, static_cast<std::int32_t>(b))) {
      const std::string name = "b" + std::to_string(b) + "." + key;
      SCOPED_TRACE(name);
      const auto it = want.find(name);
      ASSERT_NE(it, want.end());
      EXPECT_EQ(lp::digest(ours), it->second);
      ++checked;
    }
  }
  EXPECT_EQ(checked, want.size());  // every recorded field is predicted
}

}  // namespace
