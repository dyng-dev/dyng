// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_ids_test.cpp
 * @brief Container-level id parity on the committed traces and batch extracts (M8b, criteria 4
 *        and 5): every backend gives ESCHER-GPU's and MOSP_ESCHER's ids, rows and skipped counts,
 *        in lockstep with the host model; ESCHER-GPU's hypergraph and batches drawn again by
 *        generators::legacy from the containers' state (criterion 16); on CUDA, the kernel paths
 *        of A's skewed and medium traces (criterion 12; from $DYNG_SCRATCH, skipped without it).
 */
#include "parity/escher/hypergraph_ids.hpp"

#include "parity/escher/cbst_compare.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/resources.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace ids = dyng::test::hypergraph_ids;

fs::path committed() {
  return fs::path(DYNG_TEST_DATA_DIR) / "goldens" / "cbst_trace";
}

fs::path batches() {
  return fs::path(DYNG_TEST_DATA_DIR) / "goldens" / "hypergraph_batches";
}

fs::path scratch() {
  if (const char* s = std::getenv("DYNG_SCRATCH")) return s;
  const char* home = std::getenv("HOME");
  return fs::path(home ? home : "/") / "Projects" / "dyng-work";
}

std::vector<dyng::resources> backends() {
  std::vector<dyng::resources> out;
  for (const dyng::backend b : dyng::test::suite_backends()) {
    if (b == dyng::backend::sequential) out.push_back(dyng::resources::sequential());
    if (b == dyng::backend::openmp) {
      out.push_back(dyng::resources::openmp(1));
      out.push_back(dyng::resources::openmp(8));
    }
    if (b == dyng::backend::cuda) out.push_back(dyng::resources::cuda(0));
  }
  return out;
}

ids::report run(const fs::path& path, const dyng::resources& res) {
  dyng::test::cbst_compare::trace_source source(path.string(), fs::temp_directory_path().string());
  ids::options opt;
  opt.legacy_generator = true;  // ESCHER-GPU's traces: the legacy generators draw every batch
  return ids::replay(source.reader(), res, opt, path.string());
}

TEST(HypergraphIds, CommittedTracesGiveTheOriginalsIds) {
  const std::vector<fs::path> traces = {
      committed() / "escher_gpu_test" / "001-main.cbt",
      batches() / "escher_gpu_016-main.cbt",
      batches() / "mosp_escher_014-test_h2h_delta.cbt",
  };
  for (const dyng::resources& res : backends()) {
    for (const fs::path& path : traces) {
      SCOPED_TRACE(path.filename().string() + " on " +
                   std::string(dyng::to_string(res.get_backend())) + " (" +
                   std::to_string(res.num_threads()) + " threads)");
      const ids::report r = run(path, res);
      EXPECT_TRUE(r.ok()) << r.to_json();
      EXPECT_GT(r.batches, 0);
      EXPECT_GT(r.ids_compared, 0);
      if (r.repo == "ESCHER-GPU") {
        EXPECT_EQ(r.generated_batches, r.batches);
        EXPECT_EQ(r.generated_batches_equal, r.batches);
      }
      if (r.repo == "MOSP_ESCHER") {
        EXPECT_EQ(r.skipped_compared, r.batches);
        EXPECT_GT(r.vertex_batches, 0);  // the vertex-kind batches stop their sessions (M11)
      }
    }
  }
}

TEST(HypergraphIds, KernelPathsOfTheSkewedAndMediumTraces) {
  if (!dyng::backend_available(dyng::backend::cuda)) GTEST_SKIP() << "no CUDA device";
  bool cuda_suite = false;
  for (const dyng::backend b : dyng::test::suite_backends()) cuda_suite |= b == dyng::backend::cuda;
  if (!cuda_suite) GTEST_SKIP() << "the CUDA test executable runs it";
  const fs::path set = scratch() / "goldens" / "cbst_trace" / "escher_gpu_test";
  const fs::path skewed = set / "042-main" / "trace.cbt.zst";
  const fs::path medium = set / "043-main" / "trace.cbt.zst";
  if (!fs::is_regular_file(skewed) || !fs::is_regular_file(medium)) {
    GTEST_SKIP() << "the escher_gpu_test traces are not under " << set;
  }
  std::array<std::int64_t, 9> total{};
  for (const fs::path& path : {skewed, medium}) {
    SCOPED_TRACE(path.string());
    const ids::report r = run(path, dyng::resources::cuda(0));
    EXPECT_TRUE(r.ok()) << r.to_json();
    for (std::size_t i = 0; i < total.size(); ++i) total[i] += r.paths[i];
  }
  static const char* const names[9] = {"fill/thread",   "fill/warp",   "fill/block",
                                       "reuse/thread",  "reuse/warp",  "reuse/block",
                                       "unfill/thread", "unfill/warp", "unfill/block"};
  for (std::size_t i = 0; i < total.size(); ++i) {
    EXPECT_GT(total[i], 0) << names[i] << " never ran";
    std::printf("%s: %lld\n", names[i], static_cast<long long>(total[i]));
  }
}

}  // namespace
