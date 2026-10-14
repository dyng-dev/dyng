// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_device_coverage_test.cpp
 * @brief Kernel-path coverage of the dynG CBST store (M8a step 3, acceptance criterion 14; PLAN
 *        6.4.5): replaying ESCHER-GPU@abcf9b2's skewed-degree and medium-degree traces (the cases
 *        of its tests/run_tests.sh that check its block-level and warp-level kernels) runs fill,
 *        reuse and unfill at thread, warp and block level, and every operation launches exactly
 *        the degree-bin kernels ESCHER-GPU recorded (the `exact` field set compares the per-op
 *        counts).
 *
 * Labels gpu, parity, cbst and slow: the traces are golden sets in
 * $DYNG_SCRATCH/goldens/cbst_trace (escher_gpu_test, cases 042-main and 043-main); the test skips
 * when they or a device are absent.
 */
#include "parity/escher/cbst_compare.hpp"
#include "parity/escher/cbst_device_adapter.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace dyng::test::cbst_compare {
namespace {

namespace fs = std::filesystem;

fs::path scratch() {
  if (const char* s = std::getenv("DYNG_SCRATCH")) return s;
  const char* home = std::getenv("HOME");
  return fs::path(home ? home : "/") / "Projects" / "dyng-work";
}

TEST(CbstKernelPaths, SkewedAndMediumTracesRunEveryDegreeBin) {
  if (!device_available()) GTEST_SKIP() << "no CUDA device";
  const fs::path set = scratch() / "goldens" / "cbst_trace" / "escher_gpu_test";
  const fs::path skewed = set / "042-main" / "trace.cbt.zst";
  const fs::path medium = set / "043-main" / "trace.cbt.zst";
  if (!fs::is_regular_file(skewed) || !fs::is_regular_file(medium)) {
    GTEST_SKIP() << "the escher_gpu_test traces are not under " << set
                 << " (parity/export_goldens.py cbst_trace escher_gpu_test)";
  }
  std::array<std::int64_t, 9> total{};
  for (const fs::path& path : {skewed, medium}) {
    SCOPED_TRACE(path.string());
    trace_source source(path.string(), fs::temp_directory_path().string());
    device_store_adapter store("escher_gpu");
    options opt;
    opt.fields = field_set::exact;  // the per-op launch counts are compared
    const replay_report r = replay(source.reader(), store, opt, path.string());
    EXPECT_TRUE(r.ok()) << r.mismatch_count << " mismatches";
    EXPECT_EQ(r.compared, r.ops);
    for (std::size_t i = 0; i < total.size(); ++i) total[i] += store.total_bins()[i];
  }
  static const char* const names[9] = {"fill/thread",   "fill/warp",   "fill/block",
                                       "reuse/thread",  "reuse/warp",  "reuse/block",
                                       "unfill/thread", "unfill/warp", "unfill/block"};
  for (std::size_t i = 0; i < total.size(); ++i) {
    EXPECT_GT(total[i], 0) << names[i] << " never ran";
    std::printf("%s: %lld launches\n", names[i], static_cast<long long>(total[i]));
    RecordProperty(names[i], static_cast<int>(total[i]));
  }
}

}  // namespace
}  // namespace dyng::test::cbst_compare
