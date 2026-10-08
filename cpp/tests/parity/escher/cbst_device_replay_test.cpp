// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_device_replay_test.cpp
 * @brief The dynG CBST store replays the committed CBST traces (cpp/tests/data/goldens/cbst_trace)
 *        on the device (M8a step 3, ADR 0038).
 *
 * - The traces of ESCHER-GPU@abcf9b2 under escher_gpu(): no mismatch in the `exact` field set
 *   (offsets, occupancy, tail capacity, the bump pointer, the capacity, deviceBytes(), the
 *   degree-bin launches of every op, the integrity verdict), apart from the documented
 *   divergences of the port (replay_report::documented).
 * - The traces of MOSP_ESCHER@4b86159 under mosp_escher() (M8a step 5, ADR 0039): no mismatch
 *   in the `exact` field set apart from the documented divergences (constructs without row
 *   counts, a construct below its payload), the automatic compactions at the same ops.
 * - The traces of MOSP_ESCHER under escher_gpu(): every tree diverges at exactly the op and in
 *   exactly the field where the host model under the same preset diverges (the behaviour
 *   differences of ADR 0037), and nowhere else.
 * - A replay run twice gives identical digests.
 */
#include "parity/escher/cbst_compare.hpp"
#include "parity/escher/cbst_device_adapter.hpp"
#include "parity/escher/cbst_trace.hpp"
#include "support/data_paths.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test::cbst_compare {
namespace {

std::vector<std::string> committed_traces(const std::string& repo_dir) {
  std::vector<std::string> out;
  const std::filesystem::path root = data_path("goldens/cbst_trace");
  if (!std::filesystem::is_directory(root)) return out;
  for (const auto& e : std::filesystem::recursive_directory_iterator(root))
    if (e.is_regular_file() && e.path().extension() == ".cbt" &&
        e.path().parent_path().filename().string().rfind(repo_dir, 0) == 0)
      out.push_back(e.path().string());
  std::sort(out.begin(), out.end());
  return out;
}

std::string describe(const replay_report& r) {
  std::ostringstream s;
  s << r.mismatch_count << " mismatches";
  for (const auto& m : r.mismatches)
    s << "\n  op " << m.op_index << " (tree " << m.tree << " " << m.dataset << ", its op "
      << m.tree_op << ", " << m.op << ") " << m.field << " key " << m.key << ": trace "
      << m.expected << ", store " << m.actual;
  return s.str();
}

replay_report replay_with(const std::string& path, const store_factory& make, field_set fields,
                          std::size_t max_mismatches = 20) {
  auto in = cbst_trace::reader::open(path);
  options opt;
  opt.fields = fields;
  opt.max_mismatches = max_mismatches;
  return replay(*in, make, opt, path);
}

#define DYNG_SKIP_WITHOUT_DEVICE()                             \
  do {                                                         \
    if (!device_available()) GTEST_SKIP() << "no CUDA device"; \
  } while (false)

TEST(CbstDeviceReplay, EscherGpuTracesMatchExactly) {
  DYNG_SKIP_WITHOUT_DEVICE();
  const auto traces = committed_traces("escher_gpu");
  ASSERT_FALSE(traces.empty());
  std::int64_t ops = 0;
  for (const auto& path : traces) {
    SCOPED_TRACE(path);
    const replay_report r = replay_with(path, device_store_factory("escher_gpu"), field_set::exact);
    EXPECT_TRUE(r.ok()) << describe(r);
    EXPECT_EQ(r.unreplayable, 0);
    EXPECT_EQ(r.compared, r.ops);
    // The only documented divergence of ESCHER-GPU's traces: the targeted construct below its
    // payload (ADR 0038).
    for (const auto& d : r.documented) {
      EXPECT_EQ(d.rule, "construct_below_payload");
      EXPECT_EQ(d.dataset, "targeted-small_capacity");
    }
    ops += r.ops;
  }
  EXPECT_GT(ops, 40);
}

// The first divergence of every tree (op of the tree, field), or (-1, "") for none. A construct
// below its payload is one class on both sides: the host model under escher_gpu() keeps
// ESCHER-GPU's behaviour (a status mismatch against MOSP_ESCHER's exception), the store throws
// (the documented divergence construct_below_payload).
std::map<std::uint32_t, std::pair<std::int64_t, std::string>> first_divergences(
    const replay_report& r) {
  std::map<std::uint32_t, std::pair<std::int64_t, std::string>> out;
  for (const auto& [id, t] : r.trees) out[id] = {-1, ""};
  for (const auto& m : r.mismatches) {
    auto& slot = out[m.tree];
    const bool validation =
        m.op == "construct" && (m.field == "status" || m.field == "exception.type");
    if (slot.first < 0) slot = {m.tree_op, validation ? "construct validation" : m.field};
  }
  for (const auto& d : r.documented) {
    if (d.rule != "construct_below_payload") continue;  // the others exempt fields, not ops
    auto& slot = out[d.tree];
    if (slot.first < 0 || d.tree_op < slot.first) slot = {d.tree_op, "construct validation"};
  }
  return out;
}

TEST(CbstDeviceReplay, MospEscherTracesMatchUnderMospEscher) {
  DYNG_SKIP_WITHOUT_DEVICE();
  const auto traces = committed_traces("mosp_escher");
  ASSERT_FALSE(traces.empty());
  std::int64_t ops = 0, compactions = 0;
  for (const auto& path : traces) {
    SCOPED_TRACE(path);
    const replay_report r =
        replay_with(path, device_store_factory("mosp_escher"), field_set::exact);
    EXPECT_TRUE(r.ok()) << describe(r);
    EXPECT_EQ(r.unreplayable, 0);
    for (const auto& d : r.documented)
      EXPECT_TRUE(d.rule == "construct_without_row_counts" || d.rule == "construct_below_payload")
          << d.rule;
    ops += r.compared;
    compactions += r.compactions;
  }
  EXPECT_GT(ops, 40);
  EXPECT_GT(compactions, 0);  // the automatic compactions of the targeted trace
}

TEST(CbstDeviceReplay, MospEscherTracesDivergeWhereTheModelDoes) {
  DYNG_SKIP_WITHOUT_DEVICE();
  const auto traces = committed_traces("mosp_escher");
  ASSERT_FALSE(traces.empty());
  for (const char* preset : {"escher_gpu"}) {
    for (const auto& path : traces) {
      SCOPED_TRACE(std::string(preset) + " " + path);
      const replay_report dev =
          replay_with(path, device_store_factory(preset), field_set::exact, 100000);
      const replay_report model =
          replay_with(path, host_model_factory(preset), field_set::exact, 100000);
      EXPECT_EQ(dev.compared, model.compared);
      EXPECT_EQ(first_divergences(dev), first_divergences(model))
          << "device: " << describe(dev) << "\nmodel: " << describe(model);
    }
  }
}

// Every construct of both originals' committed traces is built from row counts and by the scan,
// with node arrays equal byte for byte (the adapter's twin check; a difference is a status
// mismatch of the replay).
TEST(CbstDeviceReplay, EveryConstructIsBuiltBothWaysWithIdenticalNodes) {
  DYNG_SKIP_WITHOUT_DEVICE();
  const auto traces = committed_traces("");
  ASSERT_FALSE(traces.empty());
  std::int64_t constructs = 0;
  for (const char* preset : {"escher_gpu", "mosp_escher"}) {
    for (const auto& path : traces) {
      SCOPED_TRACE(std::string(preset) + " " + path);
      auto in = cbst_trace::reader::open(path);
      device_store_adapter store(preset);
      options opt;
      opt.fields = field_set::exact;
      opt.max_mismatches = 100000;
      const replay_report r = replay(*in, store, opt, path);
      for (const auto& m : r.mismatches)
        EXPECT_EQ(m.actual.find("the node arrays built from the row counts"), std::string::npos);
      constructs += store.twin_constructs();
    }
  }
  EXPECT_GT(constructs, 40);
}

TEST(CbstDeviceReplay, ReplayingTwiceGivesIdenticalDigests) {
  DYNG_SKIP_WITHOUT_DEVICE();
  const auto traces = committed_traces("");
  ASSERT_FALSE(traces.empty());
  for (const auto& path : traces) {
    SCOPED_TRACE(path);
    std::uint64_t hashes[2] = {0, 0};
    std::string reports[2];
    for (int run = 0; run < 2; ++run) {
      auto in = cbst_trace::reader::open(path);
      device_store_adapter store("escher_gpu");
      options opt;
      opt.fields = field_set::exact;
      reports[run] = replay(*in, store, opt, path).to_json();
      hashes[run] = store.digest_hash();
    }
    EXPECT_EQ(hashes[0], hashes[1]);
    EXPECT_EQ(reports[0], reports[1]);
  }
}

}  // namespace
}  // namespace dyng::test::cbst_compare
