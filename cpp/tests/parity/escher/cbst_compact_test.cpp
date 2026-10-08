// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_compact_test.cpp
 * @brief The explicit compact() of the dynG CBST store (M8a step 4, behaviour (h); ADR 0039),
 *        proven on the operations of both originals' traces: the store and the host model
 *        (dyng::testing::cbst_host_model, whose compaction is written independently of the store,
 *        from MOSP_ESCHER@4b86159's compactCBST) replay every operation of a trace in lockstep,
 *        and both compact after every operation.
 *
 * After every operation the two states must be equal in every field (offsets included); after
 * every compaction too, and the compaction must keep what it promises: the keys, avail and
 * subtree_avail unchanged; every live row unchanged (in order) and a single segment of
 * (count + 4) / 4 * 4 slots; every deleted row its length, emptied; the store's integrity checks
 * clean. The committed traces run in every GPU build (labels gpu, parity, cbst); the small golden
 * sets in $DYNG_SCRATCH in the slow test.
 */
#include "parity/escher/cbst_compare.hpp"
#include "parity/escher/cbst_device_adapter.hpp"
#include "parity/escher/cbst_trace.hpp"
#include "support/data_paths.hpp"

#include <dyng/testing/cbst_host_model.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace dyng::test::cbst_compare {
namespace {

namespace fs = std::filesystem;
namespace tr = dyng::test::cbst_trace;

/// What one lockstep replay saw.
struct lockstep_result {
  std::int64_t ops = 0, compactions = 0, moved_back = 0, live_rows = 0, dead_rows = 0;
  std::int64_t stopped_trees = 0;
  std::vector<std::string> problems;
};

/// The first difference of two digests (empty: equal). Fields one side does not report (-1) are
/// not compared.
std::string difference(const tree_digest& a, const tree_digest& b) {
  std::ostringstream o;
  auto scalar = [&](const char* name, std::int64_t x, std::int64_t y) {
    if (o.tellp() == 0 && x != y && x >= 0 && y >= 0)
      o << name << ": store " << x << ", model " << y;
  };
  scalar("num_records", a.num_records, b.num_records);
  scalar("fixed_size", a.fixed_size, b.fixed_size);
  scalar("bump", a.bump, b.bump);
  scalar("alignment", a.alignment, b.alignment);
  scalar("scratch_items", a.scratch_items, b.scratch_items);
  scalar("scratch_payload", a.scratch_payload, b.scratch_payload);
  scalar("device_bytes", a.device_bytes, b.device_bytes);
  if (o.tellp() != 0) return o.str();
  if (a.keys != b.keys) return "keys";
  if (a.subtree_avail != b.subtree_avail) return "subtree_avail";
  if (a.records.size() != b.records.size()) return "record count";
  for (std::size_t i = 0; i < a.records.size(); ++i) {
    const key_digest &x = a.records[i], &y = b.records[i];
    const std::int64_t fx[] = {x.avail, x.length,    x.size,     x.occupancy, x.tail_capacity,
                               x.value, x.tail_base, x.segments, x.row_len,   x.terminated};
    const std::int64_t fy[] = {y.avail, y.length,    y.size,     y.occupancy, y.tail_capacity,
                               y.value, y.tail_base, y.segments, y.row_len,   y.terminated};
    static const char* const names[] = {"avail",         "length",    "size",      "occupancy",
                                        "tail_capacity", "value",     "tail_base", "segments",
                                        "row_len",       "terminated"};
    for (std::size_t f = 0; f < 10; ++f)
      if (fx[f] != fy[f])
        return "key " + std::to_string(x.key) + " " + names[f] + ": store " +
               std::to_string(fx[f]) + ", model " + std::to_string(fy[f]);
    if (x.row != y.row) return "key " + std::to_string(x.key) + " row";
  }
  return "";
}

/// The promises of a compaction (`before` and `after` are the store's states).
std::string broken_promise(const tree_digest& before, const tree_digest& after,
                           lockstep_result& r) {
  if (after.keys != before.keys) return "the keys changed";
  if (after.subtree_avail != before.subtree_avail) return "subtree_avail changed";
  if (after.records.size() != before.records.size()) return "the record count changed";
  if (after.integrity > 0 || after.integrity_subtree > 0 || after.integrity_index > 0)
    return "the integrity checks fail";
  for (std::size_t i = 0; i < after.records.size(); ++i) {
    const key_digest &b = before.records[i], &a = after.records[i];
    const std::string key = "key " + std::to_string(a.key) + ": ";
    if (a.avail != b.avail) return key + "avail changed";
    if (a.avail == 0) {
      ++r.live_rows;
      if (a.row != b.row) return key + "the live row changed";
      if (a.segments != 1) return key + "a live row of several segments";
      if (a.length != (a.row_len + 4) / 4 * 4 || a.occupancy != a.row_len ||
          a.tail_capacity != a.length - 1 || a.tail_base != a.value)
        return key + "a live row not of (count + 4) / 4 * 4 slots";
    } else {
      ++r.dead_rows;
      if (a.length != b.length) return key + "a deleted row's length changed";
      if (a.row_len != 0 || a.segments != 1) return key + "a deleted row not emptied";
    }
  }
  return "";
}

/// Replays the operations of a trace through the store and the host model, both compacting
/// after every operation that succeeded.
lockstep_result lockstep(tr::reader& in) {
  lockstep_result r;
  device_store_adapter store("escher_gpu", /*allow_compact=*/true);
  dyng::testing::cbst_host_policy policy = dyng::testing::cbst_host_policy::escher_gpu();
  policy.has_compact = true;
  host_model_adapter model(policy);
  std::set<std::uint32_t> stopped;
  op_request compact;
  compact.op = "compact";
  tr::record rec;
  while (in.next(rec)) {
    if (rec.kind == tr::record_kind::tree) {
      const auto id = static_cast<std::uint32_t>(rec.integer("tree"));
      for (store_adapter* s :
           {static_cast<store_adapter*>(&store), static_cast<store_adapter*>(&model)})
        s->create_tree(id, rec.text("dataset_name"), rec.integer("payload_capacity"),
                       rec.integer("alignment", 4));
      continue;
    }
    if (rec.kind == tr::record_kind::event && rec.text("event") == "destroy_object") {
      const auto id = static_cast<std::uint32_t>(rec.integer("tree"));
      store.destroy_tree(id);
      model.destroy_tree(id);
      continue;
    }
    if (rec.kind != tr::record_kind::op) continue;
    const auto id = static_cast<std::uint32_t>(rec.integer("tree"));
    if (stopped.count(id)) continue;
    op_request req;
    std::string missing;
    const std::string where = "tree " + std::to_string(id) + " op seq " +
                              std::to_string(rec.integer("seq")) + " (" + rec.text("op") + ")";
    if (!decode_op(rec, in, req, missing)) {
      stopped.insert(id);
      ++r.stopped_trees;
      continue;
    }
    ++r.ops;
    const op_outcome a = store.run(id, req);
    const op_outcome b = model.run(id, req);
    const bool a_ok = a.status == op_status::ok, b_ok = b.status == op_status::ok;
    if (!a_ok || !b_ok) {
      // Both fail (a fixed capacity, ...) or the documented construct below its payload (the
      // store throws, the model keeps ESCHER-GPU's tree without nodes): the tree stops here.
      const bool documented = req.op == "construct" && req.capacity < req.flat_size;
      if (a_ok != b_ok && !documented)
        r.problems.push_back(where + ": the store " + (a_ok ? "succeeded" : a.message) +
                             ", the model " + (b_ok ? "succeeded" : b.message));
      stopped.insert(id);
      ++r.stopped_trees;
      continue;
    }
    if (a.mapping != b.mapping) r.problems.push_back(where + ": the InsertMapping differs");
    if (req.op == "compact" && a.compact_return != b.compact_return)
      r.problems.push_back(where + ": compact() returned another bump pointer");
    const tree_digest before = store.digest(id);
    std::string diff = difference(before, model.digest(id));
    if (!diff.empty()) {
      r.problems.push_back(where + ": " + diff);
      stopped.insert(id);
      continue;
    }
    // The compaction after the operation.
    const op_outcome ca = store.run(id, compact);
    const op_outcome cb = model.run(id, compact);
    if (ca.status != op_status::ok || cb.status != op_status::ok) {
      r.problems.push_back(where + ": compact() failed: " + ca.message + " / " + cb.message);
      stopped.insert(id);
      continue;
    }
    ++r.compactions;
    if (ca.compact_return != cb.compact_return)
      r.problems.push_back(where + ": compact() returned " + std::to_string(ca.compact_return) +
                           ", the model " + std::to_string(cb.compact_return));
    const tree_digest after = store.digest(id);
    if (after.bump < before.bump) ++r.moved_back;
    diff = difference(after, model.digest(id));
    if (diff.empty()) diff = broken_promise(before, after, r);
    if (!diff.empty()) {
      r.problems.push_back(where + ", after compact(): " + diff);
      stopped.insert(id);
    }
  }
  return r;
}

std::string describe(const lockstep_result& r) {
  std::ostringstream o;
  o << r.problems.size() << " problems";
  for (std::size_t i = 0; i < r.problems.size() && i < 10; ++i) o << "\n  " << r.problems[i];
  return o.str();
}

#if !defined(DYNG_CBST_COMPACT_LARGE)
TEST(CbstCompactLockstep, CompactAfterEveryOpOfTheCommittedTraces) {
  if (!device_available()) GTEST_SKIP() << "no CUDA device";
  const fs::path root = data_path("goldens/cbst_trace");
  ASSERT_TRUE(fs::is_directory(root));
  std::vector<std::string> traces;
  for (const auto& e : fs::recursive_directory_iterator(root))
    if (e.is_regular_file() && e.path().extension() == ".cbt") traces.push_back(e.path().string());
  std::sort(traces.begin(), traces.end());
  ASSERT_FALSE(traces.empty());
  lockstep_result total;
  for (const auto& path : traces) {
    SCOPED_TRACE(path);
    auto in = tr::reader::open(path);
    const lockstep_result r = lockstep(*in);
    EXPECT_TRUE(r.problems.empty()) << describe(r);
    total.compactions += r.compactions;
    total.moved_back += r.moved_back;
    total.live_rows += r.live_rows;
    total.dead_rows += r.dead_rows;
  }
  EXPECT_GT(total.compactions, 40);
  EXPECT_GT(total.moved_back, 0);  // compactions that reclaimed space
  EXPECT_GT(total.dead_rows, 0);   // deleted rows were compacted too
}

#else
fs::path scratch() {
  if (const char* s = std::getenv("DYNG_SCRATCH")) return s;
  const char* home = std::getenv("HOME");
  return fs::path(home ? home : "/") / "Projects" / "dyng-work";
}

// The four small golden sets (every op of both originals' suites and targeted scenarios).
TEST(CbstCompactLockstepLarge, CompactAfterEveryOpOfTheSmallSets) {
  if (!device_available()) GTEST_SKIP() << "no CUDA device";
  const fs::path root = scratch() / "goldens" / "cbst_trace";
  if (!fs::is_directory(root)) GTEST_SKIP() << "no CBST trace golden sets under " << root;
  const fs::path tmp = scratch() / "runs";
  fs::create_directories(tmp);
  lockstep_result total;
  for (const char* set :
       {"escher_gpu_test", "escher_gpu_targeted", "mosp_escher_test", "mosp_escher_targeted"}) {
    if (!fs::is_directory(root / set)) GTEST_SKIP() << "no set " << set;
    std::vector<fs::path> traces;
    for (const auto& c : fs::directory_iterator(root / set))
      if (fs::is_regular_file(c.path() / "trace.cbt.zst"))
        traces.push_back(c.path() / "trace.cbt.zst");
    std::sort(traces.begin(), traces.end());
    for (const fs::path& path : traces) {
      SCOPED_TRACE(path.string());
      trace_source source(path.string(), tmp.string());
      const lockstep_result r = lockstep(source.reader());
      EXPECT_TRUE(r.problems.empty()) << describe(r);
      total.ops += r.ops;
      total.compactions += r.compactions;
      total.moved_back += r.moved_back;
      total.live_rows += r.live_rows;
      total.dead_rows += r.dead_rows;
      total.stopped_trees += r.stopped_trees;
    }
  }
  std::printf(
      "lockstep: %lld ops, %lld compactions (%lld reclaimed space), %lld live and %lld "
      "deleted rows checked after a compaction, %lld trees stopped at a failing op\n",
      static_cast<long long>(total.ops), static_cast<long long>(total.compactions),
      static_cast<long long>(total.moved_back), static_cast<long long>(total.live_rows),
      static_cast<long long>(total.dead_rows), static_cast<long long>(total.stopped_trees));
  EXPECT_GT(total.compactions, 10000);
}
#endif

}  // namespace
}  // namespace dyng::test::cbst_compare
