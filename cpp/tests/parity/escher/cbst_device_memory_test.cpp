// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_device_memory_test.cpp
 * @brief The early store-memory reading of M8a (acceptance criterion 15; the author's decision of
 *        2026-10-07: the memory gate of M8 is read as store accounting): on escher_gpu_coauth50k
 *        (ESCHER-GPU@abcf9b2's `build/main --input coauth.hg --batches 3 --batch-size 50000
 *        --seed 1 --check`), each tree's device_bytes() after its construct and after every
 *        operation of every batch is at most 1.05x the deviceBytes() ESCHER-GPU recorded for the
 *        same tree at the same point, and equals the bytes the store actually holds
 *        (allocated_bytes()).
 *
 * Every operation of the trace runs on a dynG store under escher_gpu() with the trace's
 * capacity. The H2H rebuilds of this set have hash-only inputs (the result of A's line-graph
 * merge; parity/goldens.toml), so their CONTENT cannot be replayed: the store is rebuilt from
 * arrays of the recorded sizes (row count, used payload, capacity) filled with an empty row
 * layout, which is all the accounting depends on. The values of the other operations are the
 * recorded ones. With $DYNG_CBST_MEMORY_JSON the table (tree, op, point, A's bytes, dynG's
 * device_bytes and allocated_bytes, ratio) is written there for parity/results/M8a.md.
 *
 * Labels gpu, parity, cbst and slow; skipped without a device or the set.
 */
#include "hypergraph/escher/cbst_store.hpp"
#include "parity/escher/cbst_compare.hpp"
#include "parity/escher/cbst_device_adapter.hpp"
#include "parity/escher/cbst_trace.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace dyng::test::cbst_compare {
namespace {

namespace fs = std::filesystem;
namespace tr = dyng::test::cbst_trace;
namespace esc = dyng::detail::escher;

fs::path scratch() {
  if (const char* s = std::getenv("DYNG_SCRATCH")) return s;
  const char* home = std::getenv("HOME");
  return fs::path(home ? home : "/") / "Projects" / "dyng-work";
}

/// The int32 input `name` of an op, or `count` zeros when the trace has only its hash.
std::vector<int> input_or_zeros(tr::reader& in, const tr::record& rec, const std::string& name,
                                bool& hashed) {
  const tr::field* f = rec.find(name);
  if (f == nullptr || f->kind != tr::field::type::array) return {};
  if (f->array.kind == tr::array_field::storage::hash_only) {
    hashed = true;
    return std::vector<int>(static_cast<std::size_t>(f->array.count), 0);
  }
  const std::vector<std::int32_t> v = in.ints(f->array);
  return std::vector<int>(v.begin(), v.end());
}

struct reading {
  std::string tree, op, point;
  std::int64_t seq = 0, a_bytes = 0, dyng_bytes = 0, held = 0;
};

TEST(CbstEarlyMemory, Coauth50kStoreBytesWithin105PercentOfEscherGpu) {
  if (!device_available()) GTEST_SKIP() << "no CUDA device";
  const fs::path set = scratch() / "goldens" / "cbst_trace" / "escher_gpu_coauth50k";
  fs::path trace;
  if (fs::is_directory(set))
    for (const auto& c : fs::directory_iterator(set))
      if (fs::is_regular_file(c.path() / "trace.cbt.zst")) trace = c.path() / "trace.cbt.zst";
  if (trace.empty()) GTEST_SKIP() << "escher_gpu_coauth50k is not under " << set;
  trace_source source(trace.string(), fs::temp_directory_path().string());
  tr::reader& in = source.reader();
  const resources res = resources::cuda();
  std::map<std::int64_t, std::string> names;
  std::map<std::int64_t, std::unique_ptr<esc::cbst_store>> trees;
  std::vector<reading> table;
  std::int64_t batch = 0;
  tr::record rec;
  while (in.next(rec)) {
    if (rec.kind == tr::record_kind::tree) {
      const std::int64_t id = rec.integer("tree");
      names[id] = rec.text("dataset_name");
      trees[id] = std::make_unique<esc::cbst_store>(
          res, names[id], static_cast<int>(rec.integer("payload_capacity")),
          static_cast<int>(rec.integer("alignment", 4)), esc::store_policy::escher_gpu());
      continue;
    }
    if (rec.kind == tr::record_kind::batch) {
      if (rec.text("hook") == "escher_gpu.update" && rec.text("phase") == "enter")
        batch = rec.integer("batch");
      continue;
    }
    if (rec.kind != tr::record_kind::op) continue;
    const std::int64_t id = rec.integer("tree");
    esc::cbst_store& s = *trees.at(id);
    const std::string op = rec.text("op");
    bool hashed = false;
    if (op == "construct") {
      std::vector<int> keys = input_or_zeros(in, rec, "in.keys", hashed);
      std::vector<int> starts = input_or_zeros(in, rec, "in.start_offsets", hashed);
      std::vector<int> flat = input_or_zeros(in, rec, "in.flat_payload", hashed);
      s.construct(keys.data(), starts.data(), static_cast<int>(rec.integer("in.num_records")),
                  flat.data(), static_cast<int>(rec.integer("in.flat_payload_size")));
    } else if (op == "rebuild") {
      const auto n = static_cast<std::size_t>(rec.integer("in.num_records"));
      const auto used = static_cast<std::size_t>(rec.integer("in.flat_payload_size"));
      const auto cap = static_cast<std::size_t>(rec.integer("in.capacity"));
      std::vector<int> starts = input_or_zeros(in, rec, "in.start_offsets", hashed);
      std::vector<int> flat = input_or_zeros(in, rec, "in.flat_payload", hashed);
      if (hashed) {
        // The accounting needs the sizes only: rows of one empty segment of 4 slots each would
        // overrun the used payload, so every row starts at the last terminator instead.
        flat.assign(used, 0);
        if (used > 0) flat[used - 1] = INT_MIN;
        starts.assign(n, used > 0 ? static_cast<int>(used - 1) : 0);
      }
      buffer<int> d_starts(res, std::max<std::size_t>(n, 1)), d_flat(res, cap);
      if (n > 0)
        ASSERT_EQ(
            cudaMemcpy(d_starts.data(), starts.data(), n * sizeof(int), cudaMemcpyHostToDevice),
            cudaSuccess);
      ASSERT_EQ(cudaMemset(d_flat.data(), 0, cap * sizeof(int)), cudaSuccess);
      if (used > 0)
        ASSERT_EQ(
            cudaMemcpy(d_flat.data(), flat.data(), used * sizeof(int), cudaMemcpyHostToDevice),
            cudaSuccess);
      // The adapter's convention: the row offsets of exactly n rows (1 for 0).
      s.rebuild_from_device(std::move(d_starts), static_cast<int>(n), std::move(d_flat),
                            static_cast<int>(used), static_cast<int>(cap));
    } else {
      std::vector<int> keys = input_or_zeros(in, rec, "in.keys", hashed);
      std::vector<int> payload = input_or_zeros(in, rec, "in.payload", hashed);
      std::vector<int> prefix = input_or_zeros(in, rec, "in.prefix", hashed);
      ASSERT_FALSE(hashed) << op << " of " << names[id] << " has hash-only inputs";
      if (op == "insert") {
        (void)s.insert(keys, payload, prefix);
      } else if (op == "fill") {
        s.append(keys, payload, prefix);
      } else if (op == "erase") {
        s.erase(keys);
      } else if (op == "unfill") {
        s.remove_values(keys, payload, prefix);
      } else {
        FAIL() << "unexpected op " << op;
      }
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    reading r;
    r.tree = names[id];
    r.op = op + (hashed ? " (hash-only inputs: sizes replayed)" : "");
    r.point = op == "construct" ? "after construct" : "batch " + std::to_string(batch);
    r.seq = rec.integer("seq");
    r.a_bytes = rec.integer("after.device_bytes");
    r.dyng_bytes = static_cast<std::int64_t>(s.device_bytes());
    r.held = static_cast<std::int64_t>(s.allocated_bytes());
    table.push_back(r);
    SCOPED_TRACE(r.tree + " " + r.op + " seq " + std::to_string(r.seq));
    EXPECT_LE(static_cast<double>(r.dyng_bytes), 1.05 * static_cast<double>(r.a_bytes));
    EXPECT_EQ(r.dyng_bytes, r.a_bytes);  // the same accounting of the same arrays
    EXPECT_EQ(r.held, r.dyng_bytes);     // and the store holds what it accounts
  }
  EXPECT_EQ(table.size(), 18U);
  if (const char* out = std::getenv("DYNG_CBST_MEMORY_JSON")) {
    std::ofstream f(out);
    f << "{\"trace\": \"" << trace.string() << "\", \"readings\": [";
    for (std::size_t i = 0; i < table.size(); ++i) {
      const reading& r = table[i];
      f << (i ? ",\n" : "\n") << "{\"seq\": " << r.seq << ", \"tree\": \"" << r.tree
        << "\", \"op\": \"" << r.op << "\", \"point\": \"" << r.point
        << "\", \"escher_gpu_device_bytes\": " << r.a_bytes
        << ", \"dyng_device_bytes\": " << r.dyng_bytes << ", \"dyng_allocated_bytes\": " << r.held
        << ", \"ratio\": "
        << (r.a_bytes ? static_cast<double>(r.dyng_bytes) / static_cast<double>(r.a_bytes) : 0.0)
        << "}";
    }
    f << "\n]}\n";
  }
}

}  // namespace
}  // namespace dyng::test::cbst_compare
