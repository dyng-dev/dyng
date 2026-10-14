// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// Targeted CBST scenarios for the trace goldens (M8a coverage, ADR 0036): one tree per path that
// the originals' own suites may not reach, driven through the original's UNCHANGED CBSTOperations
// API and recorded by the twin's recorder (this driver is linked like a twin by
// build_crossreplay.sh). Built for both originals (DYNG_REPLAY_STORE_ESCHER_GPU /
// DYNG_REPLAY_STORE_MOSP_ESCHER); the same calls run on both.
//
//   cbst_targeted [small|large|all]   (DYNG_CBST_TRACE_DIR must be set; small: every scenario
//                                     but levels and chain_fallback, large: those two)
//
// Scenarios (tree names "targeted-<name>"):
//   k0              an insert with no items (K = 0)
//   d0              an insert while no slot is deleted (D = 0): every item is surplus
//   reuse_only      deleted slots that take every inserted item
//   reuse_surplus   one item reuses a slot, two are surplus: the surplus rebuild
//   reissue_max     the deleted maximal key is too small for the items: the rebuild drops it and
//                   the surplus path issues the same key again
//   levels          fill, reuse and unfill items of 10, 100 and 2,000 values (thread / warp /
//                   block level)
//   chain_fallback  rows chained by repeated overflowing fills, then unfills of 40 and 1,100
//                   values (the warp and block kernels' serial fallback on chained rows)
//   growth          a small payload capacity and overflowing fills: ESCHER-GPU grows the payload
//   fill_compact    an overflowing fill that fits only after the unfilled overflow segments are
//                   reclaimed (MOSP_ESCHER compacts inside fill; ESCHER-GPU grows)
//   insert_compact  surplus inserts behind a full bump pointer after unfills (MOSP_ESCHER
//                   compacts inside insert)
//   capacity_throw  more values than the capacity holds after a compaction (MOSP_ESCHER throws)
//   compact         an explicit compact() (MOSP_ESCHER only) after unfills
//   small_capacity  construct with a capacity below the payload size (MOSP_ESCHER throws,
//                   ESCHER-GPU prints and returns; nothing else runs on the tree)
#include "cbst_recorder.hpp"
#include "structure.hpp"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <deque>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace rec = dyng_parity::cbst_rec;

namespace {

std::deque<std::string> g_names;

struct flat_rows {
  std::vector<int> keys, starts, values, counts;
};

// The construct layout of both originals' flatten2DVector: row i padded to ((n + 4) / 4) * 4 slots,
// zeros after the values and INT_MIN in the last slot.
flat_rows flatten(const std::vector<std::vector<int>>& rows) {
  flat_rows f;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    f.keys.push_back(static_cast<int>(i) + 1);
    f.starts.push_back(static_cast<int>(f.values.size()));
    f.counts.push_back(static_cast<int>(rows[i].size()));
    const std::size_t padded = (rows[i].size() + 4) / 4 * 4;
    f.values.insert(f.values.end(), rows[i].begin(), rows[i].end());
    f.values.insert(f.values.end(), padded - rows[i].size() - 1, 0);
    f.values.push_back(INT_MIN);
  }
  return f;
}

std::vector<int> iota_row(int n, int first) {
  std::vector<int> r(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) r[static_cast<std::size_t>(i)] = first + i;
  return r;
}

struct batch {
  std::vector<int> keys, values, prefix;
  void add(int key, const std::vector<int>& row) {
    keys.push_back(key);
    values.insert(values.end(), row.begin(), row.end());
    prefix.push_back(static_cast<int>(values.size()));
  }
};

std::unique_ptr<CBSTOperations> make(const std::string& name,
                                     const std::vector<std::vector<int>>& rows, long long extra,
                                     long long capacity = -1) {
  g_names.push_back("targeted-" + name);
  flat_rows f = flatten(rows);
  const long long cap = capacity >= 0 ? capacity : static_cast<long long>(f.values.size()) + extra;
  auto t = std::make_unique<CBSTOperations>(g_names.back().c_str(), static_cast<int>(cap), 4);
#if defined(DYNG_REPLAY_STORE_MOSP_ESCHER)
  t->construct(f.keys.data(), f.starts.data(), static_cast<int>(rows.size()), f.values.data(),
               static_cast<int>(f.values.size()), f.counts.data());
#else
  t->construct(f.keys.data(), f.starts.data(), static_cast<int>(rows.size()), f.values.data(),
               static_cast<int>(f.values.size()));
#endif
  return t;
}

void unfill(CBSTOperations& t, const batch& b) {
#if defined(DYNG_REPLAY_STORE_MOSP_ESCHER)
  unfillCBST(b.keys, b.values, b.prefix, const_cast<CBSTContext&>(t.context()));
#else
  t.unfill(b.keys, b.values, b.prefix);
#endif
}

template <class F>
void scenario(const char* name, F f) {
  try {
    f();
  } catch (const std::exception& e) {
    std::printf("%s: %s\n", name, e.what());  // recorded as the op's exception
  }
}

}  // namespace

int main(int argc, char** argv) {
  // "small": every scenario but levels and chain_fallback (small enough to commit); "large":
  // those two; default: all.
  const std::string which = argc > 1 ? argv[1] : "all";
  const bool small = which != "large", large = which != "small";
  rec::recorder* r = rec::original_recorder();
  if (r == nullptr) {
    std::fprintf(stderr, "set DYNG_CBST_TRACE_DIR: the scenarios are recorded\n");
    return 2;
  }
  std::vector<std::vector<int>> eight;
  for (int i = 0; i < 8; ++i) eight.push_back(iota_row(1 + i % 3, 10 * (i + 1)));

  if (small)
    scenario("k0", [&] {
      auto t = make("k0", eight, 64);
      (void)t->insert({}, {}, {});
    });
  if (small)
    scenario("d0", [&] {
      auto t = make("d0", eight, 64);
      batch b;
      b.add(100, {1, 2});
      b.add(101, {3});
      b.add(102, {4, 5, 6});
      (void)t->insert(b.keys, b.values, b.prefix);
    });
  if (small)
    scenario("reuse_only", [&] {
      auto t = make("reuse_only", eight, 64);
      t->erase({2, 5});
      batch b;
      b.add(2, {7});
      b.add(5, {8, 9});
      (void)t->insert(b.keys, b.values, b.prefix);
    });
  if (small)
    scenario("reuse_surplus", [&] {
      auto t = make("reuse_surplus", eight, 64);
      t->erase({3});
      batch b;
      b.add(3, {1});
      b.add(9, iota_row(9, 1));
      b.add(10, iota_row(5, 1));
      (void)t->insert(b.keys, b.values, b.prefix);
    });
  if (small)
    scenario("reissue_max", [&] {
      std::vector<std::vector<int>> rows = {iota_row(6, 1), iota_row(6, 1), iota_row(2, 1),
                                            iota_row(2, 1), iota_row(2, 1), {1}};
      auto t = make("reissue_max", rows, 64);
      t->erase({2, 6});
      batch b;
      b.add(2, iota_row(5, 1));  // fits key 2's slot (7 values)
      b.add(6, iota_row(4, 1));  // does not fit key 6's slot (3 values): surplus
      (void)t->insert(b.keys, b.values, b.prefix);
    });
  if (large)
    scenario("levels", [&] {
      std::vector<std::vector<int>> rows;
      for (int n : {10, 100, 2000, 3, 3, 3}) rows.push_back(iota_row(n, 1));
      auto t = make("levels", rows, 1 << 16);
      batch f;
      f.add(4, iota_row(10, 5000));
      f.add(5, iota_row(100, 5000));
      f.add(6, iota_row(2000, 5000));
      t->fill(f.keys, f.values, f.prefix);
      batch u;
      u.add(4, iota_row(10, 5000));
      u.add(5, iota_row(100, 5000));
      u.add(6, iota_row(2000, 5000));
      unfill(*t, u);
      t->erase({1, 2, 3});
      batch b;
      b.add(1, iota_row(10, 1));
      b.add(2, iota_row(100, 1));
      b.add(3, iota_row(2000, 1));
      (void)t->insert(b.keys, b.values, b.prefix);
    });
  if (large)
    scenario("chain_fallback", [&] {
      std::vector<std::vector<int>> rows = {iota_row(3, 1), iota_row(3, 1), iota_row(3, 1)};
      auto t = make("chain_fallback", rows, 1 << 14);
      for (int round = 0; round < 4; ++round) {
        batch f;
        f.add(1, iota_row(15, 100 + 15 * round));
        f.add(2, iota_row(400, 1000 + 400 * round));
        t->fill(f.keys, f.values, f.prefix);
      }
      batch u;
      u.add(1, iota_row(40, 100));
      u.add(2, iota_row(1100, 1000));
      unfill(*t, u);
    });
  if (small)
    scenario("growth", [&] {
      std::vector<std::vector<int>> rows;
      for (int i = 0; i < 16; ++i) rows.push_back(iota_row(3, 1));
      const long long size = static_cast<long long>(flatten(rows).values.size());
      auto t = make("growth", rows, 0, size + 64);
      for (int round = 0; round < 6; ++round) {
        batch f;
        for (int k = 1; k <= 16; k += 3) f.add(k, iota_row(6, 100 + 6 * round));
        t->fill(f.keys, f.values, f.prefix);
        batch u;
        for (int k = 1; k <= 16; k += 3) u.add(k, iota_row(6, 100 + 6 * round));
        unfill(*t, u);
      }
    });
  if (small)
    scenario("fill_compact", [&] {
      // Chained rows whose overflow segments an unfill empties; a later overflowing fill does not
      // fit behind the bump pointer, but fits after a compaction (MOSP_ESCHER: inside fill).
      std::vector<std::vector<int>> rows(4, iota_row(3, 1));
      const long long size = static_cast<long long>(flatten(rows).values.size());
      auto t = make("fill_compact", rows, 0, size + 40);
      batch f1, f2;
      f1.add(1, iota_row(8, 100));
      f2.add(1, iota_row(8, 200));
      t->fill(f1.keys, f1.values, f1.prefix);
      t->fill(f2.keys, f2.values, f2.prefix);
      batch u;
      u.add(1, [] {
        std::vector<int> v = iota_row(8, 100), w = iota_row(8, 200);
        v.insert(v.end(), w.begin(), w.end());
        return v;
      }());
      unfill(*t, u);
      batch f3;
      f3.add(2, iota_row(20, 300));
      t->fill(f3.keys, f3.values, f3.prefix);
    });
  if (small)
    scenario("insert_compact", [&] {
      // Surplus rows that do not fit behind the bump pointer after unfills emptied overflow
      // segments (MOSP_ESCHER compacts inside insert).
      std::vector<std::vector<int>> rows(12, iota_row(3, 1));
      const long long size = static_cast<long long>(flatten(rows).values.size());
      auto t = make("insert_compact", rows, 0, size + 48);
      batch f;
      for (int k = 1; k <= 4; ++k) f.add(k, iota_row(4, 100));
      t->fill(f.keys, f.values, f.prefix);
      unfill(*t, f);
      batch b;
      b.add(13, iota_row(9, 1));
      b.add(14, iota_row(9, 1));
      (void)t->insert(b.keys, b.values, b.prefix);
    });
  if (small)
    scenario("capacity_throw", [&] {
      // More values than the capacity holds even after a compaction (MOSP_ESCHER throws;
      // ESCHER-GPU grows).
      std::vector<std::vector<int>> rows(4, iota_row(3, 1));
      const long long size = static_cast<long long>(flatten(rows).values.size());
      auto t = make("capacity_throw", rows, 0, size + 16);
      batch f;
      f.add(1, iota_row(40, 100));
      t->fill(f.keys, f.values, f.prefix);
    });
  if (small)
    scenario("compact", [&] {
      std::vector<std::vector<int>> rows = {iota_row(3, 1), iota_row(3, 1)};
      auto t = make("compact", rows, 256);
      batch f;
      f.add(1, iota_row(20, 100));
      t->fill(f.keys, f.values, f.prefix);
      unfill(*t, f);
#if defined(DYNG_REPLAY_STORE_MOSP_ESCHER)
      (void)t->compact();
#endif
    });
  if (small) scenario("small_capacity", [&] { auto t = make("small_capacity", eight, 0, 8); });
  std::printf("targeted scenarios done\n");
  return 0;
}
