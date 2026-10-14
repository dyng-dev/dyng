// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_lockstep.hpp
 * @brief One CBST under a store preset, run in lockstep on the dynG device store and on the host
 *        model dyng::testing::cbst_host_model (M8a step 5, ADR 0039): the scenario tests and the
 *        random operation sequences.
 *
 * Every operation runs on both. They must agree on the outcome (status and exception type), the
 * InsertMapping, compact()'s bump pointer, the degree-bin launches and every field of the state
 * afterwards (digest_difference(): the scalars, the keys, subtree_avail, every key's record and
 * row). The model's policy is the store-equivalent one of the preset
 * (store_equivalent_host_policy()) for escher_gpu and mosp_escher, and the model's own
 * cbst_host_policy::dyng_default(), written independently of the store, for dyng_default.
 *
 * The tree also keeps the expected rows of every key, as MOSP_ESCHER@4b86159's test harness does
 * (tests/unit/test_cbst_ops.cu: a deleted key's row is empty, insert assigns the rows to the keys
 * of the InsertMapping), and check() reads the store back with its own integrity checker against
 * them (order-insensitive) and with check_subtree_avail.
 */
#pragma once

#include "hypergraph/escher/cbst_store.hpp"
#include "hypergraph/escher/integrity.hpp"
#include "parity/escher/cbst_compare.hpp"
#include "parity/escher/cbst_device_adapter.hpp"

#include <dyng/testing/cbst_host_model.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test::cbst_compare {

/// Whether the tests run their reduced sizes ($DYNG_CBST_REDUCED: the compute-sanitizer runs of
/// ci/gpu_local.sh, where the full scenarios would take hours; the sizes are named in each test).
inline bool reduced_sizes() {
  const char* v = std::getenv("DYNG_CBST_REDUCED");
  return v != nullptr && *v != '\0' && std::string(v) != "0";
}

/// The host-model policy a device-store preset is checked against.
inline dyng::testing::cbst_host_policy lockstep_model_policy(const std::string& preset) {
  if (preset == "dyng_default") return dyng::testing::cbst_host_policy::dyng_default();
  dyng::testing::cbst_host_policy p = store_equivalent_host_policy(preset);
  p.has_compact = true;  // compact() is tested under every preset
  return p;
}

/// One tree on the device store and the host model.
class lockstep_tree {
 public:
  using row = std::vector<int>;
  using rows = std::vector<row>;

  /**
   * @param[in] preset   The store preset ("escher_gpu", "mosp_escher", "dyng_default").
   * @param[in] name     The dataset name.
   * @param[in] capacity The payload capacity.
   * @param[in] model    The model's policy (default: lockstep_model_policy(preset)).
   */
  lockstep_tree(const std::string& preset, std::string name, long long capacity,
                const dyng::testing::cbst_host_policy* model = nullptr)
      : name_(std::move(name)),
        capacity_(capacity),
        store_(preset, /*allow_compact=*/true),
        model_(model ? *model : lockstep_model_policy(preset)) {
    store_.create_tree(1, name_, capacity, 4);
    model_.create_tree(1, name_, capacity, 4);
  }

  /// Runs one operation on both; returns the store's outcome.
  op_outcome run(const op_request& req, const std::string& step) {
    const op_outcome a = store_.run(1, req);
    const op_outcome b = model_.run(1, req);
    if (!problems_.empty()) return a;  // report the first divergence only
    const std::string where = name_ + " " + step + " (" + req.op + ")";
    if (a.status != b.status || a.error_type != b.error_type) {
      problem(where + ": the store " + (a.status == op_status::ok ? "succeeded" : a.error_type) +
              " " + a.message + "; the model " +
              (b.status == op_status::ok ? "succeeded" : b.error_type) + " " + b.message);
      return a;
    }
    // A failed insert has no InsertMapping (the store throws); its in-place writes are compared.
    if (a.status == op_status::ok && a.mapping != b.mapping)
      problem(where + ": the InsertMapping differs");
    if (req.op == "compact" && a.compact_return != b.compact_return)
      problem(where + ": compact() returned " + std::to_string(a.compact_return) + ", the model " +
              std::to_string(b.compact_return));
    if (b.has_bins && a.bins != b.bins) problem(where + ": the degree-bin launches differ");
    const std::string diff = digest_difference(store_.digest(1), model_.digest(1));
    if (!diff.empty()) problem(where + ": " + diff);
    return a;
  }

  /// Constructs rows of keys 1..n (with the rows' counts, or by the scan).
  op_outcome construct(const rows& values, bool with_counts = true) {
    op_request r;
    r.op = "construct";
    for (std::size_t i = 0; i < values.size(); ++i) {
      r.keys.push_back(static_cast<std::int32_t>(i) + 1);
      r.starts.push_back(static_cast<std::int32_t>(r.flat.size()));
      r.row_counts.push_back(static_cast<std::int32_t>(values[i].size()));
      const std::size_t padded = (values[i].size() + 4) / 4 * 4;
      r.flat.insert(r.flat.end(), values[i].begin(), values[i].end());
      r.flat.insert(r.flat.end(), padded - values[i].size() - 1, 0);
      r.flat.push_back(INT_MIN);
    }
    r.has_row_counts = with_counts;
    r.num_records = static_cast<std::int64_t>(values.size());
    r.flat_size = static_cast<std::int64_t>(r.flat.size());
    r.capacity = capacity_;
    r.alignment = 4;
    const op_outcome out = run(r, "construct n=" + std::to_string(values.size()));
    if (out.status == op_status::ok) expected_ = values;
    return out;
  }

  /// Appends values to rows (an empty list is skipped, as by MOSP_ESCHER's harness).
  op_outcome append(const std::map<int, row>& adds, const std::string& step = "fill") {
    op_request r = items_of("fill", adds);
    if (r.keys.empty()) return {};
    const op_outcome out = run(r, step);
    if (out.status == op_status::ok)
      for (const auto& [k, vals] : adds) {
        row& e = expected_row(k);
        e.insert(e.end(), vals.begin(), vals.end());
      }
    return out;
  }

  /// Removes every occurrence of the listed values from each row.
  op_outcome remove_values(const std::map<int, row>& removals, const std::string& step = "unfill") {
    op_request r = items_of("unfill", removals);
    if (r.keys.empty()) return {};
    const op_outcome out = run(r, step);
    if (out.status == op_status::ok)
      for (const auto& [k, vals] : removals) {
        row& e = expected_row(k);
        const std::set<int> drop(vals.begin(), vals.end());
        e.erase(std::remove_if(e.begin(), e.end(), [&](int v) { return drop.count(v) != 0; }),
                e.end());
      }
    return out;
  }

  /// Deletes keys.
  op_outcome erase(const std::vector<int>& keys, const std::string& step = "erase") {
    op_request r;
    r.op = "erase";
    r.keys.assign(keys.begin(), keys.end());
    const op_outcome out = run(r, step);
    if (out.status == op_status::ok)
      for (int k : keys) expected_row(k).clear();
    return out;
  }

  /// Inserts rows (requested keys: n + 1, n + 2, ... as MOSP_ESCHER's harness passes them, or
  /// `requested`); the InsertMapping must name distinct keys that are not live.
  op_outcome insert(const rows& items, const std::string& step = "insert",
                    const std::vector<int>* requested = nullptr) {
    op_request r;
    r.op = "insert";
    int run_size = 0;
    for (std::size_t i = 0; i < items.size(); ++i) {
      r.keys.push_back(requested ? (*requested)[i] : static_cast<int>(expected_.size() + i) + 1);
      r.payload.insert(r.payload.end(), items[i].begin(), items[i].end());
      run_size += static_cast<int>(items[i].size());
      r.prefix.push_back(run_size);
    }
    const op_outcome out = run(r, step);
    if (out.status != op_status::ok) return out;
    std::set<int> distinct;
    for (std::size_t i = 0; i < items.size() && i < out.mapping.size(); ++i) {
      const int key = out.mapping[i];
      if (key < 1 || !distinct.insert(key).second || live(key)) {
        problem(name_ + " " + step + ": item " + std::to_string(i) + " got key " +
                std::to_string(key) + " (zero, duplicate or a live key)");
        continue;
      }
      expected_row(key) = items[i];
    }
    if (out.mapping.size() != items.size()) problem(name_ + " " + step + ": mapping size");
    return out;
  }

  /// compact().
  op_outcome compact(const std::string& step = "compact") {
    op_request r;
    r.op = "compact";
    return run(r, step);
  }

  /// The store read back with its integrity checker against the expected rows
  /// (order-insensitive) and with check_subtree_avail.
  void check(const std::string& step) {
    const dyng::detail::escher::cbst_store* s = store_.store(1);
    if (s == nullptr) return problem(name_ + " " + step + ": no store");
    std::ostringstream log;
    const long long v =
        dyng::detail::escher::check_tree_rows(s->context(), expected_, true, name_.c_str(), log) +
        dyng::detail::escher::check_subtree_avail(s->context(), name_.c_str(), log);
    ++checks_;
    if (v != 0)
      problem(name_ + " " + step + ": " + std::to_string(v) + " violations\n" + log.str());
  }

  /// Whether a key holds a non-empty row (MOSP_ESCHER's harness's notion of live).
  [[nodiscard]] bool live(int key) const {
    return key >= 1 && key <= num_keys() && !expected_[static_cast<std::size_t>(key - 1)].empty();
  }
  [[nodiscard]] int num_keys() const {
    return static_cast<int>(expected_.size());
  }
  [[nodiscard]] const row& row_of(int key) const {
    return expected_[static_cast<std::size_t>(key - 1)];
  }
  [[nodiscard]] long long live_values() const {
    long long n = 0;
    for (const auto& r : expected_) n += static_cast<long long>(r.size());
    return n;
  }
  /// The store's state (scalars only).
  [[nodiscard]] tree_digest scalars() {
    return store_.scalars(1);
  }
  /// The model's growths and compactions so far (equal to the store's while problems() is empty).
  [[nodiscard]] std::pair<int, int> events() {
    int g = 0, c = 0;
    for (const auto& e : model_.events(1)) (e.second == 'g' ? g : c) += 1;
    return {g, c};
  }
  /// The degree-bin launches of the store so far (fill, reuse, unfill x thread, warp, block).
  [[nodiscard]] const std::array<std::int64_t, 9>& bins() const {
    return store_.total_bins();
  }
  [[nodiscard]] const std::vector<std::string>& problems() const {
    return problems_;
  }
  [[nodiscard]] int checks() const {
    return checks_;
  }
  [[nodiscard]] device_store_adapter& device() {
    return store_;
  }

 private:
  void problem(const std::string& what) {
    if (problems_.size() < 10) problems_.push_back(what);
  }
  row& expected_row(int key) {
    if (key > num_keys()) expected_.resize(static_cast<std::size_t>(key));
    return expected_[static_cast<std::size_t>(key - 1)];
  }
  static op_request items_of(const char* op, const std::map<int, row>& items) {
    op_request r;
    r.op = op;
    int run_size = 0;
    for (const auto& [k, vals] : items) {
      if (vals.empty()) continue;
      r.keys.push_back(k);
      r.payload.insert(r.payload.end(), vals.begin(), vals.end());
      run_size += static_cast<int>(vals.size());
      r.prefix.push_back(run_size);
    }
    return r;
  }

  std::string name_;
  long long capacity_;
  device_store_adapter store_;
  host_model_adapter model_;
  rows expected_;
  std::vector<std::string> problems_;
  int checks_ = 0;
};

/// The problems of a lockstep tree, for a test message.
inline std::string describe_problems(const std::vector<std::string>& p) {
  std::string out = std::to_string(p.size()) + " problems";
  for (const auto& s : p) out += "\n  " + s;
  return out;
}

}  // namespace dyng::test::cbst_compare
