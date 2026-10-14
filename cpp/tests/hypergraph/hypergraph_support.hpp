// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_support.hpp
 * @brief Test helpers of the hypergraph container (M8b, ADR 0041): host copies of its views, the
 *        lockstep with two dyng::testing::cbst_host_model trees (H2V and V2H, driven with the
 *        operations the container's id_reuse order makes), and random batch streams.
 *
 * The lockstep is the oracle of criterion 6: the model replays the ESCHER store's operations on
 * the host (an implementation shared with neither the CUDA store nor the host id allocator), and
 * after every batch the container's inserted ids, h2v rows by id and v2h rows (as sets) must equal
 * the model's.
 */
#pragma once

#include "hypergraph/capacity.hpp"
#include "parity/escher/cbst_compare.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/testing/cbst_host_model.hpp>
#include <dyng/testing/line_graph_oracle.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dyng::test::hyper {

using rows = std::vector<std::vector<std::int32_t>>;

/// Whether the tests run their reduced sizes ($DYNG_CBST_REDUCED: the compute-sanitizer runs of
/// ci/gpu_local.sh; the sizes are named in each test).
inline bool reduced_sizes() {
  const char* v = std::getenv("DYNG_CBST_REDUCED");
  return v != nullptr && *v != '\0' && std::string(v) != "0";
}

/// Resources of a backend (8 threads for openmp, the tests' default).
inline resources resources_of(backend b, int threads = 8) {
  switch (b) {
    case backend::sequential:
      return resources::sequential();
    case backend::openmp:
      return resources::openmp(threads);
    case backend::cuda:
      return resources::cuda(0);
  }
  return resources::sequential();
}

/// A host copy of an array in any memory space (device arrays through the stream of their device).
template <typename value_t>
std::vector<std::remove_const_t<value_t>> host_vector(array_view<value_t> v) {
  if (is_host_accessible(v.space())) {
    return std::vector<std::remove_const_t<value_t>>(v.begin(), v.end());
  }
  return to_vector(resources::cuda(v.device() < 0 ? 0 : v.device()), v);
}

/// A hyperedge list of rows.
template <typename weight_t = unweighted>
hyperedge_list<std::int32_t, weight_t> list_of(std::int32_t n, const rows& r,
                                               const std::vector<weight_t>& w = {}) {
  hyperedge_list<std::int32_t, weight_t> out;
  out.num_vertices = n;
  for (std::size_t i = 0; i < r.size(); ++i) {
    out.add_hyperedge(host_view(r[i]), w.empty() ? weight_t{} : w[i]);
  }
  return out;
}

/// The rows of a host hyperedge list (members in list order).
template <typename weight_t>
rows rows_of_list(const hyperedge_list<std::int32_t, weight_t>& l) {
  rows out(l.offsets.empty() ? 0 : l.offsets.size() - 1);
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i].assign(l.members.begin() + l.offsets[i], l.members.begin() + l.offsets[i + 1]);
  return out;
}

/// The h2v rows of a hypergraph by id (host copies).
template <typename weight_t>
rows h2v_rows(const hypergraph<std::int32_t, weight_t>& h) {
  const auto v = h.h2v();
  const auto ptr = host_vector(v.row_ptr);
  const auto cols = host_vector(v.col_ind);
  rows out(ptr.empty() ? 0 : ptr.size() - 1);
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i].assign(cols.begin() + ptr[i], cols.begin() + ptr[i + 1]);
  return out;
}

/// The v2h rows of a hypergraph by vertex (host copies).
template <typename weight_t>
std::vector<std::vector<std::int64_t>> v2h_rows(const hypergraph<std::int32_t, weight_t>& h) {
  const auto v = h.v2h();
  const auto ptr = host_vector(v.row_ptr);
  const auto cols = host_vector(v.col_ind);
  std::vector<std::vector<std::int64_t>> out(ptr.empty() ? 0 : ptr.size() - 1);
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i].assign(cols.begin() + ptr[i], cols.begin() + ptr[i + 1]);
  return out;
}

/// Rows of int64 ids (v2h, h2h).
using id_rows = std::vector<std::vector<std::int64_t>>;

/// The h2h rows of a hypergraph by id (host copies), and a check of the view's shape: rows
/// sorted and contiguous, out[r] the first entry above r. Problems go to `problems`.
template <typename weight_t>
id_rows h2h_rows(const hypergraph<std::int32_t, weight_t>& h, std::string* problems = nullptr) {
  const auto v = h.h2h();
  const auto ptr = host_vector(v.row_ptr);
  const auto cols = host_vector(v.col_ind);
  const auto out = host_vector(v.out);
  std::ostringstream os;
  if (v.row_ptr.space() != h.space() || v.col_ind.space() != h.space() ||
      v.out.space() != h.space())
    os << "the h2h view is not in the hypergraph's memory space; ";
  if (v.version != h.version()) os << "the h2h view has version " << v.version << "; ";
  if (static_cast<std::int64_t>(ptr.size()) != h.hyperedge_id_bound() + 1)
    os << "h2h has " << ptr.size() << " offsets for a bound of " << h.hyperedge_id_bound() << "; ";
  if (out.size() + 1 != ptr.size()) os << "h2h out has " << out.size() << " entries; ";
  if (!v.overlaps.empty()) os << "h2h has overlaps under escher; ";
  id_rows rows_out(ptr.empty() ? 0 : ptr.size() - 1);
  for (std::size_t i = 0; i < rows_out.size(); ++i) {
    rows_out[i].assign(cols.begin() + ptr[i], cols.begin() + ptr[i + 1]);
    if (!std::is_sorted(rows_out[i].begin(), rows_out[i].end()) ||
        std::adjacent_find(rows_out[i].begin(), rows_out[i].end()) != rows_out[i].end())
      os << "h2h row " << i << " is not sorted and distinct; ";
    if (i < out.size()) {
      const auto first_above =
          std::upper_bound(rows_out[i].begin(), rows_out[i].end(), static_cast<std::int64_t>(i));
      if (out[i] != ptr[i] + (first_above - rows_out[i].begin()))
        os << "h2h out of row " << i << " is " << out[i] << "; ";
    }
  }
  if (problems != nullptr) *problems = os.str();
  return rows_out;
}

/// The line graph of a hypergraph's current h2v rebuilt by the oracle
/// (dyng::testing::rebuild_line_graph), as rows by id.
template <typename weight_t>
id_rows oracle_h2h(const hypergraph<std::int32_t, weight_t>& h) {
  const auto v = h.h2v();
  const auto ptr = host_vector(v.row_ptr);
  const auto cols = host_vector(v.col_ind);
  const auto lg = dyng::testing::rebuild_line_graph(
      ptr.empty() ? std::vector<std::int64_t>{0} : ptr, cols, h.num_vertices());
  id_rows out(static_cast<std::size_t>(lg.num_ids()));
  for (std::int64_t i = 0; i < lg.num_ids(); ++i) out[static_cast<std::size_t>(i)] = lg.row(i);
  return out;
}

/// Compares the line graph of a hypergraph with the oracle and checks the view; returns the
/// differences (empty: none).
template <typename weight_t>
std::string check_line_graph(const hypergraph<std::int32_t, weight_t>& h) {
  std::string problems;
  const id_rows got = h2h_rows(h, &problems);
  const id_rows want = oracle_h2h(h);
  if (got != want) {
    std::ostringstream os;
    os << "h2h differs from rebuild_line_graph";
    for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
      if (got[i] != want[i]) {
        os << " (first at id " << i << ": " << got[i].size() << " vs " << want[i].size()
           << " neighbours)";
        break;
      }
    }
    if (got.size() != want.size()) os << " (" << got.size() << " vs " << want.size() << " rows)";
    problems += os.str() + "; ";
  }
  return problems;
}

/// One batch of the tests: raw deletions and raw inserted rows (repeated members allowed).
struct test_batch {
  std::vector<std::int64_t> deletions;
  rows insertions;
  std::vector<std::int32_t> weights;  ///< one per insertion for weighted hypergraphs
};

/// A test batch as a hyperedge_batch.
template <typename weight_t = unweighted>
hyperedge_batch<std::int32_t, weight_t> batch_of(const test_batch& b) {
  hyperedge_batch<std::int32_t, weight_t> out;
  for (const std::int64_t id : b.deletions) out.delete_hyperedge(id);
  for (std::size_t i = 0; i < b.insertions.size(); ++i) {
    weight_t w{};
    if constexpr (!is_unweighted_v<weight_t>) {
      w = b.weights.empty() ? weight_t{1} : static_cast<weight_t>(b.weights[i]);
    }
    out.insert_hyperedge(host_view(b.insertions[i]), w);
  }
  return out;
}

/// The model policy of a capacity preset ("escher_gpu", "mosp_escher", "dyng_default").
inline dyng::testing::cbst_host_policy model_policy(const std::string& preset) {
  if (preset == "dyng_default") return dyng::testing::cbst_host_policy::dyng_default();
  return dyng::test::cbst_compare::store_equivalent_host_policy(preset);
}

/// The name of a capacity policy's preset (empty for another policy).
inline std::string preset_name(const capacity_policy& c) {
  const std::string name = dyng::detail::hyper::to_store_policy(c).name;
  return name == "capacity_policy" ? std::string() : name;
}

/// The two model trees of a hypergraph, driven with the container's operations.
class lockstep {
 public:
  /// The models of a hypergraph built from `initial` under `props`, with the model policy of
  /// `preset` (default: the preset props.capacity names; a policy that only changes the initial
  /// sizing of a preset names that preset).
  lockstep(std::int32_t n, const rows& initial, const hypergraph_properties& props,
           std::string preset = {})
      : n_(n), props_(props) {
    if (preset.empty()) preset = preset_name(props.capacity);
    if (preset.empty()) {
      problems_.push_back("the lockstep needs a capacity preset");
      return;
    }
    const auto policy = model_policy(preset);
    rows normalized = initial;
    for (auto& r : normalized) {
      std::sort(r.begin(), r.end());
      r.erase(std::unique(r.begin(), r.end()), r.end());
    }
    h2v_ = make_tree(policy, "H2V", shifted(normalized));
    rows by_vertex(static_cast<std::size_t>(n));
    for (std::size_t id = 0; id < normalized.size(); ++id)
      for (const std::int32_t v : normalized[id])
        by_vertex[static_cast<std::size_t>(v)].push_back(static_cast<std::int32_t>(id) + 1);
    if (props.store_v2h) v2h_ = make_tree(policy, "V2H", by_vertex);
  }

  /// The expected outcome of a batch, in the container's terms.
  struct outcome {
    std::vector<std::int64_t> ids;
    std::int64_t ignored = 0, dropped = 0;
    bool ok = true;  ///< false: a store operation failed (the container is then poisoned)
    std::string error_type;
  };

  /// Applies a batch to the models with the container's rules and order.
  outcome apply(const test_batch& b) {
    outcome out;
    // Validation-free here: the tests give valid batches. Skip rules against the state before.
    std::vector<std::int32_t> keys;
    std::set<std::int32_t> seen;
    for (const std::int64_t id : b.deletions) {
      dyng::testing::cbst_key_record rec;
      const bool live = id >= 0 && id < INT_MAX &&
                        h2v_->record(static_cast<std::int32_t>(id + 1), rec) && rec.avail == 0;
      if (!live || !seen.insert(static_cast<std::int32_t>(id + 1)).second) {
        ++out.ignored;
        continue;
      }
      keys.push_back(static_cast<std::int32_t>(id + 1));
    }
    rows inserted;
    for (auto r : b.insertions) {
      std::sort(r.begin(), r.end());
      const auto last = std::unique(r.begin(), r.end());
      out.dropped += static_cast<std::int64_t>(r.end() - last);
      r.erase(last, r.end());
      for (auto& v : r) v += 1;
      inserted.push_back(r);
    }
    // The deleted rows (from the model, before the batch).
    rows deleted;
    for (const std::int32_t k : keys) {
      dyng::testing::cbst_key_record rec;
      (void)h2v_->record(k, rec);
      deleted.push_back(rec.row);
    }
    std::vector<std::int32_t> mapping;
    const auto erase_phase = [&]() {
      if (!keys.empty() && !check(h2v_->erase(keys), out)) return false;
      return remove(deleted, keys, out);
    };
    const auto insert_phase = [&]() {
      if (inserted.empty()) return true;
      std::vector<std::int32_t> requested, payload, prefix;
      for (std::size_t i = 0; i < inserted.size(); ++i) {
        requested.push_back(static_cast<std::int32_t>(i) + 1);
        payload.insert(payload.end(), inserted[i].begin(), inserted[i].end());
        prefix.push_back(static_cast<std::int32_t>(payload.size()));
      }
      const auto r = h2v_->insert(requested, payload, prefix);
      if (!check(r, out)) return false;
      mapping = r.mapping;
      return true;
    };
    const auto fill_phase = [&]() { return append(inserted, mapping, out); };
    if (props_.reuse == id_reuse::erase_first) {
      if (!erase_phase() || !insert_phase() || !fill_phase()) return out;
    } else {
      if (!insert_phase() || !erase_phase() || !fill_phase()) return out;
    }
    for (const std::int32_t k : mapping) out.ids.push_back(static_cast<std::int64_t>(k) - 1);
    return out;
  }

  /// The model's h2v rows by id (0-based members), up to its largest key.
  rows expected_h2v() const {
    rows out;
    for (const auto& rec : h2v_->records()) {
      if (static_cast<std::size_t>(rec.key) > out.size())
        out.resize(static_cast<std::size_t>(rec.key));
      if (rec.avail != 0) continue;
      auto& row = out[static_cast<std::size_t>(rec.key) - 1];
      for (const std::int32_t v : rec.row) row.push_back(v - 1);
    }
    return out;
  }

  /// The model's v2h rows by vertex, sorted (0-based ids).
  std::vector<std::vector<std::int64_t>> expected_v2h() const {
    std::vector<std::vector<std::int64_t>> out(static_cast<std::size_t>(n_));
    if (!v2h_) return out;
    for (const auto& rec : v2h_->records()) {
      auto& row = out[static_cast<std::size_t>(rec.key) - 1];
      for (const std::int32_t id : rec.row) row.push_back(static_cast<std::int64_t>(id) - 1);
      std::sort(row.begin(), row.end());
    }
    return out;
  }

  /// Compares a hypergraph with the models; returns the differences (empty: none).
  template <typename weight_t>
  std::string compare(const hypergraph<std::int32_t, weight_t>& h) const {
    std::ostringstream os;
    const rows want = expected_h2v();
    const rows got = h2v_rows(h);
    if (got.size() != want.size()) {
      os << "h2v has " << got.size() << " rows, the model " << want.size() << "; ";
    }
    for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
      if (got[i] != want[i]) {
        os << "h2v row " << i << " differs; ";
        break;
      }
    }
    if (static_cast<std::int64_t>(want.size()) != h.hyperedge_id_bound())
      os << "the id bound " << h.hyperedge_id_bound() << " is not the model's " << want.size()
         << "; ";
    if (v2h_ && v2h_rows(h) != expected_v2h()) os << "v2h differs from the model; ";
    for (const auto& p : h2v_->check()) os << "model H2V: " << p << "; ";
    return os.str();
  }

  /// Problems of the lockstep itself (empty: none).
  const std::vector<std::string>& problems() const {
    return problems_;
  }

  /// The H2V model (tests).
  const dyng::testing::cbst_host_model& h2v_model() const {
    return *h2v_;
  }

 private:
  static rows shifted(const rows& r) {
    rows out = r;
    for (auto& row : out)
      for (auto& v : row) v += 1;
    return out;
  }

  std::unique_ptr<dyng::testing::cbst_host_model> make_tree(
      const dyng::testing::cbst_host_policy& policy, const std::string& name, const rows& values) {
    std::vector<std::int32_t> keys, starts, flat, counts;
    for (std::size_t i = 0; i < values.size(); ++i) {
      keys.push_back(static_cast<std::int32_t>(i) + 1);
      starts.push_back(static_cast<std::int32_t>(flat.size()));
      counts.push_back(static_cast<std::int32_t>(values[i].size()));
      const std::size_t padded = (values[i].size() + 4) / 4 * 4;
      flat.insert(flat.end(), values[i].begin(), values[i].end());
      flat.insert(flat.end(), padded - values[i].size() - 1, 0);
      flat.push_back(INT_MIN);
    }
    const std::int64_t capacity = dyng::detail::hyper::initial_payload(
        props_.capacity, static_cast<std::int64_t>(flat.size()));
    auto tree = std::make_unique<dyng::testing::cbst_host_model>(
        policy, capacity, props_.capacity.row_alignment, name);
    const auto r = tree->construct(keys, starts, flat, &counts);
    if (r.status != dyng::testing::cbst_op_status::ok)
      problems_.push_back(name + " construct: " + r.message);
    return tree;
  }

  bool check(const dyng::testing::cbst_op_result& r, outcome& out) {
    if (r.status == dyng::testing::cbst_op_status::ok) return true;
    out.ok = false;
    out.error_type = r.error_type;
    return false;
  }

  // V2H removals: grouped by vertex, the deleted keys in batch order (groupByVertex).
  bool remove(const rows& deleted, const std::vector<std::int32_t>& keys, outcome& out) {
    if (!v2h_) return true;
    std::vector<std::int32_t> vkeys, values, prefix;
    group(deleted, keys, vkeys, values, prefix);
    if (vkeys.empty()) return true;
    return check(v2h_->remove_values(vkeys, values, prefix), out);
  }

  bool append(const rows& inserted, const std::vector<std::int32_t>& keys, outcome& out) {
    if (!v2h_) return true;
    std::vector<std::int32_t> vkeys, values, prefix;
    group(inserted, keys, vkeys, values, prefix);
    if (vkeys.empty()) return true;
    return check(v2h_->append(vkeys, values, prefix), out);
  }

  static void group(const rows& r, const std::vector<std::int32_t>& keys,
                    std::vector<std::int32_t>& vkeys, std::vector<std::int32_t>& values,
                    std::vector<std::int32_t>& prefix) {
    std::vector<std::pair<std::int32_t, std::size_t>> inc;
    for (std::size_t i = 0; i < r.size(); ++i)
      for (const std::int32_t v : r[i]) inc.emplace_back(v, i);
    std::sort(inc.begin(), inc.end());
    for (std::size_t k = 0; k < inc.size(); ++k) {
      if (k == 0 || inc[k].first != vkeys.back()) {
        if (k > 0) prefix.push_back(static_cast<std::int32_t>(values.size()));
        vkeys.push_back(inc[k].first);
      }
      values.push_back(keys[inc[k].second]);
    }
    if (!inc.empty()) prefix.push_back(static_cast<std::int32_t>(values.size()));
  }

  std::int32_t n_;
  hypergraph_properties props_;
  std::unique_ptr<dyng::testing::cbst_host_model> h2v_, v2h_;
  std::vector<std::string> problems_;
};

/// The kinds of random batch streams (criterion 6 and the conformance mixes).
enum class stream_mix {
  sparse,           ///< few changes per batch
  dense,            ///< many changes per batch
  skewed,           ///< some hyperedges of cardinality >= 1,024
  equal_sizes,      ///< many hyperedges of one size (best-fit ties)
  delete_reinsert,  ///< a deleted hyperedge inserted again in the same batch
  delete_all,       ///< every live hyperedge deleted, then re-inserted
  larger_than,      ///< a batch with more insertions than the hypergraph has hyperedges
  limits,           ///< deletions of ids at the limits (-1, INT64 min/max, the bound)
  reissue,          ///< the highest ids deleted and a row that fits no slot: the surplus rule
                    ///< drops them and gives them out again
};

/// The name of a mix.
inline const char* mix_name(stream_mix m) {
  switch (m) {
    case stream_mix::sparse:
      return "sparse";
    case stream_mix::dense:
      return "dense";
    case stream_mix::skewed:
      return "skewed";
    case stream_mix::equal_sizes:
      return "equal_sizes";
    case stream_mix::delete_reinsert:
      return "delete_reinsert";
    case stream_mix::delete_all:
      return "delete_all";
    case stream_mix::larger_than:
      return "larger_than";
    case stream_mix::limits:
      return "limits";
    case stream_mix::reissue:
      return "reissue";
  }
  return "?";
}

/// A random row of `size` members (repeats allowed when `repeats`).
inline std::vector<std::int32_t> random_row(std::mt19937_64& rng, std::int32_t n, std::int32_t size,
                                            bool repeats) {
  std::uniform_int_distribution<std::int32_t> pick(0, n - 1);
  std::vector<std::int32_t> r;
  if (repeats) {
    for (std::int32_t i = 0; i < size; ++i) r.push_back(pick(rng));
    return r;
  }
  std::set<std::int32_t> s;
  while (static_cast<std::int32_t>(s.size()) < std::min(size, n)) s.insert(pick(rng));
  r.assign(s.begin(), s.end());
  std::shuffle(r.begin(), r.end(), rng);
  return r;
}

/// An initial hypergraph of `m` random rows of size 1..max_size.
inline rows random_rows(std::mt19937_64& rng, std::int32_t n, std::int64_t m,
                        std::int32_t max_size) {
  std::uniform_int_distribution<std::int32_t> size(1, max_size);
  rows out;
  for (std::int64_t i = 0; i < m; ++i) out.push_back(random_row(rng, n, size(rng), false));
  return out;
}

/// The next batch of a stream of `mix` over a hypergraph with the live ids `live` (sorted) and
/// the id bound `bound`.
inline test_batch next_batch(std::mt19937_64& rng, stream_mix mix, std::int32_t n,
                             const std::vector<std::int64_t>& live, std::int64_t bound,
                             const rows& current) {
  test_batch b;
  std::uniform_int_distribution<std::int32_t> small(1, 6);
  std::vector<std::int64_t> pool = live;
  std::shuffle(pool.begin(), pool.end(), rng);
  const auto take = [&](std::size_t k) {
    for (std::size_t i = 0; i < std::min(k, pool.size()); ++i) b.deletions.push_back(pool[i]);
  };
  const std::size_t m = live.size();
  switch (mix) {
    case stream_mix::sparse:
      take(2);
      for (int i = 0; i < 3; ++i) b.insertions.push_back(random_row(rng, n, small(rng), true));
      break;
    case stream_mix::dense:
      take(m / 2);
      for (std::size_t i = 0; i < m / 2 + 1; ++i)
        b.insertions.push_back(random_row(rng, n, small(rng), i % 3 == 0));
      break;
    case stream_mix::skewed:
      take(m / 4);
      for (std::size_t i = 0; i < m / 4; ++i)
        b.insertions.push_back(
            random_row(rng, n, i % 7 == 0 ? std::min(n, 1100) : small(rng), false));
      break;
    case stream_mix::equal_sizes:
      take(m / 3);
      for (std::size_t i = 0; i < m / 3 + 2; ++i)
        b.insertions.push_back(random_row(rng, n, 3, false));
      break;
    case stream_mix::delete_reinsert:
      take(m / 4);
      for (const std::int64_t id : b.deletions)
        b.insertions.push_back(current[static_cast<std::size_t>(id)]);
      b.insertions.push_back(random_row(rng, n, small(rng), false));
      break;
    case stream_mix::delete_all:
      take(m);
      for (std::size_t i = 0; i < m; ++i)
        b.insertions.push_back(current[static_cast<std::size_t>(pool[i])]);
      break;
    case stream_mix::larger_than:
      // More insertions than the hypergraph has hyperedges, while it is small (the stream would
      // otherwise triple the hypergraph every batch; its line graph is checked after each).
      take(m / 5);
      for (std::size_t i = 0; i < (m < 2000 ? 2 * m + 3 : m / 4); ++i)
        b.insertions.push_back(random_row(rng, n, small(rng), false));
      break;
    case stream_mix::reissue:
      break;
    case stream_mix::limits:
      b.deletions = {-1, INT64_MIN, INT64_MAX, bound, bound - 1, 0};
      if (!pool.empty()) b.deletions.push_back(pool[0]);
      if (!pool.empty()) b.deletions.push_back(pool[0]);  // repeated
      for (int i = 0; i < 2; ++i) b.insertions.push_back(random_row(rng, n, small(rng), true));
      break;
  }
  if (mix == stream_mix::reissue) {
    for (std::size_t i = 0; i < std::min<std::size_t>(3, live.size()); ++i)
      b.deletions.push_back(live[live.size() - 1 - i]);
    b.insertions.push_back(random_row(rng, n, 40, false));
    b.insertions.push_back(random_row(rng, n, 2, false));
  }
  for (std::size_t i = 0; i < b.insertions.size(); ++i)
    b.weights.push_back(static_cast<std::int32_t>(1 + (i * 7919) % 1000));
  return b;
}

/// The live ids and rows of a hypergraph.
template <typename weight_t>
std::vector<std::int64_t> live_ids(const hypergraph<std::int32_t, weight_t>& h, rows& current) {
  current = h2v_rows(h);
  std::vector<std::int64_t> out;
  for (std::size_t i = 0; i < current.size(); ++i)
    if (!current[i].empty()) out.push_back(static_cast<std::int64_t>(i));
  return out;
}

}  // namespace dyng::test::hyper
