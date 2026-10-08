// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/integrity.cu
// Derived from MOSP_ESCHER@4b86159:escher/structure/integrity.cu (long long counts, sorted
// expected rows, checkSubtreeAvail)
/**
 * @file integrity.cu
 * @brief The CBST integrity checker: copies a tree to the host and checks it against the expected
 *        rows.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "core/cuda_runtime.hpp"
#include "hypergraph/escher/chain.cuh"
#include "hypergraph/escher/integrity.hpp"
#include "util/cuda_check.hpp"

#include <algorithm>
#include <climits>
#include <iostream>
#include <utility>

namespace dyng::detail::escher {

namespace {

struct host_tree {
  const cbst_context* ctx;
  std::vector<cbst_node> nodes;
  std::vector<int> avail;
  std::vector<int> flat;

  explicit host_tree(const cbst_context& c) : ctx(&c) {
    const int n = c.num_records;
    if (n <= 0 || c.d_nodes == nullptr) return;
    nodes.resize(n);
    avail.assign(n, 0);
    // Live rows lie in the used prefix of the payload
    flat.resize(static_cast<size_t>(c.initial_payload_size));
    const auto stream = static_cast<cudaStream_t>(c.res.stream().get());
    const scoped_device guard(c.res.device());
    DYNG_CUDA_TRY(cudaMemcpyAsync(nodes.data(), c.d_nodes, sizeof(cbst_node) * n,
                                  cudaMemcpyDeviceToHost, stream));
    if (c.d_avail)
      DYNG_CUDA_TRY(cudaMemcpyAsync(avail.data(), c.d_avail, sizeof(int) * n,
                                    cudaMemcpyDeviceToHost, stream));
    DYNG_CUDA_TRY(cudaMemcpyAsync(flat.data(), c.d_flat_payload, sizeof(int) * flat.size(),
                                  cudaMemcpyDeviceToHost, stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  int size() const {
    return static_cast<int>(nodes.size());
  }
  bool live(int t) const {
    return avail[t] == 0 && nodes[t].index >= 1;
  }

  int child(const cbst_node* p) const {
    return p == nullptr ? -1 : static_cast<int>(p - ctx->d_nodes);
  }

  // Same walk as the device kernels: compare with node->index, go left/right.
  int find(int key) const {
    int t = size() > 0 ? 0 : -1;
    for (int depth = 0; t >= 0; ++depth) {
      if (t >= size() || depth > 64) return -1;
      if (nodes[t].index == key) return t;
      t = nodes[t].index > key ? child(nodes[t].left) : child(nodes[t].right);
    }
    return -1;
  }

  // One payload segment of a row: [begin, end] inclusive.
  struct segment {
    long long begin, end;
  };

  // Reads a row following chain pointers, with the bounded walk of chain.cuh (M8a step 4).
  // Returns false if the walk leaves the payload or does not terminate.
  bool read_row(int t, std::vector<int>& row, std::vector<segment>* segments) const {
    row.clear();
    const chain_end e = walk_chain(
        flat, static_cast<long long>(flat.size()), nodes[t].value,
        [&](long long, int v) { row.push_back(v); },
        [&](long long begin, long long end, int) {
          if (segments) segments->push_back({begin, end});
        });
    return e == chain_end::terminated;
  }
};

void print_row(std::ostream& log, const std::vector<int>& r) {
  log << "[";
  for (size_t i = 0; i < r.size() && i < 12; ++i) log << (i ? " " : "") << r[i];
  if (r.size() > 12) log << " ... (" << r.size() << " values)";
  log << "]";
}

}  // namespace

long long check_tree_rows(const cbst_context& ctx, const std::vector<std::vector<int>>& expected,
                          bool order_insensitive, const char* name, std::ostream& log) {
  host_tree tree(ctx);
  const int max_details = 5;
  long long row_errors = 0, reach_errors = 0, meta_errors = 0, overlap_errors = 0;
  int details = 0;

  // 1. Rows by key, searched from the root.
  int max_key = static_cast<int>(expected.size());
  for (int t = 0; t < tree.size(); ++t)
    if (tree.live(t)) max_key = std::max(max_key, tree.nodes[t].index);
  std::vector<int> got, want;
  for (int key = 1; key <= max_key; ++key) {
    want.clear();
    if (key <= static_cast<int>(expected.size())) want = expected[key - 1];
    int t = tree.find(key);
    bool ok = true;
    got.clear();
    if (t >= 0 && tree.live(t)) ok = tree.read_row(t, got, nullptr);
    // Order-insensitive rows are compared as sorted multisets on both sides (MOSP_ESCHER,
    // escher/structure/integrity.cu@4b86159:131-134; ESCHER-GPU sorted only the rows it read, so an
    // unsorted expected row was a false failure).
    if (order_insensitive) {
      std::sort(got.begin(), got.end());
      std::sort(want.begin(), want.end());
    }
    if (!ok || got != want) {
      ++row_errors;
      if (details++ < max_details) {
        log << "[check] " << name << " key " << key << (t < 0 ? " (not found)" : "")
            << (ok ? "" : " (unterminated row)") << ": expected ";
        print_row(log, want);
        log << " got ";
        print_row(log, got);
        log << "\n";
      }
    }
  }

  // 2. Every live node is reachable by a search for its own key.
  for (int t = 0; t < tree.size(); ++t) {
    if (tree.live(t) && tree.find(tree.nodes[t].index) != t) {
      ++reach_errors;
      if (details++ < max_details)
        log << "[check] " << name << " node " << t << " (key " << tree.nodes[t].index
            << ") is not reachable by BST search\n";
    }
  }

  // 3. Tail-segment metadata and 4. disjoint payload segments.
  std::vector<std::pair<long long, long long>> extents;
  std::vector<host_tree::segment> segments;
  for (int t = 0; t < tree.size(); ++t) {
    if (!tree.live(t)) continue;
    const cbst_node& nd = tree.nodes[t];
    segments.clear();
    if (!tree.read_row(t, got, &segments)) continue;  // already reported as a row error
    const long long tb = nd.tail_base, occ = nd.occupancy, cap = nd.tail_capacity;
    bool meta_ok = segments.back().begin == tb && occ >= 0 && occ <= cap &&
                   tb + cap < static_cast<long long>(tree.flat.size());
    if (meta_ok) {
      long long live_in_tail = segments.back().end - segments.back().begin;
      meta_ok = live_in_tail == occ;
      for (long long p = tb + occ + 1; meta_ok && p <= tb + cap; ++p)
        if (tree.flat[p] > 0 || (tree.flat[p] < 0 && tree.flat[p] != INT_MIN)) meta_ok = false;
    }
    if (!meta_ok) {
      ++meta_errors;
      if (details++ < max_details)
        log << "[check] " << name << " key " << nd.index << ": tail metadata (tail_base " << tb
            << ", occupancy " << occ << ", tail_capacity " << cap
            << ") does not match the payload\n";
    }
    for (size_t s = 0; s < segments.size(); ++s) {
      long long b = segments[s].begin, e = segments[s].end;
      if (s == 0) e = std::max(e, b + nd.length - 1);  // first segment owns `length` slots
      if (s + 1 == segments.size() && meta_ok) e = std::max(e, tb + cap);
      extents.push_back({b, e});
    }
  }
  std::sort(extents.begin(), extents.end());
  for (size_t i = 1; i < extents.size(); ++i) {
    if (extents[i].first <= extents[i - 1].second) {
      ++overlap_errors;
      if (details++ < max_details)
        log << "[check] " << name << " payload segments overlap: [" << extents[i - 1].first << ", "
            << extents[i - 1].second << "] and [" << extents[i].first << ", " << extents[i].second
            << "]\n";
    }
  }

  long long total = row_errors + reach_errors + meta_errors + overlap_errors;
  log << "[check] " << name << ": " << max_key << " keys, " << row_errors << " wrong rows, "
      << reach_errors << " unreachable nodes, " << meta_errors << " metadata violations, "
      << overlap_errors << " overlapping segments" << (total ? "  <-- FAILED" : "") << "\n";
  return total;
}

namespace {

template <typename value_t>
std::vector<value_t> copy_to_host(const cbst_context& ctx, const value_t* device, size_t n) {
  std::vector<value_t> out(n);
  if (n == 0 || device == nullptr) return out;
  const scoped_device guard(ctx.res.device());
  const auto stream = static_cast<cudaStream_t>(ctx.res.stream().get());
  DYNG_CUDA_TRY(
      cudaMemcpyAsync(out.data(), device, n * sizeof(value_t), cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  return out;
}

}  // namespace

long long check_subtree_avail(const cbst_context& ctx, const char* name, std::ostream& log) {
  const int n = ctx.num_records;
  if (n <= 0 || ctx.d_avail == nullptr || ctx.d_subtree_avail == nullptr) return 0;
  const std::vector<int> avail = copy_to_host(ctx, ctx.d_avail, static_cast<size_t>(n));
  const std::vector<int> sub = copy_to_host(ctx, ctx.d_subtree_avail, static_cast<size_t>(n));
  long long bad = 0;
  for (int i = 0; i < n; ++i) {
    const long long l = 2LL * i + 1 < n ? sub[2 * i + 1] : 0;
    const long long r = 2LL * i + 2 < n ? sub[2 * i + 2] : 0;
    if (sub[i] != avail[i] + l + r) {
      if (bad < 5)
        log << "[check] " << name << " subtree_avail[" << i << "] = " << sub[i] << ", expected "
            << avail[i] + l + r << "\n";
      ++bad;
    }
  }
  log << "[check] " << name << ": " << bad << " inconsistent subtree_avail entries"
      << (bad ? "  <-- FAILED" : "") << "\n";
  return bad;
}

long long check_tree_index(const cbst_context& ctx, const char* name, std::ostream& log) {
  const int n = ctx.num_records;
  if (n <= 0 || ctx.d_nodes == nullptr) return 0;
  host_tree tree(ctx);
  const std::vector<int> avail = copy_to_host(ctx, ctx.d_avail, static_cast<size_t>(n));
  const std::vector<int> keys = copy_to_host(ctx, ctx.d_keys, static_cast<size_t>(n));
  const std::vector<int> starts = copy_to_host(ctx, ctx.d_start_offsets, static_cast<size_t>(n));
  long long flags = 0, ordered = 0, index = 0, offsets = 0, layout = 0;
  int details = 0;
  auto detail = [&](auto&&... parts) {
    if (details++ < 5) {
      log << "[check] " << name << " ";
      (log << ... << parts);
      log << "\n";
    }
  };
  for (int t = 0; t < n; ++t) {
    // The avail flags are 0 or 1 (subtree_avail counts them).
    if (!avail.empty() && avail[t] != 0 && avail[t] != 1) {
      ++flags;
      detail("avail[", t, "] = ", avail[t]);
    }
    // The heap layout of the pointers.
    const cbst_node& nd = tree.nodes[t];
    const int left = tree.child(nd.left), right = tree.child(nd.right);
    const int parent = tree.child(nd.parent);
    if (left != (2 * t + 1 < n ? 2 * t + 1 : -1) || right != (2 * t + 2 < n ? 2 * t + 2 : -1) ||
        parent != (t == 0 ? -1 : (t - 1) / 2)) {
      ++layout;
      detail("node ", t, ": its child or parent pointers are not the heap layout");
    }
  }
  // The keys array (by in-order rank) is strictly increasing, and holds the key of the node at
  // the heap position of each rank; the row offsets array its first segment's offset.
  std::vector<int> rank_of(static_cast<size_t>(n));
  {
    // The in-order walk of the heap layout: the rank of every heap position.
    std::vector<int> stack;
    int r = 0;
    int t = 0;
    while (t >= 0 || !stack.empty()) {
      while (t >= 0) {
        stack.push_back(t);
        t = 2 * t + 1 < n ? 2 * t + 1 : -1;
      }
      t = stack.back();
      stack.pop_back();
      rank_of[static_cast<size_t>(t)] = r++;
      t = 2 * t + 2 < n ? 2 * t + 2 : -1;
    }
  }
  for (int i = 1; i < n; ++i) {
    if (keys[i - 1] >= keys[i]) {
      ++ordered;
      detail("keys[", i - 1, "] = ", keys[i - 1], " >= keys[", i, "] = ", keys[i]);
    }
  }
  for (int t = 0; t < n; ++t) {
    const int r = rank_of[static_cast<size_t>(t)];
    if (keys[r] != tree.nodes[t].index) {
      ++index;
      detail("node ", t, " (rank ", r, ") holds key ", tree.nodes[t].index, ", the keys array ",
             keys[r]);
    }
    if (starts[r] != tree.nodes[t].value) {
      ++offsets;
      detail("node ", t, " (key ", tree.nodes[t].index, ") starts at ", tree.nodes[t].value,
             ", the row offsets array at ", starts[r]);
    }
  }
  const long long total = flags + ordered + index + offsets + layout;
  log << "[check] " << name << ": " << flags << " avail flags not 0/1, " << ordered
      << " unordered keys, " << index << " keys not those of their nodes, " << offsets
      << " row offsets not those of their nodes, " << layout << " nodes off the heap layout"
      << (total ? "  <-- FAILED" : "") << "\n";
  return total;
}

}  // namespace dyng::detail::escher
