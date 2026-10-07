// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/integrity.cu
/**
 * @file integrity.cu
 * @brief The CBST integrity checker: copies a tree to the host and checks it against the expected
 *        rows.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "hypergraph/escher/integrity.hpp"

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
    cudaMemcpy(nodes.data(), c.d_nodes, sizeof(cbst_node) * n, cudaMemcpyDeviceToHost);
    if (c.d_avail) cudaMemcpy(avail.data(), c.d_avail, sizeof(int) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(flat.data(), c.d_flat_payload, sizeof(int) * flat.size(), cudaMemcpyDeviceToHost);
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

  // Reads a row following chain pointers. Returns false if the walk leaves
  // the payload or does not terminate.
  bool read_row(int t, std::vector<int>& row, std::vector<segment>* segments) const {
    row.clear();
    long long loc = nodes[t].value, seg_begin = loc;
    const long long limit = static_cast<long long>(flat.size());
    for (long long steps = 0; steps <= limit; ++steps) {
      if (loc < 0 || loc >= limit) return false;
      int v = flat[loc];
      if (v < 0 && v != INT_MIN) {
        if (segments) segments->push_back({seg_begin, loc});
        loc = -static_cast<long long>(v);
        seg_begin = loc;
        continue;
      }
      if (v == 0 || v == INT_MIN) {
        if (segments) segments->push_back({seg_begin, loc});
        return true;
      }
      row.push_back(v);
      ++loc;
    }
    return false;
  }
};

void print_row(std::ostream& log, const std::vector<int>& r) {
  log << "[";
  for (size_t i = 0; i < r.size() && i < 12; ++i) log << (i ? " " : "") << r[i];
  if (r.size() > 12) log << " ... (" << r.size() << " values)";
  log << "]";
}

}  // namespace

int check_tree_rows(const cbst_context& ctx, const std::vector<std::vector<int>>& expected,
                    bool order_insensitive, const char* name, std::ostream& log) {
  host_tree tree(ctx);
  const int max_details = 5;
  long long row_errors = 0, reach_errors = 0, meta_errors = 0, overlap_errors = 0;
  int details = 0;

  // 1. Rows by key, searched from the root.
  int max_key = static_cast<int>(expected.size());
  for (int t = 0; t < tree.size(); ++t)
    if (tree.live(t)) max_key = std::max(max_key, tree.nodes[t].index);
  std::vector<int> got;
  for (int key = 1; key <= max_key; ++key) {
    static const std::vector<int> empty_row;
    const std::vector<int>& want =
        key <= static_cast<int>(expected.size()) ? expected[key - 1] : empty_row;
    int t = tree.find(key);
    bool ok = true;
    got.clear();
    if (t >= 0 && tree.live(t)) ok = tree.read_row(t, got, nullptr);
    if (ok && order_insensitive) std::sort(got.begin(), got.end());
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
  return static_cast<int>(std::min<long long>(total, INT_MAX));
}

}  // namespace dyng::detail::escher
