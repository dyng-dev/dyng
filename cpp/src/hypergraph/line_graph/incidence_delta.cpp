// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDelta.cu (deriveDelta: the net signs, the two
// candidate passes, the classification by pre / post overlap; countCandidatesKernel,
// emitCandidatesKernel, classifyPairsKernel)
/**
 * @file incidence_delta.cpp
 * @brief The host incidence_delta (ADR 0086): one source on operators::sequential_exec and
 *        openmp_exec.
 */
#include "hypergraph/line_graph/incidence_delta.hpp"

#include "hypergraph/line_graph/host_sort.hpp"
#include "operators/execution.hpp"

#include <dyng/core/error.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dyng::detail::hyper {

namespace {

namespace ops = dyng::detail::operators;

/// A candidate of an incidence with itself (o == h): counted, never a pair (MOSP_ESCHER's
/// SENTINEL; it sorts after every pair).
constexpr std::uint64_t self_pair = ~std::uint64_t{0};

struct row_ref {
  const std::int32_t* data = nullptr;
  std::int64_t size = 0;
};

row_ref post_row(const incidence_delta_rows& r, std::int64_t x) {
  if (x < 0 || x >= static_cast<std::int64_t>(r.post_rows->size())) return {};
  const auto& row = (*r.post_rows)[static_cast<std::size_t>(x)];
  return {row.data(), static_cast<std::int64_t>(row.size())};
}

row_ref pre_row(const incidence_delta_rows& r, std::int64_t x) {
  const std::int32_t slot = x >= 0 && x < static_cast<std::int64_t>(r.touched_slot->size())
                                ? (*r.touched_slot)[static_cast<std::size_t>(x)]
                                : -1;
  if (slot < 0) return post_row(r, x);  // untouched: the same before and after
  const auto begin = r.pre_rows->offsets[static_cast<std::size_t>(slot)];
  const auto end = r.pre_rows->offsets[static_cast<std::size_t>(slot) + 1];
  return {r.pre_rows->members.data() + begin, end - begin};
}

// Whether two sorted rows share a vertex (sortedOverlap).
bool overlap(row_ref a, row_ref b) {
  std::int64_t i = 0, j = 0;
  while (i < a.size && j < b.size) {
    if (a.data[i] == b.data[j]) return true;
    if (a.data[i] < b.data[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return false;
}

// The hyperedges of vertex v before the batch: [first, last) of pre_v2h_ids.
std::pair<std::int64_t, std::int64_t> pre_list(const incidence_delta_rows& r, std::int32_t v) {
  const auto& verts = *r.pre_vertices;
  const auto it = std::lower_bound(verts.begin(), verts.end(), v);
  if (it == verts.end() || *it != v) return {0, 0};
  const auto i = static_cast<std::size_t>(it - verts.begin());
  return {(*r.pre_v2h_offsets)[i], (*r.pre_v2h_offsets)[i + 1]};
}

// countCandidatesKernel: the candidates of incidence i in each pass.
struct count_candidates {
  const std::uint64_t* keys;
  const incidence_delta_rows* rows;
  std::int64_t* count0;
  std::int64_t* count1;
  void operator()(std::int64_t i) const {
    const auto v = static_cast<std::int32_t>(keys[i] >> 32);
    const auto pre = pre_list(*rows, v);
    count0[i] = pre.second - pre.first;
#if defined(DYNG_MUTATION_HSOSP_DELTA_NO_POST_PASS)
    // MUTATION (PLAN 8.4; M11a criterion 16 #8, MOSP_ESCHER CHANGES.md:493-494): the post-batch
    // candidate pass is dropped (an addition's new neighbours are never candidates).
    count1[i] = 0;
#else
    count1[i] = static_cast<std::int64_t>((*rows->post_v2h)[static_cast<std::size_t>(v)].size());
#endif
  }
};

// emitCandidatesKernel, both passes: (min(h, o), max(h, o)) for o in v's lists.
struct emit_candidates {
  const std::uint64_t* keys;
  const incidence_delta_rows* rows;
  const std::int64_t* offset0;
  const std::int64_t* offset1;
  std::int64_t total0;
  std::uint64_t* out;
  static std::uint64_t pair_of(std::int64_t h, std::int64_t o) {
    if (o == h) return self_pair;
    const auto a = static_cast<std::uint64_t>(std::min(h, o));
    const auto b = static_cast<std::uint64_t>(std::max(h, o));
    return (a << 32) | b;
  }
  void operator()(std::int64_t i) const {
    const auto v = static_cast<std::int32_t>(keys[i] >> 32);
    const auto h = static_cast<std::int64_t>(keys[i] & 0xffffffffULL);
    const auto pre = pre_list(*rows, v);
    std::uint64_t* p0 = out + offset0[i];
    for (std::int64_t k = pre.first; k < pre.second; ++k)
      *p0++ = pair_of(h, (*rows->pre_v2h_ids)[static_cast<std::size_t>(k)]);
#if !defined(DYNG_MUTATION_HSOSP_DELTA_NO_POST_PASS)
    std::uint64_t* p1 = out + total0 + offset1[i];
    for (const std::int64_t o : (*rows->post_v2h)[static_cast<std::size_t>(v)])
      *p1++ = pair_of(h, o);
#endif
  }
};

// classifyPairsKernel: -1 deleted, +1 inserted, 0 unchanged.
struct classify_pairs {
  const std::uint64_t* pairs;
  const incidence_delta_rows* rows;
  std::int8_t* cls;
  void operator()(std::int64_t i) const {
    const auto a = static_cast<std::int64_t>(pairs[i] >> 32);
    const auto b = static_cast<std::int64_t>(pairs[i] & 0xffffffffULL);
    const bool pre = overlap(pre_row(*rows, a), pre_row(*rows, b));
    const bool post = overlap(post_row(*rows, a), post_row(*rows, b));
    cls[i] = static_cast<std::int8_t>(post ? (pre ? 0 : 1) : (pre ? -1 : 0));
  }
};

}  // namespace

void net_incidence_changes(const std::vector<std::int32_t>& vertices,
                           const std::vector<std::int64_t>& ids,
                           const std::vector<std::int8_t>& signs, std::vector<std::uint64_t>& keys,
                           std::vector<std::int8_t>& net) {
  // Step 1 of deriveDelta: sorted by (vertex, hyperedge), the signs summed per incidence.
  std::vector<std::pair<std::uint64_t, std::int8_t>> changes(vertices.size());
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    changes[i] = {(static_cast<std::uint64_t>(static_cast<std::uint32_t>(vertices[i])) << 32) |
                      static_cast<std::uint32_t>(ids[i]),
                  signs[i]};
  }
  std::sort(changes.begin(), changes.end(),
            [](const auto& x, const auto& y) { return x.first < y.first; });
  keys.clear();
  net.clear();
  for (std::size_t i = 0; i < changes.size();) {
    std::size_t j = i;
    int sum = 0;
    while (j < changes.size() && changes[j].first == changes[i].first) sum += changes[j++].second;
    if (sum != 0) {
      keys.push_back(changes[i].first);
      net.push_back(static_cast<std::int8_t>(sum > 0 ? 1 : -1));
    }
    i = j;
  }
}

template <typename exec_t>
void incidence_delta(const exec_t& exec, const std::vector<std::uint64_t>& keys,
                     const incidence_delta_rows& rows, std::int64_t max_candidates,
                     line_graph_delta& out) {
  out.clear();
  const auto n = static_cast<std::int64_t>(keys.size());
  out.net_changes = n;
  if (n == 0) return;
  // 2. Candidate counts of both passes, their offsets.
  std::vector<std::int64_t> count0(static_cast<std::size_t>(n)),
      count1(static_cast<std::size_t>(n));
  exec.for_each(n, count_candidates{keys.data(), &rows, count0.data(), count1.data()});
  std::vector<std::int64_t> offset0(static_cast<std::size_t>(n)),
      offset1(static_cast<std::size_t>(n));
  std::int64_t total0 = 0, total1 = 0;
  for (std::int64_t i = 0; i < n; ++i) {
    offset0[static_cast<std::size_t>(i)] = total0;
    offset1[static_cast<std::size_t>(i)] = total1;
    total0 += count0[static_cast<std::size_t>(i)];
    total1 += count1[static_cast<std::size_t>(i)];
  }
  const std::int64_t total = total0 + total1;
  out.candidates = total;
  if (total > max_candidates) {
    throw capacity_error("dyng: hypergraph: the line-graph delta of the batch has " +
                         std::to_string(total) + " candidate pairs, more than " +
                         std::to_string(max_candidates) +
                         " (the int32 counts of the delta); split the batch");
  }
  // 3. Both passes.
  std::vector<std::uint64_t> cand(static_cast<std::size_t>(total));
  exec.for_each(
      n, emit_candidates{keys.data(), &rows, offset0.data(), offset1.data(), total0, cand.data()});
  // 4. Distinct candidates (the self pairs sort last), classified by their pre / post overlap.
  sort_keys(exec, cand);
  cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
  if (!cand.empty() && cand.back() == self_pair) cand.pop_back();
  const auto pairs = static_cast<std::int64_t>(cand.size());
  std::vector<std::int8_t> cls(static_cast<std::size_t>(pairs));
  exec.for_each(pairs, classify_pairs{cand.data(), &rows, cls.data()});
  // Deleted pairs first, then inserted ones, each in increasing order.
  for (std::int64_t i = 0; i < pairs; ++i) {
    const std::int8_t c = cls[static_cast<std::size_t>(i)];
    if (c == 0) continue;
    const auto a = static_cast<std::int64_t>(cand[static_cast<std::size_t>(i)] >> 32);
    const auto b = static_cast<std::int64_t>(cand[static_cast<std::size_t>(i)] & 0xffffffffULL);
    if (c < 0) {
      out.deleted_a.push_back(a);
      out.deleted_b.push_back(b);
    } else {
      out.inserted_a.push_back(a);
      out.inserted_b.push_back(b);
    }
  }
}

template void incidence_delta<operators::sequential_exec>(const operators::sequential_exec&,
                                                          const std::vector<std::uint64_t>&,
                                                          const incidence_delta_rows&, std::int64_t,
                                                          line_graph_delta&);
template void incidence_delta<operators::openmp_exec>(const operators::openmp_exec&,
                                                      const std::vector<std::uint64_t>&,
                                                      const incidence_delta_rows&, std::int64_t,
                                                      line_graph_delta&);

}  // namespace dyng::detail::hyper
