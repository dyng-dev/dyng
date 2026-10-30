// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/HostHypergraph.cpp (emulateLoop,
// emulateSospRecompute and emulateSospUpdate, lines 365-507)
/**
 * @file sequential.cpp
 * @brief hyper_sssp's host engine: MOSP_ESCHER's sequential emulation of its device update,
 *        ported as is with 0-based ids and parents, on the host slack_csr line graph and the
 *        applied record's pair delta (M11a step 4; ADR 0091).
 *
 * Changes against the original, each mechanical or a stated rule:
 *   - ids and parents are 0-based (B's 1-based hyperedge ids, its `id - 1` indexing, are gone);
 *     the line graph is the container's slack CSR (rows unordered: every rule below is order-free,
 *     ties go to the lower id), the weights its node weights read through hyper_sssp_graph::
 *     weight() (the virtual mode's source and target rows weigh 0, HD2);
 *   - emulateSospUpdate is split into the framework's hooks without changing its order: the roots
 *     and the children-list invalidation (identify_affected), the pulls of the invalidated
 *     hyperedges and the inserted pairs' relaxations (seed), the priority queue (loop);
 *   - the locals live in the pooled workspace (hyper_sssp_workspace): the children lists are one
 *     CSR filled in id order (B's vector of vectors, the same order), the priority queue is a
 *     vector heap (std::push_heap / pop_heap with std::greater, as std::priority_queue);
 *   - the work budget (ADR 0092): B's work units (the degree of every pulled and every expanded
 *     hyperedge), checked before each expansion, the host's deterministic point; the host engine
 *     has no push rounds, packed words or pointer jumping, so max_iterations, the 32-bit width and
 *     the jump cap do not apply here;
 *   - the public mode (a source vertex): a hyperedge holding the source is offered its own weight
 *     with parent -1 in its pulls (B's virtual source row, which wins every tie: -1 sorts first),
 *     one that lost the source is a root when its parent is -1, one that gained it is offered its
 *     weight.
 */
#include "algorithms/hyper_sssp/problem.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace dyng::detail {

void hyper_sssp_workspace::begin() noexcept {
  root_ids.clear();
  lost_source.clear();
  gained_source.clear();
  changed.clear();
  holding.clear();
  not_holding.clear();
  heap.clear();
  stack.clear();
  invalidated = 0;
  work = 0;
  expansions = 0;
  over_budget = false;
}

std::size_t hyper_sssp_workspace::bytes() const noexcept {
  return (root_ids.capacity() + lost_source.capacity() + gained_source.capacity() +
          changed.capacity() + holding.capacity() + not_holding.capacity()) *
             sizeof(std::int64_t) +
         (inv.capacity() + in_cand.capacity() + holds.capacity()) +
         (child_off.capacity() + child_ids.capacity() + stack.capacity() + candidates.capacity() +
          next.capacity() + affected.capacity() + base32.capacity() + lost32.capacity() +
          gained32.capacity() + initial32.capacity()) *
             sizeof(int) +
         heap.capacity() * sizeof(entry);
}

namespace {

constexpr long long inf = hyper_sssp_infinity;

/// Whether v holds the public mode's source vertex (`holds`: by id, or nullptr).
bool holds_source(const std::vector<std::uint8_t>* holds, std::int64_t v) {
  return holds != nullptr && static_cast<std::size_t>(v) < holds->size() &&
         (*holds)[static_cast<std::size_t>(v)] != 0;
}

/// emulateLoop: rounds of pulls over the candidates and collections of the changed ones'
/// neighbours, until no candidate is left or the cap is reached.
std::int64_t emulate_loop(const hyper_sssp_graph& g, std::vector<std::int64_t>& dist,
                          std::vector<std::int32_t>& parent, const std::vector<std::uint8_t>* holds,
                          std::int64_t max_iterations, hyper_sssp_workspace& ws) {
  const std::int64_t source = g.endpoints.source_hyperedge;
  const hyper::slack_row_view& lg = g.rows;
  std::int64_t iterations = 0;
  std::vector<int>& candidates = ws.candidates;
  std::vector<int>& affected = ws.affected;
  std::vector<std::uint8_t>& in_cand = ws.in_cand;
  reserve_scratch(in_cand, static_cast<std::size_t>(lg.num_nodes) + 1);
  in_cand.assign(static_cast<std::size_t>(lg.num_nodes) + 1, 0);
  reserve_scratch(affected, static_cast<std::size_t>(lg.num_nodes));
  reserve_scratch(ws.next, static_cast<std::size_t>(lg.num_nodes));

  while (!candidates.empty() && iterations < max_iterations) {
    ++iterations;
    affected.clear();
    for (const int v : candidates) {
      in_cand[static_cast<std::size_t>(v)] = 0;
      if (v == source) continue;
      long long best = inf;
      int best_p = -1;
      if (holds_source(holds, v)) best = g.weight(v);  // dynG: the public mode's base offer
      const std::int64_t base = lg.row_start[v];
      for (long long k = 0; k < lg.deg[v]; ++k) {
        const int p = lg.col_ind[base + k];
        const long long dp = dist[static_cast<std::size_t>(p)];
        if (dp >= inf / 2) continue;
        const long long cd = dp + g.weight(v);
        if (cd < best || (cd == best && p < best_p)) {
          best = cd;
          best_p = p;
        }
      }
      if (best != dist[static_cast<std::size_t>(v)]) {
        dist[static_cast<std::size_t>(v)] = best;
        parent[static_cast<std::size_t>(v)] = best_p;
        affected.push_back(v);
      } else {
        parent[static_cast<std::size_t>(v)] = best_p;
      }
    }
    ws.next.clear();
    for (const int u : affected) {
      const std::int64_t base = lg.row_start[u];
      for (long long k = 0; k < lg.deg[u]; ++k) {
        const int nb = lg.col_ind[base + k];
        if (nb == source) continue;
        if (!in_cand[static_cast<std::size_t>(nb)]) {
          in_cand[static_cast<std::size_t>(nb)] = 1;
          ws.next.push_back(nb);
        }
      }
    }
    std::swap(candidates, ws.next);
  }
  return candidates.empty() ? iterations : -1;
}

}  // namespace

std::int64_t hyper_sssp_host_recompute(const hyper_sssp_graph& g, std::vector<std::int64_t>& dist,
                                       std::vector<std::int32_t>& parent, std::int64_t cap,
                                       hyper_sssp_workspace& ws) {
  const auto m = static_cast<std::size_t>(g.rows.num_nodes);
  dist.assign(m, inf);
  parent.assign(m, -1);
  const hyper::slack_row_view& lg = g.rows;
  std::vector<int>& first_cands = ws.candidates;
  reserve_scratch(first_cands, m);
  first_cands.clear();
  if (g.endpoints.is_virtual()) {
    const std::int64_t s = g.endpoints.source_hyperedge;
    if (s >= 0 && static_cast<std::size_t>(s) < m) {
      dist[static_cast<std::size_t>(s)] = 0;
      for (long long k = 0; k < lg.deg[s]; ++k)
        first_cands.push_back(lg.col_ind[lg.row_start[s] + k]);
    }
    return emulate_loop(g, dist, parent, nullptr, cap, ws);
  }
  // dynG, the public mode: every live hyperedge holding the source starts at its own weight with
  // parent -1 (B's source row seeded at 0), and their neighbours are the first candidates.
  std::vector<std::uint8_t>& holds = ws.holds;
  reserve_scratch(holds, m);
  holds.assign(m, 0);
  std::vector<std::uint8_t>& listed = ws.in_cand;
  reserve_scratch(listed, m + 1);
  listed.assign(m + 1, 0);
  const std::int32_t s = g.endpoints.source_vertex;
  if (s >= 0 && static_cast<std::size_t>(s) < g.host_v2h->size()) {
    for (const std::int64_t h : (*g.host_v2h)[static_cast<std::size_t>(s)]) {
      if (!g.live(h)) continue;
      holds[static_cast<std::size_t>(h)] = 1;
      dist[static_cast<std::size_t>(h)] = g.weight(h);
    }
    for (const std::int64_t h : (*g.host_v2h)[static_cast<std::size_t>(s)]) {
      if (!g.live(h)) continue;
      for (long long k = 0; k < lg.deg[h]; ++k) {
        const int nb = lg.col_ind[lg.row_start[h] + k];
        if (!listed[static_cast<std::size_t>(nb)]) {
          listed[static_cast<std::size_t>(nb)] = 1;
          first_cands.push_back(nb);
        }
      }
    }
  }
  return emulate_loop(g, dist, parent, &holds, cap, ws);
}

void hyper_sssp_host_identify(const hyper_sssp_graph& g, const hyper_sssp_batch& b,
                              std::vector<std::int64_t>& dist, std::vector<std::int32_t>& parent,
                              hyper_sssp_workspace& ws) {
  const auto m = static_cast<std::size_t>(g.rows.num_nodes);
  if (dist.size() < m) {
    dist.resize(m, inf);
    parent.resize(m, -1);
  }
  // Roots (pre-batch tree): deleted tree edges, new / recreated / dead.
  std::vector<std::uint8_t>& inv = ws.inv;
  reserve_scratch(inv, m);
  inv.assign(m, 0);
  for (std::int64_t i = 0; i < b.num_deleted; ++i) {
    const auto a = static_cast<std::size_t>((*b.deleted_a)[static_cast<std::size_t>(i)]);
    const auto c = static_cast<std::size_t>((*b.deleted_b)[static_cast<std::size_t>(i)]);
    if (parent[c] == static_cast<std::int32_t>(a)) inv[c] = 1;
    if (parent[a] == static_cast<std::int32_t>(c)) inv[a] = 1;
  }
  for (const std::int64_t id : *b.root_ids) inv[static_cast<std::size_t>(id)] = 1;
  // dynG, the public mode: a hyperedge that lost the source while reached from it directly (B's
  // deleted pair with its source row).
  for (const std::int64_t id : *b.lost_source) {
    if (parent[static_cast<std::size_t>(id)] == -1 && dist[static_cast<std::size_t>(id)] < inf)
      inv[static_cast<std::size_t>(id)] = 1;
  }
  // Subtrees of the roots (children lists of the pre-batch tree; one CSR in id order).
  std::vector<int>& off = ws.child_off;
  std::vector<int>& kids = ws.child_ids;
  reserve_scratch(off, m + 1);
  off.assign(m + 1, 0);
  for (std::size_t id = 0; id < m; ++id)
    if (parent[id] >= 0) ++off[static_cast<std::size_t>(parent[id]) + 1];
  for (std::size_t i = 0; i < m; ++i) off[i + 1] += off[i];
  reserve_scratch(kids, static_cast<std::size_t>(off[m]));
  kids.resize(static_cast<std::size_t>(off[m]));
  {
    std::vector<int>& fill = ws.stack;  // the fill cursors, reused below as the stack
    reserve_scratch(fill, m);
    fill.assign(off.begin(), off.end() - 1);
    for (std::size_t id = 0; id < m; ++id)
      if (parent[id] >= 0)
        kids[static_cast<std::size_t>(fill[static_cast<std::size_t>(parent[id])]++)] =
            static_cast<int>(id);
  }
  std::vector<int>& stack = ws.stack;
  stack.clear();
  for (std::size_t id = 0; id < m; ++id)
    if (inv[id]) stack.push_back(static_cast<int>(id));
  while (!stack.empty()) {
    const int x = stack.back();
    stack.pop_back();
    for (int k = off[static_cast<std::size_t>(x)]; k < off[static_cast<std::size_t>(x) + 1]; ++k) {
      const int c = kids[static_cast<std::size_t>(k)];
      if (!inv[static_cast<std::size_t>(c)]) {
        inv[static_cast<std::size_t>(c)] = 1;
        stack.push_back(c);
      }
    }
  }
  ws.invalidated = 0;
  for (std::size_t id = 0; id < m; ++id) {
    if (inv[id] && static_cast<std::int64_t>(id) != g.endpoints.source_hyperedge) {
      dist[id] = inf;
      parent[id] = -1;
      ++ws.invalidated;
    }
  }
}

namespace {

/// The offer of emulateSospUpdate: v adopts (d, p) if it is smaller than its own (distance,
/// parent) (a parent of -1 compares as INT32_MAX there, and -1 as an offer sorts first: the
/// public mode's base offer wins every tie, as B's source row does).
void offer(const hyper_sssp_graph& g, std::vector<std::int64_t>& dist,
           std::vector<std::int32_t>& parent, hyper_sssp_workspace& ws, int v, long long d, int p) {
  using key = hyper_sssp_workspace::key;
  if (v == g.endpoints.source_hyperedge || !g.live(v)) return;
  const auto i = static_cast<std::size_t>(v);
  if (key(d, p) < key(dist[i], parent[i] < 0 ? INT32_MAX : parent[i])) {
    dist[i] = d;
    parent[i] = p;
    ws.heap.push_back({key(d, p), v});
    std::push_heap(ws.heap.begin(), ws.heap.end(), std::greater<hyper_sssp_workspace::entry>());
  }
}

}  // namespace

void hyper_sssp_host_seed(const hyper_sssp_graph& g, const hyper_sssp_batch& b,
                          std::vector<std::int64_t>& dist, std::vector<std::int32_t>& parent,
                          hyper_sssp_workspace& ws) {
  const hyper::slack_row_view& lg = g.rows;
  const auto m = static_cast<std::size_t>(lg.num_nodes);
  reserve_scratch(ws.heap, m + static_cast<std::size_t>(b.num_inserted) * 2);
  for (std::size_t id = 0; id < m; ++id) {
    if (!ws.inv[id] || !g.live(static_cast<std::int64_t>(id))) continue;
    const auto v = static_cast<int>(id);
    // dynG, the public mode: a hyperedge holding the source is offered its own weight (B's source
    // row's offer, the parent -1).
    if (holds_source(g.holds, v)) offer(g, dist, parent, ws, v, g.weight(v), -1);
    // dynG: B's work units (MOSP_ESCHER's pullInvalidatedKernel counts the row it reads).
    ws.work += lg.deg[v];
    for (long long k = 0; k < lg.deg[v]; ++k) {
      const int p = lg.col_ind[lg.row_start[v] + k];
      if (dist[static_cast<std::size_t>(p)] < inf)
        offer(g, dist, parent, ws, v, dist[static_cast<std::size_t>(p)] + g.weight(v), p);
    }
  }
  // dynG, the public mode: a hyperedge that gained the source (B's inserted pair with its source
  // row).
  for (const std::int64_t id : *b.gained_source)
    offer(g, dist, parent, ws, static_cast<int>(id), g.weight(id), -1);
  for (std::int64_t i = 0; i < b.num_inserted; ++i) {
    const auto a = static_cast<int>((*b.inserted_a)[static_cast<std::size_t>(i)]);
    const auto c = static_cast<int>((*b.inserted_b)[static_cast<std::size_t>(i)]);
    if (dist[static_cast<std::size_t>(a)] < inf)
      offer(g, dist, parent, ws, c, dist[static_cast<std::size_t>(a)] + g.weight(c), a);
    if (dist[static_cast<std::size_t>(c)] < inf)
      offer(g, dist, parent, ws, a, dist[static_cast<std::size_t>(c)] + g.weight(a), c);
  }
}

void hyper_sssp_host_loop(const hyper_sssp_graph& g, std::vector<std::int64_t>& dist,
                          std::vector<std::int32_t>& parent, double work_limit,
                          hyper_sssp_workspace& ws) {
  using key = hyper_sssp_workspace::key;
  const hyper::slack_row_view& lg = g.rows;
  const auto greater = std::greater<hyper_sssp_workspace::entry>();
  while (!ws.heap.empty()) {
    const auto [k, x] = ws.heap.front();
    if (k != key(dist[static_cast<std::size_t>(x)], parent[static_cast<std::size_t>(x)])) {
      std::pop_heap(ws.heap.begin(), ws.heap.end(), greater);
      ws.heap.pop_back();
      continue;
    }
    // dynG: the work budget, before each expansion (ADR 0092).
    if (static_cast<double>(ws.work) > work_limit) {
      ws.over_budget = true;
      return;
    }
    std::pop_heap(ws.heap.begin(), ws.heap.end(), greater);
    ws.heap.pop_back();
    ++ws.expansions;
    ws.work += lg.deg[x];
    for (long long e = 0; e < lg.deg[x]; ++e) {
      const int v = lg.col_ind[lg.row_start[x] + e];
      offer(g, dist, parent, ws, v, dist[static_cast<std::size_t>(x)] + g.weight(v), x);
    }
  }
}

}  // namespace dyng::detail
