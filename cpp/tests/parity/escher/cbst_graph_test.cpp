// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:graph/src/DynamicGraph.cpp
// Derived from MOSP_ESCHER@4b86159:tests/unit/test_dynamicgraph_roundtrip.cu
/**
 * @file cbst_graph_test.cpp
 * @brief The internal graph-on-CBST test (PLAN 6.1 D8, M8a step 5): a dynamic graph kept in two
 *        dynG CBST stores, as MOSP_ESCHER's DynamicGraph keeps it, must agree with dyng::graph's
 *        apply after every batch; and MOSP_ESCHER's test_dynamicgraph_roundtrip trace replays
 *        through the store under mosp_escher().
 *
 * The graph-on-CBST is a test, not a product (PLAN 6.1 D8: it shows that the store is general):
 * an edge is a record of 2 vertices plus its weights (a 2-vertex hyperedge, `[src+1, dst+1,
 * w+1...]`) in an edge store keyed by edge id, and every vertex has a row of its outgoing edge
 * ids in an adjacency store. Inserted edges take the ids of the store's InsertMapping (deleted
 * slots are reused by best fit), never ids of their own (MOSP_ESCHER's mutation #5). After every
 * batch the rows of both stores, read back, give every vertex's sorted (target, weights) list,
 * which must equal the sorted rows of dyng::graph after apply() of the same batch.
 */
#include "hypergraph/escher/cbst_store.hpp"
#include "hypergraph/escher/store_policy.hpp"
#include "parity/escher/cbst_compare.hpp"
#include "parity/escher/cbst_device_adapter.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test::cbst_compare {
namespace {

namespace esc = dyng::detail::escher;
using graph32 = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;
using batch32 = dyng::edge_batch<std::int32_t, std::int32_t>;
/// Every vertex's outgoing edges as (target, weights), sorted.
using sorted_rows = std::vector<std::vector<std::pair<int, std::vector<int>>>>;

struct edge {
  int u, v;
  std::vector<int> w;
};

/// The construct layout of rows of keys 1..n (flatten_rows) and their counts.
struct layout {
  std::vector<int> keys, starts, flat, counts;
};

layout layout_of(const std::vector<std::vector<int>>& rows) {
  layout l;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    l.keys.push_back(static_cast<int>(i) + 1);
    l.starts.push_back(static_cast<int>(l.flat.size()));
    l.counts.push_back(static_cast<int>(rows[i].size()));
    const std::size_t padded = (rows[i].size() + 4) / 4 * 4;
    l.flat.insert(l.flat.end(), rows[i].begin(), rows[i].end());
    l.flat.insert(l.flat.end(), padded - rows[i].size() - 1, 0);
    l.flat.push_back(INT_MIN);
  }
  return l;
}

/// A directed graph with K weights per edge on two CBST stores (MOSP_ESCHER's DynamicGraph: its
/// edgesCBST and outAdjCBST).
class cbst_graph {
 public:
  cbst_graph(const resources& res, int n, int k, esc::store_policy policy)
      : res_(res), n_(n), k_(k), policy_(std::move(policy)) {}

  void load(const std::vector<edge>& edges) {
    std::vector<std::vector<int>> records, out(static_cast<std::size_t>(n_));
    for (std::size_t i = 0; i < edges.size(); ++i) {
      const int id = static_cast<int>(i) + 1;
      records.push_back(record(edges[i]));
      out[static_cast<std::size_t>(edges[i].u)].push_back(id);
      id_of_[{edges[i].u, edges[i].v}] = id;
    }
    edges_ = build("edges", records);
    out_ = build("out_adj", out);
    max_id_ = static_cast<int>(edges.size());
  }

  /// Deletes, then inserts (as MOSP_ESCHER's deleteEdges and insertEdges).
  void apply(const std::vector<std::pair<int, int>>& deletes, const std::vector<edge>& inserts) {
    if (!deletes.empty()) {
      std::vector<int> ids;
      std::map<int, std::vector<int>> removals;
      for (const auto& [u, v] : deletes) {
        const int id = id_of_.at({u, v});
        id_of_.erase({u, v});
        ids.push_back(id);
        removals[u + 1].push_back(id);
      }
      edges_->erase(ids);
      const items r = items_of(removals);
      out_->remove_values(r.keys, r.payload, r.prefix);
      out_->check_errors();
    }
    if (!inserts.empty()) {
      std::vector<int> tentative, payload, prefix;
      for (std::size_t i = 0; i < inserts.size(); ++i) {
        tentative.push_back(max_id_ + static_cast<int>(i) + 1);
        const std::vector<int> rec = record(inserts[i]);
        payload.insert(payload.end(), rec.begin(), rec.end());
        prefix.push_back(static_cast<int>(payload.size()));
      }
      const esc::insert_mapping m = edges_->insert(tentative, payload, prefix);
      std::map<int, std::vector<int>> adds;
      for (std::size_t i = 0; i < inserts.size(); ++i) {
#if defined(DYNG_MUTATION_CBST_GRAPH_OWN_IDS)
        const int id = tentative[i];  // MUTATION: the graph keeps its own edge ids
#else
        const int id = m.item_to_key[i];  // the store's key: a reused slot or a surplus key
#endif
        max_id_ = std::max(max_id_, id);
        id_of_[{inserts[i].u, inserts[i].v}] = id;
        adds[inserts[i].u + 1].push_back(id);
      }
      const items a = items_of(adds);
      out_->append(a.keys, a.payload, a.prefix);
    }
  }

  /// Every vertex's outgoing (target, weights), read back from the two stores and sorted.
  sorted_rows rows() const {
    std::vector<int> vertices(static_cast<std::size_t>(n_));
    for (int u = 0; u < n_; ++u) vertices[static_cast<std::size_t>(u)] = u + 1;
    const std::vector<std::vector<int>> adjacency = out_->rows(vertices);
    sorted_rows out(static_cast<std::size_t>(n_));
    for (int u = 0; u < n_; ++u) {
      const std::vector<int>& ids = adjacency[static_cast<std::size_t>(u)];
      const std::vector<std::vector<int>> recs = edges_->rows(ids);
      for (const auto& r : recs) {
        if (r.size() != static_cast<std::size_t>(2 + k_) || r[0] != u + 1) {
          ADD_FAILURE() << "vertex " << u << ": an edge record of " << r.size()
                        << " values that does not start at the vertex";
          continue;
        }
        std::vector<int> w;
        for (int j = 0; j < k_; ++j) w.push_back(r[static_cast<std::size_t>(2 + j)] - 1);
        out[static_cast<std::size_t>(u)].emplace_back(r[1] - 1, w);
      }
      std::sort(out[static_cast<std::size_t>(u)].begin(), out[static_cast<std::size_t>(u)].end());
    }
    return out;
  }

  [[nodiscard]] const esc::cbst_store& edge_store() const {
    return *edges_;
  }

 private:
  struct items {
    std::vector<int> keys, payload, prefix;
  };
  static items items_of(const std::map<int, std::vector<int>>& by_key) {
    items it;
    for (const auto& [k, vals] : by_key) {
      it.keys.push_back(k);
      it.payload.insert(it.payload.end(), vals.begin(), vals.end());
      it.prefix.push_back(static_cast<int>(it.payload.size()));
    }
    return it;
  }
  static std::vector<int> record(const edge& e) {
    // The payload's 0 and negative values are sentinels: every value is stored + 1.
    std::vector<int> r = {e.u + 1, e.v + 1};
    for (int x : e.w) r.push_back(x + 1);
    return r;
  }
  std::unique_ptr<esc::cbst_store> build(const char* name,
                                         const std::vector<std::vector<int>>& rows) {
    layout l = layout_of(rows);
    const long long capacity =
        policy_.initial_capacity(0, static_cast<long long>(l.flat.size()), true);
    auto s = std::make_unique<esc::cbst_store>(res_, name, static_cast<int>(capacity), 4, policy_);
    s->construct(l.keys.data(), l.starts.data(), static_cast<int>(l.keys.size()), l.flat.data(),
                 static_cast<int>(l.flat.size()), l.counts.data());
    return s;
  }

  resources res_;
  int n_, k_;
  esc::store_policy policy_;
  std::unique_ptr<esc::cbst_store> edges_, out_;
  std::map<std::pair<int, int>, int> id_of_;
  int max_id_ = 0;
};

/// dyng::graph's rows, sorted.
sorted_rows rows_of(const graph32& g, int k) {
  const auto c = g.to_csr(resources::sequential());
  const std::size_t m = c.col_ind.size();
  sorted_rows out(c.row_ptr.size() - 1);
  for (std::size_t u = 0; u + 1 < c.row_ptr.size(); ++u) {
    for (auto e = static_cast<std::size_t>(c.row_ptr[u]);
         e < static_cast<std::size_t>(c.row_ptr[u + 1]); ++e) {
      std::vector<int> w;
      for (int j = 0; j < k; ++j) w.push_back(c.weights[static_cast<std::size_t>(j) * m + e]);
      out[u].emplace_back(c.col_ind[e], w);
    }
    std::sort(out[u].begin(), out[u].end());
  }
  return out;
}

graph32 graph_of(int n, int k, const std::vector<edge>& edges) {
  dyng::edge_list<std::int32_t, std::int32_t> e;
  e.num_vertices = n;
  e.num_weights = k;
  for (const auto& x : edges) {
    e.src.push_back(x.u);
    e.dst.push_back(x.v);
    e.weights.insert(e.weights.end(), x.w.begin(), x.w.end());
  }
  return graph32::from_edges(resources::sequential(), e.view(),
                             dyng::graph_properties::mosp_compatible());
}

batch32 batch_of(int k, const std::vector<std::pair<int, int>>& deletes,
                 const std::vector<edge>& inserts) {
  batch32 b(k);
  for (const auto& [u, v] : deletes) b.delete_edge(u, v);
  for (const auto& e : inserts)
    b.insert_edge(e.u, e.v, dyng::array_view<const std::int32_t>(e.w.data(), e.w.size()));
  return b;
}

class CbstGraph : public ::testing::TestWithParam<const char*> {
 protected:
  esc::store_policy policy() const {
    esc::store_policy p;
    EXPECT_TRUE(esc::store_policy::from_name(GetParam(), p));
    return p;
  }
};

// MOSP_ESCHER's 5-vertex graph: load, then delete the edges of ids 1 (0->1) and 3 (1->2) and
// insert two edges of different sizes, which reuse the deleted slots.
TEST_P(CbstGraph, TheRoundtripGraph) {
  if (!device_available()) GTEST_SKIP() << "no CUDA device";
  const int n = 5, k = 2;
  const std::vector<edge> edges = {{0, 1, {3, 5}}, {0, 3, {9, 1}}, {1, 2, {4, 2}}, {1, 4, {7, 8}},
                                   {2, 3, {2, 6}}, {3, 4, {1, 1}}, {4, 0, {5, 9}}};
  cbst_graph cg(resources::cuda(), n, k, policy());
  cg.load(edges);
  graph32 g = graph_of(n, k, edges);
  EXPECT_EQ(cg.rows(), rows_of(g, k));
  const std::vector<std::pair<int, int>> del = {{0, 1}, {1, 2}};
  const std::vector<edge> ins = {{2, 4, {11, 12}}, {0, 2, {6, 7}}};
  cg.apply(del, ins);
  (void)g.apply(resources::sequential(), batch_of(k, del, ins).view());
  EXPECT_EQ(cg.rows(), rows_of(g, k));
}

// Random rounds on a larger graph (MOSP_ESCHER's test: 60 vertices, about 10% of the pairs,
// deleted edges re-inserted later), plus rounds that add more edges than they delete (surplus
// rows and growth or compaction of the stores).
TEST_P(CbstGraph, RandomRoundsAgreeWithGraphApply) {
  if (!device_available()) GTEST_SKIP() << "no CUDA device";
  const int n = 60, k = 2;
  std::mt19937 rng(5);
  std::set<std::pair<int, int>> present;
  std::vector<edge> edges;
  for (int u = 0; u < n; ++u)
    for (int v = 0; v < n; ++v)
      if (u != v && rng() % 10 == 0) {
        edges.push_back({u, v, {static_cast<int>(rng() % 20), static_cast<int>(rng() % 20)}});
        present.insert({u, v});
      }
  cbst_graph cg(resources::cuda(), n, k, policy());
  cg.load(edges);
  graph32 g = graph_of(n, k, edges);
  ASSERT_EQ(cg.rows(), rows_of(g, k));
  std::vector<std::pair<int, int>> deleted_before;
  for (int round = 0; round < 16; ++round) {
    SCOPED_TRACE(round);
    std::vector<std::pair<int, int>> del;
    for (auto it = present.begin(); it != present.end();) {
      if (rng() % 8 == 0) {
        del.push_back(*it);
        it = present.erase(it);
      } else {
        ++it;
      }
    }
    // Re-insert the edges deleted the round before, and some new pairs (more in later rounds).
    std::vector<edge> ins;
    for (const auto& p : deleted_before)
      if (!present.count(p)) {
        ins.push_back({p.first, p.second, {static_cast<int>(rng() % 20), 7}});
        present.insert(p);
      }
    const int extra = round < 8 ? 5 : 60;
    for (int i = 0; i < extra; ++i) {
      const int u = static_cast<int>(rng() % n), v = static_cast<int>(rng() % n);
      const bool deleted_now = std::find(del.begin(), del.end(), std::make_pair(u, v)) != del.end();
      if (u == v || present.count({u, v}) || deleted_now) continue;
      ins.push_back({u, v, {static_cast<int>(rng() % 20), static_cast<int>(rng() % 20)}});
      present.insert({u, v});
    }
    cg.apply(del, ins);
    (void)g.apply(resources::sequential(), batch_of(k, del, ins).view());
    ASSERT_EQ(cg.rows(), rows_of(g, k));
    deleted_before = del;
  }
}

INSTANTIATE_TEST_SUITE_P(Presets, CbstGraph,
                         ::testing::Values("escher_gpu", "mosp_escher", "dyng_default"),
                         [](const ::testing::TestParamInfo<const char*>& p) {
                           return std::string(p.param);
                         });

// MOSP_ESCHER's own test_dynamicgraph_roundtrip trace (the golden set mosp_escher_test in
// $DYNG_SCRATCH; skipped without it) replays through the store under mosp_escher() with no
// mismatch in the exact field set.
TEST(CbstGraphTrace, MospEschersRoundtripTraceReplays) {
  if (!device_available()) GTEST_SKIP() << "no CUDA device";
  namespace fs = std::filesystem;
  const char* s = std::getenv("DYNG_SCRATCH");
  const char* home = std::getenv("HOME");
  const fs::path scratch = s ? fs::path(s) : fs::path(home ? home : "/") / "Projects" / "dyng-work";
  const fs::path trace = scratch / "goldens" / "cbst_trace" / "mosp_escher_test" /
                         "011-test_dynamicgraph_roundtrip" / "trace.cbt.zst";
  if (!fs::is_regular_file(trace)) GTEST_SKIP() << "no trace " << trace;
  fs::create_directories(scratch / "runs");
  trace_source source(trace.string(), (scratch / "runs").string());
  options opt;
  opt.fields = field_set::exact;
  const replay_report r =
      replay(source.reader(), device_store_factory("mosp_escher"), opt, trace.string());
  EXPECT_TRUE(r.ok()) << r.mismatch_count << " mismatches";
  EXPECT_EQ(r.unreplayable, 0);
  EXPECT_GT(r.mappings, 0);
  EXPECT_EQ(r.compared, r.ops);
}

}  // namespace
}  // namespace dyng::test::cbst_compare
