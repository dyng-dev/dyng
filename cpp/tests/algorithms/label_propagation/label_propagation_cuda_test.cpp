// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file label_propagation_cuda_test.cpp
 * @brief label_propagation's fused CUDA engine (M10a step 4; ADR 0075), the cases only CUDA has:
 *        LabelPropagation-CUDA@a276a3a's propagation_test (its hub graph: the degree bins, the
 *        damped rule bitwise twice and equal to the original's golden, the in_place error bounds),
 *        the connected components of G' against union_find, the deterministic stages against the
 *        sequential backend on a DynLP stream, the original's input_contract cases through
 *        update(), out_of_memory_error under a memory limit without a leak, and the moves of a
 *        result between the backends.
 *
 * The hub golden's numbers are the twin's (goldens/label_propagation_hub/info.json; the recorded
 * calls of propagation_test); DYNG_LP_HUB_INFO=<info.json> makes the test read them from the
 * golden instead of the constants below (compare.py label_propagation hub).
 */
#include "algorithms/label_propagation/fused.hpp"
#include "support/gtest_helpers.hpp"
#include "support/sha256.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/generators/vertex_arrival.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/label_propagation.hpp>
#include <dyng/testing/union_find.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace lp = dyng::label_propagation;
namespace lpc = dyng::detail::lp_cuda;
using dyng::host_view;
using lp_graph = dyng::graph<std::int32_t, std::int32_t, float>;
using stream_t = dyng::generators::vertex_arrival_stream<std::int32_t, std::int32_t>;

dyng::graph_properties slotted_props() {
  dyng::graph_properties p;
  p.directed = false;
  p.layout = dyng::row_layout::slotted;
  p.order = dyng::row_order::append;
  p.parallel_edges = dyng::multi_edges::allow;
  p.semantics = dyng::batch_semantics::net_effect();
  return p;
}

stream_t make_stream(std::int64_t n, std::int32_t batches) {
  dyng::generators::vertex_arrival_options opt;
  opt.total_vertices = n;
  opt.num_batches = batches;
  return dyng::generators::legacy::dynlp_vertex_arrival_stream<std::int32_t, std::int32_t>(
      dyng::resources::sequential(), opt);
}

lp_graph capacity_graph(const dyng::resources& res, const stream_t& s) {
  return lp_graph::with_capacity(res, static_cast<std::int32_t>(s.row_capacity.size()),
                                 host_view(s.row_capacity), slotted_props());
}

lp::options published(const stream_t& s, float delta, lp::schedule rule) {
  lp::options opt;
  opt.similarity_threshold = s.similarity_threshold;
  opt.tolerance = delta;
  opt.update_schedule = rule;
  return opt;
}

std::vector<float> host_labels(const dyng::resources& res, const lp::result<std::int32_t>& r) {
  const auto v = r.labels();
  return dyng::is_host_accessible(v.space()) ? std::vector<float>(v.begin(), v.end())
                                             : dyng::to_vector(res, v);
}

// ------------------------------------------------------------------------------------------------
// LabelPropagation-CUDA@a276a3a:tests/propagation_test.cu
// ------------------------------------------------------------------------------------------------

constexpr int hub_max_iterations = 100000;
constexpr int hub_count = 8;
constexpr int hubs[hub_count] = {200, 300, 400, 500, 600, 700, 800, 900};
constexpr int hub_degree[hub_count] = {8, 9, 32, 33, 1024, 1025, 3000, 6000};
constexpr int hub_bin[hub_count] = {lpc::bin_small,  lpc::bin_small,  lpc::bin_small,
                                    lpc::bin_medium, lpc::bin_medium, lpc::bin_large,
                                    lpc::bin_large,  lpc::bin_large};

struct hub_graph {
  int n = 0;
  std::vector<int> row_ptr, col_ind, gt_class;
  std::vector<float> values, labels;
};

/// buildGraph() of propagation_test.cu (libstdc++'s distributions, as the original's build).
hub_graph build_hub_graph() {
  const int n = 8000;
  const int num_gt = 120;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> weight(0.05F, 1.0F);
  std::uniform_int_distribution<int> vertex(0, n - 1);
  std::set<std::pair<int, int>> edges;
  auto add_edge = [&](int a, int b) {
    if (a == b) return;
    edges.insert({std::min(a, b), std::max(a, b)});
  };
  for (int v = 0; v + 1 < n; ++v) add_edge(v, v + 1);
  for (int i = 0; i < n; ++i) add_edge(vertex(rng), vertex(rng));
  std::vector<int> degree(n, 0);
  for (const auto& e : edges) {
    ++degree[e.first];
    ++degree[e.second];
  }
  const std::set<int> hub_set(std::begin(hubs), std::end(hubs));
  for (int h = 0; h < hub_count; ++h) {
    const int hub = hubs[h];
    for (int v = 0; degree[hub] < hub_degree[h] && v < n; ++v) {
      const int u = (hub * 7919 + v * 104729) % n;
      if (u == hub || hub_set.count(u) != 0) continue;
      if (edges.insert({std::min(u, hub), std::max(u, hub)}).second) {
        ++degree[hub];
        ++degree[u];
      }
    }
  }
  hub_graph g;
  g.n = n;
  std::vector<std::vector<std::pair<int, float>>> adj(n);
  for (const auto& e : edges) {
    const float w = weight(rng);
    adj[e.first].push_back({e.second, w});
    adj[e.second].push_back({e.first, w});
  }
  g.row_ptr.push_back(0);
  for (int v = 0; v < n; ++v) {
    for (const auto& p : adj[v]) {
      g.col_ind.push_back(p.first);
      g.values.push_back(p.second);
    }
    g.row_ptr.push_back(static_cast<int>(g.col_ind.size()));
  }
  g.gt_class.assign(n, -1);
  g.labels.assign(n, 0.5F);
  for (int i = 0; i < num_gt; ++i) {
    const int v = (i * (n / num_gt)) + 17;
    g.gt_class[v] = (i % 3 == 0) ? 1 : 0;
    g.labels[v] = static_cast<float>(g.gt_class[v]);
  }
  return g;
}

/// harmonic() of propagation_test.cu: Gauss-Seidel in double precision.
std::vector<double> hub_harmonic(const hub_graph& g) {
  std::vector<double> x(g.labels.begin(), g.labels.end());
  for (int sweep = 0; sweep < 200000; ++sweep) {
    double max_change = 0.0;
    for (int u = 0; u < g.n; ++u) {
      if (g.gt_class[u] >= 0) continue;
      double w = 0.0;
      double s = 0.0;
      for (int e = g.row_ptr[u]; e < g.row_ptr[u + 1]; ++e) {
        w += g.values[e];
        s += g.values[e] * x[g.col_ind[e]];
      }
      const double nx = s / w;
      max_change = std::fmax(max_change, std::fabs(nx - x[u]));
      x[u] = nx;
    }
    if (max_change < 1e-13) break;
  }
  return x;
}

/// What one runPropagation() call did.
struct hub_call {
  std::vector<float> labels;
  lpc::propagation_report report;
  int next_tag = 0;
};

/// runPropagation() of propagation_test.cu on the fused engine: every unlabelled vertex at 0.5
/// and affected, a fresh workspace (nextTag 1), the rows of a plain CSR.
hub_call run_hub(const dyng::resources& res, const hub_graph& g, bool damped, float delta) {
  const auto n = static_cast<std::size_t>(g.n);
  dyng::buffer<std::int32_t> row_ptr(res, n + 1);
  dyng::buffer<std::int32_t> col_ind(res, g.col_ind.size());
  dyng::buffer<float> values(res, g.values.size());
  dyng::buffer<std::int32_t> status(res, n);
  dyng::copy(res, host_view(g.row_ptr), row_ptr.view());
  dyng::copy(res, host_view(g.col_ind), col_ind.view());
  dyng::copy(res, host_view(g.values), values.view());
  const std::vector<std::int32_t> active(n, 0);  // vertex_status::active (seeds are labels)
  dyng::copy(res, host_view(active), status.view());
  std::vector<std::int8_t> seeds(n);
  for (std::size_t v = 0; v < n; ++v) {
    seeds[v] = static_cast<std::int8_t>(g.gt_class[v]);
  }
  lpc::state st;
  lpc::create_state(res, g.n, g.row_ptr.data(), seeds.data(), g.labels.data(), st);
  lpc::workspace ws;
  ws.reserve(res, g.n, st.capacity, 1, 1, damped);
  std::vector<std::int32_t> affected(n);
  for (std::size_t v = 0; v < n; ++v) {
    affected[v] = g.gt_class[v] < 0 ? 1 : 0;
  }
  dyng::copy(res, host_view(affected),
             dyng::array_view<std::int32_t>(ws.affected.data(), n, dyng::memory_space::device,
                                            res.device()));
  lpc::rows<std::int32_t> rows;
  rows.row_start = row_ptr.data();
  rows.row_end = row_ptr.data() + 1;
  rows.col_ind = col_ind.data();
  rows.values = values.data();
  rows.status = status.data();
  rows.n = g.n;
  hub_call out;
  out.report = lpc::launch_iterative_propagation(
      rows, st.labels.data(), st.gt_class.data(), ws.affected.data(), delta, hub_max_iterations,
      damped, st, ws, res.stream().get(), nullptr, nullptr);
  res.synchronize();
  out.labels = dyng::to_vector(res, st.labels.view());
  out.next_tag = st.next_tag;
  return out;
}

/// The golden's damped call (the twin's record of propagation_test's calls 1 and 2).
struct hub_expected {
  std::string labels_sha256 = "c1089fca7c0ca011d4f8c8207641df1ca6f03db47cfbbcba33df0c5849820e5f";
  long long iterations = 2055;
  long long frontier_visits = 14907520;
  long long persistent_iterations = 85;
  long long next_tag = 2057;
};

/// The numbers of the golden's info.json (DYNG_LP_HUB_INFO), read without a JSON library: the
/// damped calls' fields.
hub_expected expected_hub() {
  hub_expected e;
  const char* path = std::getenv("DYNG_LP_HUB_INFO");
  if (path == nullptr || *path == '\0') {
    return e;
  }
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();
  const std::size_t call = text.find("\"call\": 1");
  EXPECT_NE(call, std::string::npos) << path;
  const auto number = [&](const std::string& key) {
    const std::size_t at = text.find("\"" + key + "\":", call);
    return std::atoll(text.c_str() + at + key.size() + 3);
  };
  const std::size_t sha = text.find("\"labels_sha256\": \"", call);
  e.labels_sha256 = text.substr(sha + 18, 64);
  e.iterations = number("iterations");
  e.frontier_visits = number("frontier_visits");
  e.persistent_iterations = number("persistent_iterations");
  e.next_tag = number("next_tag");
  return e;
}

TEST(LabelPropagationCuda, HubGolden) {
  DYNG_SKIP_IF_NO_CUDA();
#if !defined(__GLIBCXX__)
  GTEST_SKIP() << "the hub graph is drawn with libstdc++'s distributions (the original's build)";
#endif
  const dyng::resources res = dyng::resources::cuda(0);
  const float delta = 1e-7F;
  const hub_graph g = build_hub_graph();
  int bins[3] = {0, 0, 0};
  for (int v = 0; v < g.n; ++v) {
    bins[lpc::degree_bin_of(g.row_ptr[v + 1] - g.row_ptr[v])]++;
  }
  EXPECT_EQ(g.col_ind.size(), 54180U);  // "8000 vertices, 54180 entries"
  EXPECT_EQ(bins[0], 7995);
  EXPECT_EQ(bins[1], 2);
  EXPECT_EQ(bins[2], 3);
  for (int h = 0; h < hub_count; ++h) {
    const int d = g.row_ptr[hubs[h] + 1] - g.row_ptr[hubs[h]];
    EXPECT_EQ(d, hub_degree[h]) << "hub " << hubs[h];
    EXPECT_EQ(lpc::degree_bin_of(d), hub_bin[h]) << "hub " << hubs[h];
  }
  const std::vector<double> opt = hub_harmonic(g);
  const hub_expected want = expected_hub();
  std::vector<float> first_damped;
  for (const bool damped : {false, true, true}) {
    const hub_call c = run_hub(res, g, damped, delta);
    double max_err = 0.0;
    double sum_err = 0.0;
    double hub_err = 0.0;
    for (int v = 0; v < g.n; ++v) {
      const double err = std::fabs(c.labels[v] - opt[v]);
      max_err = std::fmax(max_err, err);
      sum_err += err;
      if (g.row_ptr[v + 1] - g.row_ptr[v] > lpc::small_bin_max_degree) {
        hub_err = std::fmax(hub_err, err);
      }
    }
    // The original's bounds: no cap, max < 1e4 delta, mean < 1e3 delta, hubs < 1e3 delta.
    EXPECT_LT(c.report.iterations, hub_max_iterations);
    EXPECT_LT(max_err, 1e4 * delta) << (damped ? "damped" : "in_place");
    EXPECT_LT(sum_err / g.n, 1e3 * delta) << (damped ? "damped" : "in_place");
    EXPECT_LT(hub_err, 1e3 * delta) << (damped ? "damped" : "in_place");
    if (!damped) {
      continue;
    }
    EXPECT_EQ(dyng::test::sha256_hex(c.labels), want.labels_sha256);  // the original's bits
    EXPECT_EQ(c.report.iterations, want.iterations);
    EXPECT_EQ(c.report.frontier_visits, want.frontier_visits);
    EXPECT_EQ(c.report.persistent_iterations, want.persistent_iterations);
    EXPECT_EQ(c.next_tag, want.next_tag);
    if (first_damped.empty()) {
      first_damped = c.labels;
    } else {
      EXPECT_EQ(0, std::memcmp(first_damped.data(), c.labels.data(), 4 * c.labels.size()))
          << "the damped rule is bitwise reproducible";
    }
  }
}

// ------------------------------------------------------------------------------------------------
// The components of G' against union_find
// ------------------------------------------------------------------------------------------------

TEST(LabelPropagationCuda, ComponentsEqualUnionFind) {
  DYNG_SKIP_IF_NO_CUDA();
  const dyng::resources res = dyng::resources::cuda(0);
  std::mt19937 rng(11);
  for (const int edges_count : {0, 1, 50, 3000}) {
    const int n = 2000;
    // The inserted vertices: a random subset in a random order; G' among them.
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    std::vector<std::int32_t> inserted(all.begin(), all.begin() + 1200);
    std::vector<std::int32_t> src;
    std::vector<std::int32_t> dst;
    std::uniform_int_distribution<int> pick(0, 1199);
    for (int e = 0; e < edges_count; ++e) {
      src.push_back(inserted[pick(rng)]);
      dst.push_back(inserted[pick(rng)]);
    }
    lpc::workspace ws;
    const int cap[3] = {n, 0, 0};
    ws.reserve(res, n, cap, static_cast<std::int64_t>(inserted.size()), std::max(edges_count, 1),
               false);
    dyng::buffer<std::int32_t> d_ins(res, inserted.size());
    dyng::buffer<std::int32_t> d_src(res, std::max<std::size_t>(src.size(), 1));
    dyng::buffer<std::int32_t> d_dst(res, std::max<std::size_t>(dst.size(), 1));
    dyng::buffer<std::int32_t> d_count(res, 1);
    dyng::buffer<std::int32_t> d_comp(res, static_cast<std::size_t>(n));
    dyng::copy(res, host_view(inserted), d_ins.view());
    if (!src.empty()) {
      dyng::copy(res, host_view(src),
                 dyng::array_view<std::int32_t>(d_src.data(), src.size(),
                                                dyng::memory_space::device, res.device()));
      dyng::copy(res, host_view(dst),
                 dyng::array_view<std::int32_t>(d_dst.data(), dst.size(),
                                                dyng::memory_space::device, res.device()));
    }
    const std::vector<std::int32_t> count = {edges_count};
    dyng::copy(res, host_view(count), d_count.view());
    int components = 0;
    (void)lpc::launch_connected_components(d_src.data(), d_dst.data(), d_count.data(), edges_count,
                                           d_ins.data(), static_cast<int>(inserted.size()),
                                           d_comp.data(), components, ws, res.stream().get());
    const std::vector<std::int32_t> comp = dyng::to_vector(res, d_comp.view());
    dyng::testing::union_find uf(n);
    for (std::size_t e = 0; e < src.size(); ++e) {
      uf.unite(src[e], dst[e]);
    }
    std::set<std::int64_t> roots;
    for (const int v : inserted) {
      roots.insert(uf.find(v));
    }
    EXPECT_EQ(components, static_cast<int>(roots.size())) << edges_count << " edges";
    // The same partition, numbered by the smallest vertex of each in insertion order (the
    // original's numbering: the rank of a root among the roots, in insertion order).
    for (std::size_t i = 0; i < inserted.size(); ++i) {
      for (std::size_t j = i + 1; j < std::min(inserted.size(), i + 40); ++j) {
        EXPECT_EQ(uf.find(inserted[i]) == uf.find(inserted[j]),
                  comp[static_cast<std::size_t>(inserted[i])] ==
                      comp[static_cast<std::size_t>(inserted[j])]);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// The deterministic stages against the sequential backend
// ------------------------------------------------------------------------------------------------

TEST(LabelPropagationCuda, DeterministicStagesEqualSequential) {
  DYNG_SKIP_IF_NO_CUDA();
  const stream_t s = make_stream(5000, 5);
  for (const lp::schedule rule : {lp::schedule::damped, lp::schedule::in_place}) {
    const dyng::resources seq = dyng::resources::sequential();
    const dyng::resources cuda = dyng::resources::cuda(0);
    lp_graph gs = capacity_graph(seq, s);
    lp_graph gc = capacity_graph(cuda, s);
    const lp::options opt = published(s, 1e-5F, rule);
    auto rs = lp::compute(seq, gs, dyng::array_view<const std::int32_t>(),
                          dyng::array_view<const std::int8_t>(), opt);
    auto rc = lp::compute(cuda, gc, dyng::array_view<const std::int32_t>(),
                          dyng::array_view<const std::int8_t>(), opt);
    EXPECT_EQ(rc.space(), dyng::memory_space::device);
    for (std::size_t b = 0; b < s.batches.size(); ++b) {
      SCOPED_TRACE("batch " + std::to_string(b));
      const lp::stats a = lp::update(seq, gs, s.batches[b].view(), rs);
      const lp::stats c = lp::update(cuda, gc, s.batches[b].view(), rc);
      EXPECT_EQ(c.engine_used, dyng::engine::fused);
      EXPECT_EQ(a.components, c.components);
      EXPECT_EQ(a.sparsified_edges, c.sparsified_edges);
      EXPECT_EQ(a.initial_frontier, c.initial_frontier);
      EXPECT_EQ(a.batch.inserted_vertices, c.batch.inserted_vertices);
      EXPECT_EQ(a.batch.deleted_vertices, c.batch.deleted_vertices);
      EXPECT_EQ(a.batch.inserted_edges, c.batch.inserted_edges);
      EXPECT_EQ(a.batch.dropped_insertions, c.batch.dropped_insertions);
      EXPECT_TRUE(c.converged);
      EXPECT_FALSE(c.fallback_used);
      // The iterations are not a deterministic stage across backends: the engines' float sums
      // differ (CUDA's are the original's, bitwise; the sequential ones IEEE in vertex order).
      EXPECT_EQ(gs.num_edges(), gc.num_edges());
      EXPECT_EQ(gs.num_active_vertices(), gc.num_active_vertices());
      const auto seeds_s = rs.seed_labels();
      const auto seeds_c = rc.seed_labels();
      EXPECT_TRUE(std::equal(seeds_s.begin(), seeds_s.end(), seeds_c.begin(), seeds_c.end()));
      const std::vector<float> labels = host_labels(cuda, rc);
      for (std::size_t v = 0; v < labels.size(); ++v) {
        ASSERT_TRUE(labels[v] >= 0.0F && labels[v] <= 1.0F) << v;
        if (seeds_c[v] >= 0) {
          ASSERT_EQ(labels[v], static_cast<float>(seeds_c[v])) << v;
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// LabelPropagation-CUDA@a276a3a:tests/input_contract_test.cu through update()
// ------------------------------------------------------------------------------------------------

TEST(LabelPropagationCuda, OriginalInputContractThroughUpdate) {
  DYNG_SKIP_IF_NO_CUDA();
  const dyng::resources res = dyng::resources::cuda(0);
  using batch_t = dyng::edge_batch<std::int32_t, float>;
  const auto first = [] {
    batch_t b;
    b.insert_vertex(0, 0);
    b.insert_vertex(1, 1);
    b.insert_vertex(2);
    b.insert_edge(0, 2, {0.9F});
    b.insert_edge(1, 2, {0.8F});
    return b;
  };
  const auto scenario = [&](const std::string& name) {
    batch_t b0 = first();
    batch_t b1;
    batch_t b2;
    if (name == "dupedge") {
      b0.insert_edge(0, 2, {0.9F});
      return std::vector<batch_t>{b0};
    }
    if (name == "dupid") {
      b1.insert_vertex(2);
      return std::vector<batch_t>{b0, b1};
    }
    if (name == "dupsame") {
      b0.insert_vertex(2);
      return std::vector<batch_t>{b0};
    }
    if (name == "insdel" || name == "insdel2") {
      batch_t c;
      c.insert_vertex(0, 0);
      c.insert_vertex(1, 1);
      c.insert_vertex(2);
      if (name == "insdel2") {
        c.insert_vertex(2);
      }
      c.delete_vertex(2);
      return std::vector<batch_t>{c};
    }
    if (name == "insdelre") {
      b1.insert_vertex(2);
      b1.delete_vertex(2);
      return std::vector<batch_t>{b0, b1};
    }
    if (name == "range") {
      b1.delete_vertex(3);
      return std::vector<batch_t>{b0, b1};
    }
    b1.delete_vertex(2);
    if (name == "valid") {
      return std::vector<batch_t>{b0, b1};
    }
    b2.insert_vertex(2);
    b2.insert_edge(0, 2, {0.9F});
    b2.insert_edge(1, 2, {0.8F});
    return std::vector<batch_t>{b0, b1, b2};
  };
  for (const std::string name : {"valid", "insdel", "reinsert", "dupid", "dupsame", "range",
                                 "dupedge", "insdelre", "insdel2"}) {
    SCOPED_TRACE(name);
    const std::vector<std::int32_t> cap = {1, 1, 2};
    lp_graph g = lp_graph::with_capacity(res, 3, host_view(cap), slotted_props());
    lp::options opt;
    opt.similarity_threshold = 0.5F;
    opt.update_schedule = lp::schedule::damped;
    auto r = lp::compute(res, g, dyng::array_view<const std::int32_t>(),
                         dyng::array_view<const std::int8_t>(), opt);
    std::string error;
    std::vector<float> before;
    std::uint64_t version = 0;
    for (const auto& b : scenario(name)) {
      before = host_labels(res, r);
      version = g.version();
      try {
        (void)lp::update(res, g, b.view(), r);
      } catch (const dyng::capacity_error&) {
        error = "capacity";
        break;
      } catch (const dyng::invalid_argument_error&) {
        error = "invalid";
        break;
      }
    }
    if (name == "valid" || name == "insdel") {
      EXPECT_EQ(error, "");
      EXPECT_NO_THROW(g.check_integrity(res));
    } else if (name == "dupedge") {
      // Found on the device after the batch was applied: the graph is poisoned, the result too.
      EXPECT_EQ(error, "capacity");
      EXPECT_THROW((void)r.labels(), dyng::stale_result_error);
      EXPECT_THROW(g.check_integrity(res), dyng::internal_error);
    } else {
      // Refused by Step 0: nothing changed, the result is still usable.
      EXPECT_EQ(error, "invalid");
      EXPECT_EQ(g.version(), version);
      EXPECT_EQ(host_labels(res, r), before);
      EXPECT_NO_THROW(g.check_integrity(res));
    }
  }
}

// ------------------------------------------------------------------------------------------------
// out_of_memory_error under a memory limit, and no leak
// ------------------------------------------------------------------------------------------------

/// A device resource that fails once more than `limit` bytes would be live, and counts them.
class limited_resource {
 public:
  limited_resource(dyng::memory_resource_ref upstream, std::size_t limit)
      : upstream_(upstream), limit_(limit) {}
  void* allocate(dyng::stream_ref s, std::size_t b, std::size_t a) {
    if (live_.load() + b > limit_) {
      throw dyng::out_of_memory_error("limited_resource: over the test's limit");
    }
    void* p = upstream_.allocate(s, b, a);
    live_ += b;
    return p;
  }
  void deallocate(dyng::stream_ref s, void* p, std::size_t b, std::size_t a) noexcept {
    upstream_.deallocate(s, p, b, a);
    live_ -= b;
  }
  void* allocate_sync(std::size_t b, std::size_t a) {
    if (live_.load() + b > limit_) {
      throw dyng::out_of_memory_error("limited_resource: over the test's limit");
    }
    void* p = upstream_.allocate_sync(b, a);
    live_ += b;
    return p;
  }
  void deallocate_sync(void* p, std::size_t b, std::size_t a) noexcept {
    upstream_.deallocate_sync(p, b, a);
    live_ -= b;
  }
  [[nodiscard]] dyng::memory_space space() const noexcept {
    return upstream_.space();
  }
  [[nodiscard]] std::size_t live() const noexcept {
    return live_.load();
  }

 private:
  dyng::memory_resource_ref upstream_;
  std::size_t limit_;
  std::atomic<std::size_t> live_{0};
};

TEST(LabelPropagationCuda, OutOfMemoryUnderALimitWithoutALeak) {
  DYNG_SKIP_IF_NO_CUDA();
  const stream_t s = make_stream(5000, 5);
  for (const std::size_t limit :
       {std::size_t{4} << 10, std::size_t{200} << 10, std::size_t{600} << 10}) {
    limited_resource mr(dyng::default_device_memory_resource(0), limit);
    bool threw = false;
    {
      dyng::resources res = dyng::resources::cuda(0);
      res.set_memory_resource(mr);
      try {
        lp_graph g = capacity_graph(res, s);
        auto r = lp::compute(res, g, dyng::array_view<const std::int32_t>(),
                             dyng::array_view<const std::int8_t>(),
                             published(s, 1e-4F, lp::schedule::damped));
        r.reserve(res, s.max_batch_vertices, s.max_batch_edges);
        for (const auto& b : s.batches) {
          (void)lp::update(res, g, b.view(), r);
        }
      } catch (const dyng::out_of_memory_error&) {
        threw = true;
      }
      res.synchronize();
    }
    EXPECT_TRUE(threw) << limit << " bytes";
    EXPECT_EQ(mr.live(), 0U) << "device memory leaked under a limit of " << limit << " bytes";
  }
}

/// A device resource that fails the k-th allocation after arm(k), and counts the live bytes.
class failing_resource {
 public:
  explicit failing_resource(dyng::memory_resource_ref upstream) : upstream_(upstream) {}
  void arm(long long k) noexcept {
    countdown_ = k;
  }
  /// Disarm; return whether the armed allocation was reached.
  bool disarm() noexcept {
    const bool reached = countdown_.load() == 0;
    countdown_ = 0;
    return reached;
  }
  void* allocate(dyng::stream_ref s, std::size_t b, std::size_t a) {
    fail_now();
    void* p = upstream_.allocate(s, b, a);
    live_ += b;
    return p;
  }
  void deallocate(dyng::stream_ref s, void* p, std::size_t b, std::size_t a) noexcept {
    upstream_.deallocate(s, p, b, a);
    live_ -= b;
  }
  void* allocate_sync(std::size_t b, std::size_t a) {
    fail_now();
    void* p = upstream_.allocate_sync(b, a);
    live_ += b;
    return p;
  }
  void deallocate_sync(void* p, std::size_t b, std::size_t a) noexcept {
    upstream_.deallocate_sync(p, b, a);
    live_ -= b;
  }
  [[nodiscard]] dyng::memory_space space() const noexcept {
    return upstream_.space();
  }
  [[nodiscard]] std::size_t live() const noexcept {
    return live_.load();
  }
  [[nodiscard]] long long allocations() const noexcept {
    return allocations_.load();
  }

 private:
  void fail_now() {
    ++allocations_;
    long long left = countdown_.load();
    while (left > 0 && !countdown_.compare_exchange_weak(left, left - 1)) {
    }
    if (left == 1) {
      throw dyng::out_of_memory_error("failing_resource: the armed allocation");
    }
  }
  dyng::memory_resource_ref upstream_;
  std::atomic<long long> countdown_{0};
  std::atomic<long long> allocations_{0};
  std::atomic<std::size_t> live_{0};
};

/// After result::reserve() and graph::reserve() (the published setup), the device allocations of
/// the updates of the published 5k5 and 50k streams (the graph's vertex-batch arrays grow on
/// demand, with headroom after the first batch; M10a review): only the first vertex batches
/// allocate, as the reserve() documentation says.
TEST(LabelPropagationCuda, UpdatesAfterReserveAllocateOnlyInTheFirstBatches) {
  DYNG_SKIP_IF_NO_CUDA();
  for (const std::int64_t n : {std::int64_t{5000}, std::int64_t{50000}}) {
    const stream_t s = make_stream(n, n == 5000 ? 5 : 10);
    failing_resource mr(dyng::default_device_memory_resource(0));
    dyng::resources res = dyng::resources::cuda(0);
    res.set_memory_resource(mr);
    lp_graph g = capacity_graph(res, s);
    auto r = lp::compute(res, g, dyng::array_view<const std::int32_t>(),
                         dyng::array_view<const std::int8_t>(),
                         published(s, 1e-4F, lp::schedule::damped));
    r.reserve(res, s.max_batch_vertices, s.max_batch_edges);
    g.reserve(res, static_cast<std::int32_t>(s.max_batch_edges));
    std::vector<long long> counts;
    for (const auto& b : s.batches) {
      const long long before = mr.allocations();
      (void)lp::update(res, g, b.view(), r);
      counts.push_back(mr.allocations() - before);
    }
    std::string all;
    for (const long long c : counts) {
      all += std::to_string(c) + " ";
    }
    for (std::size_t b = 2; b < counts.size(); ++b) {
      EXPECT_EQ(counts[b], 0) << n << " vertices, batch " << b << "; per batch: " << all;
    }
    std::cout << n << " vertices: device allocations per update: " << all << "\n";
  }
}

/// Every device allocation of compute() and of each update of the 5k5 stream (after reserve())
/// fails once, in turn (M10a review): compute() raises out_of_memory_error; an update raises it
/// with the documented guarantee (before the commit: the graph's version and the labels unchanged,
/// and a retry succeeds with the labels of an update that never failed; after it: the result
/// poisoned); nothing leaks. The first update, a later larger batch and compute() each fail at
/// least once.
TEST(LabelPropagationCuda, EachDeviceAllocationFailureKeepsTheGuarantee) {
  DYNG_SKIP_IF_NO_CUDA();
  const stream_t s = make_stream(5000, 5);
  const lp::options opt = published(s, 1e-4F, lp::schedule::damped);
  failing_resource mr(dyng::default_device_memory_resource(0));
  // The labels after each batch of an update that never failed.
  std::vector<std::vector<float>> want;
  {
    const dyng::resources res = dyng::resources::cuda(0);
    lp_graph g = capacity_graph(res, s);
    auto r = lp::compute(res, g, dyng::array_view<const std::int32_t>(),
                         dyng::array_view<const std::int8_t>(), opt);
    r.reserve(res, s.max_batch_vertices, s.max_batch_edges);
    for (const auto& b : s.batches) {
      (void)lp::update(res, g, b.view(), r);
      want.push_back(host_labels(res, r));
    }
  }
  int compute_failures = 0;
  for (long long k = 1; k < 10000; ++k) {
    {
      dyng::resources res = dyng::resources::cuda(0);
      res.set_memory_resource(mr);
      lp_graph g = capacity_graph(res, s);
      mr.arm(k);
      try {
        (void)lp::compute(res, g, dyng::array_view<const std::int32_t>(),
                          dyng::array_view<const std::int8_t>(), opt);
      } catch (const dyng::out_of_memory_error&) {
        ++compute_failures;
      }
      const bool reached = mr.disarm();
      res.synchronize();
      if (!reached) {
        break;
      }
    }
    ASSERT_EQ(mr.live(), 0U) << "compute, allocation " << k;
  }
  EXPECT_GT(compute_failures, 0);
  std::vector<int> update_failures(s.batches.size(), 0);
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    for (long long k = 1; k < 10000; ++k) {
      SCOPED_TRACE("batch " + std::to_string(b) + ", allocation " + std::to_string(k));
      bool reached = false;
      {
        dyng::resources res = dyng::resources::cuda(0);
        res.set_memory_resource(mr);
        lp_graph g = capacity_graph(res, s);
        auto r = lp::compute(res, g, dyng::array_view<const std::int32_t>(),
                             dyng::array_view<const std::int8_t>(), opt);
        r.reserve(res, s.max_batch_vertices, s.max_batch_edges);
        for (std::size_t e = 0; e < b; ++e) {
          (void)lp::update(res, g, s.batches[e].view(), r);
        }
        const std::uint64_t version = g.version();
        const std::vector<float> before = host_labels(res, r);
        mr.arm(k);
        bool threw = false;
        try {
          (void)lp::update(res, g, s.batches[b].view(), r);
        } catch (const dyng::out_of_memory_error&) {
          threw = true;
        }
        reached = mr.disarm();
        if (!threw) {
          EXPECT_EQ(host_labels(res, r), want[b]);
        } else if (g.version() == version) {
          ++update_failures[b];
          EXPECT_EQ(host_labels(res, r), before);  // strong before the commit
          (void)lp::update(res, g, s.batches[b].view(), r);
          EXPECT_EQ(host_labels(res, r), want[b]);  // a retry succeeds
        } else {
          ++update_failures[b];
          EXPECT_EQ(g.version(), version + 1);
          EXPECT_THROW((void)r.labels(), dyng::stale_result_error);  // basic after the commit
        }
        res.synchronize();
      }
      ASSERT_EQ(mr.live(), 0U) << "device memory leaked";
      if (!reached) {
        break;
      }
    }
  }
  EXPECT_GT(update_failures[0], 0) << "the first vertex batch";
  int later = 0;
  for (std::size_t b = 1; b < update_failures.size(); ++b) {
    later += update_failures[b];
  }
  EXPECT_GT(later, 0) << "a later, larger batch";
}

// ------------------------------------------------------------------------------------------------
// A result between the backends
// ------------------------------------------------------------------------------------------------

TEST(LabelPropagationCuda, ResultsMoveBetweenTheBackends) {
  DYNG_SKIP_IF_NO_CUDA();
  const stream_t s = make_stream(3000, 4);
  const dyng::resources seq = dyng::resources::sequential();
  const dyng::resources cuda = dyng::resources::cuda(0);
  const lp::options opt = published(s, 1e-5F, lp::schedule::damped);
  lp_graph g = capacity_graph(cuda, s);
  auto r = lp::compute(cuda, g, dyng::array_view<const std::int32_t>(),
                       dyng::array_view<const std::int8_t>(), opt);
  (void)lp::update(cuda, g, s.batches[0].view(), r);
  (void)lp::update(cuda, g, s.batches[1].view(), r);
  // A host result is refused by CUDA resources and the other way round.
  EXPECT_THROW((void)lp::update(seq, g, s.batches[2].view(), r), dyng::invalid_argument_error);
  // CUDA -> CUDA: a deep copy that continues bitwise like the original.
  auto copy = r.clone(cuda);
  lp_graph g2 = g.clone(cuda);
  (void)lp::update(cuda, g, s.batches[2].view(), r);
  (void)lp::update(cuda, g2, s.batches[2].view(), copy);
  EXPECT_EQ(host_labels(cuda, r), host_labels(cuda, copy));
  // CUDA -> host -> CUDA keeps the labels and the run-wide tags.
  auto on_host = r.clone(seq);
  EXPECT_EQ(on_host.space(), dyng::memory_space::host);
  EXPECT_EQ(host_labels(seq, on_host), host_labels(cuda, r));
  auto back = on_host.clone(cuda);
  lp_graph g3 = g.clone(cuda);
  (void)lp::update(cuda, g, s.batches[3].view(), r);
  (void)lp::update(cuda, g3, s.batches[3].view(), back);
  EXPECT_EQ(host_labels(cuda, r), host_labels(cuda, back));
  // OpenMP resources: not a backend of label_propagation (AD19); operators on CUDA: M10b.
  lp::options ops = opt;
  ops.cuda_engine = dyng::engine::operators;
  EXPECT_THROW((void)lp::compute(cuda, g, dyng::array_view<const std::int32_t>(),
                                 dyng::array_view<const std::int8_t>(), ops),
               dyng::not_supported_error);
}

}  // namespace
