// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file label_propagation_operators_cuda_test.cu
 * @brief label_propagation's operators (operators/; ADR 0102) on cuda_exec against the
 *        sequential model on sequential_exec: neighbor_reduce's averages bit for bit on every
 *        mapping (this test's translation unit has no fast math, so the division is IEEE's on both
 *        sides), the hub degrees 8, 9, 32, 33, 1024, 1025, 3000 and 6000 included; the device
 *        pushes of binned_frontier (warp-aggregated, concurrent, with repeats) and expand on every
 *        lane count give the host's sets per bin and half; connected_components' Shiloach-Vishkin
 *        gives the host's canonical ids with one host synchronization per round and one for the
 *        count.
 */
#include "algorithms/label_propagation/operators/binned_frontier.cuh"
#include "algorithms/label_propagation/operators/connected_components.cuh"
#include "algorithms/label_propagation/operators/neighbor_reduce.cuh"
#include "core/budget_counters.hpp"
#include "label_propagation_operators_cases.hpp"
#include "operators/cuda_execution.cuh"
#include "support/gtest_helpers.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/resources.hpp>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <set>
#include <tuple>
#include <vector>

namespace {

namespace ops = dyng::detail::operators;
namespace lpo = dyng::detail::lp_ops;
namespace cases = dyng::test::lp_operators;

constexpr int bins = 3;

template <typename value_t>
dyng::buffer<value_t> on_device(const dyng::resources& res, const std::vector<value_t>& v) {
  dyng::buffer<value_t> b(res, std::max<std::size_t>(v.size(), 1));
  if (!v.empty()) {
    dyng::copy(
        res, dyng::host_view(v),
        dyng::array_view<value_t>(b.data(), v.size(), dyng::memory_space::device, res.device()));
  }
  return b;
}

template <typename value_t>
std::vector<value_t> on_host(const dyng::resources& res, const value_t* d, std::size_t n) {
  std::vector<value_t> out(n);
  if (n > 0) {
    dyng::copy(res, dyng::array_view<const value_t>(d, n, dyng::memory_space::device, res.device()),
               dyng::host_view(out));
  }
  return out;
}

/// The rows on the device.
struct device_rows {
  dyng::buffer<std::int32_t> row_start, row_end, col_ind, status;
  dyng::buffer<float> weights, x;
  device_rows(const dyng::resources& res, const cases::host_rows& g)
      : row_start(on_device(res, g.row_start)),
        row_end(on_device(res, g.row_end)),
        col_ind(on_device(res, g.col_ind)),
        status(on_device(res, g.status)),
        weights(on_device(res, g.weights)),
        x(on_device(res, g.x)) {}
  lpo::slot_rows<std::int32_t> view() const {
    lpo::slot_rows<std::int32_t> r;
    r.row_start = row_start.data();
    r.row_end = row_end.data();
    r.col_ind = col_ind.data();
    r.weights = weights.data();
    r.status = status.data();
    return r;
  }
};

lpo::slot_rows<std::int32_t> host_view_of(const cases::host_rows& g) {
  lpo::slot_rows<std::int32_t> r;
  r.row_start = g.row_start.data();
  r.row_end = g.row_end.data();
  r.col_ind = g.col_ind.data();
  r.weights = g.weights.data();
  r.status = g.status.data();
  return r;
}

/// Stores each average at its position (device and host).
struct store_average {
  float* out;
  std::int32_t* who;
  __host__ __device__ void operator()(int i, std::int32_t u, float average) const {
    out[i] = average;
    who[i] = u;
  }
};

constexpr lpo::reduce_mapping mappings[] = {lpo::reduce_mapping::lanes_8, lpo::reduce_mapping::warp,
                                            lpo::reduce_mapping::block_256,
                                            lpo::reduce_mapping::block_1024};

TEST(LabelPropagationOperatorsCuda, NeighborReduceIsTheSequentialModelBitForBit) {
  DYNG_SKIP_IF_NO_CUDA();
  const dyng::resources cuda = dyng::resources::cuda(0);
  const dyng::resources seq = dyng::resources::sequential();
  const cases::host_rows g = cases::make_rows(6000, 11);
  const std::vector<std::int32_t> list = cases::all_vertices(g);
  const device_rows d(cuda, g);
  const dyng::buffer<std::int32_t> d_list = on_device(cuda, list);
  const auto count = static_cast<int>(list.size());
  for (const lpo::reduce_mapping m : mappings) {
    SCOPED_TRACE(lpo::threads_of(m));
    std::vector<float> host(list.size());
    std::vector<std::int32_t> host_who(list.size());
    lpo::neighbor_reduce<ops::sequential_exec>(seq, m, host_view_of(g), g.x.data(), list.data(),
                                               count, store_average{host.data(), host_who.data()});
    dyng::buffer<float> out(cuda, list.size());
    dyng::buffer<std::int32_t> who(cuda, list.size());
    lpo::neighbor_reduce<ops::cuda_exec>(cuda, m, d.view(), d.x.data(), d_list.data(), count,
                                         store_average{out.data(), who.data()});
    const std::vector<float> device = on_host(cuda, out.data(), list.size());
    EXPECT_EQ(on_host(cuda, who.data(), list.size()), list);
    EXPECT_EQ(host_who, list);
    for (std::size_t i = 0; i < list.size(); ++i) {
      EXPECT_EQ(0, std::memcmp(&host[i], &device[i], sizeof(float)))
          << "vertex " << list[i] << " (degree " << g.row_end[list[i]] - g.row_start[list[i]]
          << "): host " << host[i] << ", device " << device[i];
    }
  }
}

/// The per-(bin, half) sets of one set of a frontier, read from the host mirror after read_size.
using bin_sets = std::vector<std::multiset<std::int32_t>>;

struct frontier_memory {
  std::vector<std::vector<std::int32_t>> host_queue;  // per bin (host test side)
  std::vector<std::int32_t> host_counts;
  int capacity[bins];
  explicit frontier_memory(int n) : host_counts(2 * bins, 0) {
    for (int b = 0; b < bins; ++b) {
      capacity[b] = n;
      host_queue.emplace_back(static_cast<std::size_t>(n), -1);
    }
  }
};

bin_sets sets_of(const std::vector<std::vector<std::int32_t>>& queue, const std::int32_t* counts,
                 const int* capacity) {
  bin_sets out(2 * bins);
  for (int b = 0; b < bins; ++b) {
    for (int half = 0; half < 2; ++half) {
      const int count = counts[b * 2 + half];
      const int offset = lpo::half_offset(half, capacity[b], count);
      out[static_cast<std::size_t>(b * 2 + half)].insert(
          queue[static_cast<std::size_t>(b)].begin() + offset,
          queue[static_cast<std::size_t>(b)].begin() + offset + count);
    }
  }
  return out;
}

/// Pushes vertex i % n (every vertex three times) with its degree.
struct push_all {
  lpo::binned_push<std::int32_t, bins> out;
  lpo::slot_rows<std::int32_t> rows;
  std::int32_t n;
  __host__ __device__ void operator()(std::int64_t i) const {
    const auto v = static_cast<std::int32_t>(i % n);
    out.push(v, rows.degree(v));
  }
};

/// A neighbour is eligible if active and its id is not a multiple of 5.
struct eligible_test {
  const std::int32_t* status;
  __host__ __device__ bool operator()(std::int32_t v) const {
    return status[v] == 0 && v % 5 != 0;
  }
};

TEST(LabelPropagationOperatorsCuda, BinnedFrontierAndExpandBuildTheHostsSets) {
  DYNG_SKIP_IF_NO_CUDA();
  const dyng::resources cuda = dyng::resources::cuda(0);
  const dyng::resources seq = dyng::resources::sequential();
  const cases::host_rows g = cases::make_rows(9000, 12);
  const std::vector<std::int32_t> list = cases::all_vertices(g);
  const device_rows d(cuda, g);
  const dyng::buffer<std::int32_t> d_list = on_device(cuda, list);
  std::vector<std::uint8_t> changed(list.size());
  for (std::size_t i = 0; i < list.size(); ++i) {
    changed[i] = (i < g.hubs.size() || i % 4 == 1) ? 1 : 0;
  }
  const dyng::buffer<std::uint8_t> d_changed = on_device(cuda, changed);
  const auto make_push = [&](std::int32_t* const* queue, std::int32_t* counts,
                             std::int32_t* stamp) {
    lpo::binned_push<std::int32_t, bins> p{};
    for (int b = 0; b < bins; ++b) {
      p.queue[b] = queue[b];
      p.capacity[b] = g.n;
    }
    p.counts = counts;
    p.stamp = stamp;
    p.tag = 3;
    p.bins.max_degree[0] = 32;
    p.bins.max_degree[1] = 1024;
    return p;
  };
  // The host's sets.
  frontier_memory hm(g.n);
  std::vector<std::int32_t> host_stamp(static_cast<std::size_t>(g.n), 0);
  std::int32_t* hq[bins] = {hm.host_queue[0].data(), hm.host_queue[1].data(),
                            hm.host_queue[2].data()};
  const auto host_push = make_push(hq, hm.host_counts.data(), host_stamp.data());
  const ops::sequential_exec host_exec(seq);
  host_exec.for_each(3 * static_cast<std::int64_t>(g.n), push_all{host_push, host_view_of(g), g.n});
  const bin_sets want_push = sets_of(hm.host_queue, hm.host_counts.data(), hm.capacity);
  // The same pushes on the device, concurrently.
  for (const int lanes : {0, 8, 32, 256, 1024}) {  // 0: the pushes; else expand with `lanes`
    SCOPED_TRACE(lanes);
    std::vector<dyng::buffer<std::int32_t>> dq;
    for (int b = 0; b < bins; ++b) {
      dq.emplace_back(cuda, static_cast<std::size_t>(g.n));
    }
    dyng::buffer<std::int32_t> counts(cuda, 2 * bins);
    dyng::buffer<std::int32_t> stamp(cuda, static_cast<std::size_t>(g.n));
    const ops::cuda_exec exec(cuda);
    exec.fill(counts.data(), 2 * bins, std::int32_t{0});
    exec.fill(stamp.data(), g.n, std::int32_t{0});
    std::int32_t* dqp[bins] = {dq[0].data(), dq[1].data(), dq[2].data()};
    const auto dev_push = make_push(dqp, counts.data(), stamp.data());
    bin_sets want = want_push;
    if (lanes == 0) {
      exec.for_each(3 * static_cast<std::int64_t>(g.n), push_all{dev_push, d.view(), g.n});
    } else {
      frontier_memory em(g.n);
      std::vector<std::int32_t> es(static_cast<std::size_t>(g.n), 0);
      std::int32_t* eq[bins] = {em.host_queue[0].data(), em.host_queue[1].data(),
                                em.host_queue[2].data()};
      lpo::expand<ops::sequential_exec>(
          seq, lanes, host_view_of(g), list.data(), changed.data(), static_cast<int>(list.size()),
          eligible_test{g.status.data()}, make_push(eq, em.host_counts.data(), es.data()));
      want = sets_of(em.host_queue, em.host_counts.data(), em.capacity);
      lpo::expand<ops::cuda_exec>(cuda, lanes, d.view(), d_list.data(), d_changed.data(),
                                  static_cast<int>(list.size()), eligible_test{d.status.data()},
                                  dev_push);
    }
    const std::vector<std::int32_t> got_counts = on_host(cuda, counts.data(), 2 * bins);
    std::vector<std::vector<std::int32_t>> got_queue;
    for (int b = 0; b < bins; ++b) {
      got_queue.push_back(
          on_host(cuda, dq[static_cast<std::size_t>(b)].data(), static_cast<std::size_t>(g.n)));
    }
    const bin_sets got = sets_of(got_queue, got_counts.data(), hm.capacity);
    for (std::size_t k = 0; k < got.size(); ++k) {
      EXPECT_EQ(got[k], want[k]) << "bin " << k / 2 << ", half " << k % 2;
    }
  }
}

TEST(LabelPropagationOperatorsCuda, ConnectedComponentsGiveTheHostsIds) {
  DYNG_SKIP_IF_NO_CUDA();
  const dyng::resources cuda = dyng::resources::cuda(0);
  const dyng::resources seq = dyng::resources::sequential();
  for (const auto& [n, listed, edges] :
       {std::tuple{50, 30, 0}, std::tuple{2000, 1500, 600}, std::tuple{200000, 120000, 110000}}) {
    SCOPED_TRACE(edges);
    const cases::edge_case c = cases::make_edges(n, listed, edges, 91 + edges);
    // The host.
    std::vector<std::int32_t> parent(static_cast<std::size_t>(n));
    std::vector<std::int32_t> root_id(static_cast<std::size_t>(n));
    std::vector<std::int32_t> want(static_cast<std::size_t>(n), -1);
    lpo::components_scratch hs;
    hs.parent = parent.data();
    hs.root_id = root_id.data();
    const std::int32_t num_edges = edges;
    const lpo::components_result hr = lpo::connected_components<ops::sequential_exec>(
        seq, c.src.data(), c.dst.data(), &num_edges, edges, c.vertices.data(), listed, hs,
        want.data());
    // The device.
    const auto d_src = on_device(cuda, c.src);
    const auto d_dst = on_device(cuda, c.dst);
    const auto d_vertices = on_device(cuda, c.vertices);
    const auto d_count = on_device(cuda, std::vector<std::int32_t>{edges});
    dyng::buffer<std::int32_t> d_parent(cuda, static_cast<std::size_t>(n));
    dyng::buffer<std::int32_t> d_root_id(cuda, static_cast<std::size_t>(n));
    dyng::buffer<std::int32_t> d_flag(cuda, static_cast<std::size_t>(listed));
    dyng::buffer<std::int32_t> d_scan(cuda, static_cast<std::size_t>(listed));
    dyng::buffer<std::int32_t> d_words(cuda, 2);
    const std::size_t temp_bytes = lpo::components_scan_temp_bytes(listed);
    dyng::buffer<unsigned char> d_temp(cuda, temp_bytes);
    auto d_component = on_device(cuda, std::vector<std::int32_t>(static_cast<std::size_t>(n), -1));
    lpo::components_scratch ds;
    ds.parent = d_parent.data();
    ds.root_id = d_root_id.data();
    ds.root_flag = d_flag.data();
    ds.root_scan = d_scan.data();
    ds.scan_temp = d_temp.data();
    ds.scan_temp_bytes = temp_bytes;
    ds.words = d_words.data();
    const std::int64_t syncs = dyng::detail::budget_snapshot().host_syncs;
    const lpo::components_result dr = lpo::connected_components<ops::cuda_exec>(
        cuda, d_src.data(), d_dst.data(), d_count.data(), edges, d_vertices.data(), listed, ds,
        d_component.data());
    if (dyng::detail::budgets_enabled()) {
      EXPECT_EQ(dyng::detail::budget_snapshot().host_syncs - syncs, dr.rounds + 1)
          << "one synchronization per round and one for the count";
    }
    EXPECT_EQ(on_host(cuda, d_component.data(), static_cast<std::size_t>(n)), want);
    EXPECT_EQ(dr.count, hr.count);
    EXPECT_EQ(dr.rounds > 0, edges > 0);
  }
}

}  // namespace
