// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file batch_binary_cuda_test.cpp
 * @brief batch_reader with the resources of the cuda backend: every record is staged through the
 *        pinned buffers and copied to the device on the resources' stream; the device views equal
 *        the host reader's, and a hypergraph applies them directly.
 */
#include "support/data_paths.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/io/batch_io.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

namespace io = dyng::io;
using hbatch = dyng::hyperedge_batch<std::int32_t, std::int32_t>;

template <typename value_t>
std::vector<value_t> host(const dyng::resources& res, dyng::array_view<const value_t> v) {
  return dyng::to_vector(res, v);
}

TEST(BatchReaderCuda, StagesEveryRecordToTheDevice) {
  const dyng::resources res = dyng::resources::cuda(0);
  dyng::test::temp_dir tmp;
  const std::string path = tmp.path("stream.dgb");
  // Batches of growing, then shrinking sizes (the pinned and device buffers grow and are reused).
  std::mt19937_64 rng(5);
  std::vector<hbatch> batches;
  std::vector<std::vector<std::int64_t>> expected;
  for (const int size : {3, 40, 2000, 7, 0, 600, 1}) {
    hbatch b;
    for (int i = 0; i < size; ++i) {
      std::vector<std::int32_t> members;
      const int card = 1 + static_cast<int>(rng() % 6);
      for (int j = 0; j < card; ++j) members.push_back(static_cast<std::int32_t>(rng() % 50));
      b.insert_hyperedge(dyng::host_view(members), 1 + static_cast<std::int32_t>(rng() % 9));
    }
    for (int i = 0; i < size / 2; ++i) b.delete_hyperedge(static_cast<std::int64_t>(rng() % 100));
    batches.push_back(std::move(b));
  }
  io::batch_writer w(path, io::hyperedge_batch_file_header<std::int32_t, std::int32_t>(
                               io::batch_file_flags::expected_ids));
  for (std::size_t k = 0; k < batches.size(); ++k) {
    std::vector<std::int64_t> ids(batches[k].num_insertions());
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<std::int64_t>(k * 10000 + i);
    io::hyperedge_batch_annotations<std::int32_t> a;
    a.expected_ids = dyng::host_view(ids);
    w.write(batches[k].view(), a);
    expected.push_back(ids);
  }
  w.close();

  io::batch_reader on_device(res, path);
  io::batch_reader on_host(path);
  std::size_t k = 0;
  while (on_device.next()) {
    ASSERT_TRUE(on_host.next());
    const auto d = on_device.hyperedge_batch<std::int32_t, std::int32_t>();
    const auto h = on_host.hyperedge_batch<std::int32_t, std::int32_t>();
    if (!d.insert_members.empty()) {
      EXPECT_EQ(d.insert_members.space(), dyng::memory_space::device);
    }
    EXPECT_EQ(host(res, d.insert_members),
              std::vector<std::int32_t>(h.insert_members.begin(), h.insert_members.end()));
    EXPECT_EQ(host(res, d.insert_offsets),
              std::vector<std::int64_t>(h.insert_offsets.begin(), h.insert_offsets.end()));
    EXPECT_EQ(host(res, d.insert_weights),
              std::vector<std::int32_t>(h.insert_weights.begin(), h.insert_weights.end()));
    EXPECT_EQ(host(res, d.delete_hyperedges), batches[k].delete_hyperedges());
    EXPECT_EQ(host(res, on_device.annotations<std::int32_t>().expected_ids), expected[k]);
    ++k;
  }
  EXPECT_EQ(k, batches.size());
  EXPECT_FALSE(on_host.next());
}

TEST(BatchReaderCuda, AHypergraphAppliesTheDeviceViews) {
  const dyng::resources res = dyng::resources::cuda(0);
  dyng::test::temp_dir tmp;
  dyng::hyperedge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = 10;
  list.add_hyperedge({0, 1, 2}, 4);
  list.add_hyperedge({3, 4}, 2);
  list.add_hyperedge({5}, 1);
  hbatch a;
  a.delete_hyperedge(1);
  a.insert_hyperedge({7, 8}, 3);
  hbatch b;
  b.delete_hyperedge(0);
  b.insert_hyperedge({9, 9, 1}, 5);
  const std::string path = tmp.path("apply.dgb");
  io::write_binary_hyperedge_batches<std::int32_t, std::int32_t>(path, {a.view(), b.view()});
  auto on_device = dyng::hypergraph<std::int32_t, std::int32_t>::from_hyperedges(res, list.view());
  auto on_host = dyng::hypergraph<std::int32_t, std::int32_t>::from_hyperedges(
      dyng::resources::sequential(), list.view());
  const auto host_batches = io::read_binary_hyperedge_batches<std::int32_t, std::int32_t>(path);
  io::batch_reader in(res, path);
  std::size_t k = 0;
  while (in.next()) {
    const auto s = on_device.apply(res, in.hyperedge_batch<std::int32_t, std::int32_t>());
    const auto t = on_host.apply(dyng::resources::sequential(), host_batches.at(k++).view());
    EXPECT_EQ(s.inserted_ids, t.inserted_ids);
    EXPECT_EQ(s.dropped_members, t.dropped_members);
    EXPECT_EQ(s.ignored_deletions, t.ignored_deletions);
  }
  EXPECT_EQ(k, 2u);
  const auto d = on_device.to_hyperedge_list(res);
  const auto h = on_host.to_hyperedge_list(dyng::resources::sequential());
  EXPECT_EQ(d.offsets, h.offsets);
  EXPECT_EQ(d.members, h.members);
  EXPECT_EQ(d.weights, h.weights);
}

}  // namespace
