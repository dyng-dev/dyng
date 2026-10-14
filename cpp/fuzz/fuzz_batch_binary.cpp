// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fuzz_batch_binary.cpp
 * @brief libFuzzer target of dyng::io::batch_reader, the `.dgb` binary batch files
 *        (cpp/fuzz/fuzz_input.hpp).
 *
 * Options: 0 the vertex count (-1, 8, 1000). One file (the bytes after the option line). The
 * reader is instantiated for the types its header names (graph batches with int32 or int64
 * vertices and no, int32 or int64 weights; hypergraph batches with int32 vertices and no or int32
 * weights; other headers are read up to their first batch view, which must be rejected). An
 * accepted file must give consistent batches (sizes, ids in range, offsets from 0 to the member
 * count, the annotations of the flags), and batch_writer followed by batch_reader must give the
 * same batches and annotations.
 */
#include "fuzz_input.hpp"

#include <dyng/core/registry.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/io/batch_io.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace io = dyng::io;

template <typename value_t>
std::vector<value_t> copy(dyng::array_view<const value_t> v) {
  return std::vector<value_t>(v.begin(), v.end());
}

/// A graph batch as owned arrays.
template <typename vertex_t, typename weight_t>
struct edge_copy {
  std::vector<vertex_t> del_src, del_dst, ins_src, ins_dst;
  std::vector<weight_t> weights;
  int k = 0;
  bool operator==(const edge_copy& o) const {
    return del_src == o.del_src && del_dst == o.del_dst && ins_src == o.ins_src &&
           ins_dst == o.ins_dst && weights == o.weights && (k == o.k || ins_src.empty());
  }
  dyng::edge_batch_view<vertex_t, weight_t> view() const {
    dyng::edge_batch_view<vertex_t, weight_t> v;
    v.delete_src = dyng::host_view(del_src);
    v.delete_dst = dyng::host_view(del_dst);
    v.insert_src = dyng::host_view(ins_src);
    v.insert_dst = dyng::host_view(ins_dst);
    v.insert_weights = dyng::host_view(weights);
    v.num_weights = k;
    return v;
  }
};

/// A hypergraph batch and its annotations as owned arrays.
template <typename vertex_t, typename weight_t>
struct hyper_copy {
  std::vector<std::int64_t> del, offsets;
  std::vector<vertex_t> members;
  std::vector<weight_t> weights;
  std::vector<std::int64_t> del_offsets, expected;
  std::vector<vertex_t> del_members;
  bool operator==(const hyper_copy& o) const {
    const auto offs = [](const std::vector<std::int64_t>& x) {
      return x.size() <= 1 ? std::vector<std::int64_t>{} : x;  // {0} and {} mean no insertion
    };
    return del == o.del && offs(offsets) == offs(o.offsets) && members == o.members &&
           weights == o.weights && del_offsets == o.del_offsets && expected == o.expected &&
           del_members == o.del_members;
  }
};

std::int64_t vertex_limit(const io::batch_reader_options& opt, std::int64_t type_max) {
  return opt.num_vertices >= 0 ? opt.num_vertices - 1 : type_max;
}

template <typename vertex_t, typename weight_t>
void run_edges(const std::string& path, const io::batch_reader_options& opt) {
  std::vector<edge_copy<vertex_t, weight_t>> batches;
  io::batch_file_header header;
  const auto read = dyng::fuzz::read_or_reject([&] {
    io::batch_reader in(path, opt);
    header = in.header();
    while (in.next()) {
      const auto v = in.edge_batch<vertex_t, weight_t>();
      edge_copy<vertex_t, weight_t> c{copy(v.delete_src),     copy(v.delete_dst),
                                      copy(v.insert_src),     copy(v.insert_dst),
                                      copy(v.insert_weights), v.num_weights};
      batches.push_back(std::move(c));
    }
    return true;
  });
  if (!read) {
    return;  // rejected with a documented exception
  }
  const std::int64_t limit =
      vertex_limit(opt, static_cast<std::int64_t>(std::numeric_limits<vertex_t>::max()));
  DYNG_FUZZ_CHECK(header.kind == dyng::container_kind::graph);
  DYNG_FUZZ_CHECK(static_cast<std::int64_t>(batches.size()) == header.num_batches);
  for (const auto& b : batches) {
    DYNG_FUZZ_CHECK(b.del_src.size() == b.del_dst.size());
    DYNG_FUZZ_CHECK(b.ins_src.size() == b.ins_dst.size());
    DYNG_FUZZ_CHECK(b.weights.size() == b.ins_src.size() * static_cast<std::size_t>(b.k));
    for (const auto* a : {&b.del_src, &b.del_dst, &b.ins_src, &b.ins_dst}) {
      for (const vertex_t v : *a) DYNG_FUZZ_CHECK(v >= 0 && v <= limit);
    }
  }
  dyng::fuzz::round_trip([&] {
    // write -> read
    const std::string out = dyng::fuzz::files().path("copy.dgb");
    {
      io::batch_writer w(out, header);
      for (const auto& b : batches) w.write(b.view());
      w.close();
    }
    dyng::fuzz::written(out);
    io::batch_reader back(out, opt);
    std::size_t i = 0;
    while (back.next()) {
      const auto v = back.edge_batch<vertex_t, weight_t>();
      edge_copy<vertex_t, weight_t> c{copy(v.delete_src),     copy(v.delete_dst),
                                      copy(v.insert_src),     copy(v.insert_dst),
                                      copy(v.insert_weights), v.num_weights};
      DYNG_FUZZ_CHECK(i < batches.size() && c == batches[i]);
      ++i;
    }
    DYNG_FUZZ_CHECK(i == batches.size());
  });
}

template <typename vertex_t, typename weight_t>
hyper_copy<vertex_t, weight_t> take(const io::batch_reader& in) {
  const auto v = in.hyperedge_batch<vertex_t, weight_t>();
  const auto a = in.annotations<vertex_t>();
  return {copy(v.delete_hyperedges), copy(v.insert_offsets),  copy(v.insert_members),
          copy(v.insert_weights),    copy(a.deleted_offsets), copy(a.expected_ids),
          copy(a.deleted_members)};
}

template <typename vertex_t, typename weight_t>
void run_hyperedges(const std::string& path, const io::batch_reader_options& opt) {
  std::vector<hyper_copy<vertex_t, weight_t>> batches;
  io::batch_file_header header;
  const auto read = dyng::fuzz::read_or_reject([&] {
    io::batch_reader in(path, opt);
    header = in.header();
    while (in.next()) batches.push_back(take<vertex_t, weight_t>(in));
    return true;
  });
  if (!read) {
    return;  // rejected with a documented exception
  }
  const std::int64_t limit =
      vertex_limit(opt, static_cast<std::int64_t>(std::numeric_limits<vertex_t>::max()));
  DYNG_FUZZ_CHECK(header.kind == dyng::container_kind::hypergraph);
  DYNG_FUZZ_CHECK(static_cast<std::int64_t>(batches.size()) == header.num_batches);
  const bool deleted = (header.flags & io::batch_file_flags::deleted_members) != 0;
  const bool expected = (header.flags & io::batch_file_flags::expected_ids) != 0;
  for (const auto& b : batches) {
    const std::size_t ni = b.offsets.empty() ? 0 : b.offsets.size() - 1;
    if (ni > 0) {
      DYNG_FUZZ_CHECK(b.offsets.front() == 0);
      DYNG_FUZZ_CHECK(b.offsets.back() == static_cast<std::int64_t>(b.members.size()));
      for (std::size_t i = 0; i < ni; ++i) DYNG_FUZZ_CHECK(b.offsets[i] < b.offsets[i + 1]);
    } else {
      DYNG_FUZZ_CHECK(b.members.empty());
    }
    for (const vertex_t v : b.members) DYNG_FUZZ_CHECK(v >= 0 && v <= limit);
    for (const std::int64_t id : b.del) DYNG_FUZZ_CHECK(id >= 0);
    DYNG_FUZZ_CHECK(dyng::is_unweighted_v<weight_t> ? b.weights.empty() : b.weights.size() == ni);
    if constexpr (!dyng::is_unweighted_v<weight_t>) {
      for (const weight_t w : b.weights) DYNG_FUZZ_CHECK(w >= 1);
    }
    DYNG_FUZZ_CHECK(deleted ? b.del_offsets.size() == b.del.size() + 1 : b.del_offsets.empty());
    DYNG_FUZZ_CHECK(expected ? b.expected.size() == ni : b.expected.empty());
    for (const vertex_t v : b.del_members) DYNG_FUZZ_CHECK(v >= 0 && v <= limit);
  }
  dyng::fuzz::round_trip([&] {
    // write -> read
    const std::string out = dyng::fuzz::files().path("copy.dgb");
    {
      io::batch_writer w(out, header);
      for (const auto& b : batches) {
        dyng::hyperedge_batch_view<vertex_t, weight_t> v;
        v.delete_hyperedges = dyng::host_view(b.del);
        v.insert_offsets = dyng::host_view(b.offsets);
        v.insert_members = dyng::host_view(b.members);
        v.insert_weights = dyng::host_view(b.weights);
        io::hyperedge_batch_annotations<vertex_t> a;
        a.deleted_offsets = dyng::host_view(b.del_offsets);
        a.deleted_members = dyng::host_view(b.del_members);
        a.expected_ids = dyng::host_view(b.expected);
        w.write(v, a);
      }
      w.close();
    }
    dyng::fuzz::written(out);
    io::batch_reader back(out, opt);
    std::size_t i = 0;
    while (back.next()) {
      DYNG_FUZZ_CHECK(i < batches.size() && (take<vertex_t, weight_t>(back) == batches[i]));
      ++i;
    }
    DYNG_FUZZ_CHECK(i == batches.size());
  });
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const dyng::fuzz::fuzz_input in(data, size, 1);
  const std::string path = dyng::fuzz::files().write("input.dgb", in.file(0));
  constexpr std::int64_t counts[] = {-1, 8, 1000};
  io::batch_reader_options opt;
  opt.num_vertices = counts[in.option(0, 3)];
  // The header picks the types (a header the reader rejects is rejected here too).
  const auto header = dyng::fuzz::read_or_reject([&] { return io::batch_reader(path).header(); });
  if (!header) {
    return 0;
  }
  using dt = io::batch_dtype;
  const bool graph = header->kind == dyng::container_kind::graph;
  const dt v = header->vertex_type;
  const dt w = header->weight_type;
  if (graph && v == dt::int32 && w == dt::none)
    run_edges<std::int32_t, dyng::unweighted>(path, opt);
  if (graph && v == dt::int32 && w == dt::int32) run_edges<std::int32_t, std::int32_t>(path, opt);
  if (graph && v == dt::int64 && w == dt::none)
    run_edges<std::int64_t, dyng::unweighted>(path, opt);
  if (graph && v == dt::int64 && w == dt::int32) run_edges<std::int64_t, std::int32_t>(path, opt);
  if (!graph && v == dt::int32 && w == dt::none) {
    run_hyperedges<std::int32_t, dyng::unweighted>(path, opt);
  }
  if (!graph && v == dt::int32 && w == dt::int32)
    run_hyperedges<std::int32_t, std::int32_t>(path, opt);
  return 0;
}
