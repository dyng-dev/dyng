// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file vertex_batch_io_test.cpp
 * @brief The I/O of 0.4's label propagation (AD13, decided by the author on 2026-10-28; ADR
 *        0105): io::write_labels / read_labels in LabelPropagation-CUDA's bytes, float edge
 *        lists, the `.dgt` vertex operations and float tokens, the `.dgb` weight code 3 (float32)
 *        and flag bit 4 (vertex_records) with their per-kind checks, and the round trip of the
 *        legacy generator's batches through both formats.
 */
#include "support/data_paths.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/registry.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/generators/vertex_arrival.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/edge_list_io.hpp>
#include <dyng/io/result_io.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using dyng::host_view;
using dyng::test::read_text;
using dyng::test::temp_dir;
using dyng::test::write_text;
namespace io = dyng::io;
using fbatch = dyng::edge_batch<std::int32_t, float>;
using fview = dyng::edge_batch_view<std::int32_t, float>;

std::uint64_t u64_at(const std::string& s, std::size_t at) {
  std::uint64_t v = 0;
  std::memcpy(&v, s.data() + at, 8);
  return v;
}

void set_u64(std::string& s, std::size_t at, std::uint64_t v) {
  std::memcpy(s.data() + at, &v, 8);
}

// CRC-32 (IEEE), bitwise: reseals a changed header so that the field's own check fires.
std::uint32_t crc32_bitwise(const unsigned char* p, std::size_t n) {
  std::uint32_t c = 0xffffffffU;
  for (std::size_t i = 0; i < n; ++i) {
    c ^= p[i];
    for (int k = 0; k < 8; ++k) c = (c & 1U) != 0 ? 0xEDB88320U ^ (c >> 1U) : c >> 1U;
  }
  return ~c;
}

void reseal_header(std::string& s) {
  const std::uint32_t crc = crc32_bitwise(reinterpret_cast<const unsigned char*>(s.data()), 60);
  std::memcpy(s.data() + 60, &crc, 4);
}

template <typename body_t>
std::string io_error_text(const body_t& body) {
  try {
    body();
  } catch (const dyng::io_error& e) {
    return e.what();
  }
  ADD_FAILURE() << "expected dyng::io_error";
  return {};
}

/// The bytes printf("%.6f") gives, the reference of O's TextWriter::putFixed6 (the same bytes as
/// iostream's fixed << setprecision(6) of the double).
std::string printf6(float x) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.6f", static_cast<double>(x));
  return buf;
}

/// Two vertex batches of a float graph: arrivals with and without labels, edges, departures.
std::vector<fbatch> sample_vertex_batches() {
  std::vector<fbatch> out;
  fbatch b0(1);
  b0.insert_vertex(0, 1);
  b0.insert_vertex(1, 0);
  b0.insert_vertex(2);
  b0.insert_edge(0, 1, {0.75F});
  b0.insert_edge(1, 2, {0.1F});
  out.push_back(std::move(b0));
  fbatch b1(1);
  b1.insert_vertex(3);
  b1.insert_vertex(4, -1);
  b1.insert_edge(3, 0, {1.0F / 3.0F});
  b1.insert_edge(4, 2, {1e-7F});
  b1.delete_vertex(2);
  b1.delete_vertex(7);
  out.push_back(std::move(b1));
  fbatch b2(1);  // departures only
  b2.delete_vertex(1);
  out.push_back(std::move(b2));
  return out;
}

std::vector<fview> views_of(const std::vector<fbatch>& batches) {
  std::vector<fview> v;
  for (const auto& b : batches) v.push_back(b.view());
  return v;
}

template <typename batch_a, typename batch_b>
void expect_same_batch(const batch_a& a, const batch_b& b) {
  EXPECT_EQ(a.insert_src(), b.insert_src());
  EXPECT_EQ(a.insert_dst(), b.insert_dst());
  ASSERT_EQ(a.insert_weights().size(), b.insert_weights().size());
  for (std::size_t i = 0; i < a.insert_weights().size(); ++i) {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::memcpy(&x, &a.insert_weights()[i], 4);
    std::memcpy(&y, &b.insert_weights()[i], 4);
    EXPECT_EQ(x, y) << "weight " << i;  // the float bits
  }
  EXPECT_EQ(a.delete_src(), b.delete_src());
  EXPECT_EQ(a.insert_vertices(), b.insert_vertices());
  EXPECT_EQ(a.insert_vertex_labels(), b.insert_vertex_labels());
  EXPECT_EQ(a.delete_vertices(), b.delete_vertices());
}

// ------------------------------------------------------------------------- write_labels -------

TEST(Labels, WriteLabelsIsPrintfFixed6ByteForByte) {
  temp_dir dir;
  // 0, 1, 0.5, the exact 6-decimal ties k/128 (0.0078125 ...: round half to even), values
  // just around the ties, tiny and negative values, and random labels in [0, 1].
  std::vector<float> labels = {0.0F,       1.0F,         0.5F,       1.0F / 128, 3.0F / 128,
                               5.0F / 128, 127.0F / 128, 0.0000005F, 0.9999995F, 0.25F,
                               -0.0F,      1e-30F,       0.1F,       0.7F};
  for (int k = 1; k < 128; k += 2) {
    const float tie = static_cast<float>(k) / 128.0F;
    labels.push_back(tie);
    labels.push_back(std::nextafter(tie, 0.0F));
    labels.push_back(std::nextafter(tie, 1.0F));
  }
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> unit(0.0F, 1.0F);
  for (int i = 0; i < 2000; ++i) labels.push_back(unit(rng));
  io::write_labels(dir.path("labels.txt"), host_view(labels));
  std::string expected;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    expected += std::to_string(i) + " " + printf6(labels[i]) + "\n";
  }
  EXPECT_EQ(read_text(dir.path("labels.txt")), expected);
  EXPECT_EQ(read_text(dir.path("labels.txt")).substr(0, 43),
            "0 0.000000\n1 1.000000\n2 0.500000\n3 0.007812");
  // Read back: within half a unit of the 6th decimal, then rounded to a float.
  const auto back =
      io::read_labels(dir.path("labels.txt"), static_cast<std::int64_t>(labels.size()));
  ASSERT_EQ(back.size(), labels.size());
  for (std::size_t i = 0; i < labels.size(); ++i) {
    EXPECT_NEAR(back[i], labels[i], 5.3e-7) << i;  // 5e-7, then half a float ulp below 1
  }
  io::write_labels(dir.path("empty.txt"), dyng::array_view<const float>());
  EXPECT_EQ(read_text(dir.path("empty.txt")), "");
  EXPECT_TRUE(io::read_labels(dir.path("empty.txt"), 0).empty());
}

TEST(Labels, ReadLabelsRejectsMalformedFiles) {
  temp_dir dir;
  const auto bad = [&](const std::string& text, std::int64_t n) {
    write_text(dir.path("bad.txt"), text);
    return io_error_text([&] { (void)io::read_labels(dir.path("bad.txt"), n); });
  };
  EXPECT_NE(bad("0 0.5\n0 0.25\n", 2).find("listed twice"), std::string::npos);
  EXPECT_NE(bad("0 0.5\n", 2).find("exactly once"), std::string::npos);
  EXPECT_NE(bad("0 0.5\n2 0.1\n", 2).find("out of range"), std::string::npos);
  EXPECT_NE(bad("0 x\n", 1).find("not a decimal number"), std::string::npos);
  EXPECT_NE(bad("0 1e999\n", 1).find("not a decimal number"), std::string::npos);
  EXPECT_NE(bad("0 0.5 1\n", 1).find("exactly 2"), std::string::npos);
  EXPECT_THROW((void)io::read_labels(dir.path("missing.txt"), 1), dyng::io_error);
  EXPECT_THROW((void)io::read_labels(dir.path("bad.txt"), -1), dyng::invalid_argument_error);
  // Out of order, blank lines and "nan" (what write_labels writes for a NaN label) read.
  write_text(dir.path("ok.txt"), "1 0.25\n\n0 nan\n");
  const auto ok = io::read_labels(dir.path("ok.txt"), 2);
  EXPECT_TRUE(std::isnan(ok[0]));
  EXPECT_EQ(ok[1], 0.25F);
}

// ---------------------------------------------------------------------- float edge lists -------

TEST(FloatEdgeLists, ReadWriteRoundTripsTheFloatBits) {
  temp_dir dir;
  dyng::edge_list<std::int32_t, float> e;
  e.num_vertices = 4;
  e.num_weights = 1;
  e.src = {0, 1, 2, 3};
  e.dst = {1, 2, 3, 0};
  e.weights = {0.1F, 1.0F / 3.0F, 1e-7F, 0.999999940F};
  io::write_edge_list(dir.path("f.txt"), e.view());
  EXPECT_EQ(read_text(dir.path("f.txt")), "0 1 0.1\n1 2 0.33333334\n2 3 1e-07\n3 0 0.99999994\n");
  io::edge_list_options o;
  o.num_weights = 1;
  o.ids = io::vertex_ids::as_is;
  o.drop_self_loops = false;
  const auto back = io::read_edge_list<std::int32_t, float>(dir.path("f.txt"), o);
  ASSERT_EQ(back.weights.size(), e.weights.size());
  for (std::size_t i = 0; i < e.weights.size(); ++i) {
    EXPECT_EQ(std::memcmp(&back.weights[i], &e.weights[i], 4), 0) << i;  // sorted = input order
  }
  // Symmetrized (an undirected similarity graph), decimal and exponent forms, '+', commas.
  write_text(dir.path("u.txt"), "# knn\n0 1 0.5\n1,2,+2.5e-1\n");
  o.symmetrize = true;
  const auto u = io::read_edge_list<std::int32_t, float>(dir.path("u.txt"), o);
  EXPECT_EQ(u.src, (std::vector<std::int32_t>{0, 1, 1, 2}));
  EXPECT_EQ(u.dst, (std::vector<std::int32_t>{1, 0, 2, 1}));
  EXPECT_EQ(u.weights, (std::vector<float>{0.5F, 0.5F, 0.25F, 0.25F}));
  // NaN, infinities, out-of-range values and junk are refused, with the line.
  for (const char* w : {"nan", "NaN", "inf", "-inf", "infinity", "1e39", "0.5x", "", "--1"}) {
    SCOPED_TRACE(w);
    write_text(dir.path("bad.txt"), std::string("0 1 ") + w + "\n");
    EXPECT_THROW(((void)io::read_edge_list<std::int32_t, float>(dir.path("bad.txt"), o)),
                 dyng::io_error);
  }
  // The integer paths are unchanged: an integer graph refuses a float token.
  write_text(dir.path("i.txt"), "0 1 0.5\n");
  EXPECT_THROW(((void)io::read_edge_list<std::int32_t, std::int32_t>(dir.path("i.txt"), o)),
               dyng::io_error);
  // A non-finite weight would not read back: the writer refuses it.
  e.weights[2] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(io::write_edge_list(dir.path("nan.txt"), e.view()), dyng::invalid_argument_error);
}

// ------------------------------------------------------------------- .dgt vertex records -------

TEST(VertexBatchText, VertexOperationsAndFloatTokensRoundTrip) {
  temp_dir dir;
  const auto batches = sample_vertex_batches();
  io::write_batches(dir.path("v.dgt"), views_of(batches));
  EXPECT_EQ(read_text(dir.path("v.dgt")),
            "%dgt 1\n%batch 0\n+v 0 1\n+v 1 0\n+v 2\n+e 0 1 0.75\n+e 1 2 0.1\n"
            "%batch 1\n+v 3\n+v 4\n-v 2\n-v 7\n+e 3 0 0.33333334\n+e 4 2 1e-07\n"
            "%batch 2\n-v 1\n");
  const auto back = io::read_batches<std::int32_t, float>(dir.path("v.dgt"));
  ASSERT_EQ(back.size(), batches.size());
  for (std::size_t i = 0; i < back.size(); ++i) {
    SCOPED_TRACE(i);
    expect_same_batch(back[i], batches[i]);
  }
  // Hand-written: labels -1, 0, 1, a label before the first edge fixes no K, comments.
  write_text(dir.path("h.dgt"), "%dgt 1 graph\n+v 5 -1\n-v 6\n# x\n+v 7 0\n+e 5 7 2.5e-1\n");
  const auto h = io::read_batches<std::int32_t, float>(dir.path("h.dgt"));
  ASSERT_EQ(h.size(), 1U);
  EXPECT_EQ(h[0].insert_vertices(), (std::vector<std::int32_t>{5, 7}));
  EXPECT_EQ(h[0].insert_vertex_labels(), (std::vector<std::int8_t>{-1, 0}));
  EXPECT_EQ(h[0].delete_vertices(), (std::vector<std::int32_t>{6}));
  EXPECT_EQ(h[0].insert_weights(), (std::vector<float>{0.25F}));
  EXPECT_EQ(h[0].num_weights(), 1);
  // Integer graphs take vertex operations too, and refuse a float token.
  write_text(dir.path("int.dgt"), "+v 0 1\n+v 1\n+e 0 1 3\n");
  const auto ints = io::read_batches<std::int32_t, std::int32_t>(dir.path("int.dgt"));
  EXPECT_EQ(ints[0].insert_vertex_labels(), (std::vector<std::int8_t>{1, -1}));
  write_text(dir.path("int_bad.dgt"), "+e 0 1 0.5\n");
  EXPECT_THROW(((void)io::read_batches<std::int32_t, std::int32_t>(dir.path("int_bad.dgt"))),
               dyng::io_error);
}

TEST(VertexBatchText, MalformedVertexOperationsAreRejected) {
  temp_dir dir;
  const auto bad = [&](const std::string& text) {
    write_text(dir.path("bad.dgt"), text);
    return io_error_text([&] { (void)io::read_batches<std::int32_t, float>(dir.path("bad.dgt")); });
  };
  EXPECT_NE(bad("+v 0 2\n").find("seed label"), std::string::npos);
  EXPECT_NE(bad("+v 0 -2\n").find("seed label"), std::string::npos);
  EXPECT_NE(bad("+v 0 1 1\n").find("unexpected"), std::string::npos);
  EXPECT_NE(bad("-v 0 1\n").find("unexpected"), std::string::npos);
  EXPECT_NE(bad("+v\n").find("incomplete"), std::string::npos);
  EXPECT_NE(bad("-v -1\n").find("vertex id"), std::string::npos);
  EXPECT_NE(bad("+e 0 1 nan\n").find("finite"), std::string::npos);
  EXPECT_NE(bad("+e 0 1 inf\n").find("finite"), std::string::npos);
  EXPECT_NE(bad("+e 0 1 1e39\n").find("finite"), std::string::npos);
  EXPECT_NE(bad("+v 0 0.5\n").find("seed label"), std::string::npos);
  // '+i' / '-i' stay hypergraph operations; '+v' / '-v' are graph operations.
  EXPECT_NE(bad("+i 0 1\n").find("hypergraph operation"), std::string::npos);
  write_text(dir.path("hv.dgt"), "%dgt 1 hypergraph\n+v 3\n");
  const std::string h = io_error_text([&] {
    (void)io::read_hyperedge_batches<std::int32_t, dyng::unweighted>(dir.path("hv.dgt"));
  });
  EXPECT_NE(h.find("belong to graph files"), std::string::npos) << h;
  // An id beyond num_vertices.
  io::batch_file_options o;
  o.num_vertices = 3;
  write_text(dir.path("r.dgt"), "+v 3\n");
  EXPECT_THROW(((void)io::read_batches<std::int32_t, float>(dir.path("r.dgt"), o)), dyng::io_error);
  // The writer refuses what would not read back.
  fbatch b(1);
  b.insert_vertex(1, 2);
  EXPECT_THROW(io::write_batches(dir.path("w.dgt"), std::vector<fview>{b.view()}),
               dyng::invalid_argument_error);
  fbatch n(1);
  n.insert_edge(0, 1, {std::numeric_limits<float>::infinity()});
  EXPECT_THROW(io::write_batches(dir.path("w.dgt"), std::vector<fview>{n.view()}),
               dyng::invalid_argument_error);
}

// ------------------------------------------------------------------- .dgb vertex records -------

TEST(VertexBatchBinary, Float32AndVertexRecordsRoundTrip) {
  temp_dir dir;
  const auto batches = sample_vertex_batches();
  io::write_binary_batches(dir.path("v.dgb"), views_of(batches));
  const std::string bytes = read_text(dir.path("v.dgb"));
  EXPECT_EQ(static_cast<unsigned char>(bytes[18]), 3U);  // weight code 3 = float32
  // AD13's on-disk flag: bit 4, the literal (the enumerator is frozen to it, ADR 0109).
  EXPECT_EQ(u64_at(bytes, 24), std::uint64_t{16});
  EXPECT_EQ(io::batch_file_flags::vertex_records, std::uint64_t{16});
  // The first record: the 48-byte count block (id, deletions, insertions, inserted vertices,
  // labels, deleted vertices), then the arrays.
  EXPECT_EQ(u64_at(bytes, 64 + 16 + 8), 0U);
  EXPECT_EQ(u64_at(bytes, 64 + 16 + 16), 2U);
  EXPECT_EQ(u64_at(bytes, 64 + 16 + 24), 3U);
  EXPECT_EQ(u64_at(bytes, 64 + 16 + 32), 3U);
  EXPECT_EQ(u64_at(bytes, 64 + 16 + 40), 0U);
  EXPECT_EQ(u64_at(bytes, 64), 48U + 8 + 8 + 8 + 16 + 8);  // the payload length
  const auto back = io::read_binary_batches<std::int32_t, float>(dir.path("v.dgb"));
  ASSERT_EQ(back.size(), batches.size());
  for (std::size_t i = 0; i < back.size(); ++i) {
    SCOPED_TRACE(i);
    expect_same_batch(back[i], batches[i]);
  }
  // .dgt and .dgb hold the same batches.
  io::write_batches(dir.path("v.dgt"), views_of(batches));
  const auto text = io::read_batches<std::int32_t, float>(dir.path("v.dgt"));
  for (std::size_t i = 0; i < text.size(); ++i) expect_same_batch(text[i], back[i]);
  // An edge-only float file has no flag and the 24-byte count block (the layout of 0.3).
  fbatch e(1);
  e.insert_edge(0, 1, {0.5F});
  io::write_binary_batches(dir.path("e.dgb"), std::vector<fview>{e.view()});
  const std::string eb = read_text(dir.path("e.dgb"));
  EXPECT_EQ(u64_at(eb, 24), 0U);
  EXPECT_EQ(u64_at(eb, 64), 24U + 8 + 8 + 8);
  // An integer graph's vertex batch: int32 weights with the flag.
  dyng::edge_batch<std::int32_t, std::int32_t> ib(1);
  ib.insert_vertex(4, 1);
  ib.insert_edge(4, 5, {9});
  io::write_binary_batches(
      dir.path("i.dgb"), std::vector<dyng::edge_batch_view<std::int32_t, std::int32_t>>{ib.view()});
  const auto ib2 = io::read_binary_batches<std::int32_t, std::int32_t>(dir.path("i.dgb"));
  expect_same_batch(ib2[0], ib);
}

TEST(VertexBatchBinary, FlagsAreFixedWhenTheFileIsOpened) {
  temp_dir dir;
  const auto batches = sample_vertex_batches();
  io::batch_writer plain(dir.path("p.dgb"), io::edge_batch_file_header<std::int32_t, float>(1));
  EXPECT_THROW(plain.write(batches[0].view()), dyng::invalid_argument_error);
  fbatch edges(1);
  edges.insert_edge(0, 1, {0.5F});
  plain.write(edges.view());
  plain.close();
  // A file opened with the flag takes edge-only batches (zero vertex counts).
  io::batch_file_header h = io::edge_batch_file_header<std::int32_t, float>(1);
  h.flags = io::batch_file_flags::vertex_records;
  io::batch_writer flagged(dir.path("f.dgb"), h);
  flagged.write(edges.view());
  flagged.write(batches[2].view());
  flagged.close();
  io::batch_reader in(dir.path("f.dgb"));
  ASSERT_TRUE(in.next());
  EXPECT_TRUE((in.edge_batch<std::int32_t, float>().insert_vertices.empty()));
  ASSERT_TRUE(in.next());
  const auto v = in.edge_batch<std::int32_t, float>();
  ASSERT_EQ(v.delete_vertices.size(), 1U);
  EXPECT_EQ(v.delete_vertices[0], 1);
  // The header is checked per kind: bit 4 and float32 belong to graph files.
  io::batch_file_header hh = io::hyperedge_batch_file_header<std::int32_t, std::int32_t>();
  hh.flags = io::batch_file_flags::vertex_records;
  EXPECT_THROW(io::batch_writer(dir.path("h.dgb"), hh), dyng::invalid_argument_error);
  hh = io::hyperedge_batch_file_header<std::int32_t, std::int32_t>();
  hh.weight_type = io::batch_dtype::float32;
  EXPECT_THROW(io::batch_writer(dir.path("h.dgb"), hh), dyng::invalid_argument_error);
  io::batch_file_header g = io::edge_batch_file_header<std::int32_t, float>(1);
  g.flags = io::batch_file_flags::incidence_edits;  // bit 3 on a graph file
  EXPECT_THROW(io::batch_writer(dir.path("g.dgb"), g), dyng::invalid_argument_error);
  g = io::edge_batch_file_header<std::int32_t, float>(1);
  g.vertex_type = io::batch_dtype::float32;  // code 3 outside the weight byte
  EXPECT_THROW(io::batch_writer(dir.path("g.dgb"), g), dyng::invalid_argument_error);
  EXPECT_EQ(io::to_string(io::batch_dtype::float32), "float32");
}

TEST(VertexBatchBinary, ReadersRejectWhatTheyDoNotKnow) {
  temp_dir dir;
  const auto batches = sample_vertex_batches();
  io::write_binary_batches(dir.path("v.dgb"), views_of(batches));
  const std::string bytes = read_text(dir.path("v.dgb"));
  const auto rejected = [&](std::string s, const char* name) {
    reseal_header(s);
    write_text(dir.path(name), s);
    return io_error_text([&] { io::batch_reader in(dir.path(name)); });
  };
  std::string s = bytes;
  set_u64(s, 24, 32);  // a flag this library does not know: the loud refusal 0.3 makes of bit 4
  EXPECT_NE(rejected(s, "f32.dgb").find("unknown flags 32"), std::string::npos);
  s = bytes;
  s[18] = 4;  // a weight code this library does not know (0.3 refuses code 3 the same way)
  EXPECT_NE(rejected(s, "w4.dgb").find("weight type 4"), std::string::npos);
  s = bytes;
  s[17] = 3;  // float32 is not a vertex type
  EXPECT_NE(rejected(s, "v3.dgb").find("vertex type 3"), std::string::npos);
  s = bytes;
  set_u64(s, 24, io::batch_file_flags::vertex_records | io::batch_file_flags::incidence_edits);
  EXPECT_NE(rejected(s, "b3.dgb").find("in a file of graph batches"), std::string::npos);
  // A hypergraph file with bit 4, and with weight code 3.
  dyng::hyperedge_batch<std::int32_t, std::int32_t> hb;
  const std::int32_t members[] = {0, 1};
  hb.insert_hyperedge(dyng::array_view<const std::int32_t>(members, 2), 3);
  io::write_binary_hyperedge_batches(
      dir.path("h.dgb"),
      std::vector<dyng::hyperedge_batch_view<std::int32_t, std::int32_t>>{hb.view()});
  const std::string hbytes = read_text(dir.path("h.dgb"));
  s = hbytes;
  set_u64(s, 24, io::batch_file_flags::vertex_records);
  EXPECT_NE(rejected(s, "h4.dgb").find("belongs to graph files"), std::string::npos);
  s = hbytes;
  s[18] = 3;
  EXPECT_NE(rejected(s, "h3.dgb").find("belong to graph files"), std::string::npos);
  // A damaged record: a label of 2, a label count that is neither 0 nor the vertex count, a NaN
  // weight (each with a correct CRC, so that the field's own check fires).
  const auto record_rejected = [&](std::string t, const char* name) {
    const std::uint64_t len = u64_at(t, 64);
    const std::uint32_t crc =
        crc32_bitwise(reinterpret_cast<const unsigned char*>(t.data()) + 80, len);
    std::memcpy(t.data() + 72, &crc, 4);
    write_text(dir.path(name), t);
    return io_error_text(
        [&] { (void)io::read_binary_batches<std::int32_t, float>(dir.path(name)); });
  };
  // Record 0: counts at 80 .. 128; arrays: ins_src (8), ins_dst (8), weights (8), vertices (16),
  // labels (8), deletions (0).
  const std::size_t labels_at = 80 + 48 + 8 + 8 + 8 + 16;
  s = bytes;
  s[labels_at] = 2;
  EXPECT_NE(record_rejected(s, "l2.dgb").find("seed label 2"), std::string::npos);
  s = bytes;
  set_u64(s, 80 + 32, 1);
  EXPECT_NE(record_rejected(s, "lc.dgb").find("seed labels for"), std::string::npos);
  s = bytes;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(s.data() + 80 + 48 + 16, &nan, 4);
  EXPECT_NE(record_rejected(s, "nan.dgb").find("not finite"), std::string::npos);
}

// --------------------------------------------- the legacy generator's batches, both formats ----

TEST(VertexBatchFiles, LegacyGeneratorBatchesRoundTrip) {
  temp_dir dir;
  const dyng::resources res = dyng::resources::sequential();
  for (const std::int64_t n : {std::int64_t{5000}, std::int64_t{50000}}) {
    SCOPED_TRACE(n);
    dyng::generators::vertex_arrival_options opt;
    opt.total_vertices = n;
    opt.num_batches = n == 5000 ? 5 : 10;
    const auto stream =
        dyng::generators::legacy::dynlp_vertex_arrival_stream<std::int32_t, std::int32_t>(res, opt);
    std::vector<fview> views;
    for (const auto& b : stream.batches) views.push_back(b.view());
    io::write_batches(dir.path("s.dgt"), views);
    io::write_binary_batches(dir.path("s.dgb"), views);
    io::batch_file_options to;
    to.num_vertices = n;
    io::batch_reader_options bo;
    bo.num_vertices = n;
    const auto text = io::read_batches<std::int32_t, float>(dir.path("s.dgt"), to);
    const auto bin = io::read_binary_batches<std::int32_t, float>(dir.path("s.dgb"), bo);
    ASSERT_EQ(text.size(), stream.batches.size());
    ASSERT_EQ(bin.size(), stream.batches.size());
    for (std::size_t i = 0; i < stream.batches.size(); ++i) {
      SCOPED_TRACE(i);
      const auto& g = stream.batches[i];
      for (const auto* back : {&text[i], &bin[i]}) {
        EXPECT_EQ(back->insert_vertices(), g.insert_vertices);
        EXPECT_EQ(back->insert_vertex_labels(), g.insert_vertex_labels);
        EXPECT_EQ(back->delete_vertices(), g.delete_vertices);
        EXPECT_EQ(back->insert_src(), g.edge_sources);
        EXPECT_EQ(back->insert_dst(), g.edge_destinations);
        ASSERT_EQ(back->insert_weights().size(), g.edge_weights.size());
        // A batch without edges has no weights, and memcmp's pointers must not be null.
        if (!g.edge_weights.empty()) {
          EXPECT_EQ(std::memcmp(back->insert_weights().data(), g.edge_weights.data(),
                                4 * g.edge_weights.size()),
                    0);
        }
      }
    }
  }
}

}  // namespace
