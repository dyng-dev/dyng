// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file batch_binary_test.cpp
 * @brief The binary batch files `.dgb`: the header and record layout, round trips of graph and
 *        hypergraph batches (with the annotations of the flags), `.dgt` <-> `.dgb`, and the
 *        rejection of truncated, size-lying, damaged, wrong-kind and future-version files.
 */
#include "support/data_paths.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/registry.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/io/batch_io.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

using dyng::test::read_text;
using dyng::test::temp_dir;
using dyng::test::write_text;
namespace io = dyng::io;
using hbatch = dyng::hyperedge_batch<std::int32_t, std::int32_t>;
using ebatch = dyng::edge_batch<std::int32_t, std::int32_t>;

template <typename body_t>
dyng::io_error expect_io_error(const body_t& body) {
  try {
    body();
  } catch (const dyng::io_error& e) {
    return e;
  }
  ADD_FAILURE() << "expected dyng::io_error";
  return dyng::io_error("none");
}

// CRC-32 (IEEE), bitwise: an independent check of the files' CRCs.
std::uint32_t crc32_bitwise(const unsigned char* p, std::size_t n) {
  std::uint32_t c = 0xffffffffU;
  for (std::size_t i = 0; i < n; ++i) {
    c ^= p[i];
    for (int k = 0; k < 8; ++k) c = (c & 1U) != 0 ? 0xEDB88320U ^ (c >> 1U) : c >> 1U;
  }
  return ~c;
}

std::uint32_t u32_at(const std::string& s, std::size_t at) {
  std::uint32_t v = 0;
  std::memcpy(&v, s.data() + at, 4);
  return v;
}
std::uint64_t u64_at(const std::string& s, std::size_t at) {
  std::uint64_t v = 0;
  std::memcpy(&v, s.data() + at, 8);
  return v;
}
void set_u32(std::string& s, std::size_t at, std::uint32_t v) {
  std::memcpy(s.data() + at, &v, 4);
}
void set_u64(std::string& s, std::size_t at, std::uint64_t v) {
  std::memcpy(s.data() + at, &v, 8);
}
// Recompute the header's CRC32 after a change, so that the check of the changed field fires.
void reseal_header(std::string& s) {
  set_u32(s, 60, crc32_bitwise(reinterpret_cast<const unsigned char*>(s.data()), 60));
}
void reseal_record(std::string& s, std::size_t record_at) {
  const std::uint64_t len = u64_at(s, record_at);
  set_u32(s, record_at + 8,
          crc32_bitwise(reinterpret_cast<const unsigned char*>(s.data()) + record_at + 16,
                        static_cast<std::size_t>(len)));
}

std::vector<hbatch> sample_hyperedge_batches() {
  std::vector<hbatch> b(3);
  b[0].delete_hyperedge(3);
  b[0].insert_hyperedge({4, 1, 1}, 6);
  b[0].insert_hyperedge({7}, 2147483647);
  b[1].delete_hyperedge(0);
  b[1].delete_hyperedge(9000000000LL);
  return b;  // b[2] is empty
}

class BinaryBatches : public ::testing::Test {
 protected:
  std::string path(const std::string& name) {
    return tmp_.path(name);
  }
  // The bytes of a two-batch hypergraph file with both annotations.
  std::string annotated_file() {
    const std::string p = path("annotated.dgb");
    io::batch_writer w(
        p, io::hyperedge_batch_file_header<std::int32_t, std::int32_t>(
               io::batch_file_flags::deleted_members | io::batch_file_flags::expected_ids));
    const auto batches = sample_hyperedge_batches();
    const std::vector<std::int64_t> off0 = {0, 2};
    const std::vector<std::int32_t> mem0 = {5, 6};
    const std::vector<std::int64_t> ids0 = {3, 11};
    io::hyperedge_batch_annotations<std::int32_t> a0{dyng::host_view(off0), dyng::host_view(mem0),
                                                     dyng::host_view(ids0)};
    w.write(batches[0].view(), a0);
    const std::vector<std::int64_t> off1 = {0, 1, 4};
    const std::vector<std::int32_t> mem1 = {9, 0, 1, 2};
    io::hyperedge_batch_annotations<std::int32_t> a1{
        dyng::host_view(off1), dyng::host_view(mem1), {}};
    w.write(batches[1].view(), a1);
    w.close();
    return read_text(p);
  }
  temp_dir tmp_;
};

TEST_F(BinaryBatches, TheHeaderAndRecordsFollowTheSpecification) {
  const std::string bytes = annotated_file();
  ASSERT_GE(bytes.size(), 64u);
  EXPECT_EQ(bytes.substr(0, 8), "DYNGBAT1");
  EXPECT_EQ(u32_at(bytes, 8), 1u);            // format version
  EXPECT_EQ(u32_at(bytes, 12), 64u);          // header size
  EXPECT_EQ(static_cast<int>(bytes[16]), 1);  // hypergraph
  EXPECT_EQ(static_cast<int>(bytes[17]), 1);  // int32 vertices
  EXPECT_EQ(static_cast<int>(bytes[18]), 1);  // int32 weights
  EXPECT_EQ(static_cast<int>(bytes[19]), 2);  // int64 hyperedge ids
  EXPECT_EQ(u32_at(bytes, 20), 1u);           // K
  EXPECT_EQ(u64_at(bytes, 24), 3u);           // flags
  EXPECT_EQ(u64_at(bytes, 32), 2u);           // batch count
  EXPECT_EQ(u32_at(bytes, 60),
            crc32_bitwise(reinterpret_cast<const unsigned char*>(bytes.data()), 60));
  // Record 0: length, CRC32, zero, payload (batch id 0, nd 1, ni 2, nm 4, ndm 2, ...).
  const std::uint64_t len = u64_at(bytes, 64);
  EXPECT_EQ(len % 8, 0u);
  EXPECT_EQ(u32_at(bytes, 72),
            crc32_bitwise(reinterpret_cast<const unsigned char*>(bytes.data()) + 80, len));
  EXPECT_EQ(u32_at(bytes, 76), 0u);
  EXPECT_EQ(u64_at(bytes, 80), 0u);
  EXPECT_EQ(u64_at(bytes, 88), 1u);
  EXPECT_EQ(u64_at(bytes, 96), 2u);
  EXPECT_EQ(u64_at(bytes, 104), 4u);
  EXPECT_EQ(u64_at(bytes, 112), 2u);
  // 40 + 8 (ids) + 24 (offsets) + 16 (members) + 8 (weights) + 16 + 8 (deleted) + 16 (expected).
  EXPECT_EQ(len, 136u);
  EXPECT_EQ(bytes.size(), 64 + 16 + len + 16 + u64_at(bytes, 64 + 16 + len));
}

TEST_F(BinaryBatches, HyperedgeBatchesRoundTripWithTheirAnnotations) {
  const std::string bytes = annotated_file();
  const std::string p = path("copy.dgb");
  write_text(p, bytes);
  io::batch_reader in(p);
  EXPECT_EQ(in.header().kind, dyng::container_kind::hypergraph);
  EXPECT_EQ(in.header().num_batches, 2);
  EXPECT_EQ(in.batch_id(), -1);
  const auto want = sample_hyperedge_batches();
  ASSERT_TRUE(in.next());
  EXPECT_EQ(in.batch_id(), 0);
  auto v = in.hyperedge_batch<std::int32_t, std::int32_t>();
  EXPECT_EQ(std::vector<std::int64_t>(v.delete_hyperedges.begin(), v.delete_hyperedges.end()),
            want[0].delete_hyperedges());
  EXPECT_EQ(std::vector<std::int64_t>(v.insert_offsets.begin(), v.insert_offsets.end()),
            want[0].insert_offsets());
  EXPECT_EQ(std::vector<std::int32_t>(v.insert_members.begin(), v.insert_members.end()),
            want[0].insert_members());
  EXPECT_EQ(std::vector<std::int32_t>(v.insert_weights.begin(), v.insert_weights.end()),
            want[0].insert_weights());
  auto a = in.annotations<std::int32_t>();
  EXPECT_EQ(std::vector<std::int32_t>(a.deleted_members.begin(), a.deleted_members.end()),
            (std::vector<std::int32_t>{5, 6}));
  EXPECT_EQ(std::vector<std::int64_t>(a.expected_ids.begin(), a.expected_ids.end()),
            (std::vector<std::int64_t>{3, 11}));
  ASSERT_TRUE(in.next());
  v = in.hyperedge_batch<std::int32_t, std::int32_t>();
  EXPECT_EQ(v.num_insertions(), 0u);
  EXPECT_EQ(v.num_deletions(), 2u);
  a = in.annotations<std::int32_t>();
  EXPECT_EQ(std::vector<std::int64_t>(a.deleted_offsets.begin(), a.deleted_offsets.end()),
            (std::vector<std::int64_t>{0, 1, 4}));
  EXPECT_TRUE(a.expected_ids.empty());
  EXPECT_FALSE(in.next());
  EXPECT_FALSE(in.next());
  // Wrong types and kinds.
  io::batch_reader again(p);
  ASSERT_TRUE(again.next());
  EXPECT_THROW((void)(again.hyperedge_batch<std::int32_t, dyng::unweighted>()),
               dyng::invalid_argument_error);
  EXPECT_THROW((void)(again.edge_batch<std::int32_t, std::int32_t>()),
               dyng::invalid_argument_error);
  // A file of the other kind is an io_error naming the right reader, as in .dgt (M8b review).
  const dyng::io_error kind =
      expect_io_error([&] { (void)(io::read_binary_batches<std::int32_t, std::int32_t>(p)); });
  EXPECT_NE(std::string(kind.what()).find("read_binary_hyperedge_batches"), std::string::npos)
      << kind.what();
  EXPECT_EQ(kind.path(), p);
}

TEST_F(BinaryBatches, WholeFilesRoundTripAndMatchTheTextFormat) {
  // Hypergraph batches: .dgt -> .dgb -> .dgt gives the same text.
  const auto batches = sample_hyperedge_batches();
  std::vector<dyng::hyperedge_batch_view<std::int32_t, std::int32_t>> views;
  for (const auto& b : batches) views.push_back(b.view());
  io::write_hyperedge_batches(path("a.dgt"), views);
  const auto from_text = io::read_hyperedge_batches<std::int32_t, std::int32_t>(path("a.dgt"));
  std::vector<dyng::hyperedge_batch_view<std::int32_t, std::int32_t>> text_views;
  for (const auto& b : from_text) text_views.push_back(b.view());
  io::write_binary_hyperedge_batches(path("a.dgb"), text_views);
  const auto from_binary =
      io::read_binary_hyperedge_batches<std::int32_t, std::int32_t>(path("a.dgb"));
  std::vector<dyng::hyperedge_batch_view<std::int32_t, std::int32_t>> binary_views;
  for (const auto& b : from_binary) binary_views.push_back(b.view());
  io::write_hyperedge_batches(path("b.dgt"), binary_views);
  EXPECT_EQ(read_text(path("a.dgt")), read_text(path("b.dgt")));
  // Unweighted hypergraph batches.
  dyng::hyperedge_batch<std::int32_t, dyng::unweighted> u;
  u.insert_hyperedge({3, 3, 2});
  io::write_binary_hyperedge_batches<std::int32_t, dyng::unweighted>(path("u.dgb"), {u.view()});
  const auto ub = io::read_binary_hyperedge_batches<std::int32_t, dyng::unweighted>(path("u.dgb"));
  ASSERT_EQ(ub.size(), 1u);
  EXPECT_EQ(ub[0].insert_members(), u.insert_members());
  // Graph batches with K = 2, int32 and int64 ids; .dgt <-> .dgb.
  ebatch a(2);
  ebatch b(2);
  a.insert_edge(0, 1, {5, 6});
  a.delete_edge(2, 3);
  b.delete_edge(4, 5);
  io::write_batches<std::int32_t, std::int32_t>(path("e.dgt"), {a.view(), b.view()});
  const auto et = io::read_batches<std::int32_t, std::int32_t>(path("e.dgt"));
  io::write_binary_batches<std::int32_t, std::int32_t>(path("e.dgb"), {et[0].view(), et[1].view()});
  const auto eb = io::read_binary_batches<std::int32_t, std::int32_t>(path("e.dgb"));
  io::write_batches<std::int32_t, std::int32_t>(path("e2.dgt"), {eb[0].view(), eb[1].view()});
  EXPECT_EQ(read_text(path("e.dgt")), read_text(path("e2.dgt")));
  dyng::edge_batch<std::int64_t, dyng::unweighted> c(0);
  c.insert_edge(5000000000LL, 1);
  io::write_binary_batches<std::int64_t, dyng::unweighted>(path("c.dgb"), {c.view()});
  const auto cb = io::read_binary_batches<std::int64_t, dyng::unweighted>(path("c.dgb"));
  ASSERT_EQ(cb.size(), 1u);
  EXPECT_EQ(cb[0].insert_src(), c.insert_src());
  EXPECT_THROW((void)(io::read_binary_batches<std::int32_t, dyng::unweighted>(path("c.dgb"))),
               dyng::invalid_argument_error);
  const dyng::io_error kind = expect_io_error([&] {
    (void)(io::read_binary_hyperedge_batches<std::int32_t, dyng::unweighted>(path("c.dgb")));
  });
  EXPECT_NE(std::string(kind.what()).find("read_binary_batches()"), std::string::npos)
      << kind.what();
  // An empty file of batches.
  io::write_binary_hyperedge_batches<std::int32_t, std::int32_t>(path("none.dgb"), {});
  EXPECT_TRUE(
      (io::read_binary_hyperedge_batches<std::int32_t, std::int32_t>(path("none.dgb")).empty()));
  EXPECT_EQ(read_text(path("none.dgb")).size(), 64u);
}

// M8b review: after an error of next() the reader stops; a retry neither skips the damaged
// record nor returns the next one as valid, and the accessors throw too.
TEST_F(BinaryBatches, TheReaderStopsAtItsFirstError) {
  std::string s = annotated_file();
  s[80 + 50] ^= 0x10;  // a payload byte of record 0
  const std::string p = path("crc_first.dgb");
  write_text(p, s);
  io::batch_reader in(p);
  const dyng::io_error first = expect_io_error([&] { (void)in.next(); });
  EXPECT_NE(std::string(first.what()).find("CRC32"), std::string::npos) << first.what();
  for (int retry = 0; retry < 2; ++retry) {
    const dyng::io_error again = expect_io_error([&] { (void)in.next(); });
    EXPECT_NE(std::string(again.what()).find("failed at batch record 0"), std::string::npos)
        << again.what();
  }
  EXPECT_THROW((void)(in.hyperedge_batch<std::int32_t, std::int32_t>()), dyng::io_error);
  EXPECT_EQ(in.batch_id(), -1);
}

// M8b review: K above 64 is reported with the value in the file.
TEST_F(BinaryBatches, AnOversizedKIsReportedAsWritten) {
  ebatch a(2);
  a.insert_edge(0, 1, {5, 6});
  io::write_binary_batches<std::int32_t, std::int32_t>(path("k.dgb"), {a.view()});
  const std::string good = read_text(path("k.dgb"));
  for (const std::uint32_t k : {65U, 4294967295U}) {
    std::string s = good;
    set_u32(s, 20, k);
    reseal_header(s);
    write_text(path("k_bad.dgb"), s);
    const dyng::io_error e = expect_io_error([&] { io::batch_reader in(path("k_bad.dgb")); });
    EXPECT_NE(std::string(e.what()).find("K = " + std::to_string(k)), std::string::npos)
        << e.what();
  }
  ebatch wide(65);
  std::vector<std::int32_t> w(65, 1);
  wide.insert_edge(0, 1, dyng::host_view(w));
  try {
    io::write_binary_batches<std::int32_t, std::int32_t>(path("wide.dgb"), {wide.view()});
    ADD_FAILURE() << "K = 65 was written";
  } catch (const dyng::invalid_argument_error& e) {
    EXPECT_NE(std::string(e.what()).find("1 to 64"), std::string::npos) << e.what();
  }
}

TEST_F(BinaryBatches, EveryTruncationIsAnIoError) {
  const std::string bytes = annotated_file();
  const std::string p = path("cut.dgb");
  for (std::size_t n = 0; n < bytes.size(); ++n) {
    SCOPED_TRACE(n);
    write_text(p, bytes.substr(0, n));
    (void)expect_io_error([&] {
      io::batch_reader in(p);
      while (in.next()) {
      }
    });
  }
}

TEST_F(BinaryBatches, DamagedSizeLyingAndFutureFilesAreIoErrors) {
  const std::string good = annotated_file();
  const std::string p = path("bad.dgb");
  const auto read_all = [&](const std::string& bytes) {
    write_text(p, bytes);
    return expect_io_error([&] {
      io::batch_reader in(p);
      while (in.next()) {
      }
    });
  };
  const auto contains = [](const dyng::io_error& e, const char* text) {
    return std::string(e.what()).find(text) != std::string::npos;
  };
  std::string s = good;
  s[3] = 'X';  // magic
  EXPECT_TRUE(contains(read_all(s), "magic"));
  s = good;
  set_u32(s, 8, 2);  // a future version
  reseal_header(s);
  const auto future = read_all(s);
  EXPECT_TRUE(contains(future, "version 2")) << future.what();
  EXPECT_EQ(future.path(), p);
  s = good;
  set_u64(s, 24, 64);  // an unknown flag
  reseal_header(s);
  EXPECT_TRUE(contains(read_all(s), "unknown flags"));
  s = good;
  s[17] = 7;  // vertex type
  reseal_header(s);
  EXPECT_TRUE(contains(read_all(s), "vertex type"));
  s = good;
  s[45] = 1;  // reserved
  reseal_header(s);
  EXPECT_TRUE(contains(read_all(s), "reserved"));
  s = good;
  s[20] = 2;  // the header changed without its CRC
  EXPECT_TRUE(contains(read_all(s), "CRC32"));
  s = good;
  set_u64(s, 32, 1000000000000ULL);  // a batch count the file cannot hold
  reseal_header(s);
  EXPECT_TRUE(contains(read_all(s), "announces"));
  s = good;
  set_u64(s, 32, 3);  // one batch more than the file has
  reseal_header(s);
  EXPECT_TRUE(contains(read_all(s), "truncated"));
  s = good;
  set_u64(s, 32, 1);  // one batch less: the second record is trailing data
  reseal_header(s);
  EXPECT_TRUE(contains(read_all(s), "follow the last"));
  s = good + std::string(8, '\0');  // trailing bytes
  EXPECT_TRUE(contains(read_all(s), "follow the last"));
  s = good;
  set_u64(s, 64, 1ULL << 62);  // a record length beyond the file
  EXPECT_TRUE(contains(read_all(s), "exceeds the rest of the file"));
  s = good;
  s[80 + 50] ^= 0x10;  // a payload byte
  EXPECT_TRUE(contains(read_all(s), "CRC32"));
  s = good;
  set_u64(s, 80 + 16, 1ULL << 60);  // an insertion count beyond the record (CRC resealed)
  reseal_record(s, 64);
  EXPECT_TRUE(contains(read_all(s), "fit"));
  s = good;
  set_u64(s, 80 + 24, 3);  // a member count that disagrees with the arrays
  reseal_record(s, 64);
  (void)read_all(s);
  s = good;
  set_u64(s, 80 + 40 + 8 + 8, 9);  // the second insertion offset beyond the members
  reseal_record(s, 64);
  (void)read_all(s);
  s = good;
  set_u64(s, 80, 5);  // batch ids: 5 then 1
  reseal_record(s, 64);
  EXPECT_TRUE(contains(read_all(s), "strictly increasing"));
  // A member out of range of the reader's vertex count.
  write_text(p, good);
  io::batch_reader_options o;
  o.num_vertices = 5;
  const auto range = expect_io_error([&] {
    io::batch_reader in(p, o);
    while (in.next()) {
    }
  });
  EXPECT_TRUE(contains(range, "out of range"));
  // Not a file at all.
  EXPECT_THROW(io::batch_reader(path("missing.dgb")), dyng::io_error);
}

// The seeds of the .dgb fuzzer are written from the specification by cpp/fuzz/make_dgb_seeds.py
// (Python's struct and zlib.crc32), independently of the library's writer: the reader must read
// them as written.
TEST_F(BinaryBatches, ReadsFilesWrittenFromTheSpecification) {
  const auto seed = [&](const std::string& name) {
    const std::string bytes =
        read_text(std::string(DYNG_TEST_DATA_DIR) + "/../../fuzz/corpus/batch_binary/" + name);
    const std::string p = path(name + ".dgb");
    write_text(p, bytes.substr(bytes.find('\n') + 1));  // without the fuzzer's option line
    return p;
  };
  {
    io::batch_reader in(seed("hyper_annotated"));
    EXPECT_EQ(in.header().flags,
              io::batch_file_flags::deleted_members | io::batch_file_flags::expected_ids);
    ASSERT_TRUE(in.next());
    const auto v = in.hyperedge_batch<std::int32_t, dyng::unweighted>();
    EXPECT_EQ(std::vector<std::int64_t>(v.delete_hyperedges.begin(), v.delete_hyperedges.end()),
              (std::vector<std::int64_t>{1, 5}));
    EXPECT_EQ(std::vector<std::int32_t>(v.insert_members.begin(), v.insert_members.end()),
              (std::vector<std::int32_t>{2, 3, 0}));
    const auto a = in.annotations<std::int32_t>();
    EXPECT_EQ(std::vector<std::int32_t>(a.deleted_members.begin(), a.deleted_members.end()),
              (std::vector<std::int32_t>{1, 2, 6, 7, 7}));
    EXPECT_EQ(std::vector<std::int64_t>(a.expected_ids.begin(), a.expected_ids.end()),
              (std::vector<std::int64_t>{1, 5}));
    ASSERT_TRUE(in.next());
    EXPECT_FALSE(in.next());
  }
  const auto g = io::read_binary_batches<std::int32_t, std::int32_t>(seed("graph_k1"));
  ASSERT_EQ(g.size(), 2u);
  EXPECT_EQ(g[0].insert_dst(), (std::vector<std::int32_t>{1, 2}));
  EXPECT_EQ(g[0].insert_weights(), (std::vector<std::int32_t>{5, 6}));
  EXPECT_EQ(g[1].delete_src(), (std::vector<std::int32_t>{4}));
  const auto w =
      io::read_binary_hyperedge_batches<std::int32_t, std::int32_t>(seed("hyper_weighted"));
  ASSERT_EQ(w.size(), 3u);
  EXPECT_EQ(w[0].insert_weights(), (std::vector<std::int32_t>{6, 2147483647}));
  EXPECT_EQ(w[1].delete_hyperedges(), (std::vector<std::int64_t>{0, 900000000000LL}));
  const auto e =
      io::read_binary_batches<std::int64_t, dyng::unweighted>(seed("graph_unweighted_int64"));
  ASSERT_EQ(e.size(), 1u);
  EXPECT_EQ(e[0].insert_src(), (std::vector<std::int64_t>{5000000000LL}));
  EXPECT_THROW(io::batch_reader(seed("crc_mismatch")).next(), dyng::io_error);
  EXPECT_THROW(io::batch_reader{seed("future_version")}, dyng::io_error);
}

TEST_F(BinaryBatches, WritersCheckTheirInput) {
  io::batch_writer w(path("w.dgb"), io::hyperedge_batch_file_header<std::int32_t, std::int32_t>());
  hbatch inc;
  inc.insert_incidence(0, 1);
  EXPECT_THROW(w.write(inc.view()), dyng::not_supported_error);
  hbatch neg;
  neg.delete_hyperedge(-3);
  EXPECT_THROW(w.write(neg.view()), dyng::invalid_argument_error);
  const std::vector<std::int64_t> ids = {1};
  io::hyperedge_batch_annotations<std::int32_t> extra{{}, {}, dyng::host_view(ids)};
  hbatch one;
  one.insert_hyperedge({1}, 1);
  EXPECT_THROW(w.write(one.view(), extra), dyng::invalid_argument_error);  // no such flag
  ebatch e(1);
  EXPECT_THROW(w.write(e.view()), dyng::invalid_argument_error);  // a graph batch
  w.write(one.view());
  EXPECT_EQ(w.num_batches(), 1);
  w.close();
  EXPECT_EQ((io::read_binary_hyperedge_batches<std::int32_t, std::int32_t>(path("w.dgb")).size()),
            1u);
  io::batch_file_header h;
  h.kind = dyng::container_kind::graph;
  h.flags = io::batch_file_flags::expected_ids;
  EXPECT_THROW(io::batch_writer(path("x.dgb"), h), dyng::invalid_argument_error);
  EXPECT_THROW((void)(io::hyperedge_batch_file_header<std::int32_t, std::int32_t>(1024)),
               dyng::invalid_argument_error);
  EXPECT_EQ(io::to_string(io::batch_dtype::int64), "int64");
  // An unclosed writer leaves a file whose count is 0 and whose record is trailing data.
  {
    io::batch_writer open(path("open.dgb"),
                          io::hyperedge_batch_file_header<std::int32_t, std::int32_t>());
    open.write(one.view());
  }
  EXPECT_THROW(
      (void)(io::read_binary_hyperedge_batches<std::int32_t, std::int32_t>(path("open.dgb"))),
      dyng::io_error);
}

}  // namespace
