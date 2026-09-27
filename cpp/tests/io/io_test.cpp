// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file io_test.cpp
 * @brief Readers and writers: round trips, formats, and the MOSP input-validation cases
 *        (MOSP-CUDA mospTest group "input-validation", re-expressed: same accept/reject
 *        decisions, dynG messages with path:line:column).
 */
#include "support/data_paths.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/csr_triplet.hpp>
#include <dyng/io/matrix_market.hpp>
#include <dyng/io/result_io.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

using dyng::test::read_text;
using dyng::test::temp_dir;
using dyng::test::write_text;
using csr32 = dyng::csr<std::int32_t, std::int32_t, std::int32_t>;

/// Runs `body`, expects an io_error, and returns it.
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

// ---------------------------------------------------------------------------------------------
// CSR triplet
// ---------------------------------------------------------------------------------------------

class CsrTriplet : public ::testing::Test {
 protected:
  /// Graph 0 -> 1 -> 2 with K = 2 (the mospTest input-validation graph).
  std::string write_graph(const std::string& rows, const std::string& values,
                          const std::string& cols = "1\n2\n") {
    const std::string prefix = tmp_.path("g" + std::to_string(count_++) + "/graphCsr");
    write_text(prefix + "RowPtr.txt", rows);
    write_text(prefix + "ColInd.txt", cols);
    write_text(prefix + "Values.txt", values);
    return prefix;
  }
  temp_dir tmp_;
  int count_ = 0;
};

TEST_F(CsrTriplet, ReadsTheMospValidationGraph) {
  const auto prefix = write_graph("0\n1\n2\n2\n", "4 5\n2147483647 7\n");
  const auto c = dyng::io::read_csr_triplet<std::int32_t, std::int32_t, std::int32_t>(prefix);
  EXPECT_EQ(c.row_ptr, (std::vector<std::int32_t>{0, 1, 2, 2}));
  EXPECT_EQ(c.col_ind, (std::vector<std::int32_t>{1, 2}));
  EXPECT_EQ(c.num_weights, 2);
  EXPECT_EQ(c.weights, (std::vector<std::int32_t>{4, 2147483647, 5, 7}));  // objective-major
}

TEST_F(CsrTriplet, RejectsWeightsOutsideOneToIntMax) {
  for (const char* values :
       {"0 5\n2 7\n", "4 5\n-3 7\n", "4294967297 5\n2 7\n", "4 5\n2147483648 7\n"}) {
    SCOPED_TRACE(values);
    const auto prefix = write_graph("0\n1\n2\n2\n", values);
    const auto e = expect_io_error([&] {
      (void)dyng::io::read_csr_triplet<std::int32_t, std::int32_t, std::int32_t>(prefix);
    });
    EXPECT_EQ(e.path(), prefix + "Values.txt");
    EXPECT_GE(e.line(), 1);
    EXPECT_GE(e.column(), 1);
  }
}

TEST_F(CsrTriplet, RejectsRowOffsetAboveIntMax) {
  const auto prefix = write_graph("0\n1\n4294967298\n2\n", "4 5\n2 7\n");
  const auto e = expect_io_error(
      [&] { (void)dyng::io::read_csr_triplet<std::int32_t, std::int32_t, std::int32_t>(prefix); });
  EXPECT_EQ(e.line(), 3);
  EXPECT_EQ(e.column(), 1);
  // With 64-bit offsets the value parses, but the offsets decrease afterwards.
  EXPECT_THROW((void)(dyng::io::read_csr_triplet<std::int32_t, std::int64_t, std::int32_t>(prefix)),
               dyng::io_error);
}

TEST_F(CsrTriplet, RejectsStructuralErrors) {
  using reader = csr32 (*)(const std::string&, const dyng::io::csr_triplet_options&);
  const reader read = &dyng::io::read_csr_triplet<std::int32_t, std::int32_t, std::int32_t>;
  const dyng::io::csr_triplet_options defaults;
  EXPECT_THROW(read(write_graph("1\n1\n2\n2\n", "4 5\n2 7\n"), defaults), dyng::io_error);
  EXPECT_THROW(read(write_graph("0\n2\n1\n2\n", "4 5\n2 7\n"), defaults), dyng::io_error);
  EXPECT_THROW(read(write_graph("0\n", ""), defaults), dyng::io_error);
  EXPECT_THROW(read(write_graph("0\n1\n2\n2\n", "4 5\n2 7\n", "1\n3\n"), defaults), dyng::io_error);
  EXPECT_THROW(read(write_graph("0\n1\n2\n2\n", "4 5\n2 7\n", "1\n"), defaults), dyng::io_error);
  EXPECT_THROW(read(write_graph("0\n1\n2\n2\n", "4 5\n2 7\n", "1\n2\n0\n"), defaults),
               dyng::io_error);
  EXPECT_THROW(read(write_graph("0\n1\n2\n2\n", "4 5\n2\n"), defaults), dyng::io_error);
  EXPECT_THROW(read(write_graph("0\n1\n2\n2\n", "4 5\n"), defaults), dyng::io_error);
  EXPECT_THROW(read(write_graph("0\n1\n2\n2\n", "4 5\n2 7\n1 1\n"), defaults), dyng::io_error);
  EXPECT_THROW(read(write_graph("0\n1\n2\n2\n", "4 5x\n2 7\n"), defaults), dyng::io_error);
  EXPECT_THROW(read(write_graph("0\n1\n2\n2\n", "4 1.5\n2 7\n"), defaults), dyng::io_error);
  EXPECT_THROW(read(write_graph("0\n1\n+2\n2\n", "4 5\n2 7\n"), defaults), dyng::io_error);
  EXPECT_THROW(read(tmp_.path("missing/graphCsr"), defaults), dyng::io_error);
  dyng::io::csr_triplet_options three;
  three.num_weights = 3;
  EXPECT_THROW(read(write_graph("0\n1\n2\n2\n", "4 5\n2 7\n"), three), dyng::io_error);
  dyng::io::csr_triplet_options negative;
  negative.num_weights = -1;
  EXPECT_THROW(read(write_graph("0\n1\n2\n2\n", "4 5\n2 7\n"), negative),
               dyng::invalid_argument_error);
}

// The three files are parsed concurrently; the reported error must still be the first one in
// file order (RowPtr, ColInd, Values) with the exact location, as if read one after the other.
TEST_F(CsrTriplet, ReportsTheFirstErrorInFileOrder) {
  using dyng::io::read_csr_triplet;
  const auto read = [](const std::string& prefix) {
    return expect_io_error(
        [&] { (void)read_csr_triplet<std::int32_t, std::int32_t, std::int32_t>(prefix); });
  };
  // Every file is wrong: the RowPtr error wins.
  auto e = read(write_graph("0\n2\n1\n2\n", "4 5\n2 x\n", "1\n9\n"));
  EXPECT_EQ(e.path(), tmp_.path("g0/graphCsrRowPtr.txt"));
  EXPECT_EQ(e.line(), 3);
  // ColInd and Values are wrong: the ColInd error, with the range [0, n - 1] of the row offsets.
  e = read(write_graph("0\n1\n2\n2\n", "4 5\n2 x\n", "1\n3\n"));
  EXPECT_EQ(e.path(), tmp_.path("g1/graphCsrColInd.txt"));
  EXPECT_EQ(e.line(), 2);
  EXPECT_EQ(e.column(), 1);
  EXPECT_NE(std::string(e.what()).find("[0, 2]"), std::string::npos) << e.what();
  // One column index too many: reported at the extra token.
  e = read(write_graph("0\n1\n2\n2\n", "4 5\n2 7\n", "1\n2\n0\n"));
  EXPECT_EQ(e.path(), tmp_.path("g2/graphCsrColInd.txt"));
  EXPECT_EQ(e.line(), 3);
  // One weight line too many: reported at that line, although its weights parse.
  e = read(write_graph("0\n1\n2\n2\n", "4 5\n2 7\n1 1\n"));
  EXPECT_EQ(e.path(), tmp_.path("g3/graphCsrValues.txt"));
  EXPECT_EQ(e.line(), 3);
  // A larger valid graph read concurrently equals the expected arrays.
  std::string rows = "0\n";
  std::string cols;
  std::string values;
  for (int v = 0; v < 1000; ++v) {
    rows += std::to_string(v + 1) + "\n";
    cols += std::to_string((v * 7 + 1) % 1000) + "\n";
    values += std::to_string(v + 1) + " " + std::to_string(1000 - v) + "\n";
  }
  rows += "1000\n";
  const auto c =
      read_csr_triplet<std::int32_t, std::int32_t, std::int32_t>(write_graph(rows, values, cols));
  ASSERT_EQ(c.num_vertices(), 1001);
  ASSERT_EQ(c.num_edges(), 1000);
  for (int e2 = 0; e2 < 1000; ++e2) {
    EXPECT_EQ(c.col_ind[static_cast<std::size_t>(e2)], (e2 * 7 + 1) % 1000);
    EXPECT_EQ(c.weights[static_cast<std::size_t>(e2)], e2 + 1);
    EXPECT_EQ(c.weights[static_cast<std::size_t>(1000 + e2)], 1000 - e2);
  }
}

TEST_F(CsrTriplet, BlankLinesAndCarriageReturnsAreAccepted) {
  const auto prefix = write_graph("0\r\n1 \n\n2\n2\n", "\n4 5\r\n\n 2\t7 \n\n", "1\n\n2");
  const auto c = dyng::io::read_csr_triplet<std::int32_t, std::int32_t, std::int32_t>(prefix);
  EXPECT_EQ(c.weights, (std::vector<std::int32_t>{4, 2, 5, 7}));
}

TEST_F(CsrTriplet, GraphWithoutEdgesNeedsTheWeightCount) {
  const auto prefix = write_graph("0\n0\n0\n", "", "");
  EXPECT_THROW((void)(dyng::io::read_csr_triplet<std::int32_t, std::int32_t, std::int32_t>(prefix)),
               dyng::io_error);
  dyng::io::csr_triplet_options options;
  options.num_weights = 3;
  const auto c =
      dyng::io::read_csr_triplet<std::int32_t, std::int32_t, std::int32_t>(prefix, options);
  EXPECT_EQ(c.num_weights, 3);
  EXPECT_EQ(c.num_vertices(), 2);
  EXPECT_EQ(c.num_edges(), 0);
}

TEST_F(CsrTriplet, WriterFormatAndRoundTrip) {
  csr32 c;
  c.row_ptr = {0, 2, 2, 3};
  c.col_ind = {2, 0, 1};
  c.num_weights = 3;
  c.weights = {1, 2, 3, 10, 20, 30, 100, 200, 300};
  const std::string prefix = tmp_.path("out/deeper/g");
  dyng::io::write_csr_triplet(prefix, c.view());
  EXPECT_EQ(read_text(prefix + "RowPtr.txt"), "0\n2\n2\n3\n");
  EXPECT_EQ(read_text(prefix + "ColInd.txt"), "2\n0\n1\n");
  EXPECT_EQ(read_text(prefix + "Values.txt"), "1 10 100\n2 20 200\n3 30 300\n");
  const auto back = dyng::io::read_csr_triplet<std::int64_t, std::int64_t, std::int32_t>(prefix);
  EXPECT_EQ(back.num_weights, 3);
  EXPECT_EQ(back.col_ind, (std::vector<std::int64_t>{2, 0, 1}));
  EXPECT_EQ(back.weights, (std::vector<std::int32_t>{1, 2, 3, 10, 20, 30, 100, 200, 300}));
}

// ---------------------------------------------------------------------------------------------
// Legacy batches
// ---------------------------------------------------------------------------------------------

class LegacyBatch : public ::testing::Test {
 protected:
  dyng::edge_batch<std::int32_t, std::int32_t> read(const std::string& insert,
                                                    const std::string& remove = "0 1\n", int k = 2,
                                                    std::int64_t n = 3) {
    const std::string dir = tmp_.path("c" + std::to_string(count_++));
    write_text(dir + "/insert.txt", insert);
    write_text(dir + "/delete.txt", remove);
    dyng::io::legacy_batch_options options;
    options.num_weights = k;
    options.num_vertices = n;
    return dyng::io::read_legacy_batch<std::int32_t, std::int32_t>(dir + "/insert.txt",
                                                                   dir + "/delete.txt", options);
  }
  temp_dir tmp_;
  int count_ = 0;
};

TEST_F(LegacyBatch, ReadsTheMospValidationBatch) {
  const auto b = read("2 0 1 2147483647\n");
  EXPECT_EQ(b.num_insertions(), 1u);
  EXPECT_EQ(b.insert_weights(), (std::vector<std::int32_t>{1, 2147483647}));
  EXPECT_EQ(b.delete_src(), (std::vector<std::int32_t>{0}));
  EXPECT_EQ(b.delete_dst(), (std::vector<std::int32_t>{1}));
  EXPECT_EQ(b.num_weights(), 2);
}

TEST_F(LegacyBatch, RejectsInvalidInsertLines) {
  for (const char* insert : {"2 0 0 3\n", "2 0 -5 3\n", "2 0 4294967297 3\n", "2 3 1 1\n"}) {
    SCOPED_TRACE(insert);
    const auto e = expect_io_error([&] { (void)read(insert); });
    EXPECT_EQ(e.line(), 1);
  }
}

TEST_F(LegacyBatch, StricterThanMosp) {
  EXPECT_THROW((void)read("2 0 1 2 3\n"), dyng::io_error);  // extra token
  EXPECT_THROW((void)read("2 0 1\n"), dyng::io_error);      // missing weight
  EXPECT_THROW((void)read("", "0\n"), dyng::io_error);      // short delete line
  EXPECT_THROW((void)read("", "0 1 2\n"), dyng::io_error);  // long delete line
  EXPECT_THROW((void)read("", "0 3\n"), dyng::io_error);    // out of range
  EXPECT_THROW((void)read("# comment\n"), dyng::io_error);  // no comments
  const auto e = expect_io_error([&] { (void)read("0 1 1 1\n\n1 x 1 1\n"); });
  EXPECT_EQ(e.line(), 3);
  EXPECT_EQ(e.column(), 3);
}

TEST_F(LegacyBatch, NoVertexCountChecksOnlySign) {
  const auto b = read("7 9 1 1\n", "12 3\n", 2, -1);
  EXPECT_EQ(b.insert_src()[0], 7);
  EXPECT_THROW((void)read("-1 9 1 1\n", "", 2, -1), dyng::io_error);
}

TEST_F(LegacyBatch, WriterFormat) {
  dyng::edge_batch<std::int64_t, std::int32_t> b(2);
  b.insert_edge(3, 1, {7, 8});
  b.insert_edge(0, 2, {1, 1});
  b.delete_edge(1, 2);
  dyng::io::write_legacy_batch(tmp_.path("w/insert.txt"), tmp_.path("w/delete.txt"), b.view());
  EXPECT_EQ(read_text(tmp_.path("w/insert.txt")), "3 1 7 8\n0 2 1 1\n");
  EXPECT_EQ(read_text(tmp_.path("w/delete.txt")), "1 2\n");
  dyng::edge_batch<std::int64_t, std::int32_t> empty(0);
  dyng::io::write_legacy_batch(tmp_.path("e/insert.txt"), tmp_.path("e/delete.txt"), empty.view());
  EXPECT_EQ(read_text(tmp_.path("e/insert.txt")), "");
}

// ---------------------------------------------------------------------------------------------
// Distances and trees
// ---------------------------------------------------------------------------------------------

TEST(ResultIo, MospValidationCases) {
  temp_dir tmp;
  const auto distances = [&](const std::string& text) {
    write_text(tmp.path("d.txt"), text);
    return dyng::io::read_distances<std::int64_t>(tmp.path("d.txt"), 3);
  };
  const auto parents = [&](const std::string& text) {
    write_text(tmp.path("p.txt"), text);
    return dyng::io::read_parents<std::int32_t>(tmp.path("p.txt"), 3);
  };
  EXPECT_EQ(distances("0 0\n1 4\n2 INF\n"),
            (std::vector<std::int64_t>{0, 4, dyng::infinite_distance<std::int64_t>()}));
  EXPECT_EQ(parents("0 -1\n1 0\n2 -1\n"), (std::vector<std::int32_t>{-1, 0, -1}));
  for (const char* text :
       {"0 0\n1 4\n", "0 0\n1 4\n1 4\n2 6\n", "0 0\n1 -4\n2 6\n", "0 0\n1 4\n3 6\n",
        "0 0\n1 4 5\n2 6\n", "0 0\n1\n2 6\n", "0 0\n1 INFINITY\n2 6\n"}) {
    SCOPED_TRACE(text);
    EXPECT_THROW((void)distances(text), dyng::io_error);
  }
  for (const char* text :
       {"0 -1\n1 0\n", "0 -1\n1 0\n1 0\n2 1\n", "0 -2\n1 0\n2 1\n", "0 -1\n1 3\n2 1\n"}) {
    SCOPED_TRACE(text);
    EXPECT_THROW((void)parents(text), dyng::io_error);
  }
  // Order does not matter; blank lines are ignored.
  EXPECT_EQ(distances("\n2 7\n0 0\n\n1 3\n"), (std::vector<std::int64_t>{0, 3, 7}));
  const auto e = expect_io_error([&] { (void)distances("0 0\n0 1\n2 2\n"); });
  EXPECT_EQ(e.line(), 2);
}

TEST(ResultIo, WritersAreMospFormat) {
  temp_dir tmp;
  const std::int64_t inf = dyng::infinite_distance<std::int64_t>();
  const std::vector<std::int64_t> d{0, 12, inf, inf / 2, inf / 2 - 1};
  dyng::io::write_distances(tmp.path("x/distances.txt"), dyng::host_view(d));
  EXPECT_EQ(read_text(tmp.path("x/distances.txt")),
            "0 0\n1 12\n2 INF\n3 INF\n4 " + std::to_string(inf / 2 - 1) + "\n");
  const std::vector<std::int32_t> p{-1, 0, 1};
  dyng::io::write_parents(tmp.path("x/tree.txt"), dyng::host_view(p));
  EXPECT_EQ(read_text(tmp.path("x/tree.txt")), "0 -1\n1 0\n2 1\n");
  const auto back = dyng::io::read_parents<std::int64_t>(tmp.path("x/tree.txt"), 3);
  EXPECT_EQ(back, (std::vector<std::int64_t>{-1, 0, 1}));
  const auto dist_back = dyng::io::read_distances<std::int64_t>(tmp.path("x/distances.txt"), 5);
  EXPECT_EQ(dist_back[2], inf);
  EXPECT_EQ(dist_back[3], inf);
  EXPECT_EQ(dist_back[4], inf / 2 - 1);
}

// ---------------------------------------------------------------------------------------------
// Matrix Market
// ---------------------------------------------------------------------------------------------

class MatrixMarket : public ::testing::Test {
 protected:
  std::string write(const std::string& text) {
    const std::string path = tmp_.path("m" + std::to_string(count_++) + ".mtx");
    write_text(path, text);
    return path;
  }
  dyng::edge_list<std::int32_t, std::int32_t> read(
      const std::string& text, const dyng::io::matrix_market_options& options = {}) {
    return dyng::io::read_matrix_market<std::int32_t, std::int32_t>(write(text), options);
  }
  temp_dir tmp_;
  int count_ = 0;
};

TEST_F(MatrixMarket, IntegerGeneralFromFile) {
  const auto e = read(
      "%%MatrixMarket matrix coordinate integer general\n% c\n\n3 3 4\n3 1 9\n1 2 5\n1 2 6\n"
      "2 2 1\n");
  EXPECT_EQ(e.num_vertices, 3);
  EXPECT_EQ(e.num_weights, 1);
  EXPECT_EQ(e.src, (std::vector<std::int32_t>{0, 2}));
  EXPECT_EQ(e.dst, (std::vector<std::int32_t>{1, 0}));
  EXPECT_EQ(e.weights, (std::vector<std::int32_t>{6, 9}));  // duplicates: the last one wins
}

TEST_F(MatrixMarket, SymmetricPatternMirrorsEntries) {
  const std::string text =
      "%%MatrixMarket matrix coordinate pattern symmetric\n3 3 3\n2 1\n3 3\n3 2\n";
  const auto e = read(text);
  EXPECT_EQ(e.num_weights, 0);
  EXPECT_EQ(e.src, (std::vector<std::int32_t>{0, 1, 1, 2}));
  EXPECT_EQ(e.dst, (std::vector<std::int32_t>{1, 0, 2, 1}));
  dyng::io::matrix_market_options raw;
  raw.drop_self_loops = false;
  raw.sort_and_dedupe = false;
  const auto f = read(text, raw);
  EXPECT_EQ(f.src, (std::vector<std::int32_t>{1, 0, 2, 2, 1}));
  EXPECT_EQ(f.dst, (std::vector<std::int32_t>{0, 1, 2, 1, 2}));
}

TEST_F(MatrixMarket, SkewSymmetricNegatesAndHermitianMirrors) {
  const auto e = read("%%MatrixMarket matrix coordinate integer skew-symmetric\n2 2 1\n2 1 4\n");
  EXPECT_EQ(e.weights, (std::vector<std::int32_t>{-4, 4}));
  dyng::io::matrix_market_options none;
  none.weights = dyng::io::matrix_market_weights::none;
  const auto h =
      read("%%MatrixMarket matrix coordinate complex hermitian\n2 2 1\n2 1 1.5 -2\n", none);
  EXPECT_EQ(h.src, (std::vector<std::int32_t>{0, 1}));
}

TEST_F(MatrixMarket, RandomWeightsAreSeeded) {
  dyng::io::matrix_market_options options;
  options.weights = dyng::io::matrix_market_weights::random;
  options.random.num_weights = 2;
  const std::string text =
      "%%MatrixMarket matrix coordinate real general\n4 4 3\n1 2 0.5\n"
      "2 3 1e3\n4 1 -2\n";
  const auto a = read(text, options);
  const auto b = read(text, options);
  EXPECT_EQ(a.num_weights, 2);
  EXPECT_EQ(a.weights.size(), 6u);
  EXPECT_EQ(a.weights, b.weights);
  for (const auto w : a.weights) {
    EXPECT_GE(w, 1);
    EXPECT_LE(w, 100);
  }
  options.random.seed = 1;
  EXPECT_NE(read(text, options).weights, a.weights);
  options.random.min = 5;
  options.random.max = 4;
  EXPECT_THROW((void)read(text, options), dyng::invalid_argument_error);
  options.random.min = 1;
  options.random.max = std::int64_t{1} << 40;
  EXPECT_THROW((void)read(text, options), dyng::invalid_argument_error);
}

TEST_F(MatrixMarket, RejectsMalformedFiles) {
  const char* bad[] = {
      "3 3 1\n1 2\n",                                                         // no header
      "%%MatrixMarket matrix array real general\n2 2\n1\n2\n3\n4\n",          // array
      "%%MatrixMarket matrix coordinate pattern general\n2 3 1\n1 2\n",       // not square
      "%%MatrixMarket matrix coordinate pattern general\n3 3 2\n1 2\n",       // missing entry
      "%%MatrixMarket matrix coordinate pattern general\n3 3 1\n1 2\n2 3\n",  // extra entry
      "%%MatrixMarket matrix coordinate pattern general\n3 3 1\n1 4\n",       // index range
      "%%MatrixMarket matrix coordinate pattern general\n3 3 1\n0 1\n",       // 0-based index
      "%%MatrixMarket matrix coordinate integer general\n3 3 1\n1 2\n",       // missing value
      "%%MatrixMarket matrix coordinate pattern general\n3 3 1\n1 2 3\n",     // extra value
      "%%MatrixMarket matrix coordinate integer general\n3 3 1\n1 2 x\n",     // bad value
      "%%MatrixMarket matrix coordinate real general\n3 3 1\n1 2 1.5\n",      // real: no weights
      "%%MatrixMarket matrix coordinate pattern wibble\n3 3 1\n1 2\n",        // symmetry
      "%%MatrixMarket matrix coordinate pattern general\n",                   // no size line
      "%%MatrixMarket matrix coordinate integer general\n2 2 1\n1 2 3000000000\n",  // weight type
  };
  for (const char* text : bad) {
    SCOPED_TRACE(text);
    EXPECT_THROW((void)read(text), dyng::io_error);
  }
  dyng::io::matrix_market_options from_file;
  from_file.weights = dyng::io::matrix_market_weights::from_file;
  EXPECT_THROW(
      (void)read("%%MatrixMarket matrix coordinate pattern general\n2 2 1\n1 2\n", from_file),
      dyng::io_error);
  const auto e = expect_io_error(
      [&] { (void)read("%%MatrixMarket matrix coordinate pattern general\n3 3 2\n1 2\n1 9\n"); });
  EXPECT_EQ(e.line(), 4);
  EXPECT_EQ(e.column(), 3);
}

TEST_F(MatrixMarket, WriterRoundTrip) {
  dyng::edge_list<std::int32_t, std::int32_t> e;
  e.num_vertices = 4;
  e.num_weights = 2;
  e.add_edge(0, 3, {5, 50});
  e.add_edge(2, 1, {6, 60});
  const std::string path = tmp_.path("w/out.mtx");
  dyng::io::write_matrix_market(path, e.view(), 1);
  EXPECT_EQ(read_text(path),
            "%%MatrixMarket matrix coordinate integer general\n4 4 2\n1 4 50\n3 2 60\n");
  const auto back = dyng::io::read_matrix_market<std::int32_t, std::int32_t>(path);
  EXPECT_EQ(back.src, e.src);
  EXPECT_EQ(back.dst, e.dst);
  EXPECT_EQ(back.weights, (std::vector<std::int32_t>{50, 60}));
  EXPECT_THROW(dyng::io::write_matrix_market(path, e.view(), 2), dyng::invalid_argument_error);
  e.num_weights = 0;
  e.weights.clear();
  dyng::io::write_matrix_market(path, e.view());
  EXPECT_EQ(read_text(path), "%%MatrixMarket matrix coordinate pattern general\n4 4 2\n1 4\n3 2\n");
}

}  // namespace
