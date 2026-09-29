// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file edge_list_io_test.cpp
 * @brief io::read_edge_list reproduces CycleEnumeration-GPU@0a976ad's parser (read_temporal_graph,
 *        and the CSR of build_directed_graph(read_graph_view)) on the committed fixtures: TUDataset
 *        `*_A.txt`, comments, commas, signs, timestamps, duplicates, self-loops, sparse ids, and
 *        Matrix Market files of every symmetry; the same accept/reject decisions and lines on
 *        malformed files; the original's GraphParserTest cases; and dynG's generalizations
 *        (weights, ids as they are, symmetrization, kept self-loops, thread count, round trips).
 *
 * Expected outputs: cpp/tests/data/cycle_enum/{parser,errors}, written by the pinned original
 * (parity/fixtures/cycle_enum/make_cycle_enum_fixtures.sh).
 */
#include "support/cycle_enum_text.hpp"
#include "support/data_paths.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/io/edge_list_io.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

using dyng::unweighted;
using dyng::io::duplicate_edges;
using dyng::io::edge_list_info;
using dyng::io::edge_list_options;
using dyng::io::vertex_ids;
using dyng::test::data_path;
using dyng::test::read_text;
using dyng::test::temp_dir;
using dyng::test::write_text;

const std::vector<std::string> parser_inputs = {
    "sample_temporal.txt", "reference_sample.txt", "tudataset_A.txt",
    "mixed.txt",           "mm_general.mtx",       "mm_symmetric.mtx",
    "mm_skew.mtx",         "mm_hermitian.mtx",     "sparse_ids.txt",
};

/// read_edge_list<int32, unweighted> (a name without a comma, for the gtest macros).
dyng::edge_list<std::int32_t, unweighted> read_u32(const std::string& path,
                                                   const edge_list_options& options = {}) {
  return dyng::io::read_edge_list<std::int32_t, unweighted>(path, options);
}

/// read_edge_list<int32, int32>.
dyng::edge_list<std::int32_t, std::int32_t> read_w32(const std::string& path,
                                                     const edge_list_options& options = {}) {
  return dyng::io::read_edge_list<std::int32_t, std::int32_t>(path, options);
}

/// The exporter's parse text of a file read with dynG.
template <typename vertex_t, typename weight_t>
std::string dyng_parse_text(const std::string& path, int threads = 0) {
  edge_list_options options;
  options.duplicates = duplicate_edges::keep;
  options.threads = threads;
  edge_list_info info;
  const auto edges = dyng::io::read_edge_list<vertex_t, weight_t>(path, options, &info);
  return dyng::test::parse_text(edges, info);
}

/// The CSR text of the cycle_enum_compatible() graph of a file.
template <typename vertex_t, typename edge_t, typename weight_t>
std::string dyng_csr_text(const std::string& path) {
  const auto res = dyng::resources::sequential();
  const auto edges = dyng::io::read_edge_list<vertex_t, weight_t>(path);
  const auto g = dyng::graph<vertex_t, edge_t, weight_t>::from_edges(
      res, edges.view(), dyng::graph_properties::cycle_enum_compatible());
  return dyng::test::csr_text(g.view().out);
}

TEST(ReadEdgeList, ParserOutputEqualsTheOriginal) {
  for (const std::string& name : parser_inputs) {
    SCOPED_TRACE(name);
    const std::string path = data_path("cycle_enum/parser/" + name);
    const std::string parse = read_text(path + ".parse");
    ASSERT_FALSE(parse.empty());
    EXPECT_EQ((dyng_parse_text<std::int32_t, unweighted>(path)), parse);
    EXPECT_EQ((dyng_parse_text<std::int64_t, unweighted>(path, 1)), parse);
    EXPECT_EQ((dyng_parse_text<std::int32_t, std::int32_t>(path, 3)), parse);
  }
}

TEST(ReadEdgeList, CsrEqualsBuildDirectedGraph) {
  for (const std::string& name : parser_inputs) {
    SCOPED_TRACE(name);
    const std::string path = data_path("cycle_enum/parser/" + name);
    const std::string csr = read_text(path + ".csr");
    ASSERT_FALSE(csr.empty());
    EXPECT_EQ((dyng_csr_text<std::int32_t, std::int32_t, unweighted>(path)), csr);
    EXPECT_EQ((dyng_csr_text<std::int32_t, std::int64_t, unweighted>(path)), csr);
    EXPECT_EQ((dyng_csr_text<std::int64_t, std::int64_t, std::int32_t>(path)), csr);
  }
}

TEST(ReadEdgeList, MalformedFilesAsTheOriginal) {
  const std::vector<std::string> names = {
      "single_field", "four_fields", "bad_timestamp", "glued_token", "plus_minus",
      "lone_plus",    "commas_only", "overflow",      "mm_array",    "mm_symmetry",
      "mm_short",     "mm_vector",   "mm_one_field",  "ok_commas",   "ok_mm_extra",
  };
  for (const std::string& name : names) {
    SCOPED_TRACE(name);
    const std::string path = data_path("cycle_enum/errors/" + name + ".txt");
    const std::string expected = read_text(data_path("cycle_enum/errors/" + name + ".expected"));
    ASSERT_FALSE(expected.empty());
    if (expected == "ok\n") {
      EXPECT_NO_THROW((void)read_u32(path));
      continue;
    }
    try {
      (void)dyng::io::read_edge_list<std::int32_t, unweighted>(path);
      ADD_FAILURE() << "expected an io_error";
    } catch (const dyng::io_error& e) {
      EXPECT_EQ("error " + std::to_string(e.line()) + "\n", expected) << e.what();
      EXPECT_EQ(e.path(), path);
    }
  }
}

// The original's GraphParserTest cases (on dynG's result).
TEST(ReadEdgeList, GroupsEdgesAndCompactsVertices) {
  edge_list_options options;
  options.duplicates = duplicate_edges::keep;
  edge_list_info info;
  const auto e = dyng::io::read_edge_list<std::int32_t, unweighted>(
      data_path("cycle_enum/parser/sample_temporal.txt"), options, &info);
  EXPECT_EQ(e.num_vertices, 3);
  EXPECT_EQ(info.external_ids, (std::vector<std::int64_t>{10, 20, 30}));  // 40 (a loop) dropped
  EXPECT_EQ(e.src, (std::vector<std::int32_t>{0, 0, 1, 2}));
  EXPECT_EQ(e.dst, (std::vector<std::int32_t>{1, 1, 0, 0}));
  EXPECT_EQ(info.timestamps, (std::vector<std::int64_t>{3, 5, 7, 6}));
  EXPECT_FALSE(info.matrix_market);

  const auto merged = dyng::io::read_edge_list<std::int32_t, unweighted>(
      data_path("cycle_enum/parser/sample_temporal.txt"));
  EXPECT_EQ(merged.src, (std::vector<std::int32_t>{0, 1, 2}));
  EXPECT_EQ(merged.dst, (std::vector<std::int32_t>{1, 0, 0}));
  EXPECT_EQ(merged.num_weights, 0);
  EXPECT_TRUE(merged.weights.empty());
}

TEST(ReadEdgeList, CompactIdsInAscendingExternalOrder) {
  temp_dir tmp;
  write_text(tmp.path("unsorted.txt"), "50 10 1\n10 30 2\n");
  edge_list_info info;
  const auto e =
      dyng::io::read_edge_list<std::int32_t, unweighted>(tmp.path("unsorted.txt"), {}, &info);
  EXPECT_EQ(info.external_ids, (std::vector<std::int64_t>{10, 30, 50}));
  EXPECT_EQ(e.src, (std::vector<std::int32_t>{0, 2}));
  EXPECT_EQ(e.dst, (std::vector<std::int32_t>{1, 0}));
}

TEST(ReadEdgeList, MissingFileAndMatrixMarketDimensions) {
  EXPECT_THROW((void)read_u32(data_path("cycle_enum/parser/missing.txt")), dyng::io_error);
  temp_dir tmp;
  // Read as data, the dimensions line would add the vertex 5 and the edge 3 -> 5.
  write_text(tmp.path("dims.mtx"),
             "%%MatrixMarket matrix coordinate pattern general\n3 5 2\n1 2\n2 3\n");
  edge_list_info info;
  const auto e =
      dyng::io::read_edge_list<std::int32_t, unweighted>(tmp.path("dims.mtx"), {}, &info);
  EXPECT_EQ(e.num_vertices, 3);
  EXPECT_EQ(e.src.size(), 2u);
  EXPECT_TRUE(info.matrix_market);
  EXPECT_FALSE(info.symmetric);
  // Weight columns are not read from a Matrix Market file.
  edge_list_options weighted;
  weighted.num_weights = 1;
  EXPECT_THROW((void)read_w32(tmp.path("dims.mtx"), weighted), dyng::io_error);
}

TEST(ReadEdgeList, WeightsTimestampsAndDuplicates) {
  temp_dir tmp;
  write_text(tmp.path("w.txt"),
             "# u v w1 w2 [ts]\n"
             "1 2 10 100\n"
             "1 2 11 110 5\n"  // a duplicate pair: the last row in file order wins
             "2 1 12 120 3\n"
             "3 3 13 130\n"  // a self-loop: dropped
             "2 3 14 140\n");
  edge_list_options options;
  options.num_weights = 2;
  options.ids = vertex_ids::as_is;
  options.index_base = 1;
  edge_list_info info;
  const auto e =
      dyng::io::read_edge_list<std::int32_t, std::int32_t>(tmp.path("w.txt"), options, &info);
  EXPECT_EQ(e.num_vertices, 3);
  EXPECT_EQ(e.num_weights, 2);
  EXPECT_EQ(e.src, (std::vector<std::int32_t>{0, 1, 1}));
  EXPECT_EQ(e.dst, (std::vector<std::int32_t>{1, 0, 2}));
  EXPECT_EQ(e.weights, (std::vector<std::int32_t>{11, 110, 12, 120, 14, 140}));
  EXPECT_TRUE(info.external_ids.empty());
  EXPECT_TRUE(info.timestamps.empty());

  options.duplicates = duplicate_edges::keep;
  options.drop_self_loops = false;
  const auto k =
      dyng::io::read_edge_list<std::int32_t, std::int32_t>(tmp.path("w.txt"), options, &info);
  EXPECT_EQ(k.src, (std::vector<std::int32_t>{0, 0, 1, 1, 2}));
  EXPECT_EQ(k.dst, (std::vector<std::int32_t>{1, 1, 0, 2, 2}));
  EXPECT_EQ(info.timestamps, (std::vector<std::int64_t>{0, 5, 3, 0, 0}));
  EXPECT_EQ(k.weights, (std::vector<std::int32_t>{10, 100, 11, 110, 12, 120, 14, 140, 13, 130}));

  // Errors: a missing weight, a weight out of range, an extra field, an id below the base.
  write_text(tmp.path("bad1.txt"), "1 2 10 100\n1 2 10\n");
  write_text(tmp.path("bad2.txt"), "1 2 10 3000000000\n");
  write_text(tmp.path("bad3.txt"), "1 2 10 100 7 8\n");
  write_text(tmp.path("bad4.txt"), "1 2 10 100\n0 2 1 1\n");
  const std::vector<std::pair<std::string, std::int64_t>> bad = {
      {"bad1.txt", 2}, {"bad2.txt", 1}, {"bad3.txt", 1}, {"bad4.txt", 2}};
  options.duplicates = duplicate_edges::merge;
  for (const auto& [name, line] : bad) {
    SCOPED_TRACE(name);
    try {
      (void)dyng::io::read_edge_list<std::int32_t, std::int32_t>(tmp.path(name), options);
      ADD_FAILURE() << "expected an io_error";
    } catch (const dyng::io_error& err) {
      EXPECT_EQ(err.line(), line) << err.what();
    }
  }
}

TEST(ReadEdgeList, SymmetrizeAndIdsAsTheyAre) {
  temp_dir tmp;
  write_text(tmp.path("u.txt"), "0 5\n5 2\n2 0\n");
  edge_list_options options;
  options.ids = vertex_ids::as_is;
  options.symmetrize = true;
  const auto e = dyng::io::read_edge_list<std::int64_t, unweighted>(tmp.path("u.txt"), options);
  EXPECT_EQ(e.num_vertices, 6);  // vertices 1, 3, 4 have no edges and are kept
  EXPECT_EQ(e.src, (std::vector<std::int64_t>{0, 0, 2, 2, 5, 5}));
  EXPECT_EQ(e.dst, (std::vector<std::int64_t>{2, 5, 0, 5, 0, 2}));
  // An id beyond the vertex type (as_is) is an error with its line.
  write_text(tmp.path("big.txt"), "0 1\n1 3000000000\n");
  try {
    (void)dyng::io::read_edge_list<std::int32_t, unweighted>(tmp.path("big.txt"), options);
    ADD_FAILURE() << "expected an io_error";
  } catch (const dyng::io_error& err) {
    EXPECT_EQ(err.line(), 2);
  }
  // Compact ids of any 64-bit value.
  const auto c = dyng::io::read_edge_list<std::int32_t, unweighted>(tmp.path("big.txt"));
  EXPECT_EQ(c.num_vertices, 3);
  edge_list_options weighted;
  weighted.num_weights = 1;
  EXPECT_THROW((void)read_u32(tmp.path("u.txt"), weighted), dyng::invalid_argument_error);
}

TEST(ReadEdgeList, ResultDoesNotDependOnTheThreadCount) {
  // A file of several MB is split into parts on several threads.
  temp_dir tmp;
  std::string text = "% a large generated edge list\n";
  std::mt19937_64 rng(5);
  for (int i = 0; i < 400000; ++i) {
    text += std::to_string(static_cast<std::int64_t>(rng() % 30000) - 100) + ", " +
            std::to_string(rng() % 30000) + " " + std::to_string(rng() % 7) + "\n";
    if (i % 1000 == 0) {
      text += "# comment\n\n";
    }
  }
  write_text(tmp.path("large.txt"), text);
  const std::string one = dyng_parse_text<std::int32_t, unweighted>(tmp.path("large.txt"), 1);
  EXPECT_EQ((dyng_parse_text<std::int32_t, unweighted>(tmp.path("large.txt"), 4)), one);
  EXPECT_EQ((dyng_parse_text<std::int32_t, unweighted>(tmp.path("large.txt"), 0)), one);
  // An error in the middle is reported at its line whatever the thread count.
  text += "1 2 3 4\n";
  const auto lines = static_cast<std::int64_t>(std::count(text.begin(), text.end(), '\n'));
  text += "5 6\n7\n";
  write_text(tmp.path("large_bad.txt"), text);
  for (const int threads : {1, 3, 8}) {
    edge_list_options options;
    options.threads = threads;
    try {
      (void)dyng::io::read_edge_list<std::int32_t, unweighted>(tmp.path("large_bad.txt"), options);
      ADD_FAILURE() << "expected an io_error";
    } catch (const dyng::io_error& err) {
      EXPECT_EQ(err.line(), lines) << threads;
    }
  }
}

TEST(WriteEdgeList, RoundTrip) {
  temp_dir tmp;
  dyng::edge_list<std::int32_t, std::int32_t> e;
  e.num_vertices = 4;
  e.num_weights = 2;
  e.add_edge(0, 3, {1, 2});
  e.add_edge(2, 2, {3, 4});
  e.add_edge(1, 0, {-5, 6});
  dyng::io::write_edge_list(tmp.path("out/w.txt"), e.view());
  EXPECT_EQ(read_text(tmp.path("out/w.txt")), "0 3 1 2\n2 2 3 4\n1 0 -5 6\n");
  edge_list_options options;
  options.num_weights = 2;
  options.ids = vertex_ids::as_is;
  options.drop_self_loops = false;
  const auto back =
      dyng::io::read_edge_list<std::int32_t, std::int32_t>(tmp.path("out/w.txt"), options);
  EXPECT_EQ(back.src, (std::vector<std::int32_t>{0, 1, 2}));
  EXPECT_EQ(back.dst, (std::vector<std::int32_t>{3, 0, 2}));
  EXPECT_EQ(back.weights, (std::vector<std::int32_t>{1, 2, -5, 6, 3, 4}));

  dyng::edge_list<std::int32_t, unweighted> u;
  u.num_vertices = 2;
  u.add_edge(1, 0);
  dyng::io::write_edge_list(tmp.path("u.txt"), u.view());
  EXPECT_EQ(read_text(tmp.path("u.txt")), "1 0\n");
  dyng::edge_list<std::int32_t, std::int32_t> bad = e;
  bad.dst.pop_back();
  EXPECT_THROW(dyng::io::write_edge_list(tmp.path("bad.txt"), bad.view()),
               dyng::invalid_argument_error);
}

}  // namespace
