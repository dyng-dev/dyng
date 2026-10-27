// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_io_test.cpp
 * @brief `.hg` files (read_hypergraph, write_hypergraph) and the hypergraph batches of `.dgt`
 *        files (read_hyperedge_batches, write_hyperedge_batches): the line rules of both
 *        originals' loaders, the option sets, the relabellings, the weights, the errors with
 *        their locations, and the round trips.
 */
#include "support/data_paths.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/types.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/hypergraph_io.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using dyng::test::read_text;
using dyng::test::temp_dir;
using dyng::test::write_text;
using rows = std::vector<std::vector<std::int32_t>>;

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

template <typename list_t>
rows rows_of(const list_t& l) {
  rows out;
  for (std::size_t i = 0; i + 1 < l.offsets.size(); ++i) {
    out.emplace_back(l.members.begin() + l.offsets[i], l.members.begin() + l.offsets[i + 1]);
  }
  return out;
}

class HypergraphFile : public ::testing::Test {
 protected:
  std::string file(const std::string& text) {
    const std::string path = tmp_.path("h" + std::to_string(count_++) + ".hg");
    write_text(path, text);
    return path;
  }
  template <typename weight_t = dyng::unweighted>
  dyng::hyperedge_list<std::int32_t, weight_t> read(
      const std::string& text, const dyng::io::hypergraph_read_options& o = {},
      dyng::io::hypergraph_file_info* info = nullptr) {
    return dyng::io::read_hypergraph<std::int32_t, weight_t>(file(text), o, info);
  }
  temp_dir tmp_;
  int count_ = 0;
};

TEST_F(HypergraphFile, ReadsTheLineRulesOfBothOriginals) {
  // Comments only at the first character; separators space, tab, CR and comma; blank lines and
  // lines of separators skipped; ids sorted and deduplicated per line; repeated lines kept.
  dyng::io::hypergraph_file_info info;
  const auto l = read("% header\n3 1 2\n\n#x\n \t,\r\n5,4\t4\r\n1 2 3\n7", {}, &info);
  EXPECT_EQ(rows_of(l), (rows{{0, 1, 2}, {3, 4}, {0, 1, 2}, {6}}));
  EXPECT_EQ(l.num_vertices, 7);  // the largest id (ESCHER-GPU's maxVertexId)
  EXPECT_TRUE(l.weights.empty());
  EXPECT_EQ(info.lines, 4);
  EXPECT_EQ(info.repeated_members, 1);
  EXPECT_TRUE(info.external_ids.empty());
  EXPECT_EQ(read("").num_hyperedges(), 0);
  EXPECT_EQ(read("").num_vertices, 0);
}

TEST_F(HypergraphFile, EscherGpuOptionsKeepIdsAndRepeatedHyperedges) {
  const auto o = dyng::io::hypergraph_read_options::escher_gpu();
  EXPECT_EQ(o.index_base, 1);
  EXPECT_FALSE(o.dedupe_hyperedges);
  EXPECT_EQ(o.max_cardinality, -1);
  EXPECT_EQ(o.relabel, dyng::io::vertex_relabel::none);
  const auto l = read("10 2\n2 10\n", o);
  EXPECT_EQ(rows_of(l), (rows{{1, 9}, {1, 9}}));
  EXPECT_EQ(l.num_vertices, 10);  // vertices 1, 3..9 of the file stay isolated
  // A zero-based file and a given vertex count.
  dyng::io::hypergraph_read_options z;
  z.index_base = 0;
  z.num_vertices = 20;
  EXPECT_EQ(rows_of(read("0 3\n", z)), (rows{{0, 3}}));
  EXPECT_EQ(read("0 3\n", z).num_vertices, 20);
  // The two deliberate differences from ESCHER-GPU's loader (M8b review; ADR 0045 section 1):
  // its largest id is 2^31 - 2 (kMaxVertexId, its keys are ids + 1), dynG's the largest that
  // leaves the vertex count in int32, 2^31 - 1; and an empty file is a hypergraph of no
  // hyperedges (A fails: "no hyperedges").
  const auto top = read("2147483646\n", o);
  EXPECT_EQ(rows_of(top), (rows{{2147483645}}));
  EXPECT_EQ(read("2147483647\n", o).num_vertices, 2147483647);
  const auto beyond = expect_io_error([&] { (void)read("2147483648\n", o); });
  EXPECT_NE(std::string(beyond.what()).find("[1, 2147483647]"), std::string::npos) << beyond.what();
  for (const char* empty : {"", "% only a comment\n", "\n\n"}) {
    const auto none = read(empty, o);
    EXPECT_EQ(none.num_hyperedges(), 0);
    EXPECT_EQ(none.num_vertices, 0);
  }
}

TEST_F(HypergraphFile, RelabelsDenselyOrByFirstAppearance) {
  dyng::io::hypergraph_read_options o;
  o.relabel = dyng::io::vertex_relabel::dense_sorted;
  dyng::io::hypergraph_file_info info;
  const auto d = read("900 -5 40\n40 7\n", o, &info);
  EXPECT_EQ(rows_of(d), (rows{{0, 2, 3}, {1, 2}}));
  EXPECT_EQ(info.external_ids, (std::vector<std::int64_t>{-5, 7, 40, 900}));
  EXPECT_EQ(d.num_vertices, 4);
  o.relabel = dyng::io::vertex_relabel::first_appearance;
  const auto f = read("900 -5 40\n40 7\n", o, &info);
  // Within a line in ascending file id: -5, 40, 900, then 7 (MOSP_ESCHER's loadHypergraphFile).
  EXPECT_EQ(rows_of(f), (rows{{0, 1, 2}, {3, 1}}));
  EXPECT_EQ(info.external_ids, (std::vector<std::int64_t>{-5, 40, 900, 7}));
  EXPECT_EQ(f.num_vertices, 4);
}

TEST_F(HypergraphFile, DropsLargeAndRepeatedHyperedges) {
  dyng::io::hypergraph_read_options o;
  o.max_cardinality = 2;
  o.dedupe_hyperedges = true;
  dyng::io::hypergraph_file_info info;
  const auto l = read("1 2 3\n2 1\n4\n1 2 2\n4\n3 3 3 3\n", o, &info);
  EXPECT_EQ(rows_of(l), (rows{{0, 1}, {3}, {2}}));
  EXPECT_EQ(info.lines, 6);
  EXPECT_EQ(info.over_cardinality, 1);
  EXPECT_EQ(info.repeated_hyperedges, 2);
  EXPECT_EQ(info.repeated_members, 4);
  EXPECT_EQ(l.num_vertices, 4);
}

TEST_F(HypergraphFile, ReadsWeightsFromTheFile) {
  const auto l = read<std::int32_t>("1 2 ; 7\n3;9\n2,1 ;5\n");
  EXPECT_EQ(rows_of(l), (rows{{0, 1}, {2}, {0, 1}}));
  EXPECT_EQ(l.weights, (std::vector<std::int32_t>{7, 9, 5}));
}

TEST_F(HypergraphFile, DrawsHsospsWeightsAfterItsTargetDraw) {
  const auto o = dyng::io::hypergraph_read_options::hsosp(42, 3);
  EXPECT_EQ(o.max_cardinality, 3);
  EXPECT_EQ(o.relabel, dyng::io::vertex_relabel::first_appearance);
  ASSERT_TRUE(o.random_weights.has_value());
  const auto l = read<std::int32_t>("5 6\n1 2 3 4\n6 7 8\n9\n", o);
  ASSERT_EQ(l.num_hyperedges(), 3);
  EXPECT_EQ(l.num_vertices, 5);
  // MOSP_ESCHER@4b86159 loadHypergraphFile: std::mt19937_64(seed); the target vertex from
  // uniform_int_distribution<int>(0, n - 1); then uniform_int_distribution<long long>(1, 100) per
  // kept hyperedge. The same draws through the standard library (libstdc++ is what the original
  // was built with; the library reproduces its algorithm, so other libraries may differ here).
#if defined(__GLIBCXX__)
  std::mt19937_64 rng(42);
  (void)std::uniform_int_distribution<int>(0, 4)(rng);
  std::uniform_int_distribution<long long> w(1, 100);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(l.weights[i], w(rng)) << i;
  }
#endif
  // Without the target draw the stream starts one draw earlier.
  auto no_draw = o;
  no_draw.random_weights->hsosp_target_draw = false;
  const auto m = read<std::int32_t>("5 6\n1 2 3 4\n6 7 8\n9\n", no_draw);
  EXPECT_NE(m.weights, l.weights);
  for (const auto x : l.weights) {
    EXPECT_GE(x, 1);
    EXPECT_LE(x, 100);
  }
}

TEST_F(HypergraphFile, RejectsMalformedFilesWithTheirLocation) {
  struct bad_case {
    const char* text;
    bool weighted;
    std::int64_t line;
    std::int64_t column;
  };
  const bad_case cases[] = {
      {"1 2\n3 x\n", false, 2, 3},         // not an integer
      {"1 2\n3 12abc\n", false, 2, 3},     // not a whole integer
      {"0 1\n", false, 1, 1},              // below index_base 1
      {"1 -2\n", false, 1, 3},             // negative under relabel none
      {"1 +2\n", false, 1, 3},             // '+' is not accepted
      {"1 2147483648\n", false, 1, 3},     // the vertex count would not fit int32
      {"1 2 ; 3\n", false, 1, 5},          // a weight in an unweighted read
      {"1 2\n", true, 1, 0},               // a weighted read needs weights
      {"1 2 ;\n", true, 1, 5},             // ';' without a weight
      {"1 2 ; 0\n", true, 1, 7},           // weight below 1
      {"1 2 ; 3 4\n", true, 1, 9},         // more after the weight
      {"; 3\n", true, 1, 1},               // a weight without vertices
      {"1 ; ; 3\n", true, 1, 3},           // two ';'
      {"1 2 ; 4294967296\n", true, 1, 7},  // weight out of int32
  };
  for (const bad_case& c : cases) {
    SCOPED_TRACE(c.text);
    const auto e = expect_io_error([&] {
      if (c.weighted) {
        (void)read<std::int32_t>(c.text);
      } else {
        (void)read(c.text);
      }
    });
    EXPECT_EQ(e.line(), c.line);
    EXPECT_EQ(e.column(), c.column);
    EXPECT_NE(std::string(e.what()).find("dyng: "), std::string::npos);
  }
  // A file with weights and random weights: the file's weights conflict.
  auto o = dyng::io::hypergraph_read_options::hsosp();
  EXPECT_THROW((void)read<std::int32_t>("1 2 ; 3\n", o), dyng::io_error);
  // Missing file.
  EXPECT_THROW(
      (void)(dyng::io::read_hypergraph<std::int32_t, dyng::unweighted>(tmp_.path("missing.hg"))),
      dyng::io_error);
  // A directory is an io_error (not out_of_memory_error from its ftell() size; M8b review), for
  // the .hg reader and the hypergraph .dgt reader.
  const auto dir = expect_io_error(
      [&] { (void)(dyng::io::read_hypergraph<std::int32_t, dyng::unweighted>(tmp_.path(""))); });
  EXPECT_NE(std::string(dir.what()).find("directory"), std::string::npos) << dir.what();
  const auto dir_batches = expect_io_error(
      [&] { (void)(dyng::io::read_hyperedge_batches<std::int32_t, std::int32_t>(tmp_.path(""))); });
  EXPECT_NE(std::string(dir_batches.what()).find("directory"), std::string::npos)
      << dir_batches.what();
  // Invalid options.
  dyng::io::hypergraph_read_options bad;
  bad.index_base = -1;
  EXPECT_THROW((void)read("1\n", bad), dyng::invalid_argument_error);
  bad = {};
  bad.max_cardinality = -2;
  EXPECT_THROW((void)read("1\n", bad), dyng::invalid_argument_error);
  bad = {};
  bad.random_weights = dyng::io::weight_spec{};
  EXPECT_THROW((void)read("1\n", bad), dyng::invalid_argument_error);  // unweighted type
  bad.random_weights->lo = 0;
  EXPECT_THROW((void)read<std::int32_t>("1\n", bad), dyng::invalid_argument_error);
}

// The common mistakes of a user name their remedy (R030 review): the weighted read in each
// interface, the index base of a 0-based file, and the files of the other formats, which are
// refused at their first line (a Matrix Market file was read as a hypergraph without an error).
TEST_F(HypergraphFile, NamesTheRemedyOfCommonMistakes) {
  const auto has = [](const dyng::io_error& e, const char* text) {
    return std::string(e.what()).find(text) != std::string::npos;
  };
  const auto weighted = expect_io_error([&] { (void)read("1 2 3 ; 5\n"); });
  EXPECT_TRUE(has(weighted, "weighted=True") && has(weighted, "--weighted")) << weighted.what();
  const auto zero = expect_io_error(
      [&] { (void)read("0 1 2\n", dyng::io::hypergraph_read_options::escher_gpu()); });
  EXPECT_TRUE(has(zero, "index_base=0") && has(zero, "--index-base 0")) << zero.what();
  // An id above the range has no such hint.
  dyng::io::hypergraph_read_options few;
  few.num_vertices = 3;
  const auto above = expect_io_error([&] { (void)read("1 9\n", few); });
  EXPECT_FALSE(has(above, "index_base")) << above.what();
  struct other_format {
    std::string text;
    const char* names;
  };
  const other_format others[] = {
      {std::string("DYNGBAT1\x01\n\0\0\x07rest", 16), ".dgb batch file"},
      {"%dgt 1 hypergraph\n%batch 0\n+h 0 1\n", ".dgt batch file"},
      {"%DGT\n", ".dgt batch file"},
      {"%%MatrixMarket matrix coordinate pattern general\n4 4 3\n1 2\n2 3\n3 4\n",
       "Matrix Market file"},
      {"%%matrixmarket matrix coordinate real symmetric\n", "Matrix Market file"},
  };
  for (const other_format& f : others) {
    SCOPED_TRACE(f.names);
    const auto e = expect_io_error([&] { (void)read(f.text); });
    EXPECT_TRUE(has(e, f.names)) << e.what();
    EXPECT_EQ(e.line(), 1);
  }
  // A comment that only resembles them is still a comment.
  EXPECT_EQ(read("%dgtx\n% MatrixMarket\n#DYNGBAT1\n1 2\n").num_hyperedges(), 1);
  // A .dgt line without the header: the first token names the batch reader.
  const auto op = expect_io_error([&] { (void)read("+h 0 1\n"); });
  EXPECT_TRUE(has(op, "read_hyperedge_batches")) << op.what();
  // A binary token is quoted with its control bytes escaped and cut after 32 bytes: the
  // message does not end at a NUL.
  const auto binary = expect_io_error(
      [&] { (void)read(std::string("1 x\x01\0y", 6) + std::string(40, 'z') + "\n"); });
  const std::string what = binary.what();
  EXPECT_NE(what.find("'x\\x01\\x00y"), std::string::npos) << what;
  EXPECT_NE(what.find("...' (expected a decimal integer)"), std::string::npos) << what;
}

TEST_F(HypergraphFile, RoundTripsThroughTheWriter) {
  dyng::hyperedge_list<std::int32_t, std::int32_t> l;
  l.num_vertices = 9;
  l.add_hyperedge({0, 4, 8}, 3);
  l.add_hyperedge({2}, 1);
  l.add_hyperedge({8, 1}, 2147483647);
  const std::string path = tmp_.path("out/w.hg");
  dyng::io::write_hypergraph(path, l.view());
  EXPECT_EQ(read_text(path), "1 5 9 ; 3\n3 ; 1\n9 2 ; 2147483647\n");
  const auto back = dyng::io::read_hypergraph<std::int32_t, std::int32_t>(path);
  EXPECT_EQ(rows_of(back), (rows{{0, 4, 8}, {2}, {1, 8}}));  // sorted by the reader
  EXPECT_EQ(back.weights, l.weights);
  EXPECT_EQ(back.num_vertices, 9);
  // Zero-based, unweighted.
  dyng::hyperedge_list<std::int32_t> u;
  u.num_vertices = 3;
  u.add_hyperedge({0, 2});
  dyng::io::write_hypergraph(path, u.view(), 0);
  EXPECT_EQ(read_text(path), "0 2\n");
  // What the writer cannot write.
  dyng::hyperedge_list<std::int32_t> empty_edge;
  empty_edge.num_vertices = 2;
  empty_edge.add_hyperedge(dyng::array_view<const std::int32_t>{});
  EXPECT_THROW(dyng::io::write_hypergraph(path, empty_edge.view()), dyng::invalid_argument_error);
  dyng::hyperedge_list<std::int32_t> out_of_range;
  out_of_range.num_vertices = 2;
  out_of_range.add_hyperedge({2});
  EXPECT_THROW(dyng::io::write_hypergraph(path, out_of_range.view()), dyng::invalid_argument_error);
  dyng::hyperedge_list<std::int32_t, std::int32_t> zero_weight;
  zero_weight.num_vertices = 2;
  zero_weight.add_hyperedge({1}, 0);
  EXPECT_THROW(dyng::io::write_hypergraph(path, zero_weight.view()), dyng::invalid_argument_error);
}

TEST(HypergraphFileNames, EveryRelabelHasAName) {
  EXPECT_EQ(dyng::io::to_string(dyng::io::vertex_relabel::none), "none");
  EXPECT_EQ(dyng::io::to_string(dyng::io::vertex_relabel::dense_sorted), "dense_sorted");
  EXPECT_EQ(dyng::io::to_string(dyng::io::vertex_relabel::first_appearance), "first_appearance");
}

// ---------------------------------------------------------------------------------------------
// Hypergraph batches in .dgt files
// ---------------------------------------------------------------------------------------------

class HyperedgeBatchText : public HypergraphFile {
 protected:
  template <typename weight_t = dyng::unweighted>
  std::vector<dyng::hyperedge_batch<std::int32_t, weight_t>> read_batches(const std::string& text,
                                                                          std::int64_t n = -1) {
    dyng::io::hyperedge_batch_file_options o;
    o.num_vertices = n;
    return dyng::io::read_hyperedge_batches<std::int32_t, weight_t>(file(text), o);
  }
};

TEST_F(HyperedgeBatchText, ReadsHyperedgeOperations) {
  const auto b = read_batches<std::int32_t>(
      "%dgt 1 hypergraph\n# c\n%batch 0\n-h 4\n+h 0 2 1 ; 7\n+h 3;9\n%batch 5\n\n-h 0\n-h 0\n");
  ASSERT_EQ(b.size(), 2u);
  EXPECT_EQ(b[0].delete_hyperedges(), (std::vector<std::int64_t>{4}));
  EXPECT_EQ(b[0].insert_offsets(), (std::vector<std::int64_t>{0, 3, 4}));
  EXPECT_EQ(b[0].insert_members(), (std::vector<std::int32_t>{0, 2, 1, 3}));  // file order kept
  EXPECT_EQ(b[0].insert_weights(), (std::vector<std::int32_t>{7, 9}));
  EXPECT_EQ(b[1].delete_hyperedges(), (std::vector<std::int64_t>{0, 0}));  // kept for apply()
  EXPECT_EQ(b[1].num_insertions(), 0u);
  // Unweighted, without a header and %batch lines: one batch; repeated members kept.
  const auto u = read_batches("+h 1 1 0\n-h 9223372036854775807\n");
  ASSERT_EQ(u.size(), 1u);
  EXPECT_EQ(u[0].insert_members(), (std::vector<std::int32_t>{1, 1, 0}));
  EXPECT_EQ(u[0].delete_hyperedges(),
            (std::vector<std::int64_t>{std::numeric_limits<std::int64_t>::max()}));
}

TEST_F(HyperedgeBatchText, RejectsMalformedFiles) {
  struct bad_case {
    const char* text;
    bool weighted;
    std::int64_t line;
  };
  const bad_case cases[] = {
      {"+e 0 1\n", false, 1},              // graph operations in a hypergraph file
      {"+h 0\n-e 0 1\n", false, 2},        // the kinds do not mix
      {"%dgt 1 graph\n", false, 1},        // a graph file
      {"+i 0 1\n", false, 1},              // incidence edits: 0.4
      {"-i 0 1\n", false, 1},              // incidence edits: 0.4
      {"+h\n", false, 1},                  // no member
      {"+h ; 3\n", true, 1},               // no member
      {"+h 0 9\n", false, 1},              // member out of range (n = 4)
      {"+h 0 -1\n", false, 1},             // negative member
      {"-h -1\n", false, 1},               // negative id
      {"-h\n", false, 1},                  // incomplete
      {"-h 1 2\n", false, 1},              // extra token
      {"+h 0 1 ; 3\n", false, 1},          // weight in an unweighted batch
      {"+h 0 1\n", true, 1},               // weighted batch without weight
      {"+h 0 1 ;\n", true, 1},             // ';' without weight
      {"+h 0 1 ; 0\n", true, 1},           // weight below 1
      {"+h 0 1 ; 3 4\n", true, 1},         // extra after the weight
      {"%batch 0\n%batch 0\n", false, 2},  // ids not increasing
      {"+h 0\n%batch 1\n", false, 2},      // operations before the first %batch
      {"+v 1\n", false, 1},                // vertex operations: 0.4
  };
  for (const bad_case& c : cases) {
    SCOPED_TRACE(c.text);
    const auto e = expect_io_error([&] {
      if (c.weighted) {
        (void)read_batches<std::int32_t>(c.text, 4);
      } else {
        (void)read_batches(c.text, 4);
      }
    });
    EXPECT_EQ(e.line(), c.line);
  }
  const auto e = expect_io_error([&] { (void)read_batches("+i 0 1\n"); });
  EXPECT_NE(std::string(e.what()).find("0.4"), std::string::npos) << e.what();
  // A member outside the hypergraph's vertex set says that the count is fixed (R030 review).
  const auto outside = expect_io_error([&] { (void)read_batches("+h 0 99\n", 4); });
  EXPECT_NE(std::string(outside.what())
                .find("vertex id 99 out of range [0, 3]: the hypergraph "
                      "has 4 vertices, fixed when it is built"),
            std::string::npos)
      << outside.what();
  // An edge reader names the hypergraph reader.
  const auto g = expect_io_error([&] {
    (void)dyng::io::read_batches<std::int32_t, std::int32_t>(file("%dgt 1 hypergraph\n"));
  });
  EXPECT_NE(std::string(g.what()).find("read_hyperedge_batches"), std::string::npos) << g.what();
}

TEST_F(HyperedgeBatchText, RoundTripsThroughTheWriter) {
  dyng::hyperedge_batch<std::int32_t, std::int32_t> a;
  dyng::hyperedge_batch<std::int32_t, std::int32_t> b;
  dyng::hyperedge_batch<std::int32_t, std::int32_t> c;
  a.delete_hyperedge(3);
  a.insert_hyperedge({4, 1, 1}, 6);
  b.insert_hyperedge({0}, 1);
  b.delete_hyperedge(0);
  b.delete_hyperedge(0);
  const std::string path = tmp_.path("hb.dgt");
  dyng::io::write_hyperedge_batches<std::int32_t, std::int32_t>(path,
                                                                {a.view(), b.view(), c.view()});
  EXPECT_EQ(read_text(path),
            "%dgt 1 hypergraph\n%batch 0\n-h 3\n+h 4 1 1 ; 6\n%batch 1\n-h 0\n-h 0\n+h 0 ; 1\n"
            "%batch 2\n");
  const auto back = dyng::io::read_hyperedge_batches<std::int32_t, std::int32_t>(path);
  ASSERT_EQ(back.size(), 3u);
  EXPECT_EQ(back[0].insert_members(), a.insert_members());
  EXPECT_EQ(back[0].insert_offsets(), a.insert_offsets());
  EXPECT_EQ(back[0].insert_weights(), a.insert_weights());
  EXPECT_EQ(back[1].delete_hyperedges(), b.delete_hyperedges());
  EXPECT_TRUE(back[2].empty());
  // What the writer refuses (the file would not read back, or it cannot hold it).
  dyng::hyperedge_batch<std::int32_t, std::int32_t> inc;
  inc.insert_incidence(0, 1);
  using wvec = std::vector<dyng::hyperedge_batch_view<std::int32_t, std::int32_t>>;
  EXPECT_THROW(dyng::io::write_hyperedge_batches(path, wvec{inc.view()}),
               dyng::not_supported_error);
  dyng::hyperedge_batch<std::int32_t, std::int32_t> neg;
  neg.delete_hyperedge(-1);
  EXPECT_THROW(dyng::io::write_hyperedge_batches(path, wvec{neg.view()}),
               dyng::invalid_argument_error);
  dyng::hyperedge_batch<std::int32_t, std::int32_t> empty_edge;
  empty_edge.insert_hyperedge(dyng::array_view<const std::int32_t>{}, 1);
  EXPECT_THROW(dyng::io::write_hyperedge_batches(path, wvec{empty_edge.view()}),
               dyng::invalid_argument_error);
  dyng::hyperedge_batch<std::int32_t, std::int32_t> zero_weight;
  zero_weight.insert_hyperedge({1}, 0);
  EXPECT_THROW(dyng::io::write_hyperedge_batches(path, wvec{zero_weight.view()}),
               dyng::invalid_argument_error);
}

}  // namespace
