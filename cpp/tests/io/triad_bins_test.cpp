// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_bins_test.cpp
 * @brief The triad-bins JSON document (io::write_triad_bins_json(), io::read_triad_bins_json()):
 *        the exact text of the writer, round trips (with and without deltas, the extremes of
 *        the value ranges, a real triad_count result), what the reader accepts (any JSON
 *        layout, escapes, unknown keys) and the errors it reports with line and column.
 */
#include "support/data_paths.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/io/result_io.hpp>
#include <dyng/triad_count.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

using dyng::test::read_text;
using dyng::test::temp_dir;
using dyng::test::write_text;
namespace io = dyng::io;

io::triad_bins motif_bins() {
  io::triad_bins b;
  b.kind = "h_motif";
  b.graph_version = 3;
  b.counts.assign(30, 0);
  b.counts[0] = 1;
  b.counts[3] = 89258;
  b.counts[29] = std::numeric_limits<std::uint64_t>::max();
  return b;
}

std::string text_of(const io::triad_bins& b) {
  std::ostringstream out;
  io::write_triad_bins_json(out, b);
  return out.str();
}

void expect_equal(const io::triad_bins& a, const io::triad_bins& b) {
  EXPECT_EQ(a.kind, b.kind);
  EXPECT_EQ(a.graph_version, b.graph_version);
  EXPECT_EQ(a.counts, b.counts);
  EXPECT_EQ(a.deltas, b.deltas);
}

/// Read `text` from a file; expect an io_error whose message contains `where` and `what`.
void expect_rejected(const std::string& text, const std::string& where, const std::string& what) {
  temp_dir dir;
  const std::string path = dir.path("bins.json");
  write_text(path, text);
  try {
    (void)io::read_triad_bins_json(path);
    ADD_FAILURE() << "accepted: " << text;
  } catch (const dyng::io_error& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find(path + ":" + where), std::string::npos) << message;
    EXPECT_NE(message.find(what), std::string::npos) << message;
  }
}

TEST(TriadBins, KindsAreTriadCountsPatterns) {
  namespace tc = dyng::triad_count;
  for (const tc::pattern p : {tc::pattern::h_motif, tc::pattern::stathyper_type1,
                              tc::pattern::stathyper_type2, tc::pattern::stathyper_type3}) {
    const int want = p == tc::pattern::h_motif ? tc::max_bins : 1;
    EXPECT_EQ(io::triad_bins_size(std::string(tc::to_string(p))), want);
  }
  EXPECT_THROW((void)io::triad_bins_size("open_triads"), dyng::invalid_argument_error);
}

TEST(TriadBins, WriterText) {
  io::triad_bins b;
  b.kind = "stathyper_type1";
  b.graph_version = 2;
  b.counts = {47093};
  EXPECT_EQ(text_of(b),
            "{\n  \"format\": \"dyng.triad_bins\",\n  \"version\": 1,\n  \"kind\": "
            "\"stathyper_type1\",\n  \"num_bins\": 1,\n  \"graph_version\": 2,\n  \"counts\": "
            "[47093]\n}\n");
  b.deltas = {{-60}, {-164}, {27}};
  EXPECT_EQ(text_of(b),
            "{\n  \"format\": \"dyng.triad_bins\",\n  \"version\": 1,\n  \"kind\": "
            "\"stathyper_type1\",\n  \"num_bins\": 1,\n  \"graph_version\": 2,\n  \"counts\": "
            "[47093],\n  \"deltas\": [\n    [-60],\n    [-164],\n    [27]\n  ]\n}\n");
}

TEST(TriadBins, RoundTrip) {
  temp_dir dir;
  io::triad_bins b = motif_bins();
  const std::string path = dir.path("sub/bins.json");
  io::write_triad_bins_json(path, b);
  EXPECT_EQ(read_text(path), text_of(b));
  expect_equal(io::read_triad_bins_json(path), b);

  b.deltas.assign(2, std::vector<std::int64_t>(30, 0));
  b.deltas[0][0] = std::numeric_limits<std::int64_t>::min();
  b.deltas[1][29] = std::numeric_limits<std::int64_t>::max();
  io::write_triad_bins_json(path, b);
  expect_equal(io::read_triad_bins_json(path), b);
}

TEST(TriadBins, RoundTripOfAResult) {
  // A result's bins: the pattern's name, its version and its counts.
  const dyng::resources res = dyng::resources::sequential();
  using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;
  dyng::hyperedge_list<std::int32_t, dyng::unweighted> list;
  list.num_vertices = 5;
  list.add_hyperedge({0, 1, 2});
  list.add_hyperedge({1, 2, 3});
  list.add_hyperedge({2, 3, 4});
  list.add_hyperedge({0, 2, 4});
  const hgraph h = hgraph::from_hyperedges(res, list.view());
  const dyng::triad_count::result r = dyng::triad_count::compute(res, h);
  io::triad_bins b;
  b.kind = std::string(dyng::triad_count::to_string(r.kind()));
  b.graph_version = r.graph_version();
  b.counts.assign(r.counts().begin(), r.counts().end());
  temp_dir dir;
  io::write_triad_bins_json(dir.path("r.json"), b);
  const io::triad_bins back = io::read_triad_bins_json(dir.path("r.json"));
  expect_equal(back, b);
  EXPECT_GT(r.total(), 0U);
}

TEST(TriadBins, ReaderAcceptsAnyJsonLayout) {
  temp_dir dir;
  const std::string path = dir.path("bins.json");
  // Keys in another order, no whitespace, escapes in strings, unknown keys of every value kind.
  write_text(path,
             "{\"counts\":[5],\"kind\":\"stathyper_\\u0074ype2\",\"extra\":{\"a\":[1,2.5e3,null,"
             "true,false,\"\\\"\\\\\\/\\b\\f\\n\\r\\t\\ud83d\\ude00\"]},\"num_bins\":1,"
             "\"version\":1,\"format\":\"dyng.triad_bins\"}");
  const io::triad_bins b = io::read_triad_bins_json(path);
  EXPECT_EQ(b.kind, "stathyper_type2");
  EXPECT_EQ(b.graph_version, 0U);  // optional
  EXPECT_EQ(b.counts, std::vector<std::uint64_t>{5});
  EXPECT_TRUE(b.deltas.empty());
}

TEST(TriadBins, WriterChecksItsInput) {
  std::ostringstream out;
  io::triad_bins b = motif_bins();
  b.kind = "motifs";
  EXPECT_THROW(io::write_triad_bins_json(out, b), dyng::invalid_argument_error);
  b = motif_bins();
  b.counts.pop_back();
  EXPECT_THROW(io::write_triad_bins_json(out, b), dyng::invalid_argument_error);
  b = motif_bins();
  b.deltas = {std::vector<std::int64_t>(29, 0)};
  EXPECT_THROW(io::write_triad_bins_json(out, b), dyng::invalid_argument_error);
  EXPECT_TRUE(out.str().empty());  // nothing written on an error
}

TEST(TriadBins, ReaderErrors) {
  const std::string head = "{\"format\": \"dyng.triad_bins\", \"version\": 1, ";
  expect_rejected("", "1:1", "expected a JSON value");
  expect_rejected("[1]", "1:1", "one JSON object");
  expect_rejected("{\"version\": 1}", "1:1", "missing key \"format\"");
  expect_rejected("{\"format\": \"other\"}", "1:12", "\"format\" must be");
  expect_rejected("{\"format\": \"dyng.triad_bins\", \"version\": 2}", "1:42",
                  "unsupported triad-bins format version 2");
  expect_rejected(head + "\"kind\": \"motif\"}", "1:53", "\"kind\" must be one of");
  expect_rejected(head + "\"kind\": \"h_motif\", \"num_bins\": 1, \"counts\": []}", "1:76",
                  "\"num_bins\" must be 30");
  expect_rejected(head + "\"kind\": \"stathyper_type1\", \"num_bins\": 1}", "1:1",
                  "missing key \"counts\"");
  expect_rejected(head + "\"kind\": \"stathyper_type1\", \"num_bins\": 1, \"counts\": [1, 2]}",
                  "1:97", "\"counts\" has 2 values");
  expect_rejected(head + "\"kind\": \"stathyper_type1\", \"num_bins\": 1, \"counts\": [-1]}",
                  "1:98", "must not be negative");
  expect_rejected(head + "\"kind\": \"stathyper_type1\", \"num_bins\": 1, \"counts\": [1.0]}",
                  "1:98", "no fraction or exponent");
  expect_rejected(head +
                      "\"kind\": \"stathyper_type1\", \"num_bins\": 1, \"counts\": "
                      "[18446744073709551616]}",
                  "1:98", "out of range");
  expect_rejected(head +
                      "\"kind\": \"stathyper_type1\", \"num_bins\": 1, \"counts\": [1], "
                      "\"deltas\": [[-9223372036854775809]]}",
                  "1:114", "out of range");
  expect_rejected(head +
                      "\"kind\": \"stathyper_type1\", \"num_bins\": 1, \"counts\": [1], "
                      "\"deltas\": [[1, 2]]}",
                  "1:113", "a row of \"deltas\" has 2 values");
  expect_rejected(head + "\"kind\": \"h_motif\", \"kind\": \"h_motif\"}", "1:64",
                  "repeated key \"kind\"");
}

TEST(TriadBins, ReaderSyntaxErrors) {
  expect_rejected("{\"format\": \"dyng.triad_bins\"}\nx", "2:1", "unexpected text after");
  expect_rejected("{\"a\": 01}", "1:8", "',' or '}'");
  expect_rejected("{\"a\": -}", "1:8", "malformed number");
  expect_rejected("{\"a\": 1.}", "1:9", "no digit after '.'");
  expect_rejected("{\"a\": 1e}", "1:9", "no digit in the exponent");
  expect_rejected("{\"a\": \"x", "1:9", "unterminated string");
  expect_rejected("{\"a\": \"\t\"}", "1:8", "a control character");
  expect_rejected("{\"a\": \"\\x\"}", "1:9", "unknown escape");
  expect_rejected("{\"a\": \"\\u12\"}", "1:", "malformed \\u escape");
  expect_rejected("{\"a\": \"\\udc00\"}", "1:", "unpaired surrogate");
  expect_rejected("{\"a\": \"\\ud800x\"}", "1:", "unpaired surrogate");
  expect_rejected("{\"a\" 1}", "1:6", "expected ':'");
  expect_rejected("{1: 1}", "1:2", "expected a string key");
  expect_rejected("{\"a\": [1 2]}", "1:10", "',' or ']'");
  expect_rejected("{\"a\": nul}", "1:7", "expected a JSON value");
  expect_rejected(std::string(100, '[') + std::string(100, ']'), "1:65", "nested deeper than 64");
}

}  // namespace
