// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:utils/utils.cpp (loadHypergraph: the line rules)
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/HypergraphGen.cpp (loadHypergraphFile: the
// cardinality limit, the first-appearance numbering and the weight stream)
/**
 * @file hypergraph_io.cpp
 * @brief `.hg` hypergraph files (PLAN Section 5.7; docs/api/file_formats.md).
 *
 * A line is a comment if its FIRST character is `#` or `%` (ESCHER-GPU's loadHypergraph);
 * separators are space, tab, carriage return and comma (both originals); every line's ids are
 * sorted and deduplicated, and a line without ids is skipped (both originals). Then, per kept
 * line in file order: the cardinality limit and the vertex numbering by first appearance of
 * MOSP_ESCHER's loadHypergraphFile, the first-occurrence rule of the originals' converters, and
 * the weights of loadHypergraphFile (one std::mt19937_64(seed), its target-vertex draw, then
 * U[lo, hi] per kept hyperedge with the algorithm of libstdc++'s uniform_int_distribution,
 * reproduced in util/rng.hpp). The tokens are dynG's: a whole decimal integer with an optional
 * leading `-` (MOSP_ESCHER's loader also accepts `+` and splits `1-2` into two ids; dynG rejects
 * both).
 */
#include "hypergraph/instantiate.hpp"
#include "util/allocation.hpp"
#include "util/parser.hpp"
#include "util/rng.hpp"
#include "util/text_writer.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/io/hypergraph_io.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dyng::io {

std::string_view to_string(vertex_relabel relabel) noexcept {
  switch (relabel) {
    case vertex_relabel::none:
      return "none";
    case vertex_relabel::dense_sorted:
      return "dense_sorted";
    case vertex_relabel::first_appearance:
      return "first_appearance";
  }
  return "unknown";
}

namespace {

constexpr bool is_separator(char c) noexcept {
  return c == ' ' || c == '\t' || c == '\r' || c == ',';
}

/// FNV-1a 64 over the raw ids of a member set (the hash of the dedupe table).
std::uint64_t hash_ids(const std::int64_t* ids, std::size_t n) noexcept {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (std::size_t i = 0; i < n; ++i) {
    auto v = static_cast<std::uint64_t>(ids[i]);
    for (int b = 0; b < 8; ++b) {
      h ^= (v & 0xffU);
      h *= 0x100000001b3ULL;
      v >>= 8U;
    }
  }
  return h;
}

/// The largest weight of `weight_t` (0 for unweighted).
template <typename weight_t>
constexpr std::int64_t largest_weight() noexcept {
  if constexpr (is_unweighted_v<weight_t>) {
    return 0;
  } else {
    return detail::max_as_int64<weight_t>();
  }
}

/// One located token of a line.
struct hg_token {
  std::string_view text;
  std::int64_t column = 0;
};

/// An ASCII letter in lower case (any other byte unchanged).
constexpr char ascii_lower(char c) noexcept {
  return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
}

/// Whether `line` starts with `prefix`, ignoring the case of ASCII letters.
bool starts_with_nocase(std::string_view line, std::string_view prefix) noexcept {
  if (line.size() < prefix.size()) return false;
  for (std::size_t i = 0; i < prefix.size(); ++i) {
    if (ascii_lower(line[i]) != ascii_lower(prefix[i])) return false;
  }
  return true;
}

/// Refuse the files of the other formats that a user may pass for a `.hg` file (R030 review): a
/// `.dgb` batch file (its magic) and a `.dgt` batch file (its header line) would fail at their
/// first token with an unhelpful message, and a Matrix Market file would be read as a hypergraph
/// without an error (its banner is a comment line, its size line a hyperedge). ESCHER-GPU's
/// loader skips the two header lines as comments; no `.hg` file of either original starts with
/// them (docs/api/file_formats.md).
void refuse_other_formats(const std::string& path, std::string_view text) {
  const std::string_view first = text.substr(0, text.find('\n'));
  if (first.substr(0, 8) == "DYNGBAT1") {
    detail::throw_io_error(
        path, 1, 1,
        "this is a .dgb batch file (the magic DYNGBAT1), not a .hg hypergraph: read its batches "
        "with io::batch_reader (Python dyng.io.BatchReader; the command line's --batches)");
  }
  if (starts_with_nocase(first, "%dgt") && (first.size() == 4 || is_separator(first[4]))) {
    detail::throw_io_error(
        path, 1, 1,
        "this is a .dgt batch file (its '%dgt' header), not a .hg hypergraph: read its batches "
        "with io::read_hyperedge_batches (Python dyng.io.read_hyperedge_batches; the command "
        "line's --batches)");
  }
  if (starts_with_nocase(first, "%%MatrixMarket")) {
    detail::throw_io_error(
        path, 1, 1,
        "this is a Matrix Market file (its '%%MatrixMarket' banner), not a .hg hypergraph: a "
        "Matrix Market file holds a graph (io::read_matrix_market; Python "
        "dyng.io.read_matrix_market)");
  }
}

}  // namespace

template <typename vertex_t, typename weight_t>
hyperedge_list<vertex_t, weight_t> read_hypergraph(const std::string& path,
                                                   const hypergraph_read_options& options,
                                                   hypergraph_file_info* info) try {
  constexpr bool weighted = !is_unweighted_v<weight_t>;
  const std::int64_t weight_limit = largest_weight<weight_t>();
  DYNG_EXPECTS(
      options.index_base >= 0 && options.index_base <= std::numeric_limits<std::int64_t>::max() -
                                                           detail::max_as_int64<vertex_t>(),
      "hypergraph_read_options::index_base must be >= 0 (and leave room for the vertex "
      "ids in int64), got ",
      options.index_base);
  DYNG_EXPECTS(options.max_cardinality >= -1,
               "hypergraph_read_options::max_cardinality must be >= -1, got ",
               options.max_cardinality);
  DYNG_EXPECTS(
      options.num_vertices >= -1 && options.num_vertices <= detail::max_as_int64<vertex_t>(),
      "hypergraph_read_options::num_vertices must be -1 or in [0, ",
      detail::max_as_int64<vertex_t>(), "], got ", options.num_vertices);
  DYNG_EXPECTS(weighted || !options.random_weights.has_value(),
               "hypergraph_read_options::random_weights needs a weighted type");
  if (options.random_weights) {
    const weight_spec& ws = *options.random_weights;
    DYNG_EXPECTS(ws.lo >= 1 && ws.lo <= ws.hi && ws.hi <= weight_limit,
                 "hypergraph_read_options::random_weights: the range [", ws.lo, ", ", ws.hi,
                 "] must lie in [1, ", weight_limit, "]");
  }
  const bool relabel_none = options.relabel == vertex_relabel::none;
  // The largest vertex id of relabel none: below num_vertices, or such that the vertex count
  // (largest + 1) fits vertex_t.
  const std::int64_t vertex_limit =
      options.num_vertices >= 0 ? options.num_vertices - 1 : detail::max_as_int64<vertex_t>() - 1;
  const bool weights_from_file = weighted && !options.random_weights.has_value();

  const std::string text = detail::read_file(path);
  refuse_other_formats(path, text);
  hypergraph_file_info local_info;
  hypergraph_file_info& out_info = info != nullptr ? *info : local_info;
  out_info = hypergraph_file_info{};

  // The kept hyperedges: their (sorted, distinct) raw ids, back to back.
  std::vector<std::int64_t> raw;
  std::vector<std::int64_t> offsets{0};
  std::vector<weight_t> weights;
  std::unordered_map<std::uint64_t, std::vector<std::int64_t>> seen;  // hash -> kept indices
  std::vector<std::int64_t> ids;
  std::vector<hg_token> tokens;
  std::int64_t largest = -1;  // relabel none: the largest vertex id

  std::size_t pos = 0;
  std::int64_t line_no = 0;
  while (pos < text.size()) {
    std::size_t end = text.find('\n', pos);
    if (end == std::string::npos) end = text.size();
    ++line_no;
    const std::string_view line(text.data() + pos, end - pos);
    pos = end + 1;
    if (!line.empty() && (line.front() == '#' || line.front() == '%')) {
      continue;  // a comment (only at the first character, as ESCHER-GPU)
    }
    // Tokens: runs of characters other than the separators and ';' (a token of its own).
    tokens.clear();
    for (std::size_t i = 0; i < line.size();) {
      if (is_separator(line[i])) {
        ++i;
        continue;
      }
      const std::size_t begin = i;
      if (line[i] == ';') {
        ++i;
      } else {
        while (i < line.size() && !is_separator(line[i]) && line[i] != ';') ++i;
      }
      tokens.push_back({line.substr(begin, i - begin), static_cast<std::int64_t>(begin) + 1});
    }
    if (tokens.empty()) {
      continue;  // a blank line
    }
    const auto fail = [&](const hg_token& t, const std::string& message) {
      detail::throw_io_error(path, line_no, t.column, message);
    };
    ids.clear();
    std::size_t t = 0;
    for (; t < tokens.size() && tokens[t].text != ";"; ++t) {
      std::int64_t value = 0;
      if (!detail::parse_int64(tokens[t].text, value)) {
        const std::string_view x = tokens[t].text;
        const bool batch_op = x == "+h" || x == "-h" || x == "+e" || x == "-e";
        fail(tokens[t], "invalid vertex id " + detail::quoted_token(x) +
                            (batch_op ? " (a line of a .dgt batch file: read its batches with "
                                        "io::read_hyperedge_batches; the command line's --batches)"
                                      : " (expected a decimal integer)"));
      }
      if (relabel_none) {
        if (value < options.index_base || value - options.index_base > vertex_limit) {
          // An id below a positive base is most often a 0-based file read with the default
          // base 1: name the option in each interface (R030 review).
          const std::string hint = value >= 0 && value < options.index_base
                                       ? " (a 0-based file needs index_base 0: Python "
                                         "index_base=0, the command line --index-base 0)"
                                       : std::string();
          fail(tokens[t], "vertex id " + std::to_string(value) + " out of range [" +
                              std::to_string(options.index_base) + ", " +
                              std::to_string(options.index_base + vertex_limit) + "]" + hint);
        }
      }
      ids.push_back(value);
    }
    bool has_weight = false;
    std::int64_t weight_value = 0;
    if (t < tokens.size()) {  // `; w`
      const hg_token& semicolon = tokens[t];
      if (ids.empty()) fail(semicolon, "a weight ('; w') without vertices");
      if (t + 1 >= tokens.size() || tokens[t + 1].text == ";") {
        fail(semicolon, "';' must be followed by the hyperedge's weight");
      }
      if (t + 2 < tokens.size()) {
        fail(tokens[t + 2], "unexpected " + detail::quoted_token(tokens[t + 2].text) +
                                " after the weight (a line is 'v1 v2 .. [; w]')");
      }
      if (!weighted) {
        fail(semicolon,
             "the file has hyperedge weights; read it as weighted (C++: a weighted weight_t; "
             "Python: weighted=True; the command line: --weighted), or remove '; w'");
      }
      if (!weights_from_file) {
        fail(semicolon, "the file has hyperedge weights and random_weights is set");
      }
      if (!detail::parse_int64(tokens[t + 1].text, weight_value)) {
        fail(tokens[t + 1], "invalid weight " + detail::quoted_token(tokens[t + 1].text) +
                                " (expected a decimal integer)");
      }
      if (weight_value < 1 || weight_value > weight_limit) {
        fail(tokens[t + 1], "weight " + std::to_string(weight_value) + " out of range [1, " +
                                std::to_string(weight_limit) + "]");
      }
      has_weight = true;
    }
    if (ids.empty()) {
      continue;
    }
    if (weights_from_file && !has_weight) {
      detail::throw_io_error(
          path, line_no, 0, "a weighted read needs the weight of every hyperedge ('v1 v2 .. ; w')");
    }
    ++out_info.lines;
    std::sort(ids.begin(), ids.end());
    const auto last = std::unique(ids.begin(), ids.end());
    out_info.repeated_members += static_cast<std::int64_t>(ids.end() - last);
    ids.erase(last, ids.end());
    if (options.max_cardinality >= 0 &&
        static_cast<std::int64_t>(ids.size()) > options.max_cardinality) {
      ++out_info.over_cardinality;
      continue;
    }
    if (options.dedupe_hyperedges) {
      auto& bucket = seen[hash_ids(ids.data(), ids.size())];
      bool repeated = false;
      for (const std::int64_t k : bucket) {
        const auto b = static_cast<std::size_t>(offsets[static_cast<std::size_t>(k)]);
        const auto e = static_cast<std::size_t>(offsets[static_cast<std::size_t>(k) + 1]);
        if (e - b == ids.size() && std::equal(ids.begin(), ids.end(), raw.begin() + b)) {
          repeated = true;
          break;
        }
      }
      if (repeated) {
        ++out_info.repeated_hyperedges;
        continue;
      }
      bucket.push_back(static_cast<std::int64_t>(offsets.size()) - 1);
    }
    if (relabel_none) largest = std::max(largest, ids.back() - options.index_base);
    raw.insert(raw.end(), ids.begin(), ids.end());
    offsets.push_back(static_cast<std::int64_t>(raw.size()));
    if constexpr (weighted) {
      if (weights_from_file) weights.push_back(static_cast<weight_t>(weight_value));
    }
  }
  seen.clear();

  hyperedge_list<vertex_t, weight_t> out;
  out.offsets = std::move(offsets);
  out.members.resize(raw.size());
  std::int64_t n = 0;
  if (relabel_none) {
    n = options.num_vertices >= 0 ? options.num_vertices : largest + 1;
    for (std::size_t i = 0; i < raw.size(); ++i) {
      out.members[i] = static_cast<vertex_t>(raw[i] - options.index_base);
    }
  } else {
    std::vector<std::int64_t>& external = out_info.external_ids;
    if (options.relabel == vertex_relabel::dense_sorted) {
      external = raw;
      std::sort(external.begin(), external.end());
      external.erase(std::unique(external.begin(), external.end()), external.end());
    }
    n = options.relabel == vertex_relabel::dense_sorted ? static_cast<std::int64_t>(external.size())
                                                        : 0;
    if (n > detail::max_as_int64<vertex_t>()) {
      detail::throw_io_error(path, 0, 0,
                             "the file has " + std::to_string(n) +
                                 " distinct vertices, more than the vertex id type holds");
    }
    if (options.relabel == vertex_relabel::dense_sorted) {
      for (std::size_t i = 0; i < raw.size(); ++i) {
        out.members[i] = static_cast<vertex_t>(
            std::lower_bound(external.begin(), external.end(), raw[i]) - external.begin());
      }
    } else {
      std::unordered_map<std::int64_t, std::int64_t> number;
      number.reserve(raw.size() / 2 + 1);
      for (std::size_t i = 0; i < raw.size(); ++i) {
        const auto [it, fresh] = number.emplace(raw[i], n);
        if (fresh) {
          if (n >= detail::max_as_int64<vertex_t>()) {
            detail::throw_io_error(path, 0, 0,
                                   "the file has more distinct vertices than the vertex id type "
                                   "holds");
          }
          external.push_back(raw[i]);
          ++n;
        }
        out.members[i] = static_cast<vertex_t>(it->second);
      }
    }
  }
  out.num_vertices = static_cast<vertex_t>(n);
  if constexpr (weighted) {
    if (options.random_weights) {
      const weight_spec& ws = *options.random_weights;
      std::mt19937_64 engine(ws.seed);
      if (ws.hsosp_target_draw && n > 0) {
        (void)detail::legacy_uniform_int64(engine, 0, static_cast<std::uint64_t>(n - 1));
      }
      const auto span = static_cast<std::uint64_t>(ws.hi - ws.lo);
      weights.resize(out.offsets.size() - 1);
      for (auto& w : weights) {
        w = static_cast<weight_t>(
            ws.lo + static_cast<std::int64_t>(detail::legacy_uniform_int64(engine, 0, span)));
      }
    }
    out.weights = std::move(weights);
  }
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::read_hypergraph (", path, ")")

template <typename vertex_t, typename weight_t>
void write_hypergraph(const std::string& path,
                      const hyperedge_list_view<vertex_t, weight_t>& hyperedges,
                      std::int64_t index_base) try {
  constexpr bool weighted = !is_unweighted_v<weight_t>;
  const auto host = [](const auto& view) {
    return view.empty() || is_host_accessible(view.space());
  };
  DYNG_EXPECTS(index_base >= 0 && index_base <= std::numeric_limits<std::int64_t>::max() -
                                                    detail::max_as_int64<vertex_t>(),
               "write_hypergraph: index_base must be >= 0 (and leave room for the vertex ids in "
               "int64), got ",
               index_base);
  DYNG_EXPECTS(host(hyperedges.offsets) && host(hyperedges.members) && host(hyperedges.weights),
               "write_hypergraph: the hyperedges must be in host-accessible memory");
  const std::int64_t m = hyperedges.num_hyperedges();
  const auto num_members = static_cast<std::int64_t>(hyperedges.members.size());
  DYNG_EXPECTS(hyperedges.offsets.empty()
                   ? num_members == 0
                   : hyperedges.offsets[0] == 0 &&
                         hyperedges.offsets[hyperedges.offsets.size() - 1] == num_members,
               "write_hypergraph: the offsets must start at 0 and end at the member count");
  DYNG_EXPECTS(
      weighted ? static_cast<std::int64_t>(hyperedges.weights.size()) == m
               : hyperedges.weights.empty(),
      "write_hypergraph: ",
      weighted ? "one weight per hyperedge is needed" : "an unweighted list has no weights");
  if constexpr (weighted) {
    for (std::int64_t h = 0; h < m; ++h) {
      const auto w = static_cast<std::int64_t>(hyperedges.weights[static_cast<std::size_t>(h)]);
      DYNG_EXPECTS(w >= 1, "write_hypergraph: the weight ", w, " of hyperedge ", h,
                   " is below 1 (read_hypergraph() reads weights in [1, max])");
    }
  }
  const std::int64_t n = static_cast<std::int64_t>(hyperedges.num_vertices);
  for (std::int64_t h = 0; h < m; ++h) {
    const std::int64_t b = hyperedges.offsets[static_cast<std::size_t>(h)];
    const std::int64_t e = hyperedges.offsets[static_cast<std::size_t>(h) + 1];
    DYNG_EXPECTS(b < e && e <= num_members, "write_hypergraph: hyperedge ", h,
                 b == e ? " is empty (a .hg file cannot hold an empty hyperedge)"
                        : " has malformed offsets");
    for (std::int64_t i = b; i < e; ++i) {
      const auto v = static_cast<std::int64_t>(hyperedges.members[static_cast<std::size_t>(i)]);
      DYNG_EXPECTS(v >= 0 && v < n, "write_hypergraph: member ", v, " of hyperedge ", h,
                   " is outside [0, ", n, ")");
    }
  }
  detail::text_writer out(path);
  for (std::int64_t h = 0; h < m; ++h) {
    const std::int64_t b = hyperedges.offsets[static_cast<std::size_t>(h)];
    const std::int64_t e = hyperedges.offsets[static_cast<std::size_t>(h) + 1];
    for (std::int64_t i = b; i < e; ++i) {
      if (i > b) out.put_char(' ');
      out.put(static_cast<std::int64_t>(hyperedges.members[static_cast<std::size_t>(i)]) +
              index_base);
    }
    if constexpr (weighted) {
      out.put(std::string_view(" ; "));
      out.put(static_cast<std::int64_t>(hyperedges.weights[static_cast<std::size_t>(h)]));
    }
    out.put_char('\n');
  }
  out.close();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::write_hypergraph (", path, ")")

#define DYNG_INSTANTIATE_HYPERGRAPH_IO(V, W)                                                 \
  template hyperedge_list<V, W> read_hypergraph<V, W>(                                       \
      const std::string&, const hypergraph_read_options&, hypergraph_file_info*);            \
  template void write_hypergraph<V, W>(const std::string&, const hyperedge_list_view<V, W>&, \
                                       std::int64_t);
DYNG_FOR_EACH_HYPERGRAPH_TYPE(DYNG_INSTANTIATE_HYPERGRAPH_IO)
#undef DYNG_INSTANTIATE_HYPERGRAPH_IO

}  // namespace dyng::io
