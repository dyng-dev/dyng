// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:src/mospPrep.cpp (mtxToCsr)
/**
 * @file matrix_market.cpp
 * @brief Matrix Market reader and writer.
 *
 * The structure and the weights of mospPrep's mtxToCsr() are reproduced exactly: entries with
 * i != j (self-loops dropped) give (i, j) and, for symmetric files, (j, i); the edges are sorted by
 * (source, destination) and deduplicated; then `K` weights per edge are drawn in edge order from
 * std::mt19937(seed) through the libstdc++ uniform_int_distribution algorithm. Unlike mtxToCsr,
 * malformed input is an error (mtxToCsr silently skipped unparsable or out-of-range entries and
 * stopped at a premature end of file).
 */
#include "io/io_instantiate.hpp"
#include "util/allocation.hpp"
#include "util/parser.hpp"
#include "util/rng.hpp"
#include "util/text_writer.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/io/matrix_market.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

namespace dyng::io {

namespace {

enum class mm_field : std::uint8_t { real, integer, complex, pattern };
enum class mm_symmetry : std::uint8_t { general, symmetric, skew_symmetric, hermitian };

std::string lower(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

bool parse_real(std::string_view text) {
  double value = 0.0;
  const char* last = text.data() + text.size();
  const auto result = std::from_chars(text.data(), last, value);
  return !text.empty() && result.ec == std::errc() && result.ptr == last;
}

struct raw_edge {
  std::int64_t u;
  std::int64_t v;
  std::int64_t value;
};

}  // namespace

template <typename vertex_t, typename weight_t>
edge_list<vertex_t, weight_t> read_matrix_market(const std::string& path,
                                                 const matrix_market_options& options) try {
  static_assert(std::is_integral_v<weight_t>, "read_matrix_market needs integral weights");
  const std::string text = detail::read_file(path);
  detail::text_scanner scanner(text, path);
  detail::token tok;

  // --- Header ----------------------------------------------------------------------------------
  if (!scanner.next_line() || !scanner.next_in_line(tok) || lower(tok.text) != "%%matrixmarket") {
    scanner.fail_line("missing '%%MatrixMarket' header");
  }
  std::string words[4];
  for (auto& word : words) {
    if (!scanner.next_in_line(tok)) {
      scanner.fail_line(
          "incomplete header; expected '%%MatrixMarket matrix coordinate <field> <symmetry>'");
    }
    word = lower(tok.text);
  }
  if (scanner.next_in_line(tok)) {
    scanner.fail(tok, "unexpected token after the Matrix Market header");
  }
  if (words[0] != "matrix") {
    scanner.fail_line("unsupported object '" + words[0] + "' (expected 'matrix')");
  }
  if (words[1] != "coordinate") {
    scanner.fail_line("unsupported format '" + words[1] +
                      "' (only 'coordinate' matrices describe graphs)");
  }
  mm_field field = mm_field::real;
  if (words[2] == "real" || words[2] == "double") {
    field = mm_field::real;
  } else if (words[2] == "integer") {
    field = mm_field::integer;
  } else if (words[2] == "complex") {
    field = mm_field::complex;
  } else if (words[2] == "pattern") {
    field = mm_field::pattern;
  } else {
    scanner.fail_line("unsupported field '" + words[2] + "'");
  }
  mm_symmetry symmetry = mm_symmetry::general;
  if (words[3] == "general") {
    symmetry = mm_symmetry::general;
  } else if (words[3] == "symmetric") {
    symmetry = mm_symmetry::symmetric;
  } else if (words[3] == "skew-symmetric") {
    symmetry = mm_symmetry::skew_symmetric;
  } else if (words[3] == "hermitian") {
    symmetry = mm_symmetry::hermitian;
  } else {
    scanner.fail_line("unsupported symmetry '" + words[3] + "'");
  }

  // --- Weight source ------------------------------------------------------------------------------
  matrix_market_weights source = options.weights;
  if (source == matrix_market_weights::automatic) {
    if (field == mm_field::integer) {
      source = matrix_market_weights::from_file;
    } else if (field == mm_field::pattern) {
      source = matrix_market_weights::none;
    } else {
      scanner.fail_line("a '" + words[2] +
                        "' matrix has no integer weights; read it with "
                        "matrix_market_weights::none or ::random");
    }
  }
  if (source == matrix_market_weights::from_file && field != mm_field::integer) {
    scanner.fail_line("matrix_market_weights::from_file needs an 'integer' matrix, not '" +
                      words[2] + "'");
  }
  if (source == matrix_market_weights::random) {
    const random_weights& r = options.random;
    DYNG_EXPECTS(r.num_weights >= 1, "random_weights::num_weights must be >= 1, got ",
                 r.num_weights);
    DYNG_EXPECTS(r.min <= r.max, "random_weights: min ", r.min, " > max ", r.max);
    DYNG_EXPECTS(r.min >= static_cast<std::int64_t>(std::numeric_limits<weight_t>::min()) &&
                     r.max <= static_cast<std::int64_t>(std::numeric_limits<weight_t>::max()),
                 "random_weights [", r.min, ", ", r.max, "] does not fit the weight type");
    DYNG_EXPECTS(
        static_cast<std::uint64_t>(r.max) - static_cast<std::uint64_t>(r.min) <= 0xffffffffULL,
        "random_weights: the range may hold at most 2^32 values");
  }

  // --- Size line -------------------------------------------------------------------------------------
  std::int64_t rows = 0;
  std::int64_t entries = 0;
  bool have_size = false;
  while (!have_size && scanner.next_line()) {
    const std::string_view line = scanner.rest_of_line();
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string_view::npos || line[first] == '%') {
      continue;
    }
    detail::token t_rows;
    detail::token t_cols;
    detail::token t_entries;
    if (!scanner.next_in_line(t_rows) || !scanner.next_in_line(t_cols) ||
        !scanner.next_in_line(t_entries)) {
      scanner.fail_line("the size line must hold '<rows> <columns> <entries>'");
    }
    if (scanner.next_in_line(tok)) {
      scanner.fail(tok, "unexpected token after the size line");
    }
    rows = detail::parse_integer<std::int64_t>(scanner, t_rows, 0, detail::max_as_int64<vertex_t>(),
                                               "row count");
    const auto cols = detail::parse_integer<std::int64_t>(
        scanner, t_cols, 0, std::numeric_limits<std::int64_t>::max(), "column count");
    entries = detail::parse_integer<std::int64_t>(
        scanner, t_entries, 0, std::numeric_limits<std::int64_t>::max(), "entry count");
    if (cols != rows) {
      scanner.fail(t_cols, "the matrix is " + std::to_string(rows) + " x " + std::to_string(cols) +
                               "; a graph needs a square matrix");
    }
    have_size = true;
  }
  if (!have_size) {
    scanner.fail_line("missing size line");
  }

  // --- Entries ---------------------------------------------------------------------------------------
  const int value_tokens = field == mm_field::pattern ? 0 : (field == mm_field::complex ? 2 : 1);
  const bool mirrored = symmetry != mm_symmetry::general;
  std::vector<raw_edge> edges;
  edges.reserve(static_cast<std::size_t>(std::min<std::int64_t>(entries, 1 << 26)) *
                (mirrored ? 2 : 1));
  std::int64_t seen = 0;
  while (scanner.next_line()) {
    const std::string_view line = scanner.rest_of_line();
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string_view::npos || line[first] == '%') {
      continue;
    }
    if (seen == entries) {
      scanner.fail_line("more entries than the " + std::to_string(entries) +
                        " announced in the size line");
    }
    detail::token t_i;
    detail::token t_j;
    if (!scanner.next_in_line(t_i) || !scanner.next_in_line(t_j)) {
      scanner.fail_line("an entry needs a row and a column index");
    }
    const auto i = detail::parse_integer<std::int64_t>(scanner, t_i, 1, rows, "row index");
    const auto j = detail::parse_integer<std::int64_t>(scanner, t_j, 1, rows, "column index");
    std::int64_t value = 0;
    for (int t = 0; t < value_tokens; ++t) {
      if (!scanner.next_in_line(tok)) {
        scanner.fail_line("an entry of a '" + words[2] + "' matrix needs " +
                          std::to_string(value_tokens) + " value(s)");
      }
      if (field == mm_field::integer) {
        value = detail::parse_integer<std::int64_t>(
            scanner, tok, std::numeric_limits<std::int64_t>::min(),
            std::numeric_limits<std::int64_t>::max(), "value");
      } else if (!parse_real(tok.text)) {
        scanner.fail(tok, "invalid value '" + std::string(tok.text) + "'");
      }
    }
    if (scanner.next_in_line(tok)) {
      scanner.fail(tok, "unexpected token after the entry");
    }
    ++seen;
    const std::int64_t u = i - 1;
    const std::int64_t v = j - 1;
    if (u == v && options.drop_self_loops) {
      continue;
    }
    edges.push_back({u, v, value});
    if (mirrored && u != v) {
      edges.push_back({v, u, symmetry == mm_symmetry::skew_symmetric ? -value : value});
    }
  }
  if (seen != entries) {
    detail::throw_io_error(path, scanner.line(), 0,
                           "the file ends after " + std::to_string(seen) + " of " +
                               std::to_string(entries) + " entries");
  }

  if (options.sort_and_dedupe) {
    std::stable_sort(edges.begin(), edges.end(), [](const raw_edge& a, const raw_edge& b) {
      return a.u != b.u ? a.u < b.u : a.v < b.v;
    });
    std::size_t kept = 0;
    for (std::size_t e = 0; e < edges.size(); ++e) {
      if (kept > 0 && edges[kept - 1].u == edges[e].u && edges[kept - 1].v == edges[e].v) {
        edges[kept - 1].value = edges[e].value;  // the last entry wins
      } else {
        edges[kept++] = edges[e];
      }
    }
    edges.resize(kept);
  }

  edge_list<vertex_t, weight_t> out;
  out.num_vertices = static_cast<vertex_t>(rows);
  out.src.resize(edges.size());
  out.dst.resize(edges.size());
  for (std::size_t e = 0; e < edges.size(); ++e) {
    out.src[e] = static_cast<vertex_t>(edges[e].u);
    out.dst[e] = static_cast<vertex_t>(edges[e].v);
  }
  if (source == matrix_market_weights::from_file) {
    out.num_weights = 1;
    out.weights.resize(edges.size());
    for (std::size_t e = 0; e < edges.size(); ++e) {
      const std::int64_t value = edges[e].value;
      if (value < static_cast<std::int64_t>(std::numeric_limits<weight_t>::min()) ||
          value > static_cast<std::int64_t>(std::numeric_limits<weight_t>::max())) {
        detail::throw_io_error(
            path, 0, 0,
            "value " + std::to_string(value) + " of edge (" + std::to_string(edges[e].u + 1) +
                ", " + std::to_string(edges[e].v + 1) + ") does not fit the weight type");
      }
      out.weights[e] = static_cast<weight_t>(value);
    }
  } else if (source == matrix_market_weights::random) {
    const random_weights& r = options.random;
    out.num_weights = r.num_weights;
    out.weights.resize(edges.size() * static_cast<std::size_t>(r.num_weights));
    std::mt19937 engine(r.seed);
    for (auto& w : out.weights) {
      w = static_cast<weight_t>(detail::legacy_uniform_int(engine, r.min, r.max));
    }
  }
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::read_matrix_market (", path, ")")

template <typename vertex_t, typename weight_t>
void write_matrix_market(const std::string& path, const edge_list_view<vertex_t, weight_t>& edges,
                         int weight_column) try {
  const std::size_t m = edges.src.size();
  const auto k_count = static_cast<std::size_t>(edges.num_weights);
  DYNG_EXPECTS(edges.dst.size() == m && edges.weights.size() == m * k_count,
               "write_matrix_market: the edge list arrays disagree in size");
  DYNG_EXPECTS(edges.num_weights == 0 || (weight_column >= 0 && weight_column < edges.num_weights),
               "write_matrix_market: weight column ", weight_column, " out of range [0, ",
               edges.num_weights, ")");
  DYNG_EXPECTS((m == 0 ||
                (is_host_accessible(edges.src.space()) && is_host_accessible(edges.dst.space()))) &&
                   (edges.weights.empty() || is_host_accessible(edges.weights.space())),
               "write_matrix_market: the edge list must be in host-accessible memory");
  detail::text_writer out(path);
  out.put(edges.num_weights == 0 ? "%%MatrixMarket matrix coordinate pattern general\n"
                                 : "%%MatrixMarket matrix coordinate integer general\n");
  out.put(static_cast<std::int64_t>(edges.num_vertices));
  out.put_char(' ');
  out.put(static_cast<std::int64_t>(edges.num_vertices));
  out.put_char(' ');
  out.put(static_cast<std::int64_t>(m));
  out.put_char('\n');
  for (std::size_t e = 0; e < m; ++e) {
    out.put(static_cast<std::int64_t>(edges.src[e]) + 1);
    out.put_char(' ');
    out.put(static_cast<std::int64_t>(edges.dst[e]) + 1);
    if (edges.num_weights > 0) {
      out.put_char(' ');
      out.put(static_cast<std::int64_t>(
          edges.weights[e * k_count + static_cast<std::size_t>(weight_column)]));
    }
    out.put_char('\n');
  }
  out.close();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::write_matrix_market (", path, ")")

#define DYNG_INSTANTIATE_MATRIX_MARKET(V, W)                                       \
  template edge_list<V, W> read_matrix_market<V, W>(const std::string&,            \
                                                    const matrix_market_options&); \
  template void write_matrix_market<V, W>(const std::string&, const edge_list_view<V, W>&, int);
DYNG_FOR_EACH_VERTEX_WEIGHT_TYPE(DYNG_INSTANTIATE_MATRIX_MARKET)
#undef DYNG_INSTANTIATE_MATRIX_MARKET

}  // namespace dyng::io
