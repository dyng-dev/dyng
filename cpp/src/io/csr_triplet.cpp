// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:src/csrGraph.cu (readCsrGraph, readIntFile, readValuesFile) and
// MOSP-OpenMP@c352151:src/csrGraph.cpp (writeCsrGraph, runConcurrently)
/**
 * @file csr_triplet.cpp
 * @brief The MOSP text CSR reader and writer.
 */
#include "graph/instantiate.hpp"
#include "util/allocation.hpp"
#include "util/concurrent.hpp"
#include "util/parser.hpp"
#include "util/text_writer.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/io/csr_triplet.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dyng::io {

namespace {

// The strict readers of the three files, one after the other: the reference semantics, and the
// error path of read_csr_triplet() (which re-runs them to report the exact first error).

/// RowPtr: n + 1 non-decreasing offsets from 0.
template <typename vertex_t, typename edge_t>
std::vector<edge_t> read_row_ptr(const std::string& path) {
  std::vector<edge_t> row_ptr;
  const std::string text = detail::read_file(path);
  detail::text_scanner scanner(text, path);
  detail::token tok;
  while (scanner.next_token(tok)) {
    const auto value = detail::parse_integer<edge_t>(scanner, tok, 0,
                                                     detail::max_as_int64<edge_t>(), "row offset");
    if (row_ptr.empty() && value != 0) {
      scanner.fail(tok, "the first row offset must be 0");
    }
    if (!row_ptr.empty() && value < row_ptr.back()) {
      scanner.fail(tok, "row offsets must not decrease (" + std::to_string(value) + " after " +
                            std::to_string(row_ptr.back()) + ")");
    }
    row_ptr.push_back(value);
  }
  if (row_ptr.size() < 2) {
    detail::throw_io_error(path, 0, 0, "a CSR needs at least 2 row offsets (one vertex)");
  }
  if (static_cast<std::int64_t>(row_ptr.size() - 1) > detail::max_as_int64<vertex_t>()) {
    detail::throw_io_error(path, 0, 0, "too many vertices for the vertex id type");
  }
  return row_ptr;
}

/// ColInd: m neighbours in [0, n).
template <typename vertex_t>
std::vector<vertex_t> read_col_ind(const std::string& path, std::int64_t n, std::size_t m) {
  std::vector<vertex_t> col_ind;
  const std::string text = detail::read_file(path);
  detail::text_scanner scanner(text, path);
  detail::token tok;
  // At most one index per two bytes: a RowPtr that announces more edges than ColInd can hold must
  // not make the reader reserve memory for them (found by fuzz_csr_triplet).
  col_ind.reserve(std::min(m, text.size() / 2 + 1));
  while (scanner.next_token(tok)) {
    if (col_ind.size() == m) {
      scanner.fail(tok, "more column indices than the " + std::to_string(m) +
                            " edges announced by the row offsets");
    }
    col_ind.push_back(detail::parse_integer<vertex_t>(scanner, tok, 0, n - 1, "column index"));
  }
  if (col_ind.size() != m) {
    detail::throw_io_error(path, 0, 0,
                           std::to_string(col_ind.size()) +
                               " column indices, but the row "
                               "offsets announce " +
                               std::to_string(m) + " edges");
  }
  return col_ind;
}

/// The weights of Values (objective-major: column c holds line i at c * lines + i) and K.
template <typename weight_t>
struct value_lines {
  std::vector<weight_t> weights;  ///< K columns of `lines` weights each
  std::size_t lines = 0;          ///< non-blank lines
  int num_weights = 0;            ///< K (0: no non-blank line)
};

/// Values: one line of K weights per edge. With `m` = SIZE_MAX the line count is not checked
/// against m (the concurrent pass, which does not know m yet); the caller checks it afterwards.
/// The weights are written straight into their objective-major columns (the graph's layout), so no
/// conversion pass is needed: the column stride is the number of lines, known in advance when `m`
/// is given and otherwise bounded by the number of line breaks (the columns are moved together at
/// the end if blank lines made the bound too large).
template <typename weight_t>
value_lines<weight_t> read_values(const std::string& path, std::size_t m, int num_weights) {
  constexpr std::size_t unknown = std::numeric_limits<std::size_t>::max();
  value_lines<weight_t> out;
  const std::string text = detail::read_file(path);
  detail::text_scanner scanner(text, path);
  detail::token tok;
  int k_count = num_weights;
  const auto weight_max = detail::max_as_int64<weight_t>();
  std::size_t stride = 0;      // column stride of out.weights (0: not allocated yet)
  std::vector<weight_t> line;  // the weights of the current line
  while (scanner.next_line()) {
    line.clear();
    while (scanner.next_in_line(tok)) {
      if (k_count > 0 && static_cast<int>(line.size()) == k_count) {
        scanner.fail(tok, "more than " + std::to_string(k_count) + " weights on the line");
      }
      line.push_back(detail::parse_integer<weight_t>(scanner, tok, 1, weight_max, "weight"));
    }
    const auto on_line = static_cast<int>(line.size());
    if (on_line == 0) {
      continue;
    }
    if (k_count == 0) {
      k_count = on_line;
    } else if (on_line != k_count) {
      scanner.fail_line(std::to_string(on_line) + " weights on the line, expected " +
                        std::to_string(k_count));
    }
    if (++out.lines > m) {
      scanner.fail_line("more weight lines than the " + std::to_string(m) + " edges");
    }
    if (stride == 0) {
      // Every non-blank line ends with a line break, except possibly the last one. With m known,
      // the stride is m unless the file cannot hold m lines: then it is the line bound (the line
      // count check below fails anyway), so a RowPtr that announces more edges than Values can
      // hold does not make the reader allocate their weights (found by fuzz_csr_triplet).
      const std::size_t line_bound =
          static_cast<std::size_t>(std::count(text.begin(), text.end(), '\n')) +
          (text.back() != '\n' ? 1 : 0);
      stride = std::min(m, line_bound);
      out.weights.resize(stride * static_cast<std::size_t>(k_count));
    }
    const std::size_t row = out.lines - 1;
    if (row >= stride) {
      DYNG_FAIL("read_csr_triplet: more weight lines than line breaks in ", path);
    }
    for (std::size_t c = 0; c < line.size(); ++c) {
      out.weights[c * stride + row] = line[c];
    }
  }
  if (m != unknown && out.lines != m) {
    detail::throw_io_error(path, 0, 0,
                           std::to_string(out.lines) + " weight lines, but the graph has " +
                               std::to_string(m) + " edges");
  }
  if (m != unknown && k_count == 0) {
    detail::throw_io_error(path, 0, 0,
                           "the graph has no edges, so the number of weight columns cannot be "
                           "inferred; set csr_triplet_options::num_weights");
  }
  if (stride > out.lines) {  // blank lines: close the gaps between the columns
    for (std::size_t c = 1; c < static_cast<std::size_t>(k_count); ++c) {
      std::copy(out.weights.begin() + static_cast<std::ptrdiff_t>(c * stride),
                out.weights.begin() + static_cast<std::ptrdiff_t>(c * stride + out.lines),
                out.weights.begin() + static_cast<std::ptrdiff_t>(c * out.lines));
    }
    out.weights.resize(out.lines * static_cast<std::size_t>(k_count));
  }
  out.num_weights = k_count;
  return out;
}

}  // namespace

// The three files are read and parsed concurrently, as MOSP's readCsrGraph() does
// (detail::run_concurrently, a port of runConcurrently). The concurrent pass cannot check what
// depends on another file (the column range needs n, the counts need m); those checks run
// afterwards, and on ANY problem the strict
// sequential readers above run again, so the result and the first reported error (RowPtr, then
// ColInd, then Values) are exactly those of reading the files one after the other.
template <typename vertex_t, typename edge_t, typename weight_t>
csr<vertex_t, edge_t, weight_t> read_csr_triplet(const std::string& prefix,
                                                 const csr_triplet_options& options) try {
  static_assert(std::is_integral_v<weight_t>, "read_csr_triplet needs integral weights");
  DYNG_EXPECTS(options.num_weights >= 0, "csr_triplet_options::num_weights must be >= 0, got ",
               options.num_weights);
  constexpr std::size_t unknown = std::numeric_limits<std::size_t>::max();
  const std::string rows_path = prefix + "RowPtr.txt";
  const std::string cols_path = prefix + "ColInd.txt";
  const std::string values_path = prefix + "Values.txt";

  csr<vertex_t, edge_t, weight_t> out;
  std::vector<vertex_t> cols;
  bool cols_ok = false;
  std::int64_t max_col = -1;  // the range check against n needs RowPtr; done afterwards
  value_lines<weight_t> values;
  bool values_ok = false;
  detail::run_concurrently({
      [&] { out.row_ptr = read_row_ptr<vertex_t, edge_t>(rows_path); },
      [&] {
        try {
          const std::string text = detail::read_file(cols_path);
          detail::text_scanner scanner(text, cols_path);
          detail::token tok;
          cols.reserve(static_cast<std::size_t>(std::count(text.begin(), text.end(), '\n') + 1));
          while (scanner.next_token(tok)) {
            const vertex_t v = detail::parse_integer<vertex_t>(
                scanner, tok, 0, detail::max_as_int64<vertex_t>(), "column index");
            max_col = std::max<std::int64_t>(max_col, v);
            cols.push_back(v);
          }
          cols_ok = true;
        } catch (const io_error&) {
          cols_ok = false;  // the strict reader reports it (after any RowPtr error)
        }
      },
      [&] {
        try {
          values = read_values<weight_t>(values_path, unknown, options.num_weights);
          values_ok = true;
        } catch (const io_error&) {
          values_ok = false;
        }
      },
  });  // rethrows a RowPtr error (the only job that throws io_error)
  const auto n = static_cast<std::int64_t>(out.row_ptr.size() - 1);
  const auto m = static_cast<std::size_t>(out.row_ptr.back());

  cols_ok = cols_ok && cols.size() == m && max_col < n;
  out.col_ind = cols_ok ? std::move(cols) : read_col_ind<vertex_t>(cols_path, n, m);

  if (!values_ok || values.lines != m || values.num_weights == 0) {
    values = read_values<weight_t>(values_path, m, options.num_weights);  // throws the error
  }
  out.num_weights = values.num_weights;
  out.weights = std::move(values.weights);  // objective-major already
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::read_csr_triplet (", prefix, ")")

template <typename vertex_t, typename edge_t, typename weight_t>
void write_csr_triplet(const std::string& prefix,
                       const csr_view<vertex_t, edge_t, weight_t>& graph) try {
  const std::size_t m = graph.col_ind.size();
  const auto k_count = static_cast<std::size_t>(graph.num_weights);
  DYNG_EXPECTS(graph.num_weights >= 0 && graph.weights.size() == m * k_count,
               "write_csr_triplet: ", graph.weights.size(), " weights for ", m, " edges and ",
               graph.num_weights, " columns");
  DYNG_EXPECTS((graph.row_ptr.empty() || is_host_accessible(graph.row_ptr.space())) &&
                   (graph.col_ind.empty() || is_host_accessible(graph.col_ind.space())) &&
                   (graph.weights.empty() || is_host_accessible(graph.weights.space())),
               "write_csr_triplet: the CSR must be in host-accessible memory");
  detail::text_writer rows(prefix + "RowPtr.txt");
  detail::text_writer cols(prefix + "ColInd.txt");
  detail::text_writer values(prefix + "Values.txt");
  for (const edge_t offset : graph.row_ptr) {
    rows.put(static_cast<std::int64_t>(offset));
    rows.put_char('\n');
  }
  for (const vertex_t v : graph.col_ind) {
    cols.put(static_cast<std::int64_t>(v));
    cols.put_char('\n');
  }
  for (std::size_t e = 0; e < m; ++e) {
    for (std::size_t k = 0; k < k_count; ++k) {
      values.put(static_cast<std::int64_t>(graph.weights[k * m + e]));
      if (k + 1 < k_count) {
        values.put_char(' ');
      }
    }
    values.put_char('\n');
  }
  rows.close();
  cols.close();
  values.close();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::write_csr_triplet (", prefix, ")")

#define DYNG_INSTANTIATE_CSR_TRIPLET(V, E, W)                                                      \
  template csr<V, E, W> read_csr_triplet<V, E, W>(const std::string&, const csr_triplet_options&); \
  template void write_csr_triplet<V, E, W>(const std::string&, const csr_view<V, E, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_CSR_TRIPLET)
#undef DYNG_INSTANTIATE_CSR_TRIPLET

}  // namespace dyng::io
