// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:src/csrGraph.cu (readCsrGraph, readIntFile, readValuesFile) and
// MOSP-OpenMP@c352151:src/csrGraph.cpp (writeCsrGraph)
/**
 * @file csr_triplet.cpp
 * @brief The MOSP text CSR reader and writer.
 */
#include "graph/instantiate.hpp"
#include "util/parser.hpp"
#include "util/text_writer.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/io/csr_triplet.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace dyng::io {

template <typename vertex_t, typename edge_t, typename weight_t>
csr<vertex_t, edge_t, weight_t> read_csr_triplet(const std::string& prefix,
                                                 const csr_triplet_options& options) {
  static_assert(std::is_integral_v<weight_t>, "read_csr_triplet needs integral weights");
  DYNG_EXPECTS(options.num_weights >= 0, "csr_triplet_options::num_weights must be >= 0, got ",
               options.num_weights);
  csr<vertex_t, edge_t, weight_t> out;
  detail::token tok;

  // --- RowPtr: n + 1 non-decreasing offsets from 0 ------------------------------------------------
  {
    const std::string path = prefix + "RowPtr.txt";
    const std::string text = detail::read_file(path);
    detail::text_scanner scanner(text, path);
    while (scanner.next_token(tok)) {
      const auto value = detail::parse_integer<edge_t>(
          scanner, tok, 0, detail::max_as_int64<edge_t>(), "row offset");
      if (out.row_ptr.empty() && value != 0) {
        scanner.fail(tok, "the first row offset must be 0");
      }
      if (!out.row_ptr.empty() && value < out.row_ptr.back()) {
        scanner.fail(tok, "row offsets must not decrease (" + std::to_string(value) + " after " +
                              std::to_string(out.row_ptr.back()) + ")");
      }
      out.row_ptr.push_back(value);
    }
    if (out.row_ptr.size() < 2) {
      detail::throw_io_error(path, 0, 0, "a CSR needs at least 2 row offsets (one vertex)");
    }
    if (static_cast<std::int64_t>(out.row_ptr.size() - 1) > detail::max_as_int64<vertex_t>()) {
      detail::throw_io_error(path, 0, 0, "too many vertices for the vertex id type");
    }
  }
  const auto n = static_cast<std::int64_t>(out.row_ptr.size() - 1);
  const auto m = static_cast<std::size_t>(out.row_ptr.back());

  // --- ColInd: m neighbours in [0, n) -------------------------------------------------------------
  {
    const std::string path = prefix + "ColInd.txt";
    const std::string text = detail::read_file(path);
    detail::text_scanner scanner(text, path);
    out.col_ind.reserve(m);
    while (scanner.next_token(tok)) {
      if (out.col_ind.size() == m) {
        scanner.fail(tok, "more column indices than the " + std::to_string(m) +
                              " edges announced by the row offsets");
      }
      out.col_ind.push_back(
          detail::parse_integer<vertex_t>(scanner, tok, 0, n - 1, "column index"));
    }
    if (out.col_ind.size() != m) {
      detail::throw_io_error(path, 0, 0,
                             std::to_string(out.col_ind.size()) +
                                 " column indices, but the row "
                                 "offsets announce " +
                                 std::to_string(m) + " edges");
    }
  }

  // --- Values: one line of K weights per edge ------------------------------------------------------
  {
    const std::string path = prefix + "Values.txt";
    const std::string text = detail::read_file(path);
    detail::text_scanner scanner(text, path);
    int k_count = options.num_weights;
    std::vector<weight_t> edge_major;
    std::size_t lines = 0;
    const auto weight_max = detail::max_as_int64<weight_t>();
    while (scanner.next_line()) {
      int on_line = 0;
      while (scanner.next_in_line(tok)) {
        if (k_count > 0 && on_line == k_count) {
          scanner.fail(tok, "more than " + std::to_string(k_count) + " weights on the line");
        }
        edge_major.push_back(
            detail::parse_integer<weight_t>(scanner, tok, 1, weight_max, "weight"));
        ++on_line;
      }
      if (on_line == 0) {
        continue;
      }
      if (k_count == 0) {
        k_count = on_line;
        edge_major.reserve(m * static_cast<std::size_t>(k_count));
      } else if (on_line != k_count) {
        scanner.fail_line(std::to_string(on_line) + " weights on the line, expected " +
                          std::to_string(k_count));
      }
      if (++lines > m) {
        scanner.fail_line("more weight lines than the " + std::to_string(m) + " edges");
      }
    }
    if (lines != m) {
      detail::throw_io_error(path, 0, 0,
                             std::to_string(lines) + " weight lines, but the graph has " +
                                 std::to_string(m) + " edges");
    }
    if (k_count == 0) {
      detail::throw_io_error(path, 0, 0,
                             "the graph has no edges, so the number of weight columns cannot be "
                             "inferred; set csr_triplet_options::num_weights");
    }
    out.num_weights = k_count;
    const auto k = static_cast<std::size_t>(k_count);
    out.weights.resize(m * k);
    for (std::size_t e = 0; e < m; ++e) {
      for (std::size_t c = 0; c < k; ++c) {
        out.weights[c * m + e] = edge_major[e * k + c];
      }
    }
  }
  return out;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void write_csr_triplet(const std::string& prefix,
                       const csr_view<vertex_t, edge_t, weight_t>& graph) {
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

#define DYNG_INSTANTIATE_CSR_TRIPLET(V, E, W)                                                      \
  template csr<V, E, W> read_csr_triplet<V, E, W>(const std::string&, const csr_triplet_options&); \
  template void write_csr_triplet<V, E, W>(const std::string&, const csr_view<V, E, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_CSR_TRIPLET)
#undef DYNG_INSTANTIATE_CSR_TRIPLET

}  // namespace dyng::io
