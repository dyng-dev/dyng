// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file batch_text.cpp
 * @brief The dynG batch text format `.dgt` (PLAN Section 5.7; docs/api/file_formats.md).
 */
#include "io/io_instantiate.hpp"
#include "util/allocation.hpp"
#include "util/parser.hpp"
#include "util/text_writer.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/io/batch_io.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace dyng::io {
namespace {

/// The smallest weight a `.dgt` file may hold for `weight_t`.
template <typename weight_t>
constexpr std::int64_t weight_min() noexcept {
  if constexpr (is_unweighted_v<weight_t>) {
    return 0;
  } else if constexpr (std::is_signed_v<weight_t>) {
    return static_cast<std::int64_t>(std::numeric_limits<weight_t>::min());
  } else {
    return 0;
  }
}

/// The largest weight a `.dgt` file may hold for `weight_t`.
template <typename weight_t>
constexpr std::int64_t weight_max() noexcept {
  if constexpr (is_unweighted_v<weight_t>) {
    return 0;
  } else {
    return detail::max_as_int64<weight_t>();
  }
}

bool is_reserved_vertex_op(std::string_view op) noexcept {
  return op == "+v" || op == "-v";
}

bool is_reserved_hypergraph_op(std::string_view op) noexcept {
  return op == "+h" || op == "-h" || op == "+i" || op == "-i";
}

}  // namespace

template <typename vertex_t, typename weight_t>
std::vector<edge_batch<vertex_t, weight_t>> read_batches(const std::string& path,
                                                         const batch_file_options& options) try {
  constexpr bool unweighted_batch = is_unweighted_v<weight_t>;
  DYNG_EXPECTS(options.num_weights >= -1, "batch_file_options::num_weights must be >= -1, got ",
               options.num_weights);
  DYNG_EXPECTS(!unweighted_batch || options.num_weights <= 0,
               "batch_file_options::num_weights must be 0 or -1 for an unweighted batch, got ",
               options.num_weights);
  const std::int64_t id_max =
      options.num_vertices >= 0 ? options.num_vertices - 1 : detail::max_as_int64<vertex_t>();
  DYNG_EXPECTS(id_max <= detail::max_as_int64<vertex_t>(),
               "batch_file_options::num_vertices does not fit the vertex id type");
  int k = unweighted_batch ? 0 : options.num_weights;  // -1 until the first insertion

  const std::string text = detail::read_file(path);
  detail::text_scanner scanner(text, path);
  std::vector<edge_batch<vertex_t, weight_t>> batches;
  std::vector<weight_t> weights;
  bool explicit_batches = false;  // the file has %batch lines
  bool seen_content = false;      // an operation or a %batch line was read
  std::int64_t last_id = -1;
  detail::token tok;
  detail::token extra;

  const auto new_batch = [&] { batches.emplace_back(k < 0 ? 0 : k); };
  const auto parse_id = [&](const detail::token& t) {
    return detail::parse_integer<vertex_t>(scanner, t, 0, id_max, "vertex id");
  };
  const auto no_more = [&](const char* form) {
    if (scanner.next_in_line(extra)) {
      scanner.fail(extra, std::string("unexpected '") + std::string(extra.text) +
                              "' (the line is '" + form + "')");
    }
  };
  const auto next_or_fail = [&](const char* form) {
    if (!scanner.next_in_line(tok)) {
      scanner.fail_line(std::string("incomplete operation (expected '") + form + "')");
    }
  };

  while (scanner.next_line()) {
    if (!scanner.next_in_line(tok) || tok.text.front() == '#') {
      continue;
    }
    const std::string_view op = tok.text;
    if (op == "%dgt") {
      if (seen_content) {
        scanner.fail(tok, "'%dgt <version>' must be the first line of the file");
      }
      next_or_fail("%dgt 1");
      if (tok.text != "1") {
        scanner.fail(tok, "unsupported .dgt version '" + std::string(tok.text) +
                              "' (this library reads version 1)");
      }
      no_more("%dgt 1");
      seen_content = true;
      continue;
    }
    if (op == "%batch") {
      if (!explicit_batches && !batches.empty()) {
        scanner.fail(tok,
                     "operations before the first '%batch' line (a file with '%batch' "
                     "lines must start every batch with one)");
      }
      next_or_fail("%batch <id>");
      const auto id = detail::parse_integer<std::int64_t>(
          scanner, tok, 0, std::numeric_limits<std::int64_t>::max(), "batch id");
      if (id <= last_id) {
        scanner.fail(tok, "batch id " + std::to_string(id) + " after " + std::to_string(last_id) +
                              " (ids must be strictly increasing)");
      }
      no_more("%batch <id>");
      last_id = id;
      explicit_batches = true;
      seen_content = true;
      new_batch();
      continue;
    }
    if (op == "+e" || op == "-e") {
      if (batches.empty()) {
        new_batch();  // a file without %batch lines: one batch
      }
      seen_content = true;
      auto& batch = batches.back();
      next_or_fail(op == "+e" ? "+e u v [w1 .. wK]" : "-e u v");
      const vertex_t u = parse_id(tok);
      next_or_fail(op == "+e" ? "+e u v [w1 .. wK]" : "-e u v");
      const vertex_t v = parse_id(tok);
      if (op == "-e") {
        no_more("-e u v");
        batch.delete_edge(u, v);
        continue;
      }
      weights.clear();
      std::vector<detail::token> weight_tokens;
      while (scanner.next_in_line(extra)) {
        weight_tokens.push_back(extra);
      }
      if (k < 0) {
        k = static_cast<int>(weight_tokens.size());  // taken from the first insertion
        for (auto& b : batches) {
          // The batches so far hold deletions only: rebuild them with K weights.
          if (b.num_weights() != k) {
            edge_batch<vertex_t, weight_t> rebuilt(k);
            rebuilt.reserve(0, b.num_deletions());
            for (std::size_t j = 0; j < b.num_deletions(); ++j) {
              rebuilt.delete_edge(b.delete_src()[j], b.delete_dst()[j]);
            }
            b = std::move(rebuilt);
          }
        }
      }
      if (weight_tokens.size() != static_cast<std::size_t>(k)) {
        scanner.fail_line(
            "an insertion '+e u v w1 .. wK' carries exactly K = " + std::to_string(k) +
            " weights here, found " + std::to_string(weight_tokens.size()));
      }
      if constexpr (unweighted_batch) {
        batch.insert_edge(u, v);
      } else {
        for (const auto& w : weight_tokens) {
          weights.push_back(detail::parse_integer<weight_t>(scanner, w, weight_min<weight_t>(),
                                                            weight_max<weight_t>(), "weight"));
        }
        if (k == 0) {
          batch.insert_edge(u, v);
        } else {
          batch.insert_edge(u, v, host_view(weights.data(), weights.size()));
        }
      }
      continue;
    }
    if (is_reserved_vertex_op(op)) {
      scanner.fail(tok, "the vertex operation '" + std::string(op) +
                            "' is reserved for the vertex batches of label_propagation (0.3); "
                            "0.1 reads edge operations ('+e', '-e') only");
    }
    if (is_reserved_hypergraph_op(op)) {
      scanner.fail(tok, "the hypergraph operation '" + std::string(op) +
                            "' is reserved for the hypergraph batches of 0.2; 0.1 reads edge "
                            "operations ('+e', '-e') only");
    }
    if (op.front() == '%') {
      scanner.fail(tok,
                   "unknown directive '" + std::string(op) + "' (expected '%dgt' or '%batch')");
    }
    scanner.fail(tok, "unknown operation '" + std::string(op) + "' (expected '+e' or '-e')");
  }
  return batches;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::read_batches (", path, ")")

template <typename vertex_t, typename weight_t>
void write_batches(const std::string& path,
                   const std::vector<edge_batch_view<vertex_t, weight_t>>& batches) try {
  int k = -1;
  for (std::size_t b = 0; b < batches.size(); ++b) {
    const auto& batch = batches[b];
    const std::size_t n = batch.insert_src.size();
    DYNG_EXPECTS(batch.insert_vertices.empty() && batch.delete_vertices.empty(),
                 "write_batches: batch ", b,
                 " has vertex operations, which the .dgt files of 0.1 do not hold");
    DYNG_EXPECTS(
        batch.num_weights >= 0 && batch.insert_dst.size() == n &&
            batch.insert_weights.size() == n * static_cast<std::size_t>(batch.num_weights) &&
            batch.delete_dst.size() == batch.delete_src.size(),
        "write_batches: the arrays of batch ", b, " disagree in size");
    const auto host = [](const auto& view) {
      return view.empty() || is_host_accessible(view.space());
    };
    DYNG_EXPECTS(host(batch.insert_src) && host(batch.insert_dst) && host(batch.insert_weights) &&
                     host(batch.delete_src) && host(batch.delete_dst),
                 "write_batches: batch ", b, " must be in host-accessible memory");
    if (n > 0) {
      DYNG_EXPECTS(k < 0 || k == batch.num_weights, "write_batches: batch ", b,
                   " has K = ", batch.num_weights, " weights per insertion, an earlier batch ", k,
                   " (a .dgt file has one K)");
      k = batch.num_weights;
    }
  }
  detail::text_writer out(path);
  out.put(std::string_view("%dgt 1\n"));
  for (std::size_t b = 0; b < batches.size(); ++b) {
    const auto& batch = batches[b];
    const auto k_count = static_cast<std::size_t>(batch.num_weights);
    out.put(std::string_view("%batch "));
    out.put(static_cast<std::int64_t>(b));
    out.put_char('\n');
    for (std::size_t j = 0; j < batch.delete_src.size(); ++j) {
      out.put(std::string_view("-e "));
      out.put(static_cast<std::int64_t>(batch.delete_src[j]));
      out.put_char(' ');
      out.put(static_cast<std::int64_t>(batch.delete_dst[j]));
      out.put_char('\n');
    }
    for (std::size_t i = 0; i < batch.insert_src.size(); ++i) {
      out.put(std::string_view("+e "));
      out.put(static_cast<std::int64_t>(batch.insert_src[i]));
      out.put_char(' ');
      out.put(static_cast<std::int64_t>(batch.insert_dst[i]));
      if constexpr (!is_unweighted_v<weight_t>) {
        for (std::size_t w = 0; w < k_count; ++w) {
          out.put_char(' ');
          out.put(static_cast<std::int64_t>(batch.insert_weights[i * k_count + w]));
        }
      }
      out.put_char('\n');
    }
  }
  out.close();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::write_batches (", path, ")")

#define DYNG_INSTANTIATE_BATCH_TEXT(V, W)                                               \
  template std::vector<edge_batch<V, W>> read_batches<V, W>(const std::string&,         \
                                                            const batch_file_options&); \
  template void write_batches<V, W>(const std::string&, const std::vector<edge_batch_view<V, W>>&);
DYNG_FOR_EACH_VERTEX_WEIGHT_TYPE(DYNG_INSTANTIATE_BATCH_TEXT)
DYNG_FOR_EACH_VERTEX_TYPE_UNWEIGHTED(DYNG_INSTANTIATE_BATCH_TEXT)
#undef DYNG_INSTANTIATE_BATCH_TEXT

}  // namespace dyng::io
