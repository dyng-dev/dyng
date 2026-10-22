// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file batch_text.cpp
 * @brief The dynG batch text format `.dgt` (PLAN Section 5.7; docs/api/file_formats.md).
 */
#include "hypergraph/instantiate.hpp"
#include "io/batch_checks.hpp"
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

bool is_hyperedge_op(std::string_view op) noexcept {
  return op == "+h" || op == "-h";
}

bool is_incidence_op(std::string_view op) noexcept {
  return op == "+i" || op == "-i";
}

/// The container kinds a `.dgt` file can hold (named on its `%dgt 1 <kind>` line).
enum class dgt_kind : std::uint8_t { graph, hypergraph };

const char* kind_name(dgt_kind kind) noexcept {
  return kind == dgt_kind::graph ? "graph" : "hypergraph";
}

/**
 * The scan shared by both readers: comments, `%dgt 1 [graph|hypergraph]`, `%batch <id>` and the
 * operations of the other container kind, which are errors (a file holds the batches of one
 * kind). `new_batch()` starts a batch; `on_op(op_token)` reads an operation of this kind
 * (returning false for an operation it does not know), creating the implicit single batch first
 * when there is none.
 */
template <typename new_batch_t, typename has_batch_t, typename on_op_t>
void scan_dgt(detail::text_scanner& scanner, dgt_kind kind, new_batch_t&& new_batch,
              has_batch_t&& has_batch, on_op_t&& on_op) {
  bool explicit_batches = false;  // the file has %batch lines
  bool seen_content = false;      // an operation or a %batch line was read
  std::int64_t last_id = -1;
  detail::token tok;
  detail::token extra;
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
  const std::string other_reader =
      kind == dgt_kind::graph ? "read_hyperedge_batches()" : "read_batches()";
  while (scanner.next_line()) {
    if (!scanner.next_in_line(tok) || tok.text.front() == '#') {
      continue;
    }
    const std::string_view op = tok.text;
    if (op == "%dgt") {
      if (seen_content) {
        scanner.fail(tok, "'%dgt <version>' must be the first line of the file");
      }
      next_or_fail("%dgt 1 [graph|hypergraph]");
      if (tok.text != "1") {
        scanner.fail(tok, "unsupported .dgt version '" + std::string(tok.text) +
                              "' (this library reads version 1)");
      }
      if (scanner.next_in_line(tok)) {
        if (tok.text != "graph" && tok.text != "hypergraph") {
          scanner.fail(tok, "unknown container kind '" + std::string(tok.text) +
                                "' (the line is '%dgt 1 [graph|hypergraph]')");
        }
        if (tok.text != kind_name(kind)) {
          scanner.fail(tok, "this file holds " + std::string(tok.text) +
                                " batches (its '%dgt' line says so): read it with " + other_reader);
        }
      }
      no_more("%dgt 1 [graph|hypergraph]");
      seen_content = true;
      continue;
    }
    if (op == "%batch") {
      if (!explicit_batches && has_batch()) {
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
    if (op.front() != '%' && !is_reserved_vertex_op(op) && !is_incidence_op(op)) {
      const bool foreign =
          kind == dgt_kind::graph ? is_hyperedge_op(op) : (op == "+e" || op == "-e");
      if (foreign) {
        scanner.fail(tok, "the " + std::string(kind == dgt_kind::graph ? "hypergraph" : "graph") +
                              " operation '" + std::string(op) + "' in a file of " +
                              kind_name(kind) + " batches: a .dgt file holds the batches of one " +
                              "container kind; read hypergraph batches with " +
                              "read_hyperedge_batches(), graph batches with read_batches()");
      }
      if (!has_batch()) {
        new_batch();  // a file without %batch lines: one batch
      }
      seen_content = true;
      if (on_op(tok)) {
        continue;
      }
    }
    if (is_reserved_vertex_op(op)) {
      scanner.fail(tok, "the vertex operation '" + std::string(op) +
                            "' is reserved for the vertex batches of label_propagation (0.4); "
                            "this release reads edge operations ('+e', '-e') and hyperedge "
                            "operations ('+h', '-h') only");
    }
    if (is_incidence_op(op)) {
      scanner.fail(tok, "the incidence operation '" + std::string(op) +
                            "' is reserved for the incidence edits of hypergraph batches (0.4); "
                            "this release reads hyperedge operations ('+h', '-h') only");
    }
    if (op.front() == '%') {
      scanner.fail(tok,
                   "unknown directive '" + std::string(op) + "' (expected '%dgt' or '%batch')");
    }
    scanner.fail(tok, "unknown operation '" + std::string(op) +
                          (kind == dgt_kind::graph ? "' (expected '+e' or '-e')"
                                                   : "' (expected '+h' or '-h')"));
  }
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
  std::vector<detail::token> weight_tokens;
  detail::token tok;
  detail::token extra;

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
  const auto on_op = [&](const detail::token& op_token) {
    const std::string_view op = op_token.text;
    if (op != "+e" && op != "-e") {
      return false;
    }
    auto& batch = batches.back();
    next_or_fail(op == "+e" ? "+e u v [w1 .. wK]" : "-e u v");
    const vertex_t u = parse_id(tok);
    next_or_fail(op == "+e" ? "+e u v [w1 .. wK]" : "-e u v");
    const vertex_t v = parse_id(tok);
    if (op == "-e") {
      no_more("-e u v");
      batch.delete_edge(u, v);
      return true;
    }
    weights.clear();
    weight_tokens.clear();
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
      scanner.fail_line("an insertion '+e u v w1 .. wK' carries exactly K = " + std::to_string(k) +
                        " weights here, found " + std::to_string(weight_tokens.size()));
    }
    if constexpr (unweighted_batch) {
      batches.back().insert_edge(u, v);
    } else {
      for (const auto& w : weight_tokens) {
        weights.push_back(detail::parse_integer<weight_t>(scanner, w, weight_min<weight_t>(),
                                                          weight_max<weight_t>(), "weight"));
      }
      if (k == 0) {
        batches.back().insert_edge(u, v);
      } else {
        batches.back().insert_edge(u, v, host_view(weights.data(), weights.size()));
      }
    }
    return true;
  };
  scan_dgt(
      scanner, dgt_kind::graph, [&] { batches.emplace_back(k < 0 ? 0 : k); },
      [&] { return !batches.empty(); }, on_op);
  return batches;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::read_batches (", path, ")")

template <typename vertex_t, typename weight_t>
void write_batches(const std::string& path,
                   const std::vector<edge_batch_view<vertex_t, weight_t>>& batches) try {
  int k = -1;
  for (std::size_t b = 0; b < batches.size(); ++b) {
    const auto& batch = batches[b];
    detail::check_edge_batch_arrays(batch, "write_batches", b);
    if (!batch.insert_src.empty()) {
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

namespace {

/// The rest of the line as tokens, with every `;` a token of its own ("1 2;5" is 1, 2, ;, 5).
void split_line(detail::text_scanner& scanner, std::vector<detail::token>& out) {
  out.clear();
  detail::token t;
  while (scanner.next_in_line(t)) {
    std::size_t begin = 0;
    const std::string_view text = t.text;
    for (std::size_t i = 0; i <= text.size(); ++i) {
      if (i == text.size() || text[i] == ';') {
        if (i > begin) {
          out.push_back(
              {text.substr(begin, i - begin), t.line, t.column + static_cast<std::int64_t>(begin)});
        }
        if (i < text.size()) {
          out.push_back({text.substr(i, 1), t.line, t.column + static_cast<std::int64_t>(i)});
        }
        begin = i + 1;
      }
    }
  }
}

}  // namespace

template <typename vertex_t, typename weight_t>
std::vector<hyperedge_batch<vertex_t, weight_t>> read_hyperedge_batches(
    const std::string& path, const hyperedge_batch_file_options& options) try {
  constexpr bool weighted = !is_unweighted_v<weight_t>;
  DYNG_EXPECTS(
      options.num_vertices >= -1 && options.num_vertices <= detail::max_as_int64<vertex_t>(),
      "hyperedge_batch_file_options::num_vertices must be -1 or in [0, ",
      detail::max_as_int64<vertex_t>(), "], got ", options.num_vertices);
  const std::int64_t id_max =
      options.num_vertices >= 0 ? options.num_vertices - 1 : detail::max_as_int64<vertex_t>();
  const std::string text = detail::read_file(path);
  detail::text_scanner scanner(text, path);
  std::vector<hyperedge_batch<vertex_t, weight_t>> batches;
  std::vector<detail::token> tokens;
  std::vector<vertex_t> members;
  const auto on_op = [&](const detail::token& op_token) {
    const std::string_view op = op_token.text;
    if (op != "+h" && op != "-h") {
      return false;
    }
    split_line(scanner, tokens);
    if (op == "-h") {
      if (tokens.empty()) scanner.fail_line("incomplete operation (expected '-h id')");
      if (tokens.size() > 1) {
        scanner.fail(tokens[1],
                     "unexpected '" + std::string(tokens[1].text) + "' (the line is '-h id')");
      }
      batches.back().delete_hyperedge(detail::parse_integer<std::int64_t>(
          scanner, tokens[0], 0, std::numeric_limits<std::int64_t>::max(), "hyperedge id"));
      return true;
    }
    members.clear();
    std::size_t i = 0;
    for (; i < tokens.size() && tokens[i].text != ";"; ++i) {
      const auto v = detail::parse_integer<vertex_t>(scanner, tokens[i], 0,
                                                     detail::max_as_int64<vertex_t>(), "vertex id");
      if (static_cast<std::int64_t>(v) > id_max) {
        // The batch's vertex count is the hypergraph's: say where it is fixed (R030 review).
        scanner.fail(tokens[i], "vertex id " + std::to_string(static_cast<std::int64_t>(v)) +
                                    " out of range [0, " + std::to_string(id_max) +
                                    "]: the hypergraph has " + std::to_string(id_max + 1) +
                                    " vertices, fixed when it is built (its num_vertices)");
      }
      members.push_back(v);
    }
    if (members.empty()) {
      scanner.fail_line("a hyperedge insertion lists at least one vertex ('+h v1 v2 .. [; w]')");
    }
    weight_t w{};
    if (i < tokens.size()) {
      if constexpr (!weighted) {
        scanner.fail(tokens[i],
                     "a weight ('; w') in an unweighted hyperedge batch; read the "
                     "file with a weighted type");
      } else {
        if (i + 1 >= tokens.size() || tokens[i + 1].text == ";") {
          scanner.fail(tokens[i], "';' must be followed by the hyperedge's weight");
        }
        if (i + 2 < tokens.size()) {
          scanner.fail(tokens[i + 2], "unexpected '" + std::string(tokens[i + 2].text) +
                                          "' after the weight (the line is '+h v1 v2 .. ; w')");
        }
        w = detail::parse_integer<weight_t>(scanner, tokens[i + 1], 1,
                                            detail::max_as_int64<weight_t>(), "weight");
      }
    } else if constexpr (weighted) {
      scanner.fail_line(
          "a weighted hyperedge batch needs the weight of every insertion "
          "('+h v1 v2 .. ; w')");
    }
    batches.back().insert_hyperedge(host_view(members.data(), members.size()), w);
    return true;
  };
  scan_dgt(
      scanner, dgt_kind::hypergraph, [&] { batches.emplace_back(); },
      [&] { return !batches.empty(); }, on_op);
  return batches;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::read_hyperedge_batches (", path, ")")

template <typename vertex_t, typename weight_t>
void write_hyperedge_batches(
    const std::string& path,
    const std::vector<hyperedge_batch_view<vertex_t, weight_t>>& batches) try {
  constexpr bool weighted = !is_unweighted_v<weight_t>;
  for (std::size_t b = 0; b < batches.size(); ++b) {
    const auto& batch = batches[b];
    if (!batch.delete_incidence_hyperedge.empty() || !batch.delete_incidence_vertex.empty() ||
        !batch.insert_incidence_hyperedge.empty() || !batch.insert_incidence_vertex.empty()) {
      throw not_supported_error(detail::concat_message(
          "dyng: write_hyperedge_batches: batch ", b,
          " has incidence edits, which the .dgt files of this release do not hold (planned for "
          "0.4)"));
    }
    detail::check_hyperedge_batch_arrays(batch, "write_hyperedge_batches", b);
  }
  detail::text_writer out(path);
  out.put(std::string_view("%dgt 1 hypergraph\n"));
  for (std::size_t b = 0; b < batches.size(); ++b) {
    const auto& batch = batches[b];
    out.put(std::string_view("%batch "));
    out.put(static_cast<std::int64_t>(b));
    out.put_char('\n');
    for (std::size_t j = 0; j < batch.delete_hyperedges.size(); ++j) {
      out.put(std::string_view("-h "));
      out.put(batch.delete_hyperedges[j]);
      out.put_char('\n');
    }
    for (std::size_t i = 0; i < batch.num_insertions(); ++i) {
      out.put(std::string_view("+h"));
      const auto e = static_cast<std::size_t>(batch.insert_offsets[i + 1]);
      for (auto j = static_cast<std::size_t>(batch.insert_offsets[i]); j < e; ++j) {
        out.put_char(' ');
        out.put(static_cast<std::int64_t>(batch.insert_members[j]));
      }
      if constexpr (weighted) {
        out.put(std::string_view(" ; "));
        out.put(static_cast<std::int64_t>(batch.insert_weights[i]));
      }
      out.put_char('\n');
    }
  }
  out.close();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::write_hyperedge_batches (", path, ")")

#define DYNG_INSTANTIATE_BATCH_TEXT(V, W)                                               \
  template std::vector<edge_batch<V, W>> read_batches<V, W>(const std::string&,         \
                                                            const batch_file_options&); \
  template void write_batches<V, W>(const std::string&, const std::vector<edge_batch_view<V, W>>&);
DYNG_FOR_EACH_VERTEX_WEIGHT_TYPE(DYNG_INSTANTIATE_BATCH_TEXT)
DYNG_FOR_EACH_VERTEX_TYPE_UNWEIGHTED(DYNG_INSTANTIATE_BATCH_TEXT)
#undef DYNG_INSTANTIATE_BATCH_TEXT

#define DYNG_INSTANTIATE_HYPEREDGE_BATCH_TEXT(V, W)                         \
  template std::vector<hyperedge_batch<V, W>> read_hyperedge_batches<V, W>( \
      const std::string&, const hyperedge_batch_file_options&);             \
  template void write_hyperedge_batches<V, W>(const std::string&,           \
                                              const std::vector<hyperedge_batch_view<V, W>>&);
DYNG_FOR_EACH_HYPERGRAPH_TYPE(DYNG_INSTANTIATE_HYPEREDGE_BATCH_TEXT)
#undef DYNG_INSTANTIATE_HYPEREDGE_BATCH_TEXT

}  // namespace dyng::io
