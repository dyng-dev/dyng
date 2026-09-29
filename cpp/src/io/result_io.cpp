// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:src/csrGraph.cu (readDistances, readParents, writeDistances,
// writeParents)
// Derived from CycleEnumeration-GPU@0a976ad:src/core/histogram.cpp (CycleHistogram::to_csv)
/**
 * @file result_io.cpp
 * @brief Distance and SSSP-tree files in the MOSP formats.
 */
#include "io/io_instantiate.hpp"
#include "util/allocation.hpp"
#include "util/parser.hpp"
#include "util/text_writer.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/types.hpp>
#include <dyng/io/result_io.hpp>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace dyng::io {

namespace {

/// The separators of the strict scanner inside a line.
inline bool is_blank(char c) noexcept {
  return c == ' ' || c == '\t' || c == '\r';
}

/// Parse the integer token at `p` in one pass: std::from_chars, and the token must end right after
/// the number (at a blank, a line break or the end), which is exactly the strict rule "the whole
/// token is one integer representable in std::int64_t" (detail::parse_int64).
inline const char* parse_token_int64(const char* p, const char* end, std::int64_t& value) noexcept {
  const auto result = std::from_chars(p, end, value);
  if (result.ec != std::errc() ||
      (result.ptr < end && !is_blank(*result.ptr) && *result.ptr != '\n')) {
    return nullptr;
  }
  return result.ptr;
}

/// One pass over a well-formed "v x" file (MOSP's readDistances / readParents way: std::from_chars
/// straight on the text). It accepts exactly what the strict reader accepts or gives up: any
/// irregularity (a malformed or out-of-range token, a missing, extra or repeated entry) returns
/// false, and the caller runs the strict reader, which reports the first problem with its line and
/// column. `parse_value(p, end, value)` parses the second token and returns the position after
/// it, or nullptr.
template <typename value_t, typename parse_value_t>
bool read_vertex_values_fast(const std::string& text, std::int64_t num_vertices,
                             std::vector<value_t>& values, std::vector<char>& seen,
                             const parse_value_t& parse_value) {
  const char* p = text.data();
  const char* const end = p + text.size();
  std::int64_t listed = 0;
  while (p < end) {
    while (p < end && is_blank(*p)) {
      ++p;
    }
    if (p == end) {
      break;
    }
    if (*p == '\n') {
      ++p;
      continue;
    }
    std::int64_t v = 0;
    p = parse_token_int64(p, end, v);
    if (p == nullptr || v < 0 || v >= num_vertices || seen[static_cast<std::size_t>(v)] != 0) {
      return false;
    }
    while (p < end && is_blank(*p)) {
      ++p;
    }
    if (p == end || *p == '\n') {
      return false;
    }
    value_t value{};
    p = parse_value(p, end, value);
    if (p == nullptr) {
      return false;
    }
    while (p < end && is_blank(*p)) {
      ++p;
    }
    if (p < end) {
      if (*p != '\n') {
        return false;  // a third token
      }
      ++p;
    }
    seen[static_cast<std::size_t>(v)] = 1;
    values[static_cast<std::size_t>(v)] = value;
    ++listed;
  }
  return listed == num_vertices;
}

/// Reads "v x" lines; `parse_value` turns the second token into a value. Every vertex exactly
/// once. `parse_fast(p, end, value)` is the same rule for the fast pass (nullptr: give up).
template <typename value_t, typename parse_value_t, typename parse_fast_t>
std::vector<value_t> read_vertex_values(const std::string& path, std::int64_t num_vertices,
                                        value_t unset, const parse_value_t& parse_value,
                                        const parse_fast_t& parse_fast) {
  DYNG_EXPECTS(num_vertices >= 0, "num_vertices must be >= 0, got ", num_vertices);
  const std::string text = detail::read_file(path);
  std::vector<value_t> values(static_cast<std::size_t>(num_vertices), unset);
  std::vector<char> seen(static_cast<std::size_t>(num_vertices), 0);
  if (read_vertex_values_fast(text, num_vertices, values, seen, parse_fast)) {
    return values;
  }
  // The strict pass: the same rules token by token, with the location of the first problem.
  std::fill(values.begin(), values.end(), unset);
  std::fill(seen.begin(), seen.end(), char{0});
  detail::text_scanner scanner(text, path);
  std::int64_t listed = 0;
  detail::token vertex_token;
  detail::token value_token;
  detail::token extra;
  while (scanner.next_line()) {
    if (!scanner.next_in_line(vertex_token)) {
      continue;
    }
    if (!scanner.next_in_line(value_token)) {
      scanner.fail_line("a line holds exactly 2 values 'v x'");
    }
    if (scanner.next_in_line(extra)) {
      scanner.fail(extra, "a line holds exactly 2 values 'v x'");
    }
    const auto v = detail::parse_integer<std::int64_t>(scanner, vertex_token, 0, num_vertices - 1,
                                                       "vertex id");
    if (seen[static_cast<std::size_t>(v)] != 0) {
      scanner.fail(vertex_token, "vertex " + std::to_string(v) + " is listed twice");
    }
    seen[static_cast<std::size_t>(v)] = 1;
    values[static_cast<std::size_t>(v)] = parse_value(scanner, value_token);
    ++listed;
  }
  if (listed != num_vertices) {
    detail::throw_io_error(path, 0, 0,
                           "the file must list each of the " + std::to_string(num_vertices) +
                               " vertices exactly once (" + std::to_string(listed) + " listed)");
  }
  return values;
}

template <typename value_t>
void expect_host_array(const array_view<const value_t>& values, const char* what) {
  DYNG_EXPECTS(values.empty() || is_host_accessible(values.space()), what,
               ": the array must be in host-accessible memory");
}

}  // namespace

template <typename distance_t>
void write_distances(const std::string& path, array_view<const distance_t> distances) try {
  expect_host_array(distances, "write_distances");
  detail::text_writer out(path);
  const distance_t unreachable = infinite_distance<distance_t>() / 2;
  for (std::size_t v = 0; v < distances.size(); ++v) {
    out.put(static_cast<std::int64_t>(v));
    out.put_char(' ');
    if (distances[v] >= unreachable) {
      out.put("INF");
    } else {
      out.put(static_cast<std::int64_t>(distances[v]));
    }
    out.put_char('\n');
  }
  out.close();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::write_distances (", path, ")")

template <typename vertex_t>
void write_parents(const std::string& path, array_view<const vertex_t> parents) try {
  expect_host_array(parents, "write_parents");
  detail::text_writer out(path);
  for (std::size_t v = 0; v < parents.size(); ++v) {
    out.put(static_cast<std::int64_t>(v));
    out.put_char(' ');
    out.put(static_cast<std::int64_t>(parents[v]));
    out.put_char('\n');
  }
  out.close();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::write_parents (", path, ")")

template <typename distance_t>
std::vector<distance_t> read_distances(const std::string& path, std::int64_t num_vertices) try {
  return read_vertex_values<distance_t>(
      path, num_vertices, infinite_distance<distance_t>(),
      [](const detail::text_scanner& scanner, const detail::token& tok) {
        if (tok.text == "INF") {
          return infinite_distance<distance_t>();
        }
        return detail::parse_integer<distance_t>(scanner, tok, 0,
                                                 detail::max_as_int64<distance_t>(), "distance");
      },
      [](const char* p, const char* end, distance_t& value) -> const char* {
        if (end - p >= 3 && p[0] == 'I' && p[1] == 'N' && p[2] == 'F' &&
            (end - p == 3 || is_blank(p[3]) || p[3] == '\n')) {
          value = infinite_distance<distance_t>();
          return p + 3;
        }
        std::int64_t parsed = 0;
        p = parse_token_int64(p, end, parsed);
        if (p == nullptr || parsed < 0 || parsed > detail::max_as_int64<distance_t>()) {
          return nullptr;
        }
        value = static_cast<distance_t>(parsed);
        return p;
      });
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::read_distances (", path, ", ", num_vertices, " vertices)")

template <typename vertex_t>
std::vector<vertex_t> read_parents(const std::string& path, std::int64_t num_vertices) try {
  return read_vertex_values<vertex_t>(
      path, num_vertices, invalid_id<vertex_t>(),
      [num_vertices](const detail::text_scanner& scanner, const detail::token& tok) {
        return detail::parse_integer<vertex_t>(scanner, tok, -1, num_vertices - 1, "parent");
      },
      [num_vertices](const char* p, const char* end, vertex_t& value) -> const char* {
        std::int64_t parsed = 0;
        p = parse_token_int64(p, end, parsed);
        if (p == nullptr || parsed < -1 || parsed > num_vertices - 1) {
          return nullptr;
        }
        value = static_cast<vertex_t>(parsed);
        return p;
      });
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::read_parents (", path, ", ", num_vertices, " vertices)")

#define DYNG_INSTANTIATE_DISTANCE_IO(D)                                      \
  template void write_distances<D>(const std::string&, array_view<const D>); \
  template std::vector<D> read_distances<D>(const std::string&, std::int64_t);
#define DYNG_INSTANTIATE_PARENT_IO(V)                                      \
  template void write_parents<V>(const std::string&, array_view<const V>); \
  template std::vector<V> read_parents<V>(const std::string&, std::int64_t);
DYNG_FOR_EACH_DISTANCE_TYPE(DYNG_INSTANTIATE_DISTANCE_IO)
DYNG_FOR_EACH_VERTEX_TYPE(DYNG_INSTANTIATE_PARENT_IO)
#undef DYNG_INSTANTIATE_DISTANCE_IO
#undef DYNG_INSTANTIATE_PARENT_IO

namespace {

/// The text of CycleHistogram::to_csv() (throws capacity_error before anything is returned).
std::string histogram_csv_text(array_view<const std::uint64_t> counts, bool include_total) {
  std::string out = "# cycle_size, num_of_cycles\n";
  std::uint64_t total = 0;
  for (std::size_t length = 0; length < counts.size(); ++length) {
    const std::uint64_t count = counts[length];
    if (count == 0) {
      continue;  // CycleHistogram keeps no entry for a zero count
    }
    out += std::to_string(length);
    out += ", ";
    out += std::to_string(count);
    out += '\n';
    if (count > std::numeric_limits<std::uint64_t>::max() - total) {
      throw capacity_error("dyng: io::write_histogram_csv: the total exceeds 2^64 - 1");
    }
    total += count;
  }
  if (include_total) {
    out += "Total, ";
    out += std::to_string(total);
    out += '\n';
  }
  return out;
}

}  // namespace

void write_histogram_csv(std::ostream& out, array_view<const std::uint64_t> counts,
                         bool include_total) try {
  expect_host_array(counts, "write_histogram_csv");
  const std::string text = histogram_csv_text(counts, include_total);
  out << text;
  if (!out) {
    throw io_error("dyng: io::write_histogram_csv: writing to the stream failed");
  }
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::write_histogram_csv (", counts.size(), " lengths)")

}  // namespace dyng::io
