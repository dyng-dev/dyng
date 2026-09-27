// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:src/csrGraph.cu (readDistances, readParents, writeDistances,
// writeParents)
/**
 * @file result_io.cpp
 * @brief Distance and SSSP-tree files in the MOSP formats.
 */
#include "io/io_instantiate.hpp"
#include "util/parser.hpp"
#include "util/text_writer.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/types.hpp>
#include <dyng/io/result_io.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace dyng::io {

namespace {

/// Reads "v x" lines; `parse_value` turns the second token into a value. Every vertex exactly
/// once.
template <typename value_t, typename parse_value_t>
std::vector<value_t> read_vertex_values(const std::string& path, std::int64_t num_vertices,
                                        value_t unset, const parse_value_t& parse_value) {
  DYNG_EXPECTS(num_vertices >= 0, "num_vertices must be >= 0, got ", num_vertices);
  const std::string text = detail::read_file(path);
  detail::text_scanner scanner(text, path);
  std::vector<value_t> values(static_cast<std::size_t>(num_vertices), unset);
  std::vector<char> seen(static_cast<std::size_t>(num_vertices), 0);
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
void write_distances(const std::string& path, array_view<const distance_t> distances) {
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

template <typename vertex_t>
void write_parents(const std::string& path, array_view<const vertex_t> parents) {
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

template <typename distance_t>
std::vector<distance_t> read_distances(const std::string& path, std::int64_t num_vertices) {
  return read_vertex_values<distance_t>(
      path, num_vertices, infinite_distance<distance_t>(),
      [](const detail::text_scanner& scanner, const detail::token& tok) {
        if (tok.text == "INF") {
          return infinite_distance<distance_t>();
        }
        return detail::parse_integer<distance_t>(scanner, tok, 0,
                                                 detail::max_as_int64<distance_t>(), "distance");
      });
}

template <typename vertex_t>
std::vector<vertex_t> read_parents(const std::string& path, std::int64_t num_vertices) {
  return read_vertex_values<vertex_t>(
      path, num_vertices, invalid_id<vertex_t>(),
      [num_vertices](const detail::text_scanner& scanner, const detail::token& tok) {
        return detail::parse_integer<vertex_t>(scanner, tok, -1, num_vertices - 1, "parent");
      });
}

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

}  // namespace dyng::io
