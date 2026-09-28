// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/core/graph.cpp (parse_matrix_market_banner,
// parse_integer, parse_rows, parse_edge_file, group_edges, read_temporal_graph)
/**
 * @file edge_list_io.cpp
 * @brief read_edge_list() and write_edge_list().
 *
 * The reader is CycleEnumeration-GPU's parser: the file is read into memory and split into
 * newline-aligned parts that are parsed on std::thread workers with std::from_chars; the first
 * malformed line in file order is reported; vertex ids are compacted in ascending order (a dense
 * presence table when the id range is small, a parallel sort otherwise); one parallel sort groups
 * the rows by (source, target, timestamp). The changes are: names, io_error with path and line
 * instead of GraphParseError, the result as an edge_list (repeated pairs merged, or kept with their
 * timestamps), and the generalizations of PLAN Section 5.7 around the unchanged core: weight
 * columns before the optional timestamp, ids kept as they are (as_is), symmetrization, optional
 * self-loops and a thread cap. With the default options the vertices, their order and the edges
 * are exactly the original's.
 */
#include "io/io_instantiate.hpp"
#include "util/allocation.hpp"
#include "util/parallel_parts.hpp"
#include "util/parser.hpp"
#include "util/text_writer.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/types.hpp>
#include <dyng/io/edge_list_io.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

namespace dyng::io {

namespace {

/// One parsed row with the ids of the file.
struct raw_row {
  std::int64_t source = 0;
  std::int64_t target = 0;
  std::int64_t timestamp = 0;
};

/// The header of a Matrix Market file, from its "%%MatrixMarket" banner line.
struct matrix_market_header {
  bool present = false;
  /// symmetric, skew-symmetric and hermitian files store each off-diagonal entry once; the graph
  /// gets both directions.
  bool symmetric = false;
};

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

/// Parse "%%MatrixMarket matrix coordinate <field> <symmetry>". Only the coordinate (sparse)
/// format describes a graph; the dense array format and unknown qualifiers are rejected.
matrix_market_header parse_matrix_market_banner(const std::string& line, const std::string& path) {
  std::istringstream banner(line);
  std::vector<std::string> tokens;
  std::string token;
  while (banner >> token) {
    tokens.push_back(lowercase(token));
  }
  if (tokens.size() < 5 || tokens[1] != "matrix") {
    detail::throw_io_error(path, 1, 0,
                           "expected '%%MatrixMarket matrix coordinate <field> <symmetry>'");
  }
  if (tokens[2] != "coordinate") {
    detail::throw_io_error(
        path, 1, 0,
        "only the MatrixMarket coordinate format is supported, not '" + tokens[2] + "'");
  }
  const std::string& symmetry = tokens[4];
  matrix_market_header header;
  header.present = true;
  if (symmetry == "general") {
    header.symmetric = false;
  } else if (symmetry == "symmetric" || symmetry == "skew-symmetric" || symmetry == "hermitian") {
    header.symmetric = true;
  } else {
    detail::throw_io_error(path, 1, 0, "unknown MatrixMarket symmetry '" + tokens[4] + "'");
  }
  return header;
}

bool is_space(char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f';
}

bool is_separator(char c) {
  return is_space(c) || c == ',';
}

/// A signed decimal integer spanning exactly [begin, end), with an optional leading '+' (as
/// accepted by operator>>).
bool parse_integer(const char* begin, const char* end, std::int64_t& value) {
  if (begin != end && *begin == '+') {
    ++begin;
    if (begin == end || *begin == '-') {
      return false;
    }
  }
  const auto [position, error] = std::from_chars(begin, end, value);
  return error == std::errc{} && position == end;
}

/// A parse failure at byte `offset` of the file.
struct parse_failure {
  std::size_t offset = std::numeric_limits<std::size_t>::max();
  std::string reason;
};

/// How the rows of one file are read.
struct row_format {
  bool matrix_market = false;    ///< value columns are ignored
  bool both_directions = false;  ///< symmetric Matrix Market or options.symmetrize
  bool drop_self_loops = true;
  int num_weights = 0;
  std::int64_t weight_min = 0;  ///< range of weight_t
  std::int64_t weight_max = 0;
  bool as_is = false;  ///< ids minus index_base must lie in [0, id_bound)
  std::int64_t index_base = 0;
  std::int64_t id_bound = 0;
};

/// Parse the rows in [begin, end) of the buffer (whole lines). Comment and blank lines are
/// skipped, commas count as whitespace, and self-loops are dropped after the row is validated.
void parse_rows(const char* data, std::size_t begin, std::size_t end, const row_format& format,
                std::vector<raw_row>& rows, std::vector<std::int64_t>& weights,
                parse_failure& failure) {
  const auto k_count = static_cast<std::size_t>(format.num_weights);
  // Up to 4 + K fields; the last one only matters as an error (Matrix Market: 4, the rest are
  // value columns and ignored).
  const std::size_t max_fields = format.matrix_market ? 4 : 4 + k_count;
  const std::size_t required = format.matrix_market ? 2 : 2 + k_count;
  std::vector<std::string_view> fields(max_fields);
  std::vector<std::int64_t> row_weights(k_count);
  const auto parse = [](std::string_view field, std::int64_t& value) {
    return parse_integer(field.data(), field.data() + field.size(), value);
  };
  while (begin < end) {
    const char* line = data + begin;
    const char* line_end = static_cast<const char*>(std::memchr(line, '\n', end - begin));
    if (line_end == nullptr) {
      line_end = data + end;
    }
    const std::size_t next = static_cast<std::size_t>(line_end - data) + 1;

    const char* cursor = line;
    while (cursor < line_end && is_space(*cursor)) {
      ++cursor;
    }
    if (cursor == line_end || *cursor == '#' || *cursor == '%') {
      begin = next;
      continue;
    }

    std::size_t count = 0;
    while (cursor < line_end && count < max_fields) {
      while (cursor < line_end && is_separator(*cursor)) {
        ++cursor;
      }
      const char* field_begin = cursor;
      while (cursor < line_end && !is_separator(*cursor)) {
        ++cursor;
      }
      if (cursor > field_begin) {
        fields[count++] =
            std::string_view(field_begin, static_cast<std::size_t>(cursor - field_begin));
      }
    }

    std::int64_t source = 0;
    std::int64_t target = 0;
    std::int64_t timestamp = 0;  // a row without a timestamp gets 0
    if (count < required || !parse(fields[0], source) || !parse(fields[1], target)) {
      failure = {begin, k_count == 0 || format.matrix_market
                            ? "expected at least two fields: source target [timestamp]"
                            : "expected at least " + std::to_string(required) +
                                  " fields: source target w1.." + "w" + std::to_string(k_count) +
                                  " [timestamp]"};
      return;
    }
    if (!format.matrix_market) {
      for (std::size_t k = 0; k < k_count; ++k) {
        std::int64_t w = 0;
        if (!parse(fields[2 + k], w) || w < format.weight_min || w > format.weight_max) {
          failure = {begin, "weight " + std::to_string(k + 1) + " '" + std::string(fields[2 + k]) +
                                "' is not an integer in [" + std::to_string(format.weight_min) +
                                ", " + std::to_string(format.weight_max) + "]"};
          return;
        }
        row_weights[k] = w;
      }
      // Matrix Market value columns (weights) are not timestamps. Otherwise the field after the
      // weights must be an integer timestamp and nothing may follow it.
      if (count >= 3 + k_count) {
        if (!parse(fields[2 + k_count], timestamp)) {
          failure = {begin, k_count == 0 ? "third field is not an integer timestamp"
                                         : "the field after the weights is not an integer "
                                           "timestamp"};
          return;
        }
        if (count == 4 + k_count) {
          failure = {begin, k_count == 0 ? "expected at most three fields: source target timestamp"
                                         : "expected at most " + std::to_string(3 + k_count) +
                                               " fields: source target w1..w" +
                                               std::to_string(k_count) + " timestamp"};
          return;
        }
      }
    }
    if (format.as_is) {
      for (std::int64_t* id : {&source, &target}) {
        if (*id < format.index_base || *id - format.index_base >= format.id_bound) {
          failure = {begin, "vertex id " + std::to_string(*id) + " out of range [" +
                                std::to_string(format.index_base) + ", " +
                                std::to_string(format.index_base) + " + " +
                                std::to_string(format.id_bound) + ")"};
          return;
        }
        *id -= format.index_base;
      }
    }

    if (source != target || !format.drop_self_loops) {
      rows.push_back(raw_row{source, target, timestamp});
      weights.insert(weights.end(), row_weights.begin(), row_weights.end());
      if (format.both_directions && source != target) {
        rows.push_back(raw_row{target, source, timestamp});
        weights.insert(weights.end(), row_weights.begin(), row_weights.end());
      }
    }
    begin = next;
  }
}

/// Rows of the file in file order and their weights (K per row), parsed in parallel.
struct parsed_file {
  std::vector<raw_row> rows;
  std::vector<std::int64_t> weights;
  matrix_market_header header;
};

parsed_file parse_edge_file(const std::string& path, row_format format, bool symmetrize,
                            int max_threads) {
  const std::string data = detail::read_file(path);
  const std::size_t size = data.size();
  const auto line_end = [&](std::size_t from) {
    const void* found = from < size ? std::memchr(data.data() + from, '\n', size - from) : nullptr;
    return found == nullptr
               ? size
               : static_cast<std::size_t>(static_cast<const char*>(found) - data.data());
  };

  // The Matrix Market banner is the first line; its dimensions line is the first non-comment
  // line after it.
  parsed_file out;
  std::size_t body = 0;
  if (size >= 14 && std::memcmp(data.data(), "%%MatrixMarket", 14) == 0) {
    const std::size_t end = line_end(0);
    out.header = parse_matrix_market_banner(data.substr(0, end), path);
    if (format.num_weights != 0) {
      detail::throw_io_error(path, 1, 0,
                             "read_edge_list reads no weight columns from a Matrix Market file "
                             "(num_weights must be 0); use read_matrix_market");
    }
    body = end + 1;
    while (body < size) {
      const std::size_t end_of_line = line_end(body);
      std::size_t first = body;
      while (first < end_of_line && is_space(data[first])) {
        ++first;
      }
      body = end_of_line + 1;
      if (first < end_of_line && data[first] != '#' && data[first] != '%') {
        break;  // the dimensions line, skipped
      }
    }
    body = std::min(body, size);
  }
  format.matrix_market = out.header.present;
  format.both_directions = out.header.symmetric || symmetrize;

  const unsigned int threads = detail::worker_count(size - body, 1U << 20, max_threads);
  std::vector<std::size_t> starts(threads + 1, size);
  starts[0] = body;
  for (unsigned int t = 1; t < threads; ++t) {
    const std::size_t guess = body + (size - body) * t / threads;
    starts[t] = std::max(starts[t - 1], std::min(line_end(guess) + 1, size));
  }

  std::vector<std::vector<raw_row>> parts(threads);
  std::vector<std::vector<std::int64_t>> part_weights(threads);
  std::vector<parse_failure> failures(threads);
  detail::parallel_parts(threads, threads, [&](unsigned int, std::size_t first, std::size_t last) {
    for (std::size_t t = first; t < last; ++t) {
      parts[t].reserve((starts[t + 1] - starts[t]) / 12);
      parse_rows(data.data(), starts[t], starts[t + 1], format, parts[t], part_weights[t],
                 failures[t]);
    }
  });
  for (const parse_failure& failure : failures) {
    if (failure.offset != std::numeric_limits<std::size_t>::max()) {
      const auto line_number =
          1 + std::count(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(failure.offset),
                         '\n');
      detail::throw_io_error(path, static_cast<std::int64_t>(line_number), 0, failure.reason);
    }
  }

  std::size_t total = 0;
  std::vector<std::size_t> offsets(threads + 1, 0);
  for (unsigned int t = 0; t < threads; ++t) {
    offsets[t] = total;
    total += parts[t].size();
  }
  const auto k_count = static_cast<std::size_t>(format.num_weights);
  out.rows.resize(total);
  out.weights.resize(total * k_count);
  detail::parallel_parts(threads, threads, [&](unsigned int, std::size_t first, std::size_t last) {
    for (std::size_t t = first; t < last; ++t) {
      std::copy(parts[t].begin(), parts[t].end(),
                out.rows.begin() + static_cast<std::ptrdiff_t>(offsets[t]));
      std::copy(part_weights[t].begin(), part_weights[t].end(),
                out.weights.begin() + static_cast<std::ptrdiff_t>(offsets[t] * k_count));
      std::vector<raw_row>().swap(parts[t]);
      std::vector<std::int64_t>().swap(part_weights[t]);
    }
  });
  return out;
}

/// One row with graph ids (CycleEnumeration-GPU's CompactEdge).
template <typename vertex_t>
struct compact_row {
  vertex_t source;
  vertex_t target;
  std::int64_t timestamp;
};

/// One row with graph ids and its index in file order (for the weights of a weighted file).
template <typename vertex_t>
struct positioned_row {
  vertex_t source;
  vertex_t target;
  std::int64_t timestamp;
  std::size_t position;
};

/// Builds a row of type row_t from the i-th row of the file.
template <typename row_t, typename vertex_t>
row_t make_row(vertex_t source, vertex_t target, std::int64_t timestamp, std::size_t i) {
  if constexpr (std::is_same_v<row_t, positioned_row<vertex_t>>) {
    return row_t{source, target, timestamp, i};
  } else {
    (void)i;
    return row_t{source, target, timestamp};
  }
}

/// The file position of a row (0 for rows that do not carry it).
template <typename row_t>
std::size_t position_of(const row_t& row) {
  if constexpr (std::is_same_v<row_t, positioned_row<decltype(row.source)>>) {
    return row.position;
  } else {
    (void)row;
    return 0;
  }
}

/// group_edges(): graph ids for the rows (compact: ascending external order), then one parallel
/// sort by (source, target, timestamp) and, for positioned rows, file order.
template <typename vertex_t, typename row_t>
std::vector<row_t> group_rows(const std::string& path, std::vector<raw_row>& raw, bool compact,
                              int max_threads, std::vector<std::int64_t>& external_ids,
                              std::int64_t& num_vertices) {
  const std::size_t n = raw.size();
  std::vector<row_t> rows(n);
  num_vertices = 0;
  if (n == 0) {
    return rows;
  }
  const unsigned int threads = detail::worker_count(n, 1U << 16, max_threads);
  const auto vertex_max = static_cast<std::int64_t>(std::numeric_limits<vertex_t>::max());
  const auto too_many = [&](std::size_t count) {
    if (static_cast<std::uint64_t>(count) > static_cast<std::uint64_t>(vertex_max)) {
      detail::throw_io_error(path, 0, 0,
                             "the graph has " + std::to_string(count) +
                                 " vertices, more than the vertex id type holds");
    }
  };

  if (!compact) {  // ids as they are (already checked against the vertex type by the parser)
    std::vector<std::int64_t> high(threads, -1);
    detail::parallel_parts(n, threads, [&](unsigned int t, std::size_t first, std::size_t last) {
      for (std::size_t i = first; i < last; ++i) {
        high[t] = std::max({high[t], raw[i].source, raw[i].target});
        rows[i] = make_row<row_t>(static_cast<vertex_t>(raw[i].source),
                                  static_cast<vertex_t>(raw[i].target), raw[i].timestamp, i);
      }
    });
    num_vertices = *std::max_element(high.begin(), high.end()) + 1;
  } else {
    std::vector<std::int64_t> low(threads, std::numeric_limits<std::int64_t>::max());
    std::vector<std::int64_t> high(threads, std::numeric_limits<std::int64_t>::min());
    detail::parallel_parts(n, threads, [&](unsigned int t, std::size_t first, std::size_t last) {
      for (std::size_t i = first; i < last; ++i) {
        low[t] = std::min({low[t], raw[i].source, raw[i].target});
        high[t] = std::max({high[t], raw[i].source, raw[i].target});
      }
    });
    const std::int64_t min_id = *std::min_element(low.begin(), low.end());
    const std::int64_t max_id = *std::max_element(high.begin(), high.end());
    const auto range = static_cast<std::uint64_t>(max_id) - static_cast<std::uint64_t>(min_id);

    if (range < std::uint64_t{4} * n + 1024) {
      // Dense ids: a presence table and its prefix count give compact ids.
      const auto slots = static_cast<std::size_t>(range) + 1;
      std::vector<std::atomic<unsigned char>> present(slots);
      detail::parallel_parts(slots, detail::worker_count(slots, 1U << 16, max_threads),
                             [&](unsigned int, std::size_t first, std::size_t last) {
                               for (std::size_t i = first; i < last; ++i) {
                                 present[i].store(0, std::memory_order_relaxed);
                               }
                             });
      detail::parallel_parts(n, threads, [&](unsigned int, std::size_t first, std::size_t last) {
        for (std::size_t i = first; i < last; ++i) {
          present[static_cast<std::size_t>(raw[i].source - min_id)].store(
              1, std::memory_order_relaxed);
          present[static_cast<std::size_t>(raw[i].target - min_id)].store(
              1, std::memory_order_relaxed);
        }
      });
      std::vector<vertex_t> compact_id(slots);
      std::size_t next = 0;
      for (std::size_t i = 0; i < slots; ++i) {
        compact_id[i] = static_cast<vertex_t>(next);
        if (present[i].load(std::memory_order_relaxed) != 0) {
          external_ids.push_back(min_id + static_cast<std::int64_t>(i));
          ++next;
          too_many(next);
        }
      }
      detail::parallel_parts(n, threads, [&](unsigned int, std::size_t first, std::size_t last) {
        for (std::size_t i = first; i < last; ++i) {
          rows[i] = make_row<row_t>(compact_id[static_cast<std::size_t>(raw[i].source - min_id)],
                                    compact_id[static_cast<std::size_t>(raw[i].target - min_id)],
                                    raw[i].timestamp, i);
        }
      });
    } else {
      // Sparse ids: sort and deduplicate them, then binary search.
      std::vector<std::int64_t> ids(2 * n);
      detail::parallel_parts(n, threads, [&](unsigned int, std::size_t first, std::size_t last) {
        for (std::size_t i = first; i < last; ++i) {
          ids[2 * i] = raw[i].source;
          ids[2 * i + 1] = raw[i].target;
        }
      });
      detail::parallel_sort(ids, std::less<std::int64_t>{}, max_threads);
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      too_many(ids.size());
      const auto compact_id = [&](std::int64_t id) {
        return static_cast<vertex_t>(std::lower_bound(ids.begin(), ids.end(), id) - ids.begin());
      };
      detail::parallel_parts(n, threads, [&](unsigned int, std::size_t first, std::size_t last) {
        for (std::size_t i = first; i < last; ++i) {
          rows[i] = make_row<row_t>(compact_id(raw[i].source), compact_id(raw[i].target),
                                    raw[i].timestamp, i);
        }
      });
      external_ids = std::move(ids);
    }
    num_vertices = static_cast<std::int64_t>(external_ids.size());
  }
  std::vector<raw_row>().swap(raw);

  // One sort groups equal (source, target) pairs and orders their timestamps (and file order).
  detail::parallel_sort(
      rows,
      [](const row_t& a, const row_t& b) {
        if (a.source != b.source) {
          return a.source < b.source;
        }
        if (a.target != b.target) {
          return a.target < b.target;
        }
        if constexpr (std::is_same_v<row_t, positioned_row<vertex_t>>) {
          if (a.timestamp != b.timestamp) {
            return a.timestamp < b.timestamp;
          }
          return a.position < b.position;
        } else {
          return a.timestamp < b.timestamp;
        }
      },
      max_threads);
  return rows;
}

/// The edges of the grouped rows: every row (keep) or one per (source, target) run (merge; the
/// weights of the run's row that is last in file order).
template <typename row_t, typename vertex_t, typename weight_t>
void fill_edges(const std::vector<row_t>& rows, const std::vector<std::int64_t>& weights,
                const edge_list_options& options, edge_list<vertex_t, weight_t>& out,
                std::vector<std::int64_t>& timestamps) {
  const auto k_count = static_cast<std::size_t>(options.num_weights);
  const bool keep = options.duplicates == duplicate_edges::keep;
  out.num_weights = options.num_weights;
  std::vector<std::size_t> chosen;
  chosen.reserve(rows.size());
  for (std::size_t i = 0; i < rows.size();) {
    std::size_t j = i + 1;
    if (keep) {
      chosen.push_back(i);
    } else {
      std::size_t last = i;
      while (j < rows.size() && rows[j].source == rows[i].source &&
             rows[j].target == rows[i].target) {
        if (position_of(rows[j]) > position_of(rows[last])) {
          last = j;
        }
        ++j;
      }
      chosen.push_back(last);
    }
    i = j;
  }
  out.src.resize(chosen.size());
  out.dst.resize(chosen.size());
  if (keep) {
    timestamps.resize(chosen.size());
  }
  for (std::size_t e = 0; e < chosen.size(); ++e) {
    const row_t& row = rows[chosen[e]];
    out.src[e] = row.source;
    out.dst[e] = row.target;
    if (keep) {
      timestamps[e] = row.timestamp;
    }
  }
  if constexpr (!is_unweighted_v<weight_t>) {
    out.weights.resize(chosen.size() * k_count);
    for (std::size_t e = 0; e < chosen.size(); ++e) {
      const std::size_t position = position_of(rows[chosen[e]]);
      for (std::size_t k = 0; k < k_count; ++k) {
        out.weights[e * k_count + k] = static_cast<weight_t>(weights[position * k_count + k]);
      }
    }
  } else {
    (void)weights;
    (void)k_count;
  }
}

}  // namespace

template <typename vertex_t, typename weight_t>
edge_list<vertex_t, weight_t> read_edge_list(const std::string& path,
                                             const edge_list_options& options,
                                             edge_list_info* info) try {
  DYNG_EXPECTS(options.num_weights >= 0, "read_edge_list: num_weights must be >= 0, got ",
               options.num_weights);
  DYNG_EXPECTS(!is_unweighted_v<weight_t> || options.num_weights == 0,
               "read_edge_list: an unweighted edge list has no weight columns, got num_weights = ",
               options.num_weights);
  row_format format;
  format.drop_self_loops = options.drop_self_loops;
  format.num_weights = options.num_weights;
  if constexpr (!is_unweighted_v<weight_t>) {
    static_assert(std::is_integral_v<weight_t>, "read_edge_list reads integer weights");
    format.weight_min = static_cast<std::int64_t>(std::numeric_limits<weight_t>::min());
    format.weight_max = detail::max_as_int64<weight_t>();
  }
  format.as_is = options.ids == vertex_ids::as_is;
  format.index_base = options.index_base;
  format.id_bound = detail::max_as_int64<vertex_t>();
  parsed_file file = parse_edge_file(path, format, options.symmetrize, options.threads);

  std::vector<std::int64_t> external_ids;
  std::int64_t num_vertices = 0;
  std::vector<std::int64_t> timestamps;
  edge_list<vertex_t, weight_t> out;
  if (options.num_weights == 0) {
    const auto rows = group_rows<vertex_t, compact_row<vertex_t>>(
        path, file.rows, options.ids == vertex_ids::compact, options.threads, external_ids,
        num_vertices);
    fill_edges(rows, file.weights, options, out, timestamps);
  } else {
    const auto rows = group_rows<vertex_t, positioned_row<vertex_t>>(
        path, file.rows, options.ids == vertex_ids::compact, options.threads, external_ids,
        num_vertices);
    fill_edges(rows, file.weights, options, out, timestamps);
  }
  out.num_vertices = static_cast<vertex_t>(num_vertices);
  if (info != nullptr) {
    info->external_ids = std::move(external_ids);
    info->timestamps = std::move(timestamps);
    info->matrix_market = file.header.present;
    info->symmetric = file.header.symmetric;
  }
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::read_edge_list (", path, ")")

template <typename vertex_t, typename weight_t>
void write_edge_list(const std::string& path, const edge_list_view<vertex_t, weight_t>& edges) try {
  const std::size_t m = edges.src.size();
  const auto k_count = static_cast<std::size_t>(edges.num_weights);
  DYNG_EXPECTS(
      edges.num_weights >= 0 && edges.dst.size() == m && edges.weights.size() == m * k_count,
      "write_edge_list: the edge list arrays disagree in size");
  DYNG_EXPECTS((m == 0 ||
                (is_host_accessible(edges.src.space()) && is_host_accessible(edges.dst.space()))) &&
                   (edges.weights.empty() || is_host_accessible(edges.weights.space())),
               "write_edge_list: the edge list must be in host-accessible memory");
  detail::text_writer out(path);
  for (std::size_t e = 0; e < m; ++e) {
    out.put(static_cast<std::int64_t>(edges.src[e]));
    out.put_char(' ');
    out.put(static_cast<std::int64_t>(edges.dst[e]));
    if constexpr (!is_unweighted_v<weight_t>) {
      for (std::size_t k = 0; k < k_count; ++k) {
        out.put_char(' ');
        out.put(static_cast<std::int64_t>(edges.weights[e * k_count + k]));
      }
    }
    out.put_char('\n');
  }
  out.close();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::write_edge_list (", path, ")")

#define DYNG_INSTANTIATE_EDGE_LIST_IO(V, W)                                                   \
  template edge_list<V, W> read_edge_list<V, W>(const std::string&, const edge_list_options&, \
                                                edge_list_info*);                             \
  template void write_edge_list<V, W>(const std::string&, const edge_list_view<V, W>&);
DYNG_FOR_EACH_VERTEX_WEIGHT_TYPE(DYNG_INSTANTIATE_EDGE_LIST_IO)
DYNG_FOR_EACH_VERTEX_TYPE_UNWEIGHTED(DYNG_INSTANTIATE_EDGE_LIST_IO)
#undef DYNG_INSTANTIATE_EDGE_LIST_IO

}  // namespace dyng::io
