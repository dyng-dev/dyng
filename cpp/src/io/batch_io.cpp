// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:src/csrGraph.cu (readChangeBatch) and
// MOSP-OpenMP@c352151:src/changeGenerator.cpp (writeChangeBatch)
/**
 * @file batch_io.cpp
 * @brief Legacy MOSP batch files (insert.txt / delete.txt).
 */
#include "io/io_instantiate.hpp"
#include "util/allocation.hpp"
#include "util/parser.hpp"
#include "util/text_writer.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/io/batch_io.hpp>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace dyng::io {

template <typename vertex_t, typename weight_t>
edge_batch<vertex_t, weight_t> read_legacy_batch(const std::string& insert_path,
                                                 const std::string& delete_path,
                                                 const legacy_batch_options& options) try {
  static_assert(std::is_integral_v<weight_t>, "read_legacy_batch needs integral weights");
  DYNG_EXPECTS(options.num_weights >= 0, "legacy_batch_options::num_weights must be >= 0, got ",
               options.num_weights);
  const std::int64_t id_max =
      options.num_vertices >= 0 ? options.num_vertices - 1 : detail::max_as_int64<vertex_t>();
  DYNG_EXPECTS(id_max <= detail::max_as_int64<vertex_t>(),
               "legacy_batch_options::num_vertices does not fit the vertex id type");
  const auto weight_max = detail::max_as_int64<weight_t>();
  const auto k_count = static_cast<std::size_t>(options.num_weights);
  edge_batch<vertex_t, weight_t> batch(options.num_weights);
  std::vector<weight_t> weights(k_count);
  detail::token tok;

  {
    const std::string text = detail::read_file(insert_path);
    detail::text_scanner scanner(text, insert_path);
    const std::size_t expected = 2 + k_count;
    while (scanner.next_line()) {
      std::size_t count = 0;
      vertex_t ends[2] = {0, 0};
      while (scanner.next_in_line(tok)) {
        if (count >= expected) {
          if (!options.mosp_lenient) {
            scanner.fail(tok, "an insertion line holds exactly " + std::to_string(expected) +
                                  " integers 'u v w1 .. wK' (K = " +
                                  std::to_string(options.num_weights) + ")");
          }
          // MOSP's readChangeBatch reads the extra integers and ignores them.
          (void)detail::parse_integer<std::int64_t>(scanner, tok, INT64_MIN, INT64_MAX, "integer");
          ++count;
          continue;
        }
        if (count < 2) {
          ends[count] = detail::parse_integer<vertex_t>(scanner, tok, 0, id_max, "vertex id");
        } else {
          weights[count - 2] =
              detail::parse_integer<weight_t>(scanner, tok, 1, weight_max, "weight");
        }
        ++count;
      }
      if (count == 0) {
        continue;
      }
      if (count < expected || (count != expected && !options.mosp_lenient)) {
        scanner.fail_line("an insertion line holds exactly " + std::to_string(expected) +
                          " integers 'u v w1 .. wK' (K = " + std::to_string(options.num_weights) +
                          "), found " + std::to_string(count));
      }
      batch.insert_edge(ends[0], ends[1], host_view(weights.data(), weights.size()));
    }
  }
  {
    const std::string text = detail::read_file(delete_path);
    detail::text_scanner scanner(text, delete_path);
    while (scanner.next_line()) {
      std::size_t count = 0;
      vertex_t ends[2] = {0, 0};
      std::int64_t values[2] = {0, 0};  // lenient mode: the first two integers
      detail::token pending[2] = {};    // ... and their tokens (for the error location)
      while (scanner.next_in_line(tok)) {
        if (count >= 2) {
          if (!options.mosp_lenient) {
            scanner.fail(tok, "a deletion line holds exactly 2 integers 'u v'");
          }
          // MOSP's readChangeBatch reads the extra integers and ignores them.
          (void)detail::parse_integer<std::int64_t>(scanner, tok, INT64_MIN, INT64_MAX, "integer");
          ++count;
          continue;
        }
        if (options.mosp_lenient) {
          // MOSP checks the range only for lines it keeps (a line with one integer is ignored).
          const auto value =
              detail::parse_integer<std::int64_t>(scanner, tok, INT64_MIN, INT64_MAX, "vertex id");
          pending[count] = tok;
          values[count++] = value;
          continue;
        }
        ends[count++] = detail::parse_integer<vertex_t>(scanner, tok, 0, id_max, "vertex id");
      }
      if (count == 0) {
        continue;
      }
      if (options.mosp_lenient) {
        if (count == 1) {
          continue;  // "shorter lines are ignored" (MOSP readChangeBatch)
        }
        for (int i = 0; i < 2; ++i) {
          if (values[i] < 0 || values[i] > id_max) {
            scanner.fail(pending[i], "vertex id " + std::to_string(values[i]) +
                                         " is out of range [0, " + std::to_string(id_max) + "]");
          }
          ends[i] = static_cast<vertex_t>(values[i]);
        }
      } else if (count != 2) {
        scanner.fail_line("a deletion line holds exactly 2 integers 'u v'");
      }
      batch.delete_edge(ends[0], ends[1]);
    }
  }
  return batch;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::read_legacy_batch (", insert_path, ", ", delete_path, ")")

template <typename vertex_t, typename weight_t>
void write_legacy_batch(const std::string& insert_path, const std::string& delete_path,
                        const edge_batch_view<vertex_t, weight_t>& batch) try {
  const std::size_t num_inserts = batch.insert_src.size();
  const auto k_count = static_cast<std::size_t>(batch.num_weights);
  DYNG_EXPECTS(batch.insert_vertices.empty() && batch.delete_vertices.empty(),
               "write_legacy_batch: vertex operations cannot be written to insert/delete files");
  DYNG_EXPECTS(batch.num_weights >= 0 && batch.insert_dst.size() == num_inserts &&
                   batch.insert_weights.size() == num_inserts * k_count &&
                   batch.delete_dst.size() == batch.delete_src.size(),
               "write_legacy_batch: the batch arrays disagree in size");
  const auto host = [](const auto& view) {
    return view.empty() || is_host_accessible(view.space());
  };
  DYNG_EXPECTS(host(batch.insert_src) && host(batch.insert_dst) && host(batch.insert_weights) &&
                   host(batch.delete_src) && host(batch.delete_dst),
               "write_legacy_batch: the batch must be in host-accessible memory");
  detail::text_writer inserts(insert_path);
  detail::text_writer deletes(delete_path);
  for (std::size_t i = 0; i < num_inserts; ++i) {
    inserts.put(static_cast<std::int64_t>(batch.insert_src[i]));
    inserts.put_char(' ');
    inserts.put(static_cast<std::int64_t>(batch.insert_dst[i]));
    for (std::size_t k = 0; k < k_count; ++k) {
      inserts.put_char(' ');
      inserts.put(static_cast<std::int64_t>(batch.insert_weights[i * k_count + k]));
    }
    inserts.put_char('\n');
  }
  for (std::size_t j = 0; j < batch.delete_src.size(); ++j) {
    deletes.put(static_cast<std::int64_t>(batch.delete_src[j]));
    deletes.put_char(' ');
    deletes.put(static_cast<std::int64_t>(batch.delete_dst[j]));
    deletes.put_char('\n');
  }
  inserts.close();
  deletes.close();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::write_legacy_batch (", insert_path, ", ", delete_path, ")")

#define DYNG_INSTANTIATE_BATCH_IO(V, W)                                                     \
  template edge_batch<V, W> read_legacy_batch<V, W>(const std::string&, const std::string&, \
                                                    const legacy_batch_options&);           \
  template void write_legacy_batch<V, W>(const std::string&, const std::string&,            \
                                         const edge_batch_view<V, W>&);
DYNG_FOR_EACH_VERTEX_WEIGHT_TYPE(DYNG_INSTANTIATE_BATCH_IO)
#undef DYNG_INSTANTIATE_BATCH_IO

}  // namespace dyng::io
