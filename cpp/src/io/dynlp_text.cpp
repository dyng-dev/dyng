// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/generateSparseGraph.cu (writeGraphFiles)
// Derived from LabelPropagation-CUDA@a276a3a:src/utils.cu (writeLabelsToFile)
/**
 * @file dynlp_text.cpp
 * @brief O's graph and label text files (see dynlp_text.hpp).
 */
#include "io/dynlp_text.hpp"

#include "util/text_writer.hpp"

#include <dyng/core/error.hpp>

#include <cstddef>

namespace dyng::detail {

template <typename vertex_t, typename edge_t>
void write_dynlp_graph_files(const std::string& prefix, const std::vector<edge_t>& row_ptr,
                             const std::vector<vertex_t>& col_ind, const std::vector<float>& values,
                             const std::vector<std::int8_t>& true_labels) {
  DYNG_EXPECTS(!row_ptr.empty() && row_ptr.size() == true_labels.size() + 1 &&
                   col_ind.size() == values.size() &&
                   static_cast<std::size_t>(row_ptr.back()) == col_ind.size(),
               "write_dynlp_graph_files: inconsistent arrays (", row_ptr.size(), " offsets, ",
               col_ind.size(), " columns, ", values.size(), " values, ", true_labels.size(),
               " labels)");
  {
    text_writer out(prefix + "RowPtr.txt");
    for (const edge_t offset : row_ptr) {
      out.put(static_cast<std::int64_t>(offset));
      out.put_char('\n');
    }
    out.close();
  }
  {
    text_writer out(prefix + "ColInd.txt");
    for (const vertex_t c : col_ind) {
      out.put(static_cast<std::int64_t>(c));
      out.put_char('\n');
    }
    out.close();
  }
  {
    text_writer out(prefix + "Values.txt");
    for (const float w : values) {
      out.put_fixed6(w);
      out.put_char('\n');
    }
    out.close();
  }
  {
    // O: trueLabels[i] >= 0.5f ? 1 : 0 of the float labels 0.0f / 1.0f.
    text_writer out(prefix + "TrueLabels.txt");
    for (std::size_t i = 0; i < true_labels.size(); ++i) {
      out.put(static_cast<std::int64_t>(i));
      out.put_char(' ');
      out.put(static_cast<std::int64_t>(true_labels[i] >= 1 ? 1 : 0));
      out.put_char('\n');
    }
    out.close();
  }
}

void write_dynlp_labels(const std::string& path, const std::vector<float>& labels) {
  text_writer out(path);
  for (std::size_t i = 0; i < labels.size(); ++i) {
    out.put(static_cast<std::int64_t>(i));
    out.put_char(' ');
    out.put_fixed6(labels[i]);
    out.put_char('\n');
  }
  out.close();
}

template void write_dynlp_graph_files<std::int32_t, std::int32_t>(const std::string&,
                                                                  const std::vector<std::int32_t>&,
                                                                  const std::vector<std::int32_t>&,
                                                                  const std::vector<float>&,
                                                                  const std::vector<std::int8_t>&);
template void write_dynlp_graph_files<std::int32_t, std::int64_t>(const std::string&,
                                                                  const std::vector<std::int64_t>&,
                                                                  const std::vector<std::int32_t>&,
                                                                  const std::vector<float>&,
                                                                  const std::vector<std::int8_t>&);

}  // namespace dyng::detail
