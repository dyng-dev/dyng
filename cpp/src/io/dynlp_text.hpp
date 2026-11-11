// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/generateSparseGraph.cu (writeGraphFiles)
// Derived from LabelPropagation-CUDA@a276a3a:src/utils.cu (writeLabelsToFile)
/**
 * @file dynlp_text.hpp
 * @brief The text files of LabelPropagation-CUDA (internal): the generated graph
 *        (`<prefix>{RowPtr,ColInd,Values,TrueLabels}.txt`) and the label file
 *        (`dynlp_labels.txt`), byte for byte as the original writes them.
 *
 * The formats (O's writeGraphFiles and writeLabelsToFile, through its TextWriter):
 *   - RowPtr.txt: one decimal offset per line, num_vertices + 1 lines;
 *   - ColInd.txt: one decimal column per line;
 *   - Values.txt: one weight per line, `std::to_chars(double(w), fixed, 6)`;
 *   - TrueLabels.txt: `i class` per vertex (class 0 or 1);
 *   - dynlp_labels.txt: `i label` per vertex, the label as Values.txt prints a weight.
 * Every line ends with '\n'. The public io::write_labels arrives in M10b (AD13 D2); these are the
 * writers the compat tool and the parity tests use.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dyng::detail {

/**
 * @brief Write the generated graph as O's four CSR text files `<prefix>RowPtr.txt`,
 *        `<prefix>ColInd.txt`, `<prefix>Values.txt` and `<prefix>TrueLabels.txt`.
 * @tparam vertex_t Column type.
 * @tparam edge_t   Offset type.
 * @param[in] prefix      Path prefix (O: `<output>/graph/csr`); missing directories are created.
 * @param[in] row_ptr     num_vertices + 1 offsets.
 * @param[in] col_ind     The columns, row_ptr.back() of them.
 * @param[in] values      The weights, one per column.
 * @param[in] true_labels The class (0 or 1) of every vertex.
 * @throws io_error if a file cannot be written.
 * @throws invalid_argument_error if the array sizes do not agree.
 */
template <typename vertex_t, typename edge_t>
void write_dynlp_graph_files(const std::string& prefix, const std::vector<edge_t>& row_ptr,
                             const std::vector<vertex_t>& col_ind, const std::vector<float>& values,
                             const std::vector<std::int8_t>& true_labels);

/**
 * @brief Write labels as O's `dynlp_labels.txt` (`i label` per vertex, 6 decimals).
 * @param[in] path   The file; its directory is created if missing.
 * @param[in] labels One label per vertex.
 * @throws io_error if the file cannot be written.
 */
void write_dynlp_labels(const std::string& path, const std::vector<float>& labels);

}  // namespace dyng::detail
