// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file names.cpp
 * @brief to_string() of the hypergraph enumerations (hypergraph_properties.hpp): the
 *        enumerators' own names, which the CLI and the Python layer accept.
 */
#include <dyng/hypergraph/hypergraph_properties.hpp>

#include <string_view>

namespace dyng {

std::string_view to_string(hypergraph_storage storage) noexcept {
  switch (storage) {
    case hypergraph_storage::escher:
      return "escher";
    case hypergraph_storage::slack_csr:
      return "slack_csr";
  }
  return "unknown";
}

std::string_view to_string(line_graph_policy policy) noexcept {
  switch (policy) {
    case line_graph_policy::none:
      return "none";
    case line_graph_policy::automatic:
      return "automatic";
    case line_graph_policy::sorted_merge:
      return "sorted_merge";
    case line_graph_policy::slack_csr:
      return "slack_csr";
  }
  return "unknown";
}

std::string_view to_string(id_reuse reuse) noexcept {
  switch (reuse) {
    case id_reuse::erase_first:
      return "erase_first";
    case id_reuse::insert_first:
      return "insert_first";
  }
  return "unknown";
}

}  // namespace dyng
