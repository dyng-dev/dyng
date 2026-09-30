// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file names.cpp
 * @brief to_string() of the graph enumerations (graph_properties.hpp): the enumerators' own
 *        names, which the CLI and the Python layer accept (`layout="compact"`, ...).
 */
#include <dyng/graph/graph_properties.hpp>

#include <string_view>

namespace dyng {

std::string_view to_string(row_layout layout) noexcept {
  switch (layout) {
    case row_layout::compact:
      return "compact";
    case row_layout::slotted:
      return "slotted";
    case row_layout::slack:
      return "slack";
  }
  return "unknown";
}

std::string_view to_string(row_order order) noexcept {
  switch (order) {
    case row_order::sorted:
      return "sorted";
    case row_order::append:
      return "append";
  }
  return "unknown";
}

std::string_view to_string(multi_edges rule) noexcept {
  switch (rule) {
    case multi_edges::forbid:
      return "forbid";
    case multi_edges::allow:
      return "allow";
  }
  return "unknown";
}

std::string_view to_string(batch_semantics::existing_insert rule) noexcept {
  switch (rule) {
    case batch_semantics::existing_insert::upsert:
      return "upsert";
    case batch_semantics::existing_insert::error:
      return "error";
    case batch_semantics::existing_insert::ignore:
      return "ignore";
  }
  return "unknown";
}

std::string_view to_string(batch_semantics::missing_delete rule) noexcept {
  switch (rule) {
    case batch_semantics::missing_delete::ignore:
      return "ignore";
    case batch_semantics::missing_delete::error:
      return "error";
  }
  return "unknown";
}

std::string_view to_string(batch_semantics::self_loop rule) noexcept {
  switch (rule) {
    case batch_semantics::self_loop::keep:
      return "keep";
    case batch_semantics::self_loop::drop:
      return "drop";
    case batch_semantics::self_loop::error:
      return "error";
  }
  return "unknown";
}

}  // namespace dyng
