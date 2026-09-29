// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_enum_text.hpp
 * @brief The text formats of the CycleEnumeration-GPU exporter (parity/exporters/cycle_enum), so
 *        the tests can compare dynG's results with the committed fixtures and digests.
 */
#pragma once

#include "support/data_paths.hpp"

#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/io/edge_list_io.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace dyng::test {

/// FNV-1a 64 of a text, as the exporter's --digest prints it: "fnv1a64 <hex> bytes <size>".
inline std::string fnv1a64_digest(const std::string& text) {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  for (const char c : text) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 0x100000001b3ULL;
  }
  char hex[17];
  std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(hash));
  return std::string("fnv1a64 ") + hex + " bytes " + std::to_string(text.size());
}

/// "n <n>\nm <m>\nrow_ptr ...\ncol_ind ...\n" of a CSR.
template <typename vertex_t, typename edge_t, typename weight_t>
std::string csr_text(const csr_view<vertex_t, edge_t, weight_t>& g) {
  std::string out = "n " + std::to_string(g.num_vertices()) + "\nm " +
                    std::to_string(g.num_edges()) + "\nrow_ptr";
  out.reserve(out.size() + 12 * (g.row_ptr.size() + g.col_ind.size()));
  if (g.row_ptr.empty()) {
    out += " 0";
  }
  for (const auto o : g.row_ptr) {
    out += ' ';
    out += std::to_string(o);
  }
  out += "\ncol_ind";
  for (const auto v : g.col_ind) {
    out += ' ';
    out += std::to_string(v);
  }
  out += '\n';
  return out;
}

/// "- u v" per deletion, then "+ u v" per insertion.
template <typename vertex_t>
std::string batch_text(const std::vector<vertex_t>& del_src, const std::vector<vertex_t>& del_dst,
                       const std::vector<vertex_t>& ins_src, const std::vector<vertex_t>& ins_dst) {
  std::ostringstream out;
  for (std::size_t i = 0; i < del_src.size(); ++i) {
    out << "- " << del_src[i] << ' ' << del_dst[i] << '\n';
  }
  for (std::size_t i = 0; i < ins_src.size(); ++i) {
    out << "+ " << ins_src[i] << ' ' << ins_dst[i] << '\n';
  }
  return out.str();
}

/// "<name> <len>:<count> ..." over the non-zero lengths >= 2.
inline std::string histogram_text(const char* name, const std::vector<std::uint64_t>& counts) {
  std::ostringstream out;
  out << name;
  for (std::size_t length = 2; length < counts.size(); ++length) {
    if (counts[length] != 0) {
      out << ' ' << length << ':' << counts[length];
    }
  }
  out << '\n';
  return out.str();
}

/// The exporter's `parse` text from a read_edge_list(duplicates = keep) result and its info.
template <typename vertex_t, typename weight_t>
std::string parse_text(const edge_list<vertex_t, weight_t>& edges, const io::edge_list_info& info) {
  std::size_t logical = 0;
  for (std::size_t i = 0; i < edges.src.size(); ++i) {
    logical += i == 0 || edges.src[i] != edges.src[i - 1] || edges.dst[i] != edges.dst[i - 1];
  }
  std::string out = "vertices " + std::to_string(info.external_ids.size()) + "\nedges " +
                    std::to_string(logical) + "\nevents " + std::to_string(edges.src.size()) + "\n";
  out.reserve(out.size() + 12 * info.external_ids.size() + 16 * edges.src.size());
  for (const std::int64_t id : info.external_ids) {
    out += "v ";
    out += std::to_string(id);
    out += '\n';
  }
  for (std::size_t i = 0; i < edges.src.size(); ++i) {
    if (i == 0 || edges.src[i] != edges.src[i - 1] || edges.dst[i] != edges.dst[i - 1]) {
      if (i != 0) {
        out += '\n';
      }
      out += "e ";
      out += std::to_string(edges.src[i]);
      out += ' ';
      out += std::to_string(edges.dst[i]);
    }
    out += ' ';
    out += std::to_string(info.timestamps[i]);
  }
  if (!edges.src.empty()) {
    out += '\n';
  }
  return out;
}

/// One random case of the fixtures: a graph on n vertices and a batch in batch order.
struct cycle_enum_case {
  std::int64_t n = 0;
  std::vector<std::int64_t> src, dst;          ///< the edges
  std::vector<std::int64_t> del_src, del_dst;  ///< deletions in batch order
  std::vector<std::int64_t> ins_src, ins_dst;  ///< insertions in batch order
};

/// Reads "n <n>", "e u v", "- u v", "+ u v" lines.
inline cycle_enum_case read_cycle_enum_case(const std::string& path) {
  std::istringstream in(read_text(path));
  cycle_enum_case c;
  std::string tag;
  while (in >> tag) {
    if (tag == "n") {
      in >> c.n;
      continue;
    }
    std::int64_t u = 0;
    std::int64_t v = 0;
    in >> u >> v;
    if (tag == "e") {
      c.src.push_back(u);
      c.dst.push_back(v);
    } else if (tag == "-") {
      c.del_src.push_back(u);
      c.del_dst.push_back(v);
    } else if (tag == "+") {
      c.ins_src.push_back(u);
      c.ins_dst.push_back(v);
    } else {
      ADD_FAILURE() << "bad tag " << tag << " in " << path;
    }
  }
  return c;
}

/// The sorted case files cases/case_NNN.txt of the fixture directory.
inline std::vector<std::string> cycle_enum_case_names() {
  std::vector<std::string> names;
  for (int i = 0;; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "case_%03d", i);
    if (!std::ifstream(data_path(std::string("cycle_enum/cases/") + name + ".txt"))) {
      break;
    }
    names.emplace_back(name);
  }
  return names;
}

/// The part of an "apply" expectation between `from` (a line) and `to` (a line; "" = the end).
inline std::string section(const std::string& text, const std::string& from,
                           const std::string& to) {
  const std::size_t begin = text.find(from + "\n");
  if (begin == std::string::npos) {
    return {};
  }
  const std::size_t body = begin + from.size() + 1;
  const std::size_t end = to.empty() ? std::string::npos : text.find(to, body);
  return text.substr(body, end == std::string::npos ? std::string::npos : end - body);
}

}  // namespace dyng::test
