// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file citation.cpp
 * @brief citation(): BibTeX entries from docs/references.bib (embedded at build time).
 */
#include <dyng/citation.hpp>
#include <dyng/core/error.hpp>

#include <array>
#include <string>
#include <string_view>
#include <utility>

namespace dyng {

namespace detail {
/// docs/references.bib, embedded by CMake (generated file references_bib.cpp).
extern const char* const references_bib;
}  // namespace detail

namespace {

constexpr std::string_view library_key = "dyng2027";

struct cite_entry {
  std::string_view name;
  std::array<std::string_view, 2> keys;  // empty string_view = unused slot
};

// Which papers to cite for each algorithm or storage name (PLAN Section 3.2).
constexpr std::array<cite_entry, 8> cite_table{{
    {"dyng", {"", ""}},
    {"sssp", {"dynamosp2025", "dynamosptpds2025"}},
    {"mosp", {"dynamosp2025", "dynamosptpds2025"}},
    {"cycle_count", {"trucy2026", ""}},
    {"triad_count", {"escher2026", "escherplus2026"}},
    {"hypergraph", {"escher2026", "escherplus2026"}},
    {"label_propagation", {"dynlp2026", ""}},
    {"hyper_sssp", {"hsosp2026", ""}},
}};

const cite_entry& find_entry(std::string_view what) {
  for (const auto& e : cite_table) {
    if (e.name == what) {
      return e;
    }
  }
  std::string known;
  for (const auto& e : cite_table) {
    known += known.empty() ? "" : ", ";
    known += e.name;
  }
  throw invalid_argument_error(
      detail::concat_message("dyng: citation(): unknown name '", what, "' (known: ", known, ")"));
}

/// The text of the BibTeX entry `key` (from its '@' line to the closing brace line).
std::string entry_text(std::string_view key) {
  const std::string_view bib(detail::references_bib);
  std::size_t pos = 0;
  while (pos < bib.size()) {
    std::size_t end = bib.find('\n', pos);
    if (end == std::string_view::npos) {
      end = bib.size();
    }
    const std::string_view line = bib.substr(pos, end - pos);
    if (!line.empty() && line.front() == '@') {
      const std::size_t brace = line.find('{');
      const std::size_t comma = line.find(',', brace);
      if (brace != std::string_view::npos && comma != std::string_view::npos &&
          line.substr(brace + 1, comma - brace - 1) == key) {
        // The entry ends at the first line that is exactly "}".
        std::size_t stop = bib.find("\n}", pos);
        stop = stop == std::string_view::npos ? bib.size() : stop + 2;
        return std::string(bib.substr(pos, stop - pos)) + "\n";
      }
    }
    pos = end + 1;
  }
  DYNG_FAIL("citation key '", key, "' missing from the embedded references.bib");
}

}  // namespace

std::vector<std::string> citation_keys(std::string_view what) {
  const cite_entry& e = find_entry(what);
  std::vector<std::string> keys{std::string(library_key)};
  for (auto k : e.keys) {
    if (!k.empty()) {
      keys.emplace_back(k);
    }
  }
  return keys;
}

std::string citation(std::string_view what) {
  std::string out;
  for (const auto& key : citation_keys(what)) {
    out += out.empty() ? "" : "\n";
    out += entry_text(key);
  }
  return out;
}

}  // namespace dyng
