// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file citation.hpp
 * @brief citation(): BibTeX for the library and the papers behind each algorithm.
 * @ingroup core
 */
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace dyng {

/**
 * @brief BibTeX entries to cite when using the library or one of its algorithms.
 *
 * The entries come from docs/references.bib, compiled into the library.
 *
 * @param[in] what "dyng" (the library only), or an algorithm or storage name: "sssp", "mosp",
 *                 "cycle_count", "triad_count", "label_propagation", "hyper_sssp", "hypergraph".
 * @return The BibTeX of the library followed by the entries of the algorithm's papers.
 * @throws invalid_argument_error if `what` is not a known name.
 * @ingroup core
 */
[[nodiscard]] std::string citation(std::string_view what = "dyng");

/**
 * @brief The citation keys (in docs/references.bib) for a name accepted by citation().
 * @param[in] what As for citation().
 * @return The keys, the library's key first.
 * @throws invalid_argument_error if `what` is not a known name.
 * @ingroup core
 */
[[nodiscard]] std::vector<std::string> citation_keys(std::string_view what = "dyng");

}  // namespace dyng
