// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.hpp
 * @brief The algorithm registry: the algorithms compiled into this library and what their
 *        manifests declare (family, container, backends, maturity, determinism, oracle, citation
 *        keys).
 * @ingroup core
 */
#pragma once

#include <dyng/core/backend.hpp>
#include <dyng/core/types.hpp>

#include <cstdint>
#include <string_view>
#include <vector>

namespace dyng {

/**
 * @brief The update family of an algorithm (the two templates of the update model).
 * @ingroup core
 */
enum class algorithm_family : std::uint8_t {
  fixed_point,      ///< a value per element, iterated until nothing changes (sssp)
  aggregate_delta,  ///< a global count, updated by one signed recount (cycle_count)
};

/**
 * @brief The container an algorithm runs on.
 * @ingroup core
 */
enum class container_kind : std::uint8_t {
  graph,       ///< dyng::graph
  hypergraph,  ///< the hypergraph container (0.2)
};

/**
 * @brief How settled an algorithm's API and behaviour are (experimental, beta, stable; docs/glossary.md).
 * @ingroup core
 */
enum class maturity_level : std::uint8_t {
  experimental,  ///< may change in any release; excluded from SemVer guarantees
  stable,        ///< SemVer applies
  deprecated,    ///< kept for at least one minor release, then removed
};

/**
 * @brief What "correct" means for an algorithm's update (the oracle the conformance kit checks it against).
 * @ingroup core
 */
enum class oracle_kind : std::uint8_t {
  compute,    ///< exact: a chain of updates equals compute() on the final graph
  reference,  ///< approximate: update and compute are within a tolerance of a converged reference
};

/**
 * @brief The name of an update family, as in the manifests ("fixed_point", "aggregate_delta").
 * @param[in] family The family.
 * @return A static string naming `family`.
 * @ingroup core
 */
[[nodiscard]] std::string_view to_string(algorithm_family family) noexcept;

/**
 * @brief The name of a container kind, as in the manifests ("graph", "hypergraph").
 * @param[in] container The container kind.
 * @return A static string naming `container`.
 * @ingroup core
 */
[[nodiscard]] std::string_view to_string(container_kind container) noexcept;

/**
 * @brief The name of a maturity level, as in the manifests ("experimental", "stable",
 *        "deprecated").
 * @param[in] maturity The maturity level.
 * @return A static string naming `maturity`.
 * @ingroup core
 */
[[nodiscard]] std::string_view to_string(maturity_level maturity) noexcept;

/**
 * @brief The name of an oracle kind, as in the manifests ("compute", "reference").
 * @param[in] oracle The oracle kind.
 * @return A static string naming `oracle`.
 * @ingroup core
 */
[[nodiscard]] std::string_view to_string(oracle_kind oracle) noexcept;

/**
 * @brief One registered algorithm, as its manifest (cpp/src/algorithms/NAME/manifest.toml)
 *        declares it.
 *
 * Every field is fixed at build time; the strings are static.
 * @ingroup core
 */
struct algorithm_info {
  std::string_view name;   ///< the algorithm's name: its namespace and header (e.g. "sssp")
  std::string_view title;  ///< a one-line title
  algorithm_family family = algorithm_family::fixed_point;  ///< the update family
  container_kind container = container_kind::graph;         ///< the container it runs on
  maturity_level maturity = maturity_level::experimental;   ///< the maturity level
  determinism determinism_level = determinism::bitwise;     ///< the determinism level
  oracle_kind oracle = oracle_kind::compute;                ///< the oracle kind
  /** @brief The backends the algorithm implements (sequential first); intersect with
   *         backend_available() for the ones this build can run. */
  std::vector<backend> backends;
  std::vector<std::string_view> cite;  ///< keys of docs/references.bib (dyng::citation(name))
};

/**
 * @brief The algorithms compiled into this library, in the order of the manifests.
 *
 * A build that selects a subset (the CMake option DYNG_ALGORITHMS) lists only that subset.
 * @return The registry (built on first use; valid for the life of the program).
 * @throws out_of_memory_error on the first call if the table cannot be allocated.
 * @ingroup core
 */
[[nodiscard]] const std::vector<algorithm_info>& algorithms();

/**
 * @brief Look up a compiled algorithm by name.
 * @param[in] name The algorithm's name (e.g. "cycle_count").
 * @return Its entry, or nullptr if this library does not contain it.
 * @throws out_of_memory_error as algorithms().
 * @ingroup core
 */
[[nodiscard]] const algorithm_info* find_algorithm(std::string_view name);

}  // namespace dyng
