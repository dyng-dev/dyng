// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file dyng.hpp
 * @brief Umbrella header: includes every public header of dynG outside dyng/testing/.
 *
 * That is the frozen 0.1 API (core, graph, update, the algorithms, citation, config, version) and
 * the headers of io/ and generators/, which are public but not frozen before the command-line
 * tools and the Python layer (0.1 API review, ADR 0023: they may still change in a minor release),
 * and the hypergraph headers (hypergraph/, ADR 0041), tracked until they are frozen for 0.3.
 * The list of includes is part of the API baseline.
 * @ingroup core
 */
#pragma once

/**
 * @defgroup core Core
 * @brief Resources, backends, streams, memory, arrays, errors, logging, profiling, version.
 */

#include <dyng/citation.hpp>
#include <dyng/config.hpp>
#include <dyng/core/array_view.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/logging.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/registry.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stats.hpp>
#include <dyng/core/stream.hpp>
#include <dyng/core/types.hpp>
#include <dyng/cycle_count.hpp>
#include <dyng/dynamic_bfs.hpp>
#include <dyng/generators/hypergraphs.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/graph/graph_view.hpp>
#include <dyng/hypergraph/hyper_apply_summary.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/hypergraph/hypergraph_properties.hpp>
#include <dyng/hypergraph/hypergraph_view.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/csr_triplet.hpp>
#include <dyng/io/edge_list_io.hpp>
#include <dyng/io/hypergraph_io.hpp>
#include <dyng/io/matrix_market.hpp>
#include <dyng/io/result_io.hpp>
#include <dyng/mosp.hpp>
#include <dyng/sssp.hpp>
#include <dyng/triangle_delta.hpp>
#include <dyng/update.hpp>
#include <dyng/version.hpp>
