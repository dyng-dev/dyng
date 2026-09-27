// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file dyng.hpp
 * @brief Umbrella header: includes every stable public header of dynG.
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
#include <dyng/core/resources.hpp>
#include <dyng/core/stats.hpp>
#include <dyng/core/stream.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/graph/graph_view.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/csr_triplet.hpp>
#include <dyng/io/matrix_market.hpp>
#include <dyng/io/result_io.hpp>
#include <dyng/sssp.hpp>
#include <dyng/update.hpp>
#include <dyng/version.hpp>
