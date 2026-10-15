// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file counter_shims.hpp
 * @brief The parts of ESCHER-GPU's driver interface (include/driver.hpp, include/
 *        dynamic_hypergraph.hpp, include/batch.hpp) that the transplanted counters read: the
 *        snapshot's row views and key count, the batch's deleted and inserted keys, and the
 *        counter interface without the CPU reference and the printing (transplant hunk of
 *        parity/escher_port/manual_hunks_triad.patch; removed when the counters take the
 *        workspace, ADR 0053).
 */
#pragma once

#include "hypergraph/row_views.hpp"

#include <vector>

namespace dyng::detail::triad {

// The row views a counter reads (ESCHER-GPU's Snapshot: h2vRows, h2hRows, numHyperedgeKeys).
struct counter_snapshot {
  escher::row_view h2v_rows;
  escher::row_view h2h_rows;
  int num_hyperedge_keys = 0;
};

// The keys of a batch (ESCHER-GPU's Batch: deletedKeys, insertedKeys after the CBST insert).
struct counter_batch {
  std::vector<int> deleted_keys;
  std::vector<int> inserted_keys;
};

// ESCHER-GPU's DynamicCounter without reference(), print() and needsVertexGraph().
class dynamic_counter {
 public:
  virtual ~dynamic_counter() = default;
  virtual int num_values() const = 0;
  virtual std::vector<long long> count(const counter_snapshot& snapshot) = 0;
  virtual void subtract(const counter_snapshot& before, const counter_batch& batch,
                        std::vector<long long>& delta) = 0;
  virtual void add(const counter_snapshot& after, const counter_batch& batch,
                   std::vector<long long>& delta) = 0;
};

}  // namespace dyng::detail::triad
