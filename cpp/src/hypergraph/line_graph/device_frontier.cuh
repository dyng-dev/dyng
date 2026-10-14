// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:include/frontier.hpp
/**
 * @file device_frontier.cuh
 * @brief The frontier of one delta phase on the device: the changed ids and a membership flag per
 *        id (ESCHER-GPU's DeviceFrontier).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_line_graph.toml
 * (M8b, ADR 0042; the store's encoding is described in cpp/src/hypergraph/escher/README.md).
 */
#pragma once

#include "util/cuda_check.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>

#include <cuda_runtime.h>

#include <vector>

namespace dyng::detail::escher {

// The frontier of one delta phase on the device: the ids of the changed
// hyperedges (or vertices) and a membership flag for every id 0..max_id.
//
// Exact frontier-incident delta: a triad changes only if it contains a
// changed hyperedge, so the delta is
//     -#{old triads containing a deleted hyperedge}
//     +#{new triads containing an inserted hyperedge}.
// Each triad is counted once, at its smallest frontier member f: from f the
// kernels skip every other frontier member smaller than f.
class device_frontier {
 public:
  // The arrays from the memory resource of `res` (dynG: buffers instead of cudaMalloc).
  device_frontier(const resources& res, const std::vector<int>& ids, int max_id)
      : ids_(res, ids.empty() ? 1 : ids.size()),
        flags_(res, static_cast<size_t>(max_id) + 1),
        size_(static_cast<int>(ids.size())) {
    std::vector<unsigned char> flags(static_cast<size_t>(max_id) + 1, 0);
    for (int id : ids)
      if (id >= 0 && id <= max_id) flags[id] = 1;
    if (!ids.empty())
      DYNG_CUDA_TRY(
          cudaMemcpy(ids_.data(), ids.data(), sizeof(int) * ids.size(), cudaMemcpyHostToDevice));
    DYNG_CUDA_TRY(cudaMemcpy(flags_.data(), flags.data(), flags.size(), cudaMemcpyHostToDevice));
  }
  device_frontier(const device_frontier&) = delete;
  device_frontier& operator=(const device_frontier&) = delete;

  const int* ids() const {
    return ids_.data();
  }
  const unsigned char* flags() const {
    return flags_.data();
  }
  int size() const {
    return size_;
  }

 private:
  buffer<int> ids_;
  buffer<unsigned char> flags_;
  int size_;
};

}  // namespace dyng::detail::escher
