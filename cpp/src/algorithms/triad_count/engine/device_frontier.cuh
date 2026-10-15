// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:include/frontier.hpp
/**
 * @file device_frontier.cuh
 * @brief The frontier of one delta phase on the device: the changed keys and a membership flag per
 *        key (ESCHER-GPU's DeviceFrontier, include/frontier.hpp).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 */
#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

namespace dyng::detail::triad {

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
  device_frontier(const std::vector<int>& ids, int max_id) : size_(static_cast<int>(ids.size())) {
    std::vector<unsigned char> flags(static_cast<size_t>(max_id) + 1, 0);
    for (int id : ids)
      if (id >= 0 && id <= max_id) flags[id] = 1;
    check(cudaMalloc(&ids_, sizeof(int) * (ids.empty() ? 1 : ids.size())));
    check(cudaMalloc(&flags_, flags.size()));
    if (!ids.empty())
      check(cudaMemcpy(ids_, ids.data(), sizeof(int) * ids.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(flags_, flags.data(), flags.size(), cudaMemcpyHostToDevice));
  }
  ~device_frontier() {
    cudaFree(ids_);
    cudaFree(flags_);
  }
  device_frontier(const device_frontier&) = delete;
  device_frontier& operator=(const device_frontier&) = delete;

  const int* ids() const {
    return ids_;
  }
  const unsigned char* flags() const {
    return flags_;
  }
  int size() const {
    return size_;
  }

 private:
  static void check(cudaError_t result) {
    if (result != cudaSuccess) {
      std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
      std::exit(-1);
    }
  }
  int* ids_ = nullptr;
  unsigned char* flags_ = nullptr;
  int size_;
};

}  // namespace dyng::detail::triad
