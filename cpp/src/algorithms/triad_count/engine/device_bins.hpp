// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:include/device_bins.hpp
/**
 * @file device_bins.hpp
 * @brief Device-side 64-bit counters of a count or a delta (ESCHER-GPU's DeviceBins).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 */
#pragma once

#include "util/cuda_check.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace dyng::detail::triad {

// Device-side 64-bit counters for a count or a delta. Signed deltas are
// accumulated in two's complement (unsigned atomicAdd of -x).
class device_bins {
 public:
  explicit device_bins(int n) : n_(n) {
    DYNG_CUDA_TRY(cudaMalloc(&data_, sizeof(unsigned long long) * n_));
    zero();
  }
  ~device_bins() {
    DYNG_CUDA_TRY_NO_THROW(cudaFree(data_));
  }
  device_bins(const device_bins&) = delete;
  device_bins& operator=(const device_bins&) = delete;

  unsigned long long* data() {
    return data_;
  }
  void zero() {
    DYNG_CUDA_TRY(cudaMemset(data_, 0, sizeof(unsigned long long) * n_));
  }
  std::vector<long long> read() const {
    std::vector<long long> out(n_);
    DYNG_CUDA_TRY(cudaMemcpy(out.data(), data_, sizeof(long long) * n_, cudaMemcpyDeviceToHost));
    return out;
  }

 private:
  int n_;
  unsigned long long* data_ = nullptr;
};

}  // namespace dyng::detail::triad
