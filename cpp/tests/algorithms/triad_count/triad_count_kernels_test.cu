// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_count_kernels_test.cu
 * @brief Device tests of triad_count's wedge engine (M9a criterion 9): the class of every one of
 *        the 128 Venn-region patterns through MotifOp and the engine's bins handle on the device,
 *        where a pattern of no class raises the device error word and never writes bins[-1]; the
 *        same patterns through the host copy of the class map; the exact arithmetic of the
 *        operations and the capacity bit of the device bins (ADR 0057).
 */
#include "algorithms/triad_count/engine/exact.hpp"
#include "algorithms/triad_count/engine/motif_ops.hpp"
#include "algorithms/triad_count/engine/wedge.cuh"
#include "util/device_error_flags.hpp"

#include <dyng/testing/triad_reference.hpp>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace {

namespace triad = dyng::detail::triad;

/// The region sizes of a Venn id (bit set: one vertex in the region) as the classification reads
/// them: the degrees, the pairwise intersections and the common part.
struct regions {
  int deg_a, deg_b, deg_c, c_ab, c_bc, c_ca, g;
};

__host__ __device__ regions regions_of(int id) {
  const int a = (id >> 0) & 1, b = (id >> 1) & 1, c = (id >> 2) & 1;
  const int d = (id >> 3) & 1, e = (id >> 4) & 1, f = (id >> 5) & 1, g = (id >> 6) & 1;
  return {a + d + f + g, b + d + e + g, c + e + f + g, d + g, e + g, f + g, g};
}

/// One block per Venn id: the class through compute_motif_index and the bins handle, which holds
/// one guard slot before the 30 bins (bins[-1]) and one after them.
__global__ void classify_kernel(unsigned long long* out_bins, unsigned long long* out_error) {
  __shared__ unsigned long long shared[1 + 30 + 1];
  const int id = static_cast<int>(blockIdx.x);
  for (int i = static_cast<int>(threadIdx.x); i < 32; i += static_cast<int>(blockDim.x)) {
    shared[i] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const regions r = regions_of(id);
    triad::wedge::block_bins<30, false> bins{shared + 1, out_error + id};
    bins.add(triad::compute_motif_index(r.deg_a, r.deg_b, r.deg_c, r.c_ab, r.c_bc, r.c_ca, r.g),
             1ULL);
  }
  __syncthreads();
  for (int i = static_cast<int>(threadIdx.x); i < 32; i += static_cast<int>(blockDim.x)) {
    out_bins[id * 32 + i] = shared[i];
  }
}

TEST(TriadCountKernels, EveryVennPatternGetsItsClassOrRaisesTheErrorWord) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "no CUDA device";
  }
  constexpr int ids = 128;
  unsigned long long* d_bins = nullptr;
  unsigned long long* d_error = nullptr;
  ASSERT_EQ(cudaMalloc(&d_bins, sizeof(unsigned long long) * ids * 32), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_error, sizeof(unsigned long long) * ids), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_error, 0, sizeof(unsigned long long) * ids), cudaSuccess);
  classify_kernel<<<ids, 64>>>(d_bins, d_error);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::vector<unsigned long long> bins(static_cast<std::size_t>(ids) * 32);
  std::vector<unsigned long long> error(ids);
  ASSERT_EQ(cudaMemcpy(bins.data(), d_bins, sizeof(unsigned long long) * bins.size(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(error.data(), d_error, sizeof(unsigned long long) * error.size(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_bins);
  cudaFree(d_error);
  // The independent class map (derived by the oracle from the permutation orbits): 0-based class,
  // -1 for no class.
  const std::array<std::int8_t, ids> map = dyng::testing::triad_motif_class_map();
  const auto internal =
      static_cast<unsigned long long>(dyng::detail::bits_of(dyng::detail::device_error::internal));
  int classified = 0;
  for (int id = 0; id < ids; ++id) {
    SCOPED_TRACE("Venn id " + std::to_string(id));
    const unsigned long long* b = bins.data() + static_cast<std::size_t>(id) * 32;
    EXPECT_EQ(b[0], 0U) << "bins[-1] was written";
    EXPECT_EQ(b[31], 0U) << "bins[30] was written";
    const int expected = map[static_cast<std::size_t>(id)] - 1;
    const regions r = regions_of(id);
    EXPECT_EQ(triad::compute_motif_index(r.deg_a, r.deg_b, r.deg_c, r.c_ab, r.c_bc, r.c_ca, r.g),
              expected)
        << "the host copy of the class map";
    if (expected < 0) {
      EXPECT_EQ(error[static_cast<std::size_t>(id)], internal) << "no class: the error word";
      for (int k = 1; k <= 30; ++k) EXPECT_EQ(b[k], 0U) << "bin " << k - 1;
    } else {
      ++classified;
      EXPECT_EQ(error[static_cast<std::size_t>(id)], 0U);
      for (int k = 1; k <= 30; ++k) {
        EXPECT_EQ(b[k], k - 1 == expected ? 1U : 0U) << "bin " << k - 1;
      }
    }
  }
  EXPECT_EQ(classified, 96);  // 96 ids in the 30 classes, 32 in none
}

/// The exact arithmetic of the operations (engine/exact.hpp, ADR 0057) on the device: C(n, 3)
/// and the checked products and sums, against the host's.
__global__ void exact_kernel(const int* n, int count, unsigned long long* values, int* overflows) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  overflows[i] = triad::choose3_overflows(n[i], values[i]) ? 1 : 0;
}

__global__ void checked_kernel(unsigned long long a, unsigned long long b, int* overflows) {
  unsigned long long out = 0;
  overflows[0] = triad::mul_overflows(a, b, out) ? 1 : 0;
  overflows[1] = triad::add_overflows(a, b, out) ? 1 : 0;
}

/// An operation of one bin without a sum of its own (for flush_bins()).
struct one_bin_op {
  static constexpr int num_bins = 1;
  __device__ void flush(triad::wedge::block_bins<1, true>& /*bins*/) {}
};

/// The checked bins handle: two adds whose sum carries past 2^64 - 1 set the capacity bit, as
/// does overflow(); the flush into the call's bins carries too.
__global__ void carry_kernel(unsigned long long* out, unsigned long long* error, int which) {
  __shared__ unsigned long long shared[1];
  if (threadIdx.x == 0) shared[0] = 0;
  __syncthreads();
  triad::wedge::block_bins<1, true> bins{shared, error};
  one_bin_op op;
  if (threadIdx.x == 0) {
    if (which == 0) {
      bins.add(0, 1ULL << 63);
      bins.add(0, 1ULL << 63);  // carries
    } else if (which == 1) {
      bins.overflow();
    } else {
      bins.add(0, 1ULL << 63);  // out already holds 2^63: the flush carries
    }
  }
  triad::wedge::flush_bins(op, bins, out);
}

TEST(TriadCountKernels, ExactArithmeticAndTheCapacityBitOnTheDevice) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "no CUDA device";
  }
  const std::vector<int> n = {
      0,         2,         3,         4,         1000,      2'097'151,    2'097'152,
      2'100'000, 2'642'246, 3'000'000, 4'801'280, 4'801'281, 2'147'483'647};
  int* d_n = nullptr;
  unsigned long long* d_values = nullptr;
  int* d_over = nullptr;
  ASSERT_EQ(cudaMalloc(&d_n, sizeof(int) * n.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_values, sizeof(unsigned long long) * n.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_over, sizeof(int) * n.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_n, n.data(), sizeof(int) * n.size(), cudaMemcpyHostToDevice), cudaSuccess);
  exact_kernel<<<1, 32>>>(d_n, static_cast<int>(n.size()), d_values, d_over);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::vector<unsigned long long> values(n.size());
  std::vector<int> over(n.size());
  ASSERT_EQ(cudaMemcpy(values.data(), d_values, sizeof(unsigned long long) * n.size(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(over.data(), d_over, sizeof(int) * n.size(), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (std::size_t i = 0; i < n.size(); ++i) {
    SCOPED_TRACE("n = " + std::to_string(n[i]));
    unsigned long long host = 0;
    const bool host_over = triad::choose3_overflows(n[i], host);
    EXPECT_EQ(over[i] != 0, host_over);
    if (!host_over) EXPECT_EQ(values[i], host);
  }
  EXPECT_EQ(values[7], 1543497795000700000ULL);    // C(2,100,000, 3): above 2^63 - 1 as n^3 / 6
  EXPECT_EQ(values[9], 4499995500001000000ULL);    // C(3,000,000, 3)
  EXPECT_EQ(values[10], 18446738006366306560ULL);  // C(4,801,280, 3): the largest that fits
  EXPECT_EQ(over[11], 1);                          // C(4,801,281, 3) > 2^64 - 1
  EXPECT_EQ(over[12], 1);

  for (const auto& [a, b, mul, add] :
       {std::tuple<unsigned long long, unsigned long long, int, int>{1ULL << 32, 1ULL << 31, 0, 0},
        {1ULL << 32, 1ULL << 32, 1, 0},
        {~0ULL, 1, 0, 1},
        {1ULL << 63, 2, 1, 0}}) {
    checked_kernel<<<1, 1>>>(a, b, d_over);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    std::array<int, 2> got{};
    ASSERT_EQ(cudaMemcpy(got.data(), d_over, sizeof(int) * 2, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(got[0], mul) << a << " * " << b;
    EXPECT_EQ(got[1], add) << a << " + " << b;
  }

  const auto capacity =
      static_cast<unsigned long long>(dyng::detail::bits_of(dyng::detail::device_error::capacity));
  unsigned long long* d_out = nullptr;
  unsigned long long* d_error = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, sizeof(unsigned long long)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_error, sizeof(unsigned long long)), cudaSuccess);
  for (int which = 0; which < 3; ++which) {
    SCOPED_TRACE("case " + std::to_string(which));
    const unsigned long long start = which == 2 ? 1ULL << 63 : 0;
    ASSERT_EQ(cudaMemcpy(d_out, &start, sizeof start, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_error, 0, sizeof(unsigned long long)), cudaSuccess);
    carry_kernel<<<1, 32>>>(d_out, d_error, which);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    unsigned long long error = 0;
    ASSERT_EQ(cudaMemcpy(&error, d_error, sizeof error, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(error, capacity);
  }
  cudaFree(d_out);
  cudaFree(d_error);
  cudaFree(d_n);
  cudaFree(d_values);
  cudaFree(d_over);
}

}  // namespace
