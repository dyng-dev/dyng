// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file host_sort.hpp
 * @brief sort_keys: a sort of 64-bit keys on the host executors (operators::sequential_exec,
 *        openmp_exec) for the line-graph delta and the slack CSR apply (ADR 0086).
 *
 * The keys' order is unique, so the result does not depend on the thread count: one std::sort
 * per chunk, then rounds of pairwise merges into a second buffer (nothing allocates inside the
 * regions; the buffer is allocated before them). Below 2^20 keys, or on one thread, one std::sort.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace dyng::detail::hyper {

/**
 * @brief Sorts `v` in increasing order.
 * @tparam exec_t operators::sequential_exec or operators::openmp_exec.
 * @param[in]     exec The executor.
 * @param[in,out] v    The keys.
 */
template <typename exec_t>
void sort_keys(const exec_t& exec, std::vector<std::uint64_t>& v) {
  const auto n = static_cast<std::int64_t>(v.size());
  const int threads = exec.threads();
  if (threads <= 1 || n < (std::int64_t{1} << 20)) {
    std::sort(v.begin(), v.end());
    return;
  }
  const std::int64_t chunks = std::min<std::int64_t>(4 * threads, n);
  std::vector<std::int64_t> bound(static_cast<std::size_t>(chunks) + 1);
  for (std::int64_t c = 0; c <= chunks; ++c) bound[static_cast<std::size_t>(c)] = n * c / chunks;
  std::uint64_t* data = v.data();
  const std::int64_t* b = bound.data();
  exec.for_each(chunks, [data, b](std::int64_t c) { std::sort(data + b[c], data + b[c + 1]); });
  std::vector<std::uint64_t> other(v.size());
  std::uint64_t* from = v.data();
  std::uint64_t* to = other.data();
  for (std::int64_t width = 1; width < chunks; width *= 2) {
    const std::int64_t merges = (chunks + 2 * width - 1) / (2 * width);
    exec.for_each(merges, [from, to, b, width, chunks](std::int64_t m) {
      const std::int64_t lo = 2 * width * m;
      const std::int64_t mid = std::min(lo + width, chunks);
      const std::int64_t hi = std::min(lo + 2 * width, chunks);
      std::merge(from + b[lo], from + b[mid], from + b[mid], from + b[hi], to + b[lo]);
    });
    std::swap(from, to);
  }
  if (from != v.data()) v.swap(other);
}

}  // namespace dyng::detail::hyper
