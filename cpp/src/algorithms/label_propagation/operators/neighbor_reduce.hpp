// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/iterativePropagation.cu (accumulateNeighbors,
// processVertexGroup, blockSum, processVertexBlock: the lane mappings, the sums and the guards of
// updateLabel)
/**
 * @file neighbor_reduce.hpp
 * @brief neighbor_reduce (pull; PLAN 4.5.3): for every vertex u of a list, the weighted average of
 *        its neighbours' values with the original's guards, in the original's association order
 *        for a lane mapping; its host half (the sequential model). The CUDA half is
 *        neighbor_reduce.cuh.
 *
 * Per vertex u the operator sums, over the entries e of u's row whose column v is >= 0 (not a
 * soft-deleted entry) and whose vertex is not deleted, w_e and w_e x[v] in float, and calls
 * apply(i, u, average) once, with average = (sum w > 1e-10f) ? sum (w x) / sum w : x[u] (the
 * original's guard: a vertex without live weight keeps its value).
 *
 * The association order is a lane mapping's (reduce_mapping): with L lanes per vertex (8, or a
 * warp of 32) lane l sums the entries l, l + L, l + 2L, ... in order (a weighted sum as one fused
 * multiply-add per entry: fmaf(w, x, sum)), then the lanes are added as the original's shuffle
 * tree: for offset L/2, L/4, ..., 1, p[l] += p[l + offset]; lane 0 holds the sums. With a block of
 * B threads (256 or 1024) thread t sums the entries t, t + B, ..., each warp adds its lanes by the
 * same tree (L = 32), and the warps' sums are added in warp order from 0.0f. The sequential model
 * computes exactly that order, so on a non-fast-math build its sums and averages are the CUDA
 * operator's bits; label_propagation's target compiles with --use_fast_math (flush-to-zero, an
 * approximate division), where the CUDA operator's bits are the original kernels'.
 *
 * Rule of two (ADR 0102): label_propagation is the only user (sssp's pull, the named second user,
 * has its own packed argmin); the operator stays in this folder until a second algorithm uses it.
 */
#pragma once

#include "algorithms/label_propagation/operators/binned_frontier.hpp"
#include "operators/execution.hpp"

#include <dyng/core/resources.hpp>

#include <cmath>
#include <cstdint>

namespace dyng::detail::lp_ops {

/// The lane mapping of a reduction: threads per vertex (8 or 32 within a warp; 256 or 1024 as a
/// block per vertex).
enum class reduce_mapping : int {
  lanes_8 = 8,        ///< 8 lanes per vertex (the original's small bin)
  warp = 32,          ///< a warp per vertex (the medium bin)
  block_256 = 256,    ///< a block of 256 threads per vertex (the large bin, multi-block kernels)
  block_1024 = 1024,  ///< a block of 1024 threads (the large bin in the persistent kernel)
};

/// The threads per vertex of a mapping.
constexpr int threads_of(reduce_mapping m) noexcept {
  return static_cast<int>(m);
}

/// The guard of the average (the original's updateLabel: wAll > 1e-10f).
inline constexpr float min_weight = 1e-10F;

/**
 * @brief The two sums of one vertex for a mapping, in the documented association order (the
 *        sequential model; DYNG_HD so a test can run it on the device too).
 * @tparam edge_t Edge offset type.
 * @param[in]  rows    The rows.
 * @param[in]  x       The values.
 * @param[in]  u       The vertex.
 * @param[in]  m       The mapping.
 * @param[out] w_sum   Sum of the live weights.
 * @param[out] wx_sum  Sum of w x.
 */
template <typename edge_t>
DYNG_HD void model_sums(const slot_rows<edge_t>& rows, const float* x, std::int32_t u,
                        reduce_mapping m, float& w_sum, float& wx_sum) {
  const int threads = threads_of(m);
  const int lanes = threads <= 32 ? threads : 32;
  const int warps = threads <= 32 ? 1 : threads / 32;
  float block_w = 0.0F;
  float block_wx = 0.0F;
  for (int warp = 0; warp < warps; ++warp) {
    float pw[32];
    float pwx[32];
    for (int l = 0; l < lanes; ++l) {
      const int t = warp * 32 + l;  // the thread's index in the vertex's group or block
      float w = 0.0F;
      float wx = 0.0F;
      for (edge_t e = rows.row_start[u] + t; e < rows.row_end[u]; e += threads) {
        const std::int32_t v = rows.col_ind[e];
        if (v < 0) continue;
        if (rows.status != nullptr && rows.status[v] == 1) continue;
        const float we = rows.weights[e];
        w += we;
#if defined(__CUDA_ARCH__)
        wx = fmaf(we, x[v], wx);
#else
        wx = std::fma(we, x[v], wx);
#endif
      }
      pw[l] = w;
      pwx[l] = wx;
    }
    for (int offset = lanes / 2; offset > 0; offset >>= 1) {
      for (int l = 0; l < offset; ++l) {
        pw[l] += pw[l + offset];
        pwx[l] += pwx[l + offset];
      }
    }
    if (threads <= 32) {
      w_sum = pw[0];
      wx_sum = pwx[0];
      return;
    }
    block_w += pw[0];
    block_wx += pwx[0];
  }
  w_sum = block_w;
  wx_sum = block_wx;
}

/**
 * @brief The guarded average (the original's updateLabel before its rule).
 * @param[in] w_sum  Sum of the live weights.
 * @param[in] wx_sum Sum of w x.
 * @param[in] own    The vertex's own value.
 * @return wx_sum / w_sum, or `own` when w_sum <= 1e-10f.
 */
DYNG_HD inline float guarded_average(float w_sum, float wx_sum, float own) {
  return (w_sum > min_weight) ? wx_sum / w_sum : own;
}

/**
 * @brief neighbor_reduce on one backend (specialised per executor: the host here, cuda_exec in
 *        neighbor_reduce.cuh).
 * @tparam exec_t The executor.
 */
template <typename exec_t>
struct neighbor_reduce_impl;

/// neighbor_reduce on the host: the sequential model, the positions in order.
template <>
struct neighbor_reduce_impl<operators::sequential_exec> {
  /// See neighbor_reduce().
  template <typename edge_t, typename apply_t>
  static void run(const resources& /*res*/, reduce_mapping m, const slot_rows<edge_t>& rows,
                  const float* x, const std::int32_t* list, int count, apply_t apply) {
    for (int i = 0; i < count; ++i) {
      const std::int32_t u = list[i];
      float w = 0.0F;
      float wx = 0.0F;
      model_sums(rows, x, u, m, w, wx);
      apply(i, u, guarded_average(w, wx, x[u]));
    }
  }
};

/**
 * @brief neighbor_reduce: apply(i, list[i], average) for every position i of the list, average
 *        the guarded weighted average of list[i]'s live neighbours' values in the association
 *        order of `m` (see the file comment). On CUDA one launch on the stream of `res`; apply
 *        runs on the vertex's first thread.
 * @tparam exec_t  The executor (operators::sequential_exec, or operators::cuda_exec with
 *                 neighbor_reduce.cuh).
 * @tparam edge_t  Edge offset type.
 * @tparam apply_t `DYNG_HD void operator()(int i, std::int32_t u, float average) const`.
 * @param[in] res   The resources (the executor's backend).
 * @param[in] m     The mapping.
 * @param[in] rows  The rows.
 * @param[in] x     The values.
 * @param[in] list  The vertices.
 * @param[in] count Their number.
 * @param[in] apply What to do with each average.
 */
template <typename exec_t, typename edge_t, typename apply_t>
void neighbor_reduce(const resources& res, reduce_mapping m, const slot_rows<edge_t>& rows,
                     const float* x, const std::int32_t* list, int count, apply_t apply) {
  neighbor_reduce_impl<exec_t>::run(res, m, rows, x, list, count, apply);
}

}  // namespace dyng::detail::lp_ops
