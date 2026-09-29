// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:include/cycle_enum/cuda/cuda_dfs.cuh (CsrView,
// lower_bound_u32, find_edge) and src/dynamic/update_cuda_kernel.cu (DeviceEdge)
/**
 * @file device_csr.cuh
 * @brief Device building blocks over sorted CSR rows shared by the device set apply
 *        (graph/apply_set_device.cu) and the cycle_count kernels: a read-only CSR view, a
 *        lower_bound over a sorted range and a directed edge.
 *
 * CUDA translation units only. Rows are sorted by neighbour id and hold distinct neighbours, so
 * one lower_bound both tests an edge and splits a row at a vertex id. The view reads vertex ids as
 * unsigned 32-bit values and row offsets as the unsigned type of the graph's edge_t (uint32_t for
 * int32_t offsets, exactly CycleEnumeration-GPU's 32-bit CsrView; uint64_t for int64_t offsets):
 * the ids and offsets of a graph are non-negative, so the unsigned reading is the same value.
 */
#pragma once

#include <cstdint>
#include <type_traits>

namespace dyng::detail {

/// A vertex id as the device kernels read it (CycleEnumeration-GPU's 32-bit VertexId).
using device_vertex = std::uint32_t;

/**
 * @brief The unsigned offset type the device kernels read for an edge offset type.
 * @tparam edge_t Edge offset type (int32_t or int64_t).
 */
template <typename edge_t>
using device_offset_t = std::make_unsigned_t<edge_t>;

/**
 * @brief Read-only CSR view (CycleEnumeration-GPU's CsrView, templated on the offset type).
 * @tparam offset_t Unsigned row offset type (uint32_t: the original's layout).
 */
template <typename offset_t>
struct device_csr {
  std::uint32_t vertex_count = 0;            ///< n
  const offset_t* offsets = nullptr;         ///< n + 1 row offsets
  const device_vertex* neighbors = nullptr;  ///< sorted out-neighbours per row
};

/**
 * @brief A directed edge of a change list (CycleEnumeration-GPU's DeviceEdge).
 */
struct device_edge {
  device_vertex source = 0;  ///< tail
  device_vertex target = 0;  ///< head
};

/**
 * @brief First position in the sorted range `values[begin, end)` not less than `target`
 *        (lower_bound_u32).
 * @tparam offset_t Position type.
 * @param[in] values The array.
 * @param[in] begin  First position.
 * @param[in] end    One past the last position.
 * @param[in] target The value searched.
 * @return The position.
 */
template <typename offset_t>
__device__ __forceinline__ offset_t lower_bound_u32(const device_vertex* __restrict__ values,
                                                    offset_t begin, offset_t end,
                                                    const device_vertex target) {
  while (begin < end) {
    const offset_t mid = begin + ((end - begin) >> 1);
    if (__ldg(values + mid) < target) {
      begin = mid + 1;
    } else {
      end = mid;
    }
  }
  return begin;
}

/**
 * @brief Whether the directed edge `from -> to` exists; `position` receives the lower_bound of
 *        `to` in the row of `from` (find_edge).
 * @tparam offset_t Offset type.
 * @param[in]  graph    The CSR.
 * @param[in]  from     Tail.
 * @param[in]  to       Head.
 * @param[out] position lower_bound of `to` in the row.
 * @param[out] row_end  End of the row.
 * @return true if the edge exists.
 */
template <typename offset_t>
__device__ __forceinline__ bool find_edge(const device_csr<offset_t> graph,
                                          const device_vertex from, const device_vertex to,
                                          offset_t& position, offset_t& row_end) {
  const offset_t begin = __ldg(graph.offsets + from);
  row_end = __ldg(graph.offsets + from + 1);
  position = lower_bound_u32(graph.neighbors, begin, row_end, to);
  return position < row_end && __ldg(graph.neighbors + position) == to;
}

}  // namespace dyng::detail
