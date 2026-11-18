// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/connectedComponents.cu (initParentKernel,
// hookKernel, compressKernel, markRootsKernel, numberRootsKernel, assignComponentIdsKernel,
// launchConnectedComponents)
/**
 * @file connected_components.cuh
 * @brief connected_components' CUDA half (connected_components.hpp): Shiloach-Vishkin written with
 *        cuda_exec's for_each (one grid-stride kernel per phase), the root flags scanned with CUB
 *        on the stream of the resources. One host synchronization per round and one for the
 *        count. CUDA translation units only.
 */
#pragma once

#include "algorithms/label_propagation/operators/connected_components.hpp"
#include "operators/cuda_execution.cuh"
#include "util/cuda_check.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <cstdint>

namespace dyng::detail::lp_ops {

/// par[v] = v for every listed vertex (initParentKernel).
struct cc_init_parent {
  std::int32_t* parent;          ///< the parents
  const std::int32_t* vertices;  ///< the list
  /// Position i.
  __device__ void operator()(std::int64_t i) const {
    const std::int32_t v = vertices[i];
    parent[v] = v;
  }
};

/// Hook the larger root of an edge under the smaller (hookKernel; the count read on the device).
struct cc_hook {
  const std::int32_t* src;        ///< first endpoints
  const std::int32_t* dst;        ///< second endpoints
  const std::int32_t* num_edges;  ///< the edges (device word)
  std::int32_t* parent;           ///< the parents
  std::int32_t* changed;          ///< set when a hook changed a parent
  /// Edge e.
  __device__ void operator()(std::int64_t e) const {
    if (e >= *num_edges) return;
    const std::int32_t ru = parent[src[e]];  // after a compress every parent is a root
    const std::int32_t rv = parent[dst[e]];
    if (ru != rv) {
      const std::int32_t large = max(ru, rv);
      const std::int32_t small = min(ru, rv);
      if (atomicMin(&parent[large], small) != small) *changed = 1;
    }
  }
};

/// Point every listed vertex at its root (compressKernel).
struct cc_compress {
  std::int32_t* parent;          ///< the parents
  const std::int32_t* vertices;  ///< the list
  /// Position i.
  __device__ void operator()(std::int64_t i) const {
    const std::int32_t v = vertices[i];
    std::int32_t root = parent[v];
    while (parent[root] != root) root = parent[root];
    parent[v] = root;
  }
};

/// Flag the listed roots (markRootsKernel).
struct cc_mark_roots {
  const std::int32_t* parent;    ///< the parents
  const std::int32_t* vertices;  ///< the list
  std::int32_t* root_flag;       ///< per position
  /// Position i.
  __device__ void operator()(std::int64_t i) const {
    const std::int32_t v = vertices[i];
    root_flag[i] = parent[v] == v ? 1 : 0;
  }
};

/// A root's id is its rank among the roots; the last position writes the count
/// (numberRootsKernel).
struct cc_number_roots {
  const std::int32_t* vertices;   ///< the list
  std::int64_t n;                 ///< its length
  const std::int32_t* root_flag;  ///< per position
  const std::int32_t* root_scan;  ///< the roots before each position
  std::int32_t* root_id;          ///< per vertex id
  std::int32_t* count;            ///< the components (device word)
  /// Position i.
  __device__ void operator()(std::int64_t i) const {
    if (root_flag[i] != 0) root_id[vertices[i]] = root_scan[i];
    if (i == n - 1) *count = root_scan[i] + root_flag[i];
  }
};

/// Every listed vertex takes its root's id (assignComponentIdsKernel).
struct cc_assign {
  const std::int32_t* parent;    ///< the parents
  const std::int32_t* root_id;   ///< per root its id
  std::int32_t* component_id;    ///< per vertex id
  const std::int32_t* vertices;  ///< the list
  /// Position i.
  __device__ void operator()(std::int64_t i) const {
    const std::int32_t v = vertices[i];
    component_id[v] = root_id[parent[v]];
  }
};

/// connected_components on cuda_exec: Shiloach-Vishkin.
template <>
struct components_impl<operators::cuda_exec> {
  /// See connected_components().
  static components_result run(const resources& res, const std::int32_t* src,
                               const std::int32_t* dst, const std::int32_t* num_edges,
                               int max_edges, const std::int32_t* vertices, int num_vertices,
                               const components_scratch& s, std::int32_t* component_id,
                               int max_rounds) {
    const operators::cuda_exec exec(res);
    const auto stream = static_cast<cudaStream_t>(res.stream().get());
    std::int32_t* changed = s.words;
    std::int32_t* count = s.words + 1;
    exec.for_each(num_vertices, cc_init_parent{s.parent, vertices});
    components_result r;
    while (max_edges > 0) {
      if (r.rounds == max_rounds) {
        DYNG_FAIL("label_propagation: the connected components of G' did not converge in ",
                  max_rounds, " Shiloach-Vishkin rounds (a malfunction guard)");
      }
      ++r.rounds;
      exec.write(changed, std::int32_t{0});
      exec.for_each(max_edges, cc_hook{src, dst, num_edges, s.parent, changed});
      exec.for_each(num_vertices, cc_compress{s.parent, vertices});
      if (exec.read(changed) == 0) {
        break;
      }
    }
    exec.for_each(num_vertices, cc_mark_roots{s.parent, vertices, s.root_flag});
    std::size_t bytes = s.scan_temp_bytes;
    DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(s.scan_temp, bytes, s.root_flag, s.root_scan,
                                                num_vertices, stream));
    exec.for_each(num_vertices, cc_number_roots{vertices, num_vertices, s.root_flag, s.root_scan,
                                                s.root_id, count});
    exec.for_each(num_vertices, cc_assign{s.parent, s.root_id, component_id, vertices});
    r.count = exec.read(count);
    return r;
  }
};

/**
 * @brief The bytes of the CUDA scan's temporary storage for up to `max_vertices` vertices.
 * @param[in] max_vertices The longest list.
 * @return The bytes (at least 1).
 */
inline std::size_t components_scan_temp_bytes(int max_vertices) {
  std::size_t bytes = 0;
  DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
      nullptr, bytes, static_cast<const std::int32_t*>(nullptr),
      static_cast<std::int32_t*>(nullptr), max_vertices > 0 ? max_vertices : 1));
  return bytes > 0 ? bytes : 1;
}

}  // namespace dyng::detail::lp_ops
