// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:src/sospUpdateGpu.cu (SospWorkspace::reserve / nextGeneration,
// runPersistent, choosePacking, sospUpdateGpu, sospFromScratchGpu)
/**
 * @file cuda.cu
 * @brief The CUDA backend of sssp: the fused engine (enact_fused, Tier B). The host side of
 *        MOSP-CUDA's sospUpdateGpu() / sospFromScratchGpu(), ported straight; the kernel is in
 *        fused.cuh.
 *
 * Mechanical changes only: names, templates on the index types, namespace dyng::detail,
 * exceptions instead of `bool` + `cerr` (invalid_argument_error for a tree that does not belong to
 * the graph, not_supported_error when the kernel cannot run cooperatively), the stream and device
 * of `resources` instead of the legacy default stream, the workspace leased from the pool of
 * `resources` (dyng scratch buffers; the pinned control-block mirror from its staging resource),
 * the grid size computed per kernel instantiation (cached in the workspace). Two host-side
 * additions read the control block the kernel already writes: `affected` (see fused.cuh) and a
 * parent-cycle check (pointer jumping still active in its last round is impossible in a forest;
 * the check only matters for trees imported with validate_inputs = false).
 */
#include "algorithms/sssp/fused.cuh"
#include "algorithms/sssp/problem.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "graph/instantiate.hpp"
#include "util/cuda_check.hpp"
#include "util/device_error_flags.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace dyng::detail {

namespace {

using sssp_fused::block_size;
using sssp_fused::packed_inf;
using sssp_fused::u64;

cudaStream_t native(const resources& res) noexcept {
  return static_cast<cudaStream_t>(res.stream().get());
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// Workspace (MOSP-CUDA@e220ee2:src/sospUpdateGpu.cu SospWorkspace)
// ------------------------------------------------------------------------------------------------

template <typename vertex_t>
void sssp_cuda_workspace<vertex_t>::reserve(const resources& res, std::int64_t requested) {
  if (requested <= capacity) {
    return;
  }
  const scoped_device guard(res.device());
  const auto n = static_cast<std::size_t>(std::max<std::int64_t>(requested, 1));
  packed.reserve(res, n);
  stamp.reserve(res, n);
  in_far.reserve(res, n);
  flag.reserve(res, n);
  ancestor.reserve(res, n);
  list_a.reserve(res, n);
  list_b.reserve(res, n);
  far_a.reserve(res, n);
  far_b.reserve(res, n);
  candidates.reserve(res, n);
  frontier.reserve(res, n);
  control.reserve(res, sizeof(sssp_fused::control<vertex_t>));
  if (host_control.size() < sizeof(sssp_fused::control<vertex_t>)) {
    host_control = buffer<unsigned char>(sizeof(sssp_fused::control<vertex_t>), res.stream(),
                                         resources_access::staging_memory(res), res.device());
  }
  const cudaStream_t stream = native(res);
  DYNG_CUDA_TRY(cudaMemsetAsync(stamp.data(), 0, n * sizeof(int), stream));
  DYNG_CUDA_TRY(cudaMemsetAsync(in_far.data(), 0, n * sizeof(int), stream));
  DYNG_CUDA_TRY(cudaMemsetAsync(flag.data(), 0, n * sizeof(int), stream));
  capacity = requested;
  generation = 0;
}

template <typename vertex_t>
int sssp_cuda_workspace<vertex_t>::next_generation(const resources& res) {
  // An update uses a few generations plus one per push iteration and threshold increase; restart
  // well before the counter could wrap.
  if (generation > INT_MAX / 2) {
    const scoped_device guard(res.device());
    DYNG_CUDA_TRY(cudaMemsetAsync(stamp.data(), 0, static_cast<std::size_t>(capacity) * sizeof(int),
                                  native(res)));
    generation = 0;
  }
  return ++generation;
}

template <typename vertex_t>
std::size_t sssp_cuda_workspace<vertex_t>::bytes() const noexcept {
  return packed.bytes() + stamp.bytes() + in_far.bytes() + flag.bytes() + ancestor.bytes() +
         list_a.bytes() + list_b.bytes() + far_a.bytes() + far_b.bytes() + candidates.bytes() +
         frontier.bytes() + control.bytes() + host_control.size() +
         (changed_from.capacity() + changed_to.capacity()) * sizeof(vertex_t) +
         device_changed_from.bytes() + device_changed_to.bytes() + device_insert_heads.bytes();
}

template struct sssp_cuda_workspace<std::int32_t>;
template struct sssp_cuda_workspace<std::int64_t>;

// ------------------------------------------------------------------------------------------------
// The persistent launch (MOSP-CUDA@e220ee2:src/sospUpdateGpu.cu runPersistent, choosePacking)
// ------------------------------------------------------------------------------------------------

namespace {

/// Co-resident blocks of the kernel instantiation (cooperative launch), computed once per
/// workspace and kernel, as SospWorkspace::reserve computes gridBlocks.
template <typename vertex_t, typename edge_t, typename weight_t>
int grid_blocks(const resources& res, sssp_cuda_workspace<vertex_t>& ws) {
  const auto kernel = &sssp_fused::sssp_persistent_kernel<vertex_t, edge_t, weight_t>;
  const void* key = reinterpret_cast<const void*>(kernel);
  if (ws.grid_kernel != key) {
    const cuda_device_properties& props = resources_access::device_properties(res);
    if (!props.cooperative_launch) {
      throw not_supported_error(concat_message(
          "dyng: sssp: the fused engine needs cooperative launch, which CUDA device ",
          props.ordinal, " (", props.name, ") does not support"));
    }
    int per_sm = 0;
    DYNG_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, block_size, 0));
    if (per_sm < 1) {
      throw not_supported_error(
          concat_message("dyng: sssp: the persistent kernel does not fit on an SM of CUDA device ",
                         props.ordinal, " (", props.name, ")"));
    }
    ws.grid_blocks = per_sm * props.multiprocessor_count;
    ws.grid_kernel = key;
  }
  return ws.grid_blocks;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void run_persistent(const resources& res, sssp_fused::params<vertex_t, edge_t, weight_t>& params,
                    sssp_cuda_workspace<vertex_t>& ws, sssp_counters& stats) {
  using control_type = sssp_fused::control<vertex_t>;
  params.packed = ws.packed.data();
  params.stamp = ws.stamp.data();
  params.in_far = ws.in_far.data();
  params.flag = ws.flag.data();
  params.ancestor = ws.ancestor.data();
  params.candidates = ws.candidates.data();
  params.frontier = ws.frontier.data();
  params.near_a = ws.list_a.data();
  params.near_b = ws.list_b.data();
  params.far_a = ws.far_a.data();
  params.far_b = ws.far_b.data();
  params.ctl = reinterpret_cast<control_type*>(ws.control.data());
  params.generation = ws.next_generation(res);
  const int blocks = grid_blocks<vertex_t, edge_t, weight_t>(res, ws);

  const cudaStream_t stream = native(res);
  auto* mirror = reinterpret_cast<control_type*>(ws.host_control.data());
  control_type initial{};
  initial.minimum = packed_inf;
  *mirror = initial;
  DYNG_CUDA_TRY(cudaMemcpyAsync(ws.control.data(), mirror, sizeof(control_type),
                                cudaMemcpyHostToDevice, stream));
  void* args[] = {&params};
  DYNG_CUDA_TRY(cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(
          &sssp_fused::sssp_persistent_kernel<vertex_t, edge_t, weight_t>),
      dim3(static_cast<unsigned int>(blocks)), dim3(block_size), args, 0, stream));
  DYNG_CUDA_TRY(cudaMemcpyAsync(mirror, ws.control.data(), sizeof(control_type),
                                cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  const control_type result = *mirror;
  if (result.overflow != 0) {
    throw_device_errors(bits_of(device_error::invalid_input), "sssp::update",
                        "an input distance is negative or larger than (n - 1) * max weight; the "
                        "initial tree does not belong to this graph");
  }
  if (!params.from_scratch && result.rounds == params.max_rounds && result.rounds > 0 &&
      result.active[(result.rounds - 1) % 3] != 0) {
    throw_device_errors(bits_of(device_error::parent_cycle), "sssp::update",
                        "the input shortest-path tree has a parent cycle (pointer jumping did not "
                        "reach a root in ceil(log2 n) + 1 rounds)");
  }
  ws.generation = std::max(ws.generation, result.generation);
  stats.invalidated = static_cast<std::int64_t>(result.invalidated);
  stats.packed_parents = params.packed_format.has_parents();
  stats.iterations = result.iterations;
  stats.epochs = result.epochs;
  stats.pushes = result.pushes;
  stats.affected = static_cast<std::int64_t>(result.affected);
}

/// Choose the packing for n vertices and the distance bound (n - 1) * max_weight; fails only if
/// distances could overflow 62 bits. The kernel drops candidates above the bound, so every packed
/// distance stays within it.
template <typename vertex_t, typename edge_t, typename weight_t>
void choose_packing(std::int64_t n, std::int64_t max_weight,
                    sssp_fused::params<vertex_t, edge_t, weight_t>& params) {
  const u64 weight = static_cast<u64>(std::max<std::int64_t>(max_weight, 1));
  const u64 hops = static_cast<u64>(std::max<std::int64_t>(n - 1, 1));
  DYNG_EXPECTS(weight <= sssp_fused::output_max_distance / hops, "sssp: distances up to ", weight,
               " * ", hops, " do not fit in 62 bits");
  params.max_distance = weight * hops;
  params.packed_format = sssp_fused::make_packing(n, params.max_distance);
}

template <typename vertex_t, typename edge_t, typename weight_t>
sssp_fused::device_csr<vertex_t, edge_t, weight_t> out_csr(
    const sssp_graph<vertex_t, edge_t, weight_t>& g) {
  sssp_fused::device_csr<vertex_t, edge_t, weight_t> out;
  out.number_of_nodes = g.num_vertices;
  out.row_ptr = g.out_row_ptr;
  out.col_ind = g.out_col_ind;
  out.weights = g.out_weights;
  return out;
}

template <typename vertex_t, typename edge_t, typename weight_t>
sssp_fused::device_csr<vertex_t, edge_t, weight_t> in_csr(
    const sssp_graph<vertex_t, edge_t, weight_t>& g) {
  sssp_fused::device_csr<vertex_t, edge_t, weight_t> in;
  in.number_of_nodes = g.num_vertices;
  in.row_ptr = g.in_row_ptr;
  in.col_ind = g.in_col_ind;
  in.weights = g.in_weights;
  return in;
}

template <typename vertex_t>
vertex_t list_length(std::size_t count, const char* what) {
  DYNG_EXPECTS(count <= static_cast<std::size_t>(std::numeric_limits<vertex_t>::max()),
               "sssp: ", count, " ", what, " exceed the index range of the vertex type");
  return static_cast<vertex_t>(count);
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// Entry points (MOSP-CUDA@e220ee2:src/sospUpdateGpu.cu sospUpdateGpu, sospFromScratchGpu)
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_cuda_update(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run) {
  scoped_stage stage(res, "sssp.enact_fused");
  const scoped_device guard(res.device());
  const vertex_t n = run.graph.num_vertices;
  run.counters = sssp_counters{};
  if (n == 0) {
    return;
  }
  sssp_cuda_workspace<vertex_t>& ws = *run.cuda_ws;
  ws.reserve(res, n);
  DYNG_EXPECTS(run.delta > 0, "sssp: the near-far width delta must be > 0, got ", run.delta);
  sssp_fused::params<vertex_t, edge_t, weight_t> params{};
  choose_packing(n, run.max_weight, params);
  int rounds = 0;
  while ((1LL << rounds) < static_cast<long long>(n)) {
    ++rounds;
  }
  const sssp_changes<vertex_t>& changes = *run.changes;
  params.out = out_csr(run.graph);
  params.in = in_csr(run.graph);
  params.changes.changed_from = changes.changed_from;
  params.changes.changed_to = changes.changed_to;
  params.changes.number_of_changed = list_length<vertex_t>(changes.num_changed, "changed edges");
  params.changes.insert_heads = changes.insert_heads;
  params.changes.number_of_insert_heads =
      list_length<vertex_t>(changes.num_insert_heads, "insertion heads");
  params.source = run.source;
  params.from_scratch = false;
  params.delta = static_cast<u64>(run.delta);
  params.max_rounds = rounds + 1;
  params.distances = reinterpret_cast<long long*>(run.distances);
  params.parent = run.parents;
  run_persistent(res, params, ws, run.counters);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_cuda_compute(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run) {
  scoped_stage stage(res, "sssp.enact_fused");
  const scoped_device guard(res.device());
  const vertex_t n = run.graph.num_vertices;
  run.counters = sssp_counters{};
  if (n == 0) {
    return;
  }
  DYNG_EXPECTS(run.source >= 0 && run.source < n, "sssp: source ", run.source,
               " is out of range [0, ", n, ")");
  sssp_cuda_workspace<vertex_t>& ws = *run.cuda_ws;
  ws.reserve(res, n);
  DYNG_EXPECTS(run.delta > 0, "sssp: the near-far width delta must be > 0, got ", run.delta);
  sssp_fused::params<vertex_t, edge_t, weight_t> params{};
  choose_packing(n, run.max_weight, params);
  params.out = out_csr(run.graph);
  params.in = params.out;
  params.source = run.source;
  params.from_scratch = true;
  params.delta = static_cast<u64>(run.delta);
  params.distances = reinterpret_cast<long long*>(run.distances);
  params.parent = run.parents;
  run_persistent(res, params, ws, run.counters);
}

#define DYNG_INSTANTIATE_SSSP_CUDA(V, E, W)                                      \
  template void sssp_cuda_update<V, E, W>(const resources&, sssp_run<V, E, W>&); \
  template void sssp_cuda_compute<V, E, W>(const resources&, sssp_run<V, E, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_SSSP_CUDA)
#undef DYNG_INSTANTIATE_SSSP_CUDA

// One registration per line (the registrar's name is made unique by __LINE__).
DYNG_REGISTER_KERNEL(sssp_fused::sssp_persistent_kernel<std::int32_t, std::int32_t, std::int32_t>);
DYNG_REGISTER_KERNEL(sssp_fused::sssp_persistent_kernel<std::int32_t, std::int64_t, std::int32_t>);
DYNG_REGISTER_KERNEL(sssp_fused::sssp_persistent_kernel<std::int64_t, std::int64_t, std::int32_t>);

}  // namespace dyng::detail
