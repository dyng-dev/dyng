// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/iterativePropagation.cu (initFrontierKernel,
// updateLabel's damped rule, commitLabelsKernel, enforceGroundTruthKernel and the iteration of
// launchIterativePropagation) and src/dynLP.cu (the order of one batch's steps)
/**
 * @file operators_engine.cu
 * @brief The operators engine of label_propagation on CUDA (operators_engine.hpp; ADR 0101).
 */
#include "algorithms/label_propagation/operators/binned_frontier.cuh"
#include "algorithms/label_propagation/operators/connected_components.cuh"
#include "algorithms/label_propagation/operators/neighbor_reduce.cuh"
#include "algorithms/label_propagation/operators_engine.hpp"
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "graph/graph_impl.hpp"
#include "operators/cuda_execution.cuh"
#include "util/cuda_check.hpp"

#include <dyng/core/error.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace dyng::detail::lp_cuda {

namespace {

using frontier_t = lp_ops::binned_frontier<std::int32_t, num_degree_bins>;
using push_t = lp_ops::binned_push<std::int32_t, num_degree_bins>;

constexpr std::int32_t vertex_active = 0;  // vertex_status::active

cudaStream_t native(const resources& res) noexcept {
  return static_cast<cudaStream_t>(res.stream().get());
}

template <typename edge_t>
lp_ops::slot_rows<edge_t> slot_rows_of(const device_slotted<std::int32_t, edge_t, float>& d) {
  lp_ops::slot_rows<edge_t> r;
  r.row_start = d.row_ptr.data();
  r.row_end = d.row_end.data();
  r.col_ind = d.col_ind.data();
  r.weights = d.weights.data();
  r.status = d.status.data();
  return r;
}

template <typename edge_t>
rows<edge_t> fused_rows_of(const device_slotted<std::int32_t, edge_t, float>& d) {
  rows<edge_t> r;
  r.row_start = d.row_ptr.data();
  r.row_end = d.row_end.data();
  r.col_ind = d.col_ind.data();
  r.values = d.weights.data();
  r.status = d.status.data();
  r.n = static_cast<std::int64_t>(d.num_vertices);
  return r;
}

/// The first set: the affected, active vertices without a seed label (initFrontierKernel).
template <typename edge_t>
struct seed_frontier {
  const std::int32_t* affected;    ///< the affected flags
  const std::int32_t* gt_class;    ///< the seed classes
  lp_ops::slot_rows<edge_t> rows;  ///< the rows (states, degrees)
  push_t out;                      ///< the first set
  /// Vertex v.
  __device__ void operator()(std::int64_t i) const {
    const auto v = static_cast<std::int32_t>(i);
    if (affected[v] == 1 && rows.status[v] == vertex_active && gt_class[v] < 0) {
      out.push(v, rows.degree(v));
    }
  }
};

/// The damped rule of one queued vertex (updateLabel, damped: F' = 0.5f (F + average), clamped,
/// into the side array; changed when it moved by more than delta).
struct damped_rule {
  const float* labels;    ///< the iteration's labels
  float* new_labels;      ///< per position the new label (the bin's side array)
  std::uint8_t* changed;  ///< per position whether it changed by more than delta
  float delta;            ///< the tolerance
  /// Position i, vertex u, its neighbours' guarded average.
  __device__ void operator()(int i, std::int32_t u, float average) const {
    const float old_label = labels[u];
    float new_label = average;
#if !defined(DYNG_MUTATION_LP_DAMPED_NO_DAMPING)
    new_label = 0.5f * (old_label + new_label);
#endif  // MUTATION (AD9) when defined: F' = the average, without the damping
    new_label = fminf(fmaxf(new_label, 0.0f), 1.0f);
    new_labels[i] = new_label;
    changed[i] = fabsf(new_label - old_label) > delta ? 1 : 0;
  }
};

/// A neighbour enters the next set if it is active and has no seed label (appendNextFrontier).
struct eligible_neighbour {
  const std::int32_t* status;    ///< the states
  const std::int32_t* gt_class;  ///< the seed classes
  /// Neighbour v.
  __device__ bool operator()(std::int32_t v) const {
    return status[v] == vertex_active && gt_class[v] < 0;
  }
};

/// The damped commit of one list (commitLabelsKernel).
struct commit_labels {
  float* labels;             ///< the labels
  const float* new_labels;   ///< per position the new label
  const std::int32_t* list;  ///< the list
  /// Position i.
  __device__ void operator()(std::int64_t i) const {
    labels[list[i]] = new_labels[i];
  }
};

/// The seeds keep their classes (enforceGroundTruthKernel).
struct enforce_seeds {
  float* labels;                 ///< the labels
  const std::int32_t* gt_class;  ///< the seed classes
  /// Vertex v.
  __device__ void operator()(std::int64_t v) const {
    if (gt_class[v] == 0) {
      labels[v] = 0.0f;
    } else if (gt_class[v] == 1) {
      labels[v] = 1.0f;
    }
  }
};

/// The reduction mapping of a bin (the fused engine's: 8 lanes, a warp, a block of 1024 threads
/// while the frontier fits its persistent kernel, else of 256).
lp_ops::reduce_mapping mapping_of(int bin, std::int64_t frontier) noexcept {
  switch (bin) {
    case bin_small:
      return lp_ops::reduce_mapping::lanes_8;
    case bin_medium:
      return lp_ops::reduce_mapping::warp;
    default:
      return frontier <= small_frontier_limit ? lp_ops::reduce_mapping::block_1024
                                              : lp_ops::reduce_mapping::block_256;
  }
}

/// The settle of the graph's report (after the first synchronization since the commit).
template <typename edge_t>
void settle(const graph<std::int32_t, edge_t, float>& g, operators_batch& out) {
  if (out.settled) {
    return;
  }
  const device_slotted_settled s = graph_access::settle_device_apply(g, out.error_detail);
  out.graph_errors = s.error_bits;
  out.appended = s.appended;
  out.dropped = s.dropped;
  out.settled = true;
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// The workspace
// ------------------------------------------------------------------------------------------------

void operators_workspace::reserve(const resources& res, const int (&capacity)[num_degree_bins]) {
  const scoped_device guard(res.device());
  for (int b = 0; b < num_degree_bins; ++b) {
    if (capacity[b] > sized[b] || sized[b] == 0) {
      changed[b].reserve(res, static_cast<std::size_t>(std::max(capacity[b], 1)));
      sized[b] = std::max(capacity[b], 1);
    }
  }
}

std::size_t operators_workspace::bytes() const noexcept {
  std::size_t b = 0;
  for (const auto& c : changed) {
    b += c.bytes();
  }
  return b;
}

// ------------------------------------------------------------------------------------------------
// The hooks
// ------------------------------------------------------------------------------------------------

template <typename edge_t>
void ops_identify_affected(const resources& res, const graph<std::int32_t, edge_t, float>& g,
                           state& st, workspace& ws, const batch_args& a, operators_batch& out) {
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  const auto& dev = graph_access::slotted_device(res, g);
  const int n = static_cast<int>(st.n);
  const int num_inserted = static_cast<int>(dev.num_inserted);
  const int num_edges = static_cast<int>(dev.num_edges);
  std::int32_t* before_bits = ws.root_id.data();  // free once the components are numbered
  // The seeds' classes, then the inserted vertices affected and stamped (the fused engine's
  // passes: setGroundTruthKernel's classes, markInsertedKernel).
  launch_set_seed_classes(dev.inserted.data(),
                          dev.has_labels ? dev.inserted_labels.data() : nullptr, num_inserted,
                          st.gt_class.data(), stream);
  launch_mark_inserted(dev.inserted.data(), num_inserted, ws.affected.data(), st.new_stamp.data(),
                       a.batch_stamp, stream);
  if (num_inserted == 0) {
    // A batch without insertions: the endpoints of its edges are affected (ADR 0072's fix).
    launch_mark_edge_endpoints(ws.affected.data(), dev.edge_src.data(), dev.edge_dst.data(),
                               num_edges, n, dev.status.data(), stream);
    launch_keep_labels(st.labels.data(), before_bits, n, stream);
    return;
  }
  // G' (the strong new-to-new edges; the endpoints of every batch edge affected), then its
  // components over Ins_eff with connected_components.
  launch_build_sparsified_subgraph(ws.affected.data(), dev.edge_src.data(), dev.edge_dst.data(),
                                   dev.edge_weight.data(), num_edges, n, dev.status.data(),
                                   st.new_stamp.data(), a.batch_stamp, a.tau, ws.sub_src.data(),
                                   ws.sub_dst.data(), ws.words.data() + 2, stream);
  lp_ops::components_scratch s;
  s.parent = ws.par.data();
  s.root_id = ws.root_id.data();
  s.root_flag = ws.root_flag.data();
  s.root_scan = ws.root_scan.data();
  s.scan_temp = ws.scan_temp.data();
  s.scan_temp_bytes = ws.scan_temp_bytes;
  s.words = ws.words.data();
  const lp_ops::components_result cc = lp_ops::connected_components<operators::cuda_exec>(
      res, ws.sub_src.data(), ws.sub_dst.data(), ws.words.data() + 2, num_edges,
      dev.inserted.data(), num_inserted, s, ws.component_id.data());
  out.components = cc.count;
  out.hook_rounds = cc.rounds;
  out.host_syncs += cc.rounds + 1;
  launch_keep_labels(st.labels.data(), before_bits, n, stream);
}

template <typename edge_t>
void ops_seed(const resources& res, const graph<std::int32_t, edge_t, float>& g, state& st,
              workspace& ws, const operators_batch& b) {
  const scoped_device guard(res.device());
  const auto& dev = graph_access::slotted_device(res, g);
  launch_label_initialization(fused_rows_of(dev), st.labels.data(), ws.component_id.data(),
                              b.components, dev.inserted.data(), static_cast<int>(dev.num_inserted),
                              st.gt_class.data(), ws, native(res));
}

template <typename edge_t>
propagation_report ops_iterate(const resources& res, const rows<edge_t>& g, float* d_labels,
                               const std::int32_t* gt_class, const std::int32_t* affected,
                               float delta, int max_iterations, state& st, workspace& ws,
                               operators_workspace& ows, void (*after_first_sync)(void*),
                               void* arg) {
  const scoped_device guard(res.device());
  const operators::cuda_exec exec(res);
  lp_ops::slot_rows<edge_t> r;
  r.row_start = g.row_start;
  r.row_end = g.row_end;
  r.col_ind = g.col_ind;
  r.weights = g.values;
  r.status = g.status;
  const std::int64_t n = g.n;
  ows.reserve(res, st.capacity);
  frontier_t::storage fs{};
  for (int b = 0; b < num_degree_bins; ++b) {
    fs.queue[0][b] = ws.queue[0][b].data();
    fs.queue[1][b] = ws.queue[1][b].data();
    fs.capacity[b] = st.capacity[b];
  }
  fs.counts = ws.counts.data();
  fs.host_counts = ws.host_counts.data();
  fs.stamp = st.stamp.data();
  fs.bins.max_degree[0] = small_bin_max_degree;
  fs.bins.max_degree[1] = medium_bin_max_degree;
  const frontier_t f(fs);
  float* labels = d_labels;
#if defined(DYNG_MUTATION_LP_D_C_STALE_BUFFER)
  // MUTATION (D-c, AD9): the damped commit goes to the other of two full label arrays, whose
  // entries not updated in the iteration are two iterations old (the original's refactor before
  // 8419483); the labels copied back at the end.
  const cudaStream_t stream = native(res);
  float* stale_other = nullptr;
  DYNG_CUDA_TRY(cudaMallocAsync(&stale_other, static_cast<std::size_t>(n) * sizeof(float), stream));
  DYNG_CUDA_TRY(cudaMemcpyAsync(stale_other, labels, static_cast<std::size_t>(n) * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
#endif
  propagation_report rep;
  rep.tag_before = st.next_tag;
  // The first set: the affected, active vertices without a seed label.
  int cur = 0;
  int tag = st.next_tag++;
  f.clear(exec, cur);
  exec.for_each(n, seed_frontier<edge_t>{affected, gt_class, r, f.into(cur, tag)});
  std::int64_t total = f.read_size(exec, cur);
  ++rep.host_syncs;
  if (after_first_sync != nullptr) {
    after_first_sync(arg);
  }
  rep.initial_frontier = total;
  const eligible_neighbour eligible{g.status, gt_class};
  while (total > 0 && rep.iterations < max_iterations) {
    const int next = cur ^ 1;
    f.clear(exec, next);
    rep.frontier_visits += total;
    const push_t into = f.into(next, tag + 1);
    // neighbor_reduce: every queued vertex's new label from this iteration's labels.
    for (int half = 0; half < 2; ++half) {
      for (int b = 0; b < num_degree_bins; ++b) {
        const int count = f.count(cur, b, half);
        const int offset = f.offset(b, half, count);
        lp_ops::neighbor_reduce<operators::cuda_exec>(
            res, mapping_of(b, total), r, labels, f.list(cur, b, half), count,
            damped_rule{labels, ws.new_labels[b].data() + offset, ows.changed[b].data() + offset,
                        delta});
      }
    }
    // expand: the changed vertices and their active unlabelled neighbours into the next set.
    for (int half = 0; half < 2; ++half) {
      for (int b = 0; b < num_degree_bins; ++b) {
        const int count = f.count(cur, b, half);
        const int offset = f.offset(b, half, count);
        lp_ops::expand<operators::cuda_exec>(res, lp_ops::threads_of(mapping_of(b, total)), r,
                                             f.list(cur, b, half), ows.changed[b].data() + offset,
                                             count, eligible, into);
      }
    }
    // The damped commit, after both half-sweeps read this iteration's labels.
    for (int half = 0; half < 2; ++half) {
      for (int b = 0; b < num_degree_bins; ++b) {
        const int count = f.count(cur, b, half);
        const int offset = f.offset(b, half, count);
#if defined(DYNG_MUTATION_LP_D_C_STALE_BUFFER)
        exec.for_each(count, commit_labels{stale_other, ws.new_labels[b].data() + offset,
                                           f.list(cur, b, half)});  // MUTATION: the other buffer
#else
        exec.for_each(
            count, commit_labels{labels, ws.new_labels[b].data() + offset, f.list(cur, b, half)});
#endif
      }
    }
#if defined(DYNG_MUTATION_LP_D_C_STALE_BUFFER)
    std::swap(labels, stale_other);  // MUTATION: the buffers swap
#endif
    total = f.read_size(exec, next);
    ++rep.host_syncs;
    cur = next;
    ++tag;
    ++rep.iterations;
  }
  st.next_tag = tag + 1;
  rep.capped = total > 0;
#if defined(DYNG_MUTATION_LP_D_C_STALE_BUFFER)
  if (labels != d_labels) {  // MUTATION: the current buffer back into the result's labels
    DYNG_CUDA_TRY(cudaMemcpyAsync(d_labels, labels, static_cast<std::size_t>(n) * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));
    stale_other = labels;
    labels = d_labels;
  }
  DYNG_CUDA_TRY(cudaFreeAsync(stale_other, stream));
#endif
  // The seeds keep their classes (enforceGroundTruthKernel).
  exec.for_each(n, enforce_seeds{labels, gt_class});
  return rep;
}

template <typename edge_t>
void ops_propagate(const resources& res, const graph<std::int32_t, edge_t, float>& g, state& st,
                   workspace& ws, operators_workspace& ows, const batch_args& a,
                   operators_batch& out) {
  const scoped_device guard(res.device());
  const auto& dev = graph_access::slotted_device(res, g);
  struct settle_arg {
    const graph<std::int32_t, edge_t, float>* g;
    operators_batch* out;
    static void call(void* p) {
      auto* s = static_cast<settle_arg*>(p);
      settle(*s->g, *s->out);
    }
  } arg{&g, &out};
  out.propagation =
      ops_iterate(res, fused_rows_of(dev), st.labels.data(), st.gt_class.data(), ws.affected.data(),
                  a.delta, a.max_iterations, st, ws, ows, &settle_arg::call, &arg);
  out.host_syncs += out.propagation.host_syncs;
  out.capped = out.propagation.capped;
  if (out.capped && a.fallback) {
    // on_limit::fallback_recompute: IrLP of the current graph replaces the labels.
    auto lease = resources_access::workspaces(res).template acquire<irlp_workspace>(res);
    out.irlp_iterations = run_irlp(res, fused_rows_of(dev), st.gt_class.data(), st.labels.data(),
                                   a.delta, a.max_iterations, lease.get(), out.irlp_capped);
    out.host_syncs += out.irlp_iterations + 1;
    out.recomputed = true;
  }
}

template <typename edge_t>
void ops_finalize(const resources& res, const graph<std::int32_t, edge_t, float>& g, state& st,
                  workspace& ws, operators_batch& out) {
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  const auto& dev = graph_access::slotted_device(res, g);
  // The stats' affected (the vertices whose label bits changed) and G''s size, in one read.
  launch_count_changed(st.labels.data(), ws.root_id.data(), static_cast<int>(st.n),
                       ws.words.data() + 3, stream);
  DYNG_CUDA_TRY(cudaMemcpyAsync(ws.host_words.data() + 2, ws.words.data() + 2,
                                2 * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  note_host_sync();
  ++out.host_syncs;
  out.affected = ws.host_words.data()[3];
  out.sparsified = dev.num_inserted > 0 ? ws.host_words.data()[2] : 0;
  settle(g, out);  // a no-op: the propagation's first read settled it
}

#define DYNG_INSTANTIATE_LP_OPERATORS(E)                                                           \
  template void ops_identify_affected<E>(const resources&, const graph<std::int32_t, E, float>&,   \
                                         state&, workspace&, const batch_args&, operators_batch&); \
  template void ops_seed<E>(const resources&, const graph<std::int32_t, E, float>&, state&,        \
                            workspace&, const operators_batch&);                                   \
  template void ops_propagate<E>(const resources&, const graph<std::int32_t, E, float>&, state&,   \
                                 workspace&, operators_workspace&, const batch_args&,              \
                                 operators_batch&);                                                \
  template propagation_report ops_iterate<E>(                                                      \
      const resources&, const rows<E>&, float*, const std::int32_t*, const std::int32_t*, float,   \
      int, state&, workspace&, operators_workspace&, void (*)(void*), void*);                      \
  template void ops_finalize<E>(const resources&, const graph<std::int32_t, E, float>&, state&,    \
                                workspace&, operators_batch&);
DYNG_INSTANTIATE_LP_OPERATORS(std::int32_t)
DYNG_INSTANTIATE_LP_OPERATORS(std::int64_t)
#undef DYNG_INSTANTIATE_LP_OPERATORS

}  // namespace dyng::detail::lp_cuda
