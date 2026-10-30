// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/dynLP.cu (runDynLP: the device arrays, the order
// of one batch's steps, the transfers and their events, the per-batch evaluation's reads),
// src/iterativePropagation.cu (createPropagationWorkspace's queue capacities),
// src/connectedComponents.cu (createCCWorkspace) and src/labelInitialization.cu
// (createLabelInitWorkspace)
/**
 * @file cuda.cu
 * @brief The fused CUDA engine's batch on the framework's hooks (fused.hpp; ADR 0075): the
 *        run-wide state, the pooled workspace, before_apply, after_commit (the rest of the batch)
 *        and compute() (IrLP), with the library's stage trace (ADR 0073) read from the device.
 *
 * runDynLP keeps, for the whole run, the device graph, the labels, statuses, classes, the
 * affected flags and the step workspaces, and per batch: copies the ids then the edges on a copy
 * stream (events idsReady and edgesReady), clears the affected flags, marks the deletion
 * neighbours, applies the vertex changes, marks the inserted vertices, appends the edges, builds
 * G', its components and the initial labels when vertices were inserted, and propagates. Here
 * (AD21, decided by the author on 2026-10-28: the library never creates a stream) everything runs
 * on the resources' one stream in that order: Step 0 copies the ids, then the edges (the
 * graph's upload_vertex_batch), before_apply clears and marks, the commit applies the vertex
 * changes and appends, and after_commit runs the rest. The original's overlap of the edge copy
 * with the deletion marks and the status kernels is gone; its cost is measured by the early gate
 * (parity/results/M10a-early.json).
 *
 * Mechanical changes: the arrays of the result (labels, classes, batch stamps, queue stamps,
 * nextTag) in the result's state, the scratch from the pool (sized by reserve(), as the original
 * sizes it from the whole run's batches), the seeds' classes set by this engine (the graph has no
 * ground-truth state), the prof:: counters returned in the stats and the profiler. Additions: the
 * graph's report settled after the propagation's first synchronization; the labels before the
 * update kept (in the CC scratch, free by then) and the changed ones counted (stats::affected), in
 * one more synchronization at the end.
 */
#include "algorithms/label_propagation/fused.hpp"
#include "algorithms/label_propagation/trace.hpp"
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "graph/graph_impl.hpp"
#include "util/cuda_check.hpp"
#include "util/device_error_flags.hpp"

#include <dyng/core/error.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace dyng::detail::lp_cuda {

namespace {

cudaStream_t native(const resources& res) noexcept {
  return static_cast<cudaStream_t>(res.stream().get());
}

template <typename value_t>
std::vector<value_t> download(const value_t* d, std::size_t count, cudaStream_t stream) {
  std::vector<value_t> out(count);
  if (count > 0) {
    DYNG_CUDA_TRY(
        cudaMemcpyAsync(out.data(), d, count * sizeof(value_t), cudaMemcpyDeviceToHost, stream));
  }
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  return out;
}

/// The rows of a slotted device copy as the kernels read them.
template <typename edge_t>
rows<edge_t> device_rows(const device_slotted<std::int32_t, edge_t, float>& d) {
  rows<edge_t> r;
  r.row_start = d.row_ptr.data();
  r.row_end = d.row_end.data();
  r.col_ind = d.col_ind.data();
  r.values = d.weights.data();
  r.status = d.status.data();
  r.n = static_cast<std::int64_t>(d.num_vertices);
  return r;
}

/// The original's codes of the current states and seeds (by name, ADR 0069; the stage trace).
std::vector<std::int32_t> original_statuses(const std::vector<std::int32_t>& status,
                                            const std::vector<std::int32_t>& gt_class) {
  std::vector<std::int32_t> out(status.size());
  for (std::size_t v = 0; v < status.size(); ++v) {
    switch (status[v]) {
      case 0:
        out[v] = gt_class[v] >= 0 ? lp_trace::original_ground_truth : lp_trace::original_active;
        break;
      case 1:
        out[v] = lp_trace::original_deleted;
        break;
      default:
        out[v] = lp_trace::original_inactive;
        break;
    }
  }
  return out;
}

/// The graph's report, settled once after the first synchronization since the commit.
template <typename edge_t>
struct settle_point {
  const graph<std::int32_t, edge_t, float>* g = nullptr;
  batch_result* out = nullptr;
  bool done = false;
  std::string detail;

  void settle() {
    if (done) {
      return;
    }
    const device_slotted_settled s = graph_access::settle_device_apply(*g, detail);
    out->graph_errors = s.error_bits;
    out->appended = s.appended;
    out->dropped = s.dropped;
    out->error_detail = detail.c_str();
    done = true;
  }

  static void callback(void* p) {
    static_cast<settle_point*>(p)->settle();
  }
};

}  // namespace

// ------------------------------------------------------------------------------------------------
// The run-wide state and the workspace
// ------------------------------------------------------------------------------------------------

template <typename edge_t>
void create_state(const resources& res, std::int64_t n, const edge_t* row_ptr,
                  const std::int8_t* seeds, const float* labels, state& out) {
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  const auto nn = static_cast<std::size_t>(std::max<std::int64_t>(n, 1));
  out.device = res.device();
  out.n = n;
  out.labels = buffer<float>(res, nn);
  out.gt_class = buffer<std::int32_t>(res, nn);
  out.new_stamp = buffer<std::int32_t>(res, nn);
  out.stamp = buffer<std::int32_t>(res, nn);
  out.next_tag = 1;
  // createPropagationWorkspace: a vertex's degree grows up to its row capacity, so it can reach
  // every bin up to degreeBin(capacity).
  out.capacity[bin_small] = static_cast<int>(n);
  out.capacity[bin_medium] = 0;
  out.capacity[bin_large] = 0;
  for (std::int64_t v = 0; v < n; ++v) {
    const int top = degree_bin_of(static_cast<std::int64_t>(row_ptr[v + 1] - row_ptr[v]));
    if (top >= bin_medium) {
      out.capacity[bin_medium]++;
    }
    if (top >= bin_large) {
      out.capacity[bin_large]++;
    }
  }
  DYNG_CUDA_TRY(cudaMemsetAsync(out.new_stamp.data(), 0, nn * sizeof(std::int32_t), stream));
  DYNG_CUDA_TRY(cudaMemsetAsync(out.stamp.data(), 0, nn * sizeof(std::int32_t), stream));
  if (n > 0) {
    std::vector<std::int32_t> classes(static_cast<std::size_t>(n));
    for (std::size_t v = 0; v < classes.size(); ++v) {
      classes[v] = seeds[v];
    }
    DYNG_CUDA_TRY(cudaMemcpyAsync(out.gt_class.data(), classes.data(),
                                  classes.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                  stream));
    DYNG_CUDA_TRY(cudaMemcpyAsync(out.labels.data(), labels, classes.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream));  // the pageable sources
  }
}

void upload_state(const resources& res, std::int64_t n, const int (&capacity)[num_degree_bins],
                  const std::int8_t* seeds, const float* labels, const std::int64_t* stamp,
                  std::int64_t next_tag, state& out) {
  if (next_tag > std::numeric_limits<int>::max()) {
    throw capacity_error(
        concat_message("dyng: label_propagation::result::clone: the result's propagation tag ",
                       next_tag, " does not fit the 32-bit queue stamps of the CUDA engine"));
  }
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  const auto nn = static_cast<std::size_t>(std::max<std::int64_t>(n, 1));
  out.device = res.device();
  out.n = n;
  out.labels = buffer<float>(res, nn);
  out.gt_class = buffer<std::int32_t>(res, nn);
  out.new_stamp = buffer<std::int32_t>(res, nn);
  out.stamp = buffer<std::int32_t>(res, nn);
  out.next_tag = static_cast<int>(next_tag);
  std::copy(std::begin(capacity), std::end(capacity), std::begin(out.capacity));
  // The batch stamps only identify the vertices of the current batch: every later batch stamp is
  // larger than any earlier one, so 0 is safe for a copy.
  DYNG_CUDA_TRY(cudaMemsetAsync(out.new_stamp.data(), 0, nn * sizeof(std::int32_t), stream));
  if (n > 0) {
    const auto count = static_cast<std::size_t>(n);
    std::vector<std::int32_t> classes(count);
    std::vector<std::int32_t> stamps(count);
    for (std::size_t v = 0; v < count; ++v) {
      classes[v] = seeds[v];
      stamps[v] = static_cast<std::int32_t>(stamp[v]);
    }
    DYNG_CUDA_TRY(cudaMemcpyAsync(out.gt_class.data(), classes.data(), count * sizeof(std::int32_t),
                                  cudaMemcpyHostToDevice, stream));
    DYNG_CUDA_TRY(cudaMemcpyAsync(out.stamp.data(), stamps.data(), count * sizeof(std::int32_t),
                                  cudaMemcpyHostToDevice, stream));
    DYNG_CUDA_TRY(cudaMemcpyAsync(out.labels.data(), labels, count * sizeof(float),
                                  cudaMemcpyHostToDevice, stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  }
}

void download_state(const state& in, std::vector<float>& labels, std::vector<std::int64_t>& stamp) {
  const scoped_device guard(in.device);
  const auto stream = static_cast<cudaStream_t>(in.labels.stream().get());
  const auto count = static_cast<std::size_t>(in.n);
  labels.resize(count);
  stamp.resize(count);
  std::vector<std::int32_t> stamps(count);
  if (count > 0) {
    DYNG_CUDA_TRY(cudaMemcpyAsync(labels.data(), in.labels.data(), count * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
    DYNG_CUDA_TRY(cudaMemcpyAsync(stamps.data(), in.stamp.data(), count * sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost, stream));
  }
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  note_host_sync();
  for (std::size_t v = 0; v < count; ++v) {
    stamp[v] = stamps[v];
  }
}

void copy_state(const resources& res, const state& in, state& out) {
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  const auto nn = static_cast<std::size_t>(std::max<std::int64_t>(in.n, 1));
  out.device = res.device();
  out.n = in.n;
  out.labels = buffer<float>(res, nn);
  out.gt_class = buffer<std::int32_t>(res, nn);
  out.new_stamp = buffer<std::int32_t>(res, nn);
  out.stamp = buffer<std::int32_t>(res, nn);
  out.next_tag = in.next_tag;
  std::copy(std::begin(in.capacity), std::end(in.capacity), std::begin(out.capacity));
  DYNG_CUDA_TRY(cudaMemcpyAsync(out.labels.data(), in.labels.data(), nn * sizeof(float),
                                cudaMemcpyDefault, stream));
  DYNG_CUDA_TRY(cudaMemcpyAsync(out.gt_class.data(), in.gt_class.data(), nn * sizeof(std::int32_t),
                                cudaMemcpyDefault, stream));
  DYNG_CUDA_TRY(cudaMemcpyAsync(out.new_stamp.data(), in.new_stamp.data(),
                                nn * sizeof(std::int32_t), cudaMemcpyDefault, stream));
  DYNG_CUDA_TRY(cudaMemcpyAsync(out.stamp.data(), in.stamp.data(), nn * sizeof(std::int32_t),
                                cudaMemcpyDefault, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  note_host_sync();
}

void workspace::reserve(const resources& res, std::int64_t n,
                        const int (&capacity)[num_degree_bins], std::int64_t inserted,
                        std::int64_t edges, bool damped, bool exact) {
  const scoped_device guard(res.device());
  const auto nn = static_cast<std::size_t>(std::max<std::int64_t>(n, 1));
  if (n > sized_vertices) {
    affected.reserve(res, nn);
    component_id.reserve(res, nn);
    par.reserve(res, nn);
    root_id.reserve(res, nn);
    sized_vertices = n;
  }
  if (inserted > sized_inserted) {
    // createCCWorkspace / createLabelInitWorkspace (maxComponents = maxInserted), with headroom.
    // Exactly the reservation (result::reserve: the original's sizes), with headroom when an
    // update finds a larger batch.
    const auto ins = static_cast<std::size_t>(std::max<std::int64_t>(inserted, 1));
    const auto grow = [&](auto& b) {
      if (exact) {
        (void)b.reserve(res, ins);
      } else {
        (void)b.reserve_with_headroom(res, ins);
      }
    };
    grow(root_flag);
    grow(root_scan);
    grow(weight_l0);
    grow(weight_l1);
    scan_temp_bytes = components_scan_bytes(static_cast<int>(root_flag.capacity()));
    scan_temp.reserve(res, std::max<std::size_t>(scan_temp_bytes, 1));
    sized_inserted = static_cast<std::int64_t>(root_flag.capacity());
  }
  if (edges > sized_edges) {
    const auto e = static_cast<std::size_t>(std::max<std::int64_t>(edges, 1));
    if (exact) {
      (void)sub_src.reserve(res, e);
      (void)sub_dst.reserve(res, e);
    } else {
      (void)sub_src.reserve_with_headroom(res, e);
      (void)sub_dst.reserve_with_headroom(res, e);
    }
    sized_edges = static_cast<std::int64_t>(sub_src.capacity());
  }
  for (int b = 0; b < num_degree_bins; ++b) {
    if (capacity[b] > sized_capacity[b] || (damped && !sized_damped)) {
      const auto c = static_cast<std::size_t>(std::max(capacity[b], 1));
      queue[0][b].reserve(res, c);
      queue[1][b].reserve(res, c);
      if (damped) {
        new_labels[b].reserve(res, c);
      }
      sized_capacity[b] = std::max(capacity[b], sized_capacity[b]);
    }
  }
  sized_damped = sized_damped || damped;
  counts.reserve(res, 2 * counters_per_set);
  control.reserve(res, 4);
  words.reserve(res, 4);
  if (host_counts.size() < 2 * counters_per_set) {
    note_reservation();
    host_counts = buffer<std::int32_t>(2 * counters_per_set, res.stream(),
                                       resources_access::staging_memory(res), res.device());
    host_control =
        buffer<long long>(4, res.stream(), resources_access::staging_memory(res), res.device());
    host_words =
        buffer<std::int32_t>(4, res.stream(), resources_access::staging_memory(res), res.device());
    DYNG_CUDA_TRY(cudaMemsetAsync(counts.data(), 0, 2 * counters_per_set * sizeof(std::int32_t),
                                  native(res)));
  }
}

std::size_t workspace::bytes() const noexcept {
  std::size_t b = affected.bytes() + component_id.bytes() + par.bytes() + root_id.bytes() +
                  root_flag.bytes() + root_scan.bytes() + scan_temp.bytes() + weight_l0.bytes() +
                  weight_l1.bytes() + sub_src.bytes() + sub_dst.bytes() + counts.bytes() +
                  control.bytes() + words.bytes() + host_counts.size() * sizeof(std::int32_t) +
                  host_control.size() * sizeof(long long) +
                  host_words.size() * sizeof(std::int32_t);
  for (int k = 0; k < num_degree_bins; ++k) {
    b += queue[0][k].bytes() + queue[1][k].bytes() + new_labels[k].bytes();
  }
  return b;
}

// ------------------------------------------------------------------------------------------------
// One batch
// ------------------------------------------------------------------------------------------------

template <typename edge_t>
void before_apply(const resources& res, const graph<std::int32_t, edge_t, float>& g,
                  const state& st, workspace& ws, std::int64_t inserted, std::int64_t edges,
                  bool damped) {
  const scoped_device guard(res.device());
  const auto& dev = graph_access::slotted_device(res, g);
#if defined(DYNG_MUTATION_LP_INPLACE_PLAIN_JACOBI)
  damped = true;  // MUTATION (AD9): in_place uses the damped side arrays (plain Jacobi)
#endif
  ws.reserve(res, st.n, st.capacity, std::max(inserted, dev.num_inserted),
             std::max(edges, dev.num_edges), damped);
  const cudaStream_t stream = native(res);
  // Step 1a: neighbors of deleted vertices are affected (Algorithm 2, lines 3-6), read from the
  // pre-deletion adjacency.
  DYNG_CUDA_TRY(cudaMemsetAsync(ws.affected.data(), 0,
                                static_cast<std::size_t>(st.n) * sizeof(std::int32_t), stream));
  launch_mark_deletions(dev.row_ptr.data(), dev.row_end.data(), dev.col_ind.data(),
                        ws.affected.data(), dev.deleted_effective_ptr,
                        static_cast<int>(dev.num_deleted_effective),
                        static_cast<int>(dev.num_vertices), stream);
}

template <typename edge_t>
batch_result after_commit(const resources& res, const graph<std::int32_t, edge_t, float>& g,
                          state& st, workspace& ws, const batch_args& a) {
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  const auto& dev = graph_access::slotted_device(res, g);
  const rows<edge_t> r = device_rows(dev);
  const int totalNodes = static_cast<int>(st.n);
  const int numInserted = static_cast<int>(dev.num_inserted);
  const int numNewEdges = static_cast<int>(dev.num_edges);
  batch_result out;
  settle_point<edge_t> point;
  point.g = &g;
  point.out = &out;
  lp_trace::writer* trace = lp_trace::writer::get();

  // The seeds' classes (launchApplyVertexChanges' setGroundTruthKernel: d_gtClass), then the
  // inserted vertices are affected and stamped as new in this batch (launchMarkInserted).
  launch_set_seed_classes(dev.inserted.data(),
                          dev.has_labels ? dev.inserted_labels.data() : nullptr, numInserted,
                          st.gt_class.data(), stream);
  launch_mark_inserted(dev.inserted.data(), numInserted, ws.affected.data(), st.new_stamp.data(),
                       a.batch_stamp, stream);
  if (trace != nullptr) {
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
    point.settle();
    const auto status = download(dev.status.data(), static_cast<std::size_t>(totalNodes), stream);
    const auto gt = download(st.gt_class.data(), static_cast<std::size_t>(totalNodes), stream);
    const std::vector<std::int32_t> codes = original_statuses(status, gt);
    lp_trace::record rv = trace->begin("apply_vertex_changes", a.batch_no);
    trace->add_i32(rv, "num_deleted", a.num_deleted);
    trace->add_i32(rv, "num_inserted", numInserted);
    trace->add_array(rv, "inserted", lp_trace::i32, a.inserted, a.num_inserted);
    trace->add_array(rv, "status", lp_trace::i32, codes.data(), codes.size());
    trace->add_array(rv, "gt_class", lp_trace::i32, gt.data(), gt.size());
    trace->end_record(rv);
    const auto slots = static_cast<std::size_t>(dev.num_slots);
    const auto row_end = download(dev.row_end.data(), static_cast<std::size_t>(totalNodes), stream);
    const auto cols = download(dev.col_ind.data(), slots, stream);
    const auto vals = download(dev.weights.data(), slots, stream);
    std::vector<std::int32_t> ends(row_end.begin(), row_end.end());
    lp_trace::record ra = trace->begin("append_edges", a.batch_no);
    trace->add_i32(ra, "num_new_edges", numNewEdges);
    trace->add_i64(ra, "capacity", static_cast<std::int64_t>(slots));
    if (a.batch_no == 0) {
      const auto row_ptr =
          download(dev.row_ptr.data(), static_cast<std::size_t>(totalNodes) + 1, stream);
      std::vector<std::int32_t> ptr(row_ptr.begin(), row_ptr.end());
      trace->add_array(ra, "row_ptr", lp_trace::i32, ptr.data(), ptr.size());
    }
    trace->add_array(ra, "row_end", lp_trace::i32, ends.data(), ends.size());
    trace->add_array(ra, "slot_col_ind", lp_trace::i32, cols.data(), cols.size());
    trace->add_array(ra, "slot_values", lp_trace::f32, vals.data(), vals.size());
    // The original's flags: [0] a row overflowed, [1] 1 + a re-inserted id.
    const std::int32_t flags[2] = {
        (out.graph_errors & bits_of(device_error::capacity)) != 0 ? 1 : 0, 0};
    trace->add(ra, "error_flags", lp_trace::i32, flags, 2);
    trace->end_record(ra);
  }

  std::int32_t* before_bits = ws.root_id.data();  // free once the components are numbered
  if (numInserted > 0) {
    // Step 1b: G' and its components (only the strong new-to-new edges; new vertices without G'
    // edges are singleton components).
    launch_build_sparsified_subgraph(
        ws.affected.data(), dev.edge_src.data(), dev.edge_dst.data(), dev.edge_weight.data(),
        numNewEdges, totalNodes, dev.status.data(), st.new_stamp.data(), a.batch_stamp, a.tau,
        ws.sub_src.data(), ws.sub_dst.data(), ws.words.data() + 2, stream);
    if (trace != nullptr) {
      const int count = download(ws.words.data() + 2, 1, stream)[0];
      const auto src = download(ws.sub_src.data(), static_cast<std::size_t>(count), stream);
      const auto dst = download(ws.sub_dst.data(), static_cast<std::size_t>(count), stream);
      std::vector<std::pair<std::int32_t, std::int32_t>> pairs(static_cast<std::size_t>(count));
      for (std::size_t i = 0; i < pairs.size(); ++i) {
        pairs[i] = {std::min(src[i], dst[i]), std::max(src[i], dst[i])};
      }
      std::sort(pairs.begin(), pairs.end());
      std::vector<std::int32_t> flat;
      for (const auto& [x, y] : pairs) {
        flat.push_back(x);
        flat.push_back(y);
      }
      lp_trace::record rs = trace->begin("sparsify", a.batch_no);
      trace->add_f32(rs, "tau", a.tau);
      trace->add_i32(rs, "batch_stamp", a.batch_stamp);
      trace->add_i32(rs, "num_new_edges", numNewEdges);
      trace->add_i32(rs, "num_sub_edges", count);
      trace->add_array(rs, "sub_edges_sorted", lp_trace::i32, flat.data(), flat.size());
      trace->end_record(rs);
    }
    int numComponents = 0;
    out.hook_rounds = launch_connected_components(
        ws.sub_src.data(), ws.sub_dst.data(), ws.words.data() + 2, numNewEdges, dev.inserted.data(),
        numInserted, ws.component_id.data(), numComponents, ws, stream);
    out.host_syncs += out.hook_rounds + 1;
    out.components = numComponents;
    out.sparsified = ws.host_words.data()[2];
    if (trace != nullptr) {
      const auto all =
          download(ws.component_id.data(), static_cast<std::size_t>(totalNodes), stream);
      std::vector<std::int32_t> ids(a.num_inserted);
      for (std::size_t i = 0; i < a.num_inserted; ++i) {
        ids[i] = all[static_cast<std::size_t>(a.inserted[i])];
      }
      lp_trace::record rc = trace->begin("connected_components", a.batch_no);
      trace->add_i32(rc, "num_components", numComponents);
      trace->add_i32(rc, "num_inserted", numInserted);
      trace->add_i64(rc, "hook_rounds", out.hook_rounds);
      trace->add_array(rc, "component_of_inserted", lp_trace::i32, ids.data(), ids.size());
      trace->end_record(rc);
    }
    launch_keep_labels(st.labels.data(), before_bits, totalNodes, stream);
    // Step 2: label initialization.
    launch_label_initialization(r, st.labels.data(), ws.component_id.data(), numComponents,
                                dev.inserted.data(), numInserted, st.gt_class.data(), ws, stream);
    if (trace != nullptr) {
      const auto w0 =
          download(ws.weight_l0.data(), static_cast<std::size_t>(numComponents), stream);
      const auto w1 =
          download(ws.weight_l1.data(), static_cast<std::size_t>(numComponents), stream);
      const auto labels = download(st.labels.data(), static_cast<std::size_t>(totalNodes), stream);
      std::vector<float> init(a.num_inserted);
      for (std::size_t i = 0; i < a.num_inserted; ++i) {
        init[i] = labels[static_cast<std::size_t>(a.inserted[i])];
      }
      lp_trace::record rl = trace->begin("label_init", a.batch_no);
      trace->add_i32(rl, "num_components", numComponents);
      trace->add_array(rl, "w0", lp_trace::u64, w0.data(), w0.size());
      trace->add_array(rl, "w1", lp_trace::u64, w1.data(), w1.size());
      trace->add_array(rl, "initial_labels_of_inserted", lp_trace::f32, init.data(), init.size());
      trace->end_record(rl);
    }
  } else {
    // A batch without insertions: the endpoints of its edges are affected (ADR 0072's fix).
    launch_mark_edge_endpoints(ws.affected.data(), dev.edge_src.data(), dev.edge_dst.data(),
                               numNewEdges, totalNodes, dev.status.data(), stream);
    launch_keep_labels(st.labels.data(), before_bits, totalNodes, stream);
  }

  // Step 3: iterative propagation.
  if (trace != nullptr) {
    const auto aff = download(ws.affected.data(), static_cast<std::size_t>(totalNodes), stream);
    const auto status = download(dev.status.data(), static_cast<std::size_t>(totalNodes), stream);
    const auto gt = download(st.gt_class.data(), static_cast<std::size_t>(totalNodes), stream);
    const auto labels = download(st.labels.data(), static_cast<std::size_t>(totalNodes), stream);
    std::vector<std::int32_t> frontier;
    for (int v = 0; v < totalNodes; ++v) {
      const auto vv = static_cast<std::size_t>(v);
      if (aff[vv] == 1 && status[vv] == 0 && gt[vv] < 0) {
        frontier.push_back(v);
      }
    }
    lp_trace::record rb = trace->begin("propagation_begin", a.batch_no);
    trace->add_i32(rb, "num_nodes", totalNodes);
    trace->add_f32(rb, "delta", a.delta);
    trace->add_i32(rb, "max_iterations", a.max_iterations);
    trace->add_i32(rb, "update_rule", a.damped ? 1 : 0);
    trace->add_i32(rb, "next_tag", st.next_tag);
    trace->add_i32(rb, "frontier_size", static_cast<std::int32_t>(frontier.size()));
    trace->add_array(rb, "initial_frontier", lp_trace::i32, frontier.data(), frontier.size());
    trace->add_array(rb, "labels", lp_trace::f32, labels.data(), labels.size());
    trace->end_record(rb);
  }
  out.propagation = launch_iterative_propagation(
      r, st.labels.data(), st.gt_class.data(), ws.affected.data(), a.delta, a.max_iterations,
      a.damped, st, ws, stream, &settle_point<edge_t>::callback, &point);
  out.host_syncs += out.propagation.host_syncs;
  out.capped = out.propagation.capped;
  if (out.capped && a.fallback) {
    // on_limit::fallback_recompute: IrLP of the current graph replaces the labels.
    auto lease = resources_access::workspaces(res).template acquire<irlp_workspace>(res);
    bool irlp_capped = false;
    out.irlp_iterations = run_irlp(res, r, st.gt_class.data(), st.labels.data(), a.delta,
                                   a.max_iterations, lease.get(), irlp_capped);
    out.host_syncs += out.irlp_iterations + 1;
    out.recomputed = true;
  }
  // The stats' affected: the vertices whose label bits changed in this update.
  launch_count_changed(st.labels.data(), before_bits, totalNodes, ws.words.data() + 3, stream);
  DYNG_CUDA_TRY(cudaMemcpyAsync(ws.host_words.data() + 3, ws.words.data() + 3, sizeof(std::int32_t),
                                cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  note_host_sync();
  ++out.host_syncs;
  out.affected = ws.host_words.data()[3];
  point.settle();  // nothing pending: a no-op unless the trace settled it
  if (trace != nullptr) {
    const auto labels = download(st.labels.data(), static_cast<std::size_t>(totalNodes), stream);
    lp_trace::record re = trace->begin("propagation_end", a.batch_no);
    trace->add_i32(re, "iterations", out.propagation.iterations);
    trace->add_i64(re, "frontier_visits", out.propagation.frontier_visits);
    trace->add_i64(re, "persistent_iterations", out.propagation.persistent_iterations);
    trace->add_i32(re, "next_tag", st.next_tag);
    trace->add_array(re, "labels", lp_trace::f32, labels.data(), labels.size());
    trace->end_record(re);
  }
  // The detail outlives `point`: copy it into storage that lives as long as the thread's next
  // update (a static per thread; empty unless the graph recorded an error).
  static thread_local std::string detail;
  detail = point.detail;
  out.error_detail = detail.c_str();
  return out;
}

// ------------------------------------------------------------------------------------------------
// compute() (runIrLPSnapshot)
// ------------------------------------------------------------------------------------------------

template <typename edge_t>
int compute(const resources& res, const graph<std::int32_t, edge_t, float>& g,
            const std::int8_t* seeds, float delta, int max_iterations, state& st, bool& capped) {
  const scoped_device guard(res.device());
  const auto& impl = graph_access::impl(g);
  const std::int64_t n = g.num_vertices();
  // The starting labels (runIrLPSnapshot): 0.5, a seed its class (compute()'s seeds are active).
  std::vector<float> start(static_cast<std::size_t>(n), 0.5F);
  for (std::size_t v = 0; v < start.size(); ++v) {
    if (seeds[v] >= 0) {
      start[v] = static_cast<float>(seeds[v]);
    }
  }
  create_state(res, n, impl.host_storage().row_ptr.data(), seeds, start.data(), st);
  capped = false;
  if (g.num_edges() == 0) {
    // runIrLPValidation skips an empty snapshot (the labels stay as they start); no scratch is
    // leased for it (the published flow's compute() on the empty capacity graph).
    if (lp_trace::writer* trace = lp_trace::writer::get()) {
      lp_trace::record rr = trace->begin("irlp_snapshot", -1);
      trace->add_i32(rr, "returned", 1);
      trace->add_i32(rr, "iterations", 0);
      trace->add_f32(rr, "delta", delta);
      trace->add_array(rr, "labels", lp_trace::f32, start.data(), start.size());
      trace->end_record(rr);
    }
    return 0;
  }
  rows<edge_t> r;
  if (impl.is_slotted()) {
    r = device_rows(graph_access::slotted_device(res, g));
  } else {
    const auto& d = graph_access::device_out(res, g);
    r.row_start = d.out_row_ptr.data();
    r.row_end = d.out_row_ptr.data() + 1;
    r.col_ind = d.out_col_ind.data();
    r.values = d.out_column(0);
    r.status = nullptr;
    r.n = n;
  }
  auto lease = resources_access::workspaces(res).template acquire<irlp_workspace>(res);
  const int iterations = run_irlp(res, r, st.gt_class.data(), st.labels.data(), delta,
                                  max_iterations, lease.get(), capped);
  if (lp_trace::writer* trace = lp_trace::writer::get()) {
    const auto labels = download(st.labels.data(), static_cast<std::size_t>(n), native(res));
    lp_trace::record rr = trace->begin("irlp_snapshot", -1);
    trace->add_i32(rr, "returned", 1);
    trace->add_i32(rr, "iterations", iterations);
    trace->add_f32(rr, "delta", delta);
    trace->add_array(rr, "labels", lp_trace::f32, labels.data(), labels.size());
    trace->end_record(rr);
  }
  return iterations;
}

#define DYNG_INSTANTIATE_LP_CUDA(E)                                                             \
  template void create_state<E>(const resources&, std::int64_t, const E*, const std::int8_t*,   \
                                const float*, state&);                                          \
  template void before_apply<E>(const resources&, const graph<std::int32_t, E, float>&,         \
                                const state&, workspace&, std::int64_t, std::int64_t, bool);    \
  template batch_result after_commit<E>(const resources&, const graph<std::int32_t, E, float>&, \
                                        state&, workspace&, const batch_args&);                 \
  template int compute<E>(const resources&, const graph<std::int32_t, E, float>&,               \
                          const std::int8_t*, float, int, state&, bool&);
DYNG_INSTANTIATE_LP_CUDA(std::int32_t)
DYNG_INSTANTIATE_LP_CUDA(std::int64_t)
#undef DYNG_INSTANTIATE_LP_CUDA

}  // namespace dyng::detail::lp_cuda
