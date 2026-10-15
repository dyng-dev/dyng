// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:src/type1.cu
// Derived from ESCHER-GPU@abcf9b2:src/type2.cu
// Derived from ESCHER-GPU@abcf9b2:include/motif.hpp
// Derived from ESCHER-GPU@abcf9b2:include/motif_update.hpp
// Derived from ESCHER-GPU@abcf9b2:src/main.cu
// Derived from ESCHER-GPU@abcf9b2:src/type3.cu
/**
 * @file counters.cuh
 * @brief The four dynamic counters: the full count and the subtract / add phases of the h-motif
 *        counts (MotifCounter, computeMotifCounts, the delta accumulator) and of StatHyper's types
 *        1-3 (Type1Counter .. Type3Counter), as ESCHER-GPU's executables run them.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 */
#pragma once

#include <cuda_runtime.h>

#include <vector>
// Include our utility functions
#include "algorithms/triad_count/engine/counter_shims.hpp"
#include "algorithms/triad_count/engine/device_bins.hpp"
#include "algorithms/triad_count/engine/device_frontier.cuh"

// Device helpers and kernels
#include "algorithms/triad_count/engine/count_ops.cuh"
// dynG (transplant hunk): Type2Op and Type3Op are in stathyper_ops.cuh.
#include "algorithms/triad_count/engine/stathyper_ops.cuh"
#include "algorithms/triad_count/engine/wedge.cuh"
#include "hypergraph/row_views.hpp"

#include <vector>

namespace dyng::detail::triad {

// 30-bin h-motif counts of the triangles of the line graph (H2H rows),
// classified with the vertex sets of the hyperedges (H2V rows).
std::vector<long long> compute_motif_counts(const escher::row_view& h2v,
                                            const escher::row_view& h2h);
// Device-side accumulator for motif delta counts (30 bins, 64-bit).
// Persists across subtract and add phases so both accumulate into the same buffer.
struct motif_delta_accumulator {
  unsigned long long* d_counts = nullptr;
  void init();                                // allocates and zeros device buffer
  void read_to(std::vector<long long>& out);  // copies result to host and frees
};

// Subtract motif counts for triangles touching the deleted frontier.
// Must be called BEFORE modifying data structures so old H2V/H2H are intact.
void compute_motif_subtract(const escher::row_view& h2v, const escher::row_view& h2h,
                            const std::vector<int>& deleted_ids, motif_delta_accumulator& acc);

// Add motif counts for triangles touching the inserted frontier.
// Must be called AFTER building new H2V/H2H snapshots.
void compute_motif_add(const escher::row_view& h2v, const escher::row_view& h2h,
                       const std::vector<int>& inserted_ids, motif_delta_accumulator& acc);
// 30-bin hyperedge-triad (h-motif) counts over the triangles of the line
// graph H2H, maintained under batches of hyperedge deletions and insertions.
class motif_counter : public dynamic_counter {
 public:
  int num_values() const override {
    return 30;
  }

  std::vector<long long> count(const counter_snapshot& snapshot) override {
    return compute_motif_counts(snapshot.h2v_rows, snapshot.h2h_rows);
  }

  // CountUpdate() phase A: subtract BEFORE modifying the data structures.
  void subtract(const counter_snapshot& before, const counter_batch& batch,
                std::vector<long long>& delta) override {
    (void)delta;
    acc_.init();
    compute_motif_subtract(before.h2v_rows, before.h2h_rows, batch.deleted_keys, acc_);
  }

  // CountUpdate() phase B: add on the updated snapshot, using the keys the
  // CBST insert actually assigned.
  void add(const counter_snapshot& after, const counter_batch& batch,
           std::vector<long long>& delta) override {
    compute_motif_add(after.h2v_rows, after.h2h_rows, batch.inserted_keys, acc_);
    std::vector<long long> d;
    acc_.read_to(d);
    for (int i = 0; i < 30; ++i) delta[i] += d[i];
  }

 private:
  motif_delta_accumulator acc_;
};
// ---------------------------------------------------------------------------
// Type-1 Motif:  For each hyperedge e, count C(|e|, 3) = |e|*(|e|-1)*(|e|-2)/6
// Uses H2V only (row lengths from the H2V row view).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Type-1 counter: sum_e C(|e|, 3). The delta only involves the changed
// hyperedges: -C(|d|,3) for each deleted d (old snapshot) and +C(|e|,3) for
// each inserted e (new snapshot).
// ---------------------------------------------------------------------------
class type1_counter : public dynamic_counter {
 public:
  int num_values() const override {
    return 1;
  }

  std::vector<long long> count(const counter_snapshot& snapshot) override {
    device_bins bins(1);
    wedge::count_keys(nullptr, snapshot.num_hyperedge_keys, make_choose3_op(snapshot.h2v_rows), 1,
                      bins.data());
    return bins.read();
  }

  void subtract(const counter_snapshot& before, const counter_batch& batch,
                std::vector<long long>& delta) override {
    (void)delta;
    delta_.zero();
    phase(before, batch.deleted_keys, -1);
  }

  void add(const counter_snapshot& after, const counter_batch& batch,
           std::vector<long long>& delta) override {
    phase(after, batch.inserted_keys, 1);
    delta[0] += delta_.read()[0];
  }

 private:
  void phase(const counter_snapshot& snapshot, const std::vector<int>& ids, long long sign) {
    if (ids.empty()) return;
    device_frontier f(ids, snapshot.num_hyperedge_keys);
    wedge::count_keys(f.ids(), f.size(), make_choose3_op(snapshot.h2v_rows), sign, delta_.data());
  }

  device_bins delta_{1};
};
// ---------------------------------------------------------------------------
// Type-2 counter: pairs (i, j) sharing >= 2 vertices, C(a,2)*b*c. The delta
// counts every pair containing a changed hyperedge once, at its smaller
// frontier member (the original frontier kernel only counted j > f).
// ---------------------------------------------------------------------------
class type2_counter : public dynamic_counter {
 public:
  int num_values() const override {
    return 1;
  }

  std::vector<long long> count(const counter_snapshot& snapshot) override {
    device_bins bins(1);
    wedge::count_pairs(snapshot.h2h_rows, nullptr, snapshot.num_hyperedge_keys, nullptr,
                       make_type2_op(snapshot.h2v_rows), 1, bins.data());
    return bins.read();
  }

  void subtract(const counter_snapshot& before, const counter_batch& batch,
                std::vector<long long>& delta) override {
    (void)delta;
    delta_.zero();
    phase(before, batch.deleted_keys, -1);
  }

  void add(const counter_snapshot& after, const counter_batch& batch,
           std::vector<long long>& delta) override {
    phase(after, batch.inserted_keys, 1);
    delta[0] += delta_.read()[0];
  }

 private:
  void phase(const counter_snapshot& snapshot, const std::vector<int>& ids, long long sign) {
    if (ids.empty()) return;
    device_frontier f(ids, snapshot.num_hyperedge_keys);
    wedge::count_pairs(snapshot.h2h_rows, f.ids(), f.size(), f.flags(),
                       make_type2_op(snapshot.h2v_rows), sign, delta_.data());
  }

  device_bins delta_{1};
};
// ---------------------------------------------------------------------------
// Type-3 counter: H2H triangles (i, j, k), x*y*z of the pairwise-only
// overlaps. The delta counts every triangle containing a changed hyperedge
// once, at its smallest frontier member.
// ---------------------------------------------------------------------------
class type3_counter : public dynamic_counter {
 public:
  int num_values() const override {
    return 1;
  }

  std::vector<long long> count(const counter_snapshot& snapshot) override {
    device_bins bins(1);
    wedge::count_triangles(snapshot.h2h_rows, nullptr, snapshot.num_hyperedge_keys, nullptr,
                           make_type3_op(snapshot.h2v_rows), 1, bins.data());
    return bins.read();
  }

  void subtract(const counter_snapshot& before, const counter_batch& batch,
                std::vector<long long>& delta) override {
    (void)delta;
    delta_.zero();
    phase(before, batch.deleted_keys, -1);
  }

  void add(const counter_snapshot& after, const counter_batch& batch,
           std::vector<long long>& delta) override {
    phase(after, batch.inserted_keys, 1);
    delta[0] += delta_.read()[0];
  }

 private:
  void phase(const counter_snapshot& snapshot, const std::vector<int>& ids, long long sign) {
    if (ids.empty()) return;
    device_frontier f(ids, snapshot.num_hyperedge_keys);
    wedge::count_triangles(snapshot.h2h_rows, f.ids(), f.size(), f.flags(),
                           make_type3_op(snapshot.h2v_rows), sign, delta_.data());
  }

  device_bins delta_{1};
};

}  // namespace dyng::detail::triad
