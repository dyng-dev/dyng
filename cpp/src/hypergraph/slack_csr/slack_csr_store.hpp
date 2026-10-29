// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/include/hsosp.cuh
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospInternal.cuh
/**
 * @file slack_csr/slack_csr_store.hpp
 * @brief The device side of slack_csr storage: the incidence mirror (hyperedge and vertex slack
 *        rows), the slack CSR line graph with its node weights, and the last batch's pair delta on
 *        the device (MOSP_ESCHER's DeviceIncidence, DeviceH2H and DeviceDelta).
 *
 * Ported from MOSP_ESCHER@4b86159 with the renames of parity/escher_port/rename_map_hsosp.toml
 * (M11a, ADR 0089; the layout rules are described in
 * cpp/src/hypergraph/slack_csr/slack_layout.hpp).
 */
#pragma once

/*
 * The h2h line graph of the hypergraph has a special property under the
 * meeting's cost model (stepping into hyperedge h_j costs w_j): every
 * in-edge of node j carries the same weight w_j. The device graph therefore
 * stores ONE symmetric adjacency (col_ind) plus a per-node weight array,
 * halving memory versus the in/out CSR pair the MOSP kernels use.
 *
 * The kernels are node-weighted adaptations of the MOSP project's
 * parallelSOSPUpdate kernels (collectCandidates / updateDistances with
 * atomicCAS dedup + atomicAdd worklist compaction); the originals in
 * mosp/src are untouched.
 */

#include "hypergraph/slack_csr/slack_layout.hpp"
#include "util/cuda_check.hpp"
#include "util/device_error_flags.hpp"

#include <dyng/core/error.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dyng::detail::hyper::slack {

// dynG (transplant hunk): the host incidence model, its line graph and the batch's incidence
// changes as the transplanted code reads them (MOSP_ESCHER's HostHypergraph, LineGraphCSR and
// IncidenceBatch). dynG keeps the host model in the slack_csr host store (hypergraph ids are
// 0-based there; vertices 0-based on both sides); the device store builds these views over it.

/// The rows by 0-based id (an id at or above the rows is dead: an empty row).
struct host_rows {
  const std::vector<std::vector<std::int32_t>>* rows = nullptr;  ///< by id
  /// @brief The row of an id. @param[in] i The id. @return Its members (empty if dead).
  const std::vector<std::int32_t>& operator[](std::size_t i) const {
    static const std::vector<std::int32_t> none;
    return i < rows->size() ? (*rows)[i] : none;
  }
};

/// Whether an id is live (its row is not empty).
struct host_alive {
  host_rows rows;  ///< the rows
  /// @brief Liveness. @param[in] i The id. @return Live.
  bool operator[](std::size_t i) const {
    return !rows[i].empty();
  }
};

/// The hyperedges of every vertex (0-based ids, increasing).
struct host_v2h {
  const std::vector<std::vector<std::int64_t>>* rows = nullptr;  ///< by vertex
  /// @brief The hyperedges of a vertex. @param[in] v The vertex. @return Their ids.
  const std::vector<std::int64_t>& operator[](std::size_t v) const {
    return (*rows)[v];
  }
};

/// The weights by 0-based id (an id at or above them weighs 0).
struct host_weights {
  const std::vector<std::int64_t>* w = nullptr;  ///< by id
  /// @brief The weight of an id. @param[in] i The id. @return Its weight.
  long long operator[](std::size_t i) const {
    return i < w->size() ? (*w)[i] : 0;
  }
};

/// MOSP_ESCHER's HostHypergraph as the transplanted code reads it.
struct host_hypergraph {
  int num_vertices = 0;  ///< vertices
  int bound = 0;         ///< ids 1..bound (B's maxId())
  host_rows he_verts;    ///< sorted members by 0-based id
  host_v2h v2h;          ///< 0-based ids by vertex
  host_weights he_w;     ///< weights by 0-based id
  host_alive alive;      ///< liveness by 0-based id
  /// @brief The largest id (1-based). @return bound.
  int max_id() const {
    return bound;
  }
};

/// MOSP_ESCHER's LineGraphCSR as the transplanted code reads it: rows by 1-based id, neighbours
/// 0-based (dynG's host line graph).
struct line_graph_csr {
  const std::int64_t* row_start = nullptr;  ///< by 0-based id
  const std::int32_t* deg = nullptr;        ///< by 0-based id
  const std::int32_t* nbr = nullptr;        ///< the neighbours (0-based ids)
  /// @brief The degree of a 1-based id. @param[in] id The id. @return Its degree.
  long long degree(int id) const {
    return deg[id - 1];
  }
  /// @brief The row of a 1-based id. @param[in] id The id. @return Its neighbours.
  const int* row(int id) const {
    return nbr + row_start[id - 1];
  }
};

/// MOSP_ESCHER's IncidenceBatch: the batch's incidence changes as the device delta reads them
/// (hyperedge ids 1-based, vertices 0-based).
struct incidence_batch {
  std::vector<std::uint64_t> inc_key;    ///< (vertex << 32 | heId) of every changed incidence
  std::vector<int> inc_sign;             ///< +1 added, -1 removed
  std::vector<int> touched;              ///< the hyperedges whose vertex list changed (unique)
  std::vector<int> pre_off, pre_vals;    ///< their vertex lists before the batch (CSR)
  std::vector<int> post_off, post_vals;  ///< and after it (dead: empty)
  std::vector<int> touched_vertices;     ///< the vertices whose list changed (unique)
  std::vector<int> touched_vertex_len;   ///< their list lengths after the batch
  std::vector<int> new_he;               ///< the ids the batch inserted (recycled included)
  std::vector<long long> new_w;          ///< their weights
  std::vector<int> dead_he;              ///< the ids it deleted that stay dead
};

/**
 * The last batch delta in device memory (0-based node indices), derived by
 * applyBatch and read by hsospUpdate: deleted pairs first, then inserted
 * pairs; ids = new (inserted or recreated) nodes, then dead ones.
 */
struct device_delta {
  int2* d_pairs = nullptr;
  int* d_ids = nullptr;
  long long* d_new_w = nullptr;  ///< weights of the new nodes
  long long pair_capacity = 0;
  long long id_capacity = 0;
  int num_del = 0;
  int num_ins = 0;
  int num_new = 0;
  int num_dead = 0;

  // The scratch of the device-side grouping of the CSR apply: directed
  // row keys (double buffered for the radix sort), row runs and offsets,
  // and the CUB temporary storage. Grown on demand, reused across batches.
  unsigned long long* d_keys = nullptr;
  unsigned long long* d_keys_alt = nullptr;
  int* d_rows = nullptr;
  int* d_row_len = nullptr;
  int* d_row_off = nullptr;
  int* d_num_rows = nullptr;
  void* d_temp = nullptr;
  std::size_t temp_bytes = 0;
  long long key_capacity = 0;
  /// After the CSR apply: the sorted directed keys (row << 33 |
  /// isInsert << 32 | col), 2 per pair (one of d_keys / d_keys_alt).
  const unsigned long long* d_sorted_keys = nullptr;
  long long num_sorted_keys = 0;
  /// Pinned host copy of the sorted keys (for the CBST maintenance).
  unsigned long long* h_sorted_keys = nullptr;
  long long host_key_capacity = 0;

  void reserve(long long pairs, long long ids);
  void reserve_keys(long long keys);
  void reserve_temp(std::size_t bytes);
  void free();
};

/**
 * Device mirror of the incidence model, input of the line-graph delta
 * derivation: the vertex list of every hyperedge and the hyperedge list of
 * every vertex, as slack rows. The host decides the layout (it knows every
 * row length) and keeps a copy of the offsets and capacities; rows that
 * outgrow their capacity move to the tail, and a full tail triggers a
 * re-upload with more capacity.
 */
struct device_incidence {
  int max_nodes = 0;
  int num_vertices = 0;
  // Hyperedge -> vertices (node-indexed).
  long long* d_he_off = nullptr;
  int* d_he_len = nullptr;
  int* d_he_val = nullptr;
  long long he_capacity = 0;
  long long he_tail = 0;
  std::vector<long long> he_off;
  std::vector<int> he_cap;
  // Vertex -> hyperedges (0-based node indices).
  long long* d_v_off = nullptr;
  int* d_v_len = nullptr;
  int* d_v_val = nullptr;
  long long v_capacity = 0;
  long long v_tail = 0;
  std::vector<long long> v_off;
  std::vector<int> v_cap;
  // Per-batch scratch, grown on demand.
  int* d_touched_idx = nullptr;  ///< node -> touched slot or -1
  int* d_post_len = nullptr;     ///< vertex -> post-batch length
  struct scratch_arrays;
  scratch_arrays* scratch = nullptr;

  device_incidence() = default;
  ~device_incidence();
  device_incidence(const device_incidence&) = delete;
  device_incidence& operator=(const device_incidence&) = delete;
  device_incidence(device_incidence&&) noexcept;
  device_incidence& operator=(device_incidence&&) noexcept;

  /** Uploads the incidence of @p hg; the value arrays get free tail
     *  space of (headroom - 1) x their initial size (+64 entries). */
  void build(const host_hypergraph& hg, int max_nodes, double headroom);
  long long device_bytes() const;
  void free();
};

// dynG (transplant hunk): the scratch of the delta, defined here (MOSP_ESCHER defined it in
// hsospDelta.cu, the one translation unit of the mirror and the delta).
using u64 = unsigned long long;

struct device_incidence::scratch_arrays {
  u64* inc_keys = nullptr;
  u64* inc_keys_alt = nullptr;
  int* signs = nullptr;
  int* signs_alt = nullptr;
  u64* uniq = nullptr;
  int* net = nullptr;
  long long* cnt0 = nullptr;
  long long* cnt1 = nullptr;
  long long* off0 = nullptr;
  long long* off1 = nullptr;
  int* seg_len = nullptr;
  int* seg_off = nullptr;
  int* seg_vert = nullptr;
  long long inc_cap = 0, inc_cap2 = 0, inc_cap3 = 0, inc_cap4 = 0, inc_cap5 = 0, inc_cap6 = 0,
            inc_cap7 = 0, inc_cap8 = 0, inc_cap9 = 0, inc_cap10 = 0, inc_cap11 = 0, inc_cap12 = 0,
            inc_cap13 = 0;
  u64* cand = nullptr;
  u64* cand_alt = nullptr;
  signed char* cls = nullptr;
  u64* selected = nullptr;
  long long cand_cap = 0, cand_cap2 = 0, cls_cap = 0, sel_cap = 0;
  int* touched = nullptr;
  int* pre_off = nullptr;
  int* post_off = nullptr;
  long long* he_write_off = nullptr;
  long long t_cap = 0, t_cap2 = 0, t_cap3 = 0, t_cap4 = 0;
  int* pre_vals = nullptr;
  int* post_vals = nullptr;
  long long val_cap = 0, val_cap2 = 0;
  int* t_verts = nullptr;
  int* t_vert_len = nullptr;
  int* reloc_verts = nullptr;
  long long* reloc_off = nullptr;
  long long tv_cap = 0, tv_cap2 = 0, r_cap = 0, r_cap2 = 0;
  long long* totals = nullptr;
  int* counts = nullptr;             // [0] uniq, [1] segs, [2] cand, [3] sel
  long long* host_totals = nullptr;  // pinned
  int* host_counts = nullptr;        // pinned
  void* temp = nullptr;
  std::size_t temp_cap = 0;

  scratch_arrays() {
    DYNG_CUDA_TRY(cudaMalloc(&totals, sizeof(long long) * 2));
    DYNG_CUDA_TRY(cudaMalloc(&counts, sizeof(int) * 4));
    DYNG_CUDA_TRY(cudaMallocHost(&host_totals, sizeof(long long) * 2));
    DYNG_CUDA_TRY(cudaMallocHost(&host_counts, sizeof(int) * 4));
  }
  ~scratch_arrays() {
    for (void* p : {static_cast<void*>(inc_keys),     static_cast<void*>(inc_keys_alt),
                    static_cast<void*>(signs),        static_cast<void*>(signs_alt),
                    static_cast<void*>(uniq),         static_cast<void*>(net),
                    static_cast<void*>(cnt0),         static_cast<void*>(cnt1),
                    static_cast<void*>(off0),         static_cast<void*>(off1),
                    static_cast<void*>(seg_len),      static_cast<void*>(seg_off),
                    static_cast<void*>(seg_vert),     static_cast<void*>(cand),
                    static_cast<void*>(cand_alt),     static_cast<void*>(cls),
                    static_cast<void*>(selected),     static_cast<void*>(touched),
                    static_cast<void*>(pre_off),      static_cast<void*>(post_off),
                    static_cast<void*>(he_write_off), static_cast<void*>(pre_vals),
                    static_cast<void*>(post_vals),    static_cast<void*>(t_verts),
                    static_cast<void*>(t_vert_len),   static_cast<void*>(reloc_verts),
                    static_cast<void*>(reloc_off),    static_cast<void*>(totals),
                    static_cast<void*>(counts),       temp})
      if (p) DYNG_CUDA_TRY_NO_THROW(cudaFree(p));
    if (host_totals) DYNG_CUDA_TRY_NO_THROW(cudaFreeHost(host_totals));
    if (host_counts) DYNG_CUDA_TRY_NO_THROW(cudaFreeHost(host_counts));
  }

  void reserve_temp(std::size_t bytes) {
    if (bytes <= temp_cap) return;
    if (temp) DYNG_CUDA_TRY(cudaFree(temp));
    temp_cap = bytes + bytes / 2 + 4096;
    DYNG_CUDA_TRY(cudaMalloc(&temp, temp_cap));
  }
};

/** Resident device h2h graph. Node index = heId - 1. */
struct device_h2h {
  int max_nodes = 0;           ///< capacity of the node-indexed arrays
  int num_nodes = 0;           ///< nodes in use ( == host max_id() )
  long long cap_entries = 0;   ///< capacity of col_ind
  long long used_entries = 0;  ///< host mirror of the tail cursor

  long long* d_row_start = nullptr;
  int* d_deg = nullptr;
  int* d_cap = nullptr;
  int* d_col_ind = nullptr;
  long long* d_node_w = nullptr;
  unsigned long long* d_tail_cursor = nullptr;
  int* d_overflow_flag = nullptr;
  /// dynG: the sticky device error word (PLAN 4.7.3) of the delta and the CSR apply: a deletion
  /// that finds no entry is a broken invariant (device_error::internal), read with the overflow
  /// flag.
  std::uint32_t* d_error = nullptr;
  long long num_entries = 0;  ///< live adjacency entries (2 x pairs)

  device_delta delta;    ///< last applied batch delta
  device_incidence inc;  ///< incidence mirror for the delta

  device_h2h() = default;
  ~device_h2h();
  device_h2h(const device_h2h&) = delete;
  device_h2h& operator=(const device_h2h&) = delete;
  device_h2h(device_h2h&&) noexcept;
  device_h2h& operator=(device_h2h&&) noexcept;

  long long device_bytes() const;
  void free();
};

/**
 * Build (or rebuild) the resident device graph: the CSR from the line
 * graph @p lg of @p hg, and the incidence mirror from @p hg.
 *
 * @param hg            host hypergraph (incidence + weights + liveness).
 * @param lg            its line graph (host_hypergraph::line_graph).
 * @param max_nodes      node capacity; must cover every id the run will see.
 * @param entry_headroom col_ind capacity = initial slack entries *
 *                      entry_headroom (+4096); growth room for relocations.
 *                      Must be >= 1 (invalid_argument_error otherwise).
 */
void build_device_h2h(device_h2h& dev, const host_hypergraph& hg, const line_graph_csr& lg,
                      int max_nodes, double entry_headroom);

/** Number of rows of the incidence mirror (hyperedge -> vertices and
 *  vertex -> hyperedges, compared as sorted lists) that differ from @p hg;
 *  a test check, it copies the whole mirror to the host. */
long long incidence_mirror_mismatches(const device_incidence& inc, const host_hypergraph& hg);

/** Copies the rows of nodes 0..m-1 to the host as sorted 1-based
 *  hyperedge id lists (duplicates kept), for checks against an oracle. */
std::vector<std::vector<int>> download_rows(const device_h2h& dev, int m);

inline int grid_for(long long n, int block) {
  return static_cast<int>((n + block - 1) / block);
}

/** The CSR part of build_device_h2h (the incidence mirror is kept); also
 *  the rebuild after a tail overflow. */
void build_csr(device_h2h& dev, const host_hypergraph& hg, const line_graph_csr& lg, int max_nodes,
               double entry_headroom);

/**
 * Applies the pairs in dev.delta to the CSR (device sort by row + one warp
 * per row) and scatters the weights of the new nodes; afterwards
 * dev.delta.d_sorted_keys holds the sorted directed keys. Returns false if
 * the tail region overflowed.
 */
bool apply_delta_pairs(device_h2h& dev);

// dynG (transplant hunk): shared by the mirror (slack_csr_store.cu) and the delta
// (incidence_delta.cu), which MOSP_ESCHER kept in one translation unit.

/// (Re)uploads the hyperedge rows of the mirror from the host, with free tail space.
void upload_he_rows(device_incidence& di, const host_hypergraph& hg, double growth);
/// (Re)uploads the vertex rows of the mirror from the host, with free tail space.
void upload_vertex_rows(device_incidence& di, const host_hypergraph& hg, double growth);

/** Derives the batch's net line-graph delta into dev.delta; returns the
 *  number of candidate pairs examined and counts mirror re-uploads. More than
 *  @p max_candidates candidates (INT_MAX, B's limit; lower in a test) throw. */
long long derive_delta(device_h2h& dev, const host_hypergraph& hg, const incidence_batch& inc,
                       int& reuploads, long long max_candidates);

}  // namespace dyng::detail::hyper::slack
