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

#include "core/resources_access.hpp"
#include "hypergraph/slack_csr/slack_layout.hpp"
#include "util/cuda_check.hpp"
#include "util/device_error_flags.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
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
  buffer<int2> d_pairs;
  buffer<int> d_ids;
  buffer<long long> d_new_w;  ///< weights of the new nodes
  long long pair_capacity = 0;
  long long id_capacity = 0;
  int num_del = 0;
  int num_ins = 0;
  int num_new = 0;
  int num_dead = 0;

  // The scratch of the device-side grouping of the CSR apply: directed
  // row keys (double buffered for the radix sort), row runs and offsets,
  // and the CUB temporary storage. Grown on demand, reused across batches.
  buffer<unsigned long long> d_keys;
  buffer<unsigned long long> d_keys_alt;
  buffer<int> d_rows;
  buffer<int> d_row_len;
  buffer<int> d_row_off;
  buffer<int> d_num_rows;
  buffer<char> d_temp;
  std::size_t temp_bytes = 0;
  long long key_capacity = 0;
  /// After the CSR apply: the sorted directed keys (row << 33 |
  /// isInsert << 32 | col), 2 per pair (one of d_keys / d_keys_alt).
  const unsigned long long* d_sorted_keys = nullptr;
  long long num_sorted_keys = 0;

  void reserve(const resources& res, long long pairs, long long ids);
  void reserve_keys(const resources& res, long long keys);
  void reserve_temp(const resources& res, std::size_t bytes);
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
  cudaStream_t stream = nullptr;  ///< dynG: the store's stream (every operation on it)
  int max_nodes = 0;
  int num_vertices = 0;
  // Hyperedge -> vertices (node-indexed).
  buffer<long long> d_he_off;
  buffer<int> d_he_len;
  buffer<int> d_he_val;
  long long he_capacity = 0;
  long long he_tail = 0;
  std::vector<long long> he_off;
  std::vector<int> he_cap;
  // Vertex -> hyperedges (0-based node indices).
  buffer<long long> d_v_off;
  buffer<int> d_v_len;
  buffer<int> d_v_val;
  long long v_capacity = 0;
  long long v_tail = 0;
  std::vector<long long> v_off;
  std::vector<int> v_cap;
  // Per-batch scratch, grown on demand.
  buffer<int> d_touched_idx;  ///< node -> touched slot or -1
  buffer<int> d_post_len;     ///< vertex -> post-batch length
  struct scratch_arrays;
  std::unique_ptr<scratch_arrays> scratch;
  const resources* res = nullptr;  ///< dynG: the store's resources (every array from them)

  device_incidence();
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
  buffer<u64> inc_keys;
  buffer<u64> inc_keys_alt;
  buffer<int> signs;
  buffer<int> signs_alt;
  buffer<u64> uniq;
  buffer<int> net;
  buffer<long long> cnt0;
  buffer<long long> cnt1;
  buffer<long long> off0;
  buffer<long long> off1;
  buffer<int> seg_len;
  buffer<int> seg_off;
  buffer<int> seg_vert;
  long long inc_cap = 0, inc_cap2 = 0, inc_cap3 = 0, inc_cap4 = 0, inc_cap5 = 0, inc_cap6 = 0,
            inc_cap7 = 0, inc_cap8 = 0, inc_cap9 = 0, inc_cap10 = 0, inc_cap11 = 0, inc_cap12 = 0,
            inc_cap13 = 0;
  buffer<u64> cand;
  buffer<u64> cand_alt;
  buffer<signed char> cls;
  buffer<u64> selected;
  long long cand_cap = 0, cand_cap2 = 0, cls_cap = 0, sel_cap = 0;
  buffer<int> touched;
  buffer<int> pre_off;
  buffer<int> post_off;
  buffer<long long> he_write_off;
  long long t_cap = 0, t_cap2 = 0, t_cap3 = 0, t_cap4 = 0;
  buffer<int> pre_vals;
  buffer<int> post_vals;
  long long val_cap = 0, val_cap2 = 0;
  buffer<int> t_verts;
  buffer<int> t_vert_len;
  buffer<int> reloc_verts;
  buffer<long long> reloc_off;
  long long tv_cap = 0, tv_cap2 = 0, r_cap = 0, r_cap2 = 0;
  buffer<long long> totals;
  buffer<int> counts;             // [0] uniq, [1] segs, [2] cand, [3] sel
  buffer<long long> host_totals;  // pinned
  buffer<int> host_counts;        // pinned
  buffer<char> temp;
  std::size_t temp_cap = 0;

  const resources* res = nullptr;

  explicit scratch_arrays(const resources& r)
      : totals(r, 2),
        counts(r, 4),
        host_totals(2, r.stream(), resources_access::staging_memory(r), -1),
        host_counts(4, r.stream(), resources_access::staging_memory(r), -1),
        res(&r) {}
  void reserve_temp(std::size_t bytes) {
    if (bytes <= temp_cap) return;
    temp = buffer<char>();
    temp_cap = bytes + bytes / 2 + 4096;
    temp = buffer<char>(*res, temp_cap);
  }
};

/** Resident device h2h graph. Node index = heId - 1. */
struct device_h2h {
  cudaStream_t stream = nullptr;  ///< dynG: the store's stream (every operation on it)
  int max_nodes = 0;              ///< capacity of the node-indexed arrays
  int num_nodes = 0;              ///< nodes in use ( == host max_id() )
  long long cap_entries = 0;      ///< capacity of col_ind
  long long used_entries = 0;     ///< host mirror of the tail cursor

  buffer<long long> d_row_start;
  buffer<int> d_deg;
  buffer<int> d_cap;
  buffer<int> d_col_ind;
  buffer<long long> d_node_w;
  buffer<unsigned long long> d_tail_cursor;
  buffer<int> d_overflow_flag;
  /// dynG: the sticky device error word (PLAN 4.7.3) of the delta and the CSR apply: a deletion
  /// that finds no entry is a broken invariant (device_error::internal), read with the overflow
  /// flag.
  buffer<std::uint32_t> d_error;
  long long num_entries = 0;  ///< live adjacency entries (2 x pairs)

  device_delta delta;              ///< last applied batch delta
  device_incidence inc;            ///< incidence mirror for the delta
  const resources* res = nullptr;  ///< dynG: the store's resources (every array from them)

  device_h2h() = default;
  ~device_h2h() = default;
  device_h2h(const device_h2h&) = delete;
  device_h2h& operator=(const device_h2h&) = delete;
  device_h2h(device_h2h&&) noexcept = default;
  device_h2h& operator=(device_h2h&&) noexcept = default;

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
