// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/delta_merge.cuh
// Derived from ESCHER-GPU@abcf9b2:include/batch.hpp
// Derived from ESCHER-GPU@abcf9b2:include/graph_maintenance.hpp
/**
 * @file sorted_merge.cuh
 * @brief The sorted delta merge of the line graph (H2H): the row delta, the batch it reads, the
 *        kernels and updateLineGraph (ESCHER-GPU's delta_merge.cuh and graph_maintenance.hpp, H2H
 *        part).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_line_graph.toml
 * (M8b, ADR 0042; the store's encoding is described in cpp/src/hypergraph/escher/README.md).
 */
#pragma once

#include "hypergraph/escher/cbst_store.hpp"
#include "hypergraph/escher/row_view.cuh"

#include <cstdint>
#include <vector>

namespace dyng::detail::escher {

// Delta of a set of sorted rows (keys 1..num_keys), as sorted pair arrays
// ((key << 32) | value) with per-key counts and offsets:
//  - replaced rows (replaced[key] != 0) take the replacement row `rep`;
//  - other rows keep their old entries minus `rem` plus `add`.
struct row_delta {
  const unsigned char* replaced;
  const int* rep_count;
  const long long* rep_off;
  const unsigned long long* rep;
  const int* rem_count;
  const long long* rem_off;
  const unsigned long long* rem;
  const int* add_count;
  const long long* add_off;
  const unsigned long long* add;
};
// One batch of hyperedge updates. Deletions and insertions replace whole
// hyperedges; an incident-vertex edit is a deletion plus an insertion.
struct line_graph_batch {
  std::vector<int> deleted_keys;                // sorted keys of deleted hyperedges
  std::vector<std::vector<int>> deleted_rows;   // their vertex sets (same order)
  std::vector<std::vector<int>> inserted_rows;  // vertex sets of inserted hyperedges
  std::vector<int> inserted_keys;               // keys assigned by the CBST insert
                                                // (filled in by the data-structure update)
};
// head[key] = payload offset of the first segment of each live row.
__global__ void row_heads_kernel(const cbst_node* nodes, const int* avail, int num_nodes,
                                 int max_key, int* head);
// New H2H rows: (e, h) for every hyperedge h != e sharing a vertex with the
// inserted hyperedge e (rows[i] = vertices of keys[i]).
// The V2H rows are walked with chain.cuh's bounded walk over its `v2h_used` slots; a broken
// chain raises `v2h_control`, the V2H tree's device error word.
__global__ void neighbourhood_count_kernel(const int* keys, const int* row_offsets,
                                           const int* vertices, int num_rows, const int* v2h_head,
                                           const int* v2h_flat, long long v2h_used,
                                           std::uint32_t* v2h_control, long long* count);
__global__ void neighbourhood_gather_kernel(const int* keys, const int* row_offsets,
                                            const int* vertices, int num_rows, const int* v2h_head,
                                            const int* v2h_flat, long long v2h_used,
                                            std::uint32_t* v2h_control, const long long* offsets,
                                            unsigned long long* out);
// H2H removals: (u, d) for every surviving neighbour u of a deleted d.
__global__ void removal_count_kernel(row_view old, const int* deleted, int num_deleted,
                                     const unsigned char* is_deleted, long long* count);
__global__ void removal_gather_kernel(row_view old, const int* deleted, int num_deleted,
                                      const unsigned char* is_deleted, const long long* offsets,
                                      unsigned long long* out);
// H2H additions: (h, e) for every entry (e, h) of a new row with h not inserted.
__global__ void mirror_additions_kernel(const unsigned long long* rows, long long n,
                                        const unsigned char* is_inserted, unsigned long long* out,
                                        unsigned long long* num_out);
__global__ void pair_histogram_kernel(const unsigned long long* pairs, long long n, int* count);
__global__ void merged_row_slots_kernel(row_view old, row_delta d, int num_keys, int* len,
                                        long long* slot);
__global__ void write_merged_rows_kernel(row_view old, row_delta d, int num_keys,
                                         const long long* start, const int* len, int* out,
                                         std::uint32_t* control);
// GPU maintenance of the derived adjacency trees after the ESCHER
// operations of a batch have updated H2V and V2H. Each tree
// is updated with an exact, out-of-place sorted delta merge: the delta
// (removed entries, added entries, replaced rows) is computed on the GPU
// from the old rows and the updated V2H, merged row by row into a new
// payload, and the CBST is rebuilt over it. This replaces the per-batch
// host rebuild of the whole snapshot, its re-upload and the second resident
// copy of the trees.

// Line graph H2H (keys 1..num_keys after the batch). `old_rows` is the view of
// the current H2H tree, `v2h` the updated V2H tree.
// dynG: the merged payload must fit `max_payload` entries (capacity_error before it is allocated;
// the tree is unchanged then); a broken chain of V2H or a delta that breaks the merge's
// requirements raises V2H's device error word, which the caller reads (cbst_store::check_errors).
// Throws cuda_error for an error of the CUDA runtime.
void update_line_graph(cbst_store& h2h, const row_view& old_rows, const cbst_context& v2h,
                       const line_graph_batch& batch, int num_keys, long long max_payload);

}  // namespace dyng::detail::escher
