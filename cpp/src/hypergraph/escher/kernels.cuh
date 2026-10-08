// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/kernels.cuh
// Derived from MOSP_ESCHER@4b86159:escher/kernel/kernels.cuh (the compaction kernels)
/**
 * @file kernels.cuh
 * @brief Declarations of the CBST store's kernels.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#pragma once

#include "hypergraph/escher/cbst_store.hpp"

namespace dyng::detail::escher {

// Build tree
__global__ void build_empty_tree_kernel(cbst_node* nodes, int n);
__global__ void rewire_tree_kernel(cbst_node* nodes, int n);
__global__ void store_items_kernel(cbst_node* nodes, int* indices, int* values, int n,
                                   int total_size);
__global__ void init_tail_metadata_kernel(cbst_node* nodes, const int* flat_values, int n);
__global__ void set_initial_occupancy_kernel(cbst_node* nodes, const int* row_counts, int n);
__global__ void count_occupancy_mismatches_kernel(const cbst_node* nodes, const int* flat_values,
                                                  int n, int* mismatches);
__global__ void place_node_records_kernel(cbst_node* nodes, const cbst_node* sorted_records, int n);
__global__ void rank_order_nodes_kernel(const cbst_node* nodes, const int* avail, int n,
                                        cbst_node* ranked, int* ranked_live);
__global__ void record_keys_and_starts_kernel(const cbst_node* records, int n, int* keys,
                                              int* starts);

// Payload ops
__global__ void allocate_space_kernel(int* partial_solution, int* flat_values,
                                      int space_available_from, int* insert_indices,
                                      int* insert_values, int* insert_sizes, int insert_size);
__global__ void compute_next_multiple_of_4_kernel(int* partial_solution, int* tmp, int K);
__global__ void update_partial_solution_kernel(int* partial_solution, int* tmp, int K);

// Delete / availability (two-phase: locate then apply)
__global__ void locate_delete_targets_kernel(cbst_node* nodes, int* delete_indices, int delete_size,
                                             int* out_positions);
__global__ void apply_deletes_kernel(cbst_node* nodes, int* positions, int delete_size, int* avail);
__global__ void reduce_avail_level_kernel(int level_start, int level_end, int num_records,
                                          int* avail, int* subtree_avail);

// Lookup / find
__global__ void find_node_kernel(const cbst_node* nodes, const int* search_indices, int search_size,
                                 int* positions);
__global__ void find_row_lengths_kernel(const cbst_node* nodes, const int* avail, const int* flat,
                                        long long used, const int* keys, int num_keys, int* lengths,
                                        std::uint32_t* control);
__global__ void find_row_values_kernel(const cbst_node* nodes, const int* avail, const int* flat,
                                       long long used, const int* keys, int num_keys,
                                       const long long* offsets, const int* lengths, int* out);

// Insert reuse (two-phase: locate then apply)
__global__ void locate_reusable_slots_kernel(int* subtree_avail, int* avail, int num_records,
                                             int* out_positions, int K);

// Best-fit metadata extraction
__global__ void extract_slot_capacities_kernel(cbst_node* nodes, int* positions,
                                               int* out_capacities, int D);
__global__ void compute_item_sizes_kernel(int* prefix_sizes, int* out_sizes, int K);

// GPU-parallel best-fit matching
__global__ void lower_bound_kernel(int* sorted_caps, int D, int* sorted_sizes, int M, int* out_lo);
__global__ void compute_b_in_place_kernel(int* lo, int M);
__global__ void compute_assigned_kernel(int* prefix_max, int* assigned, int M);

// Recover original keys for deleted-slot positions via CBST layout formula
__global__ void extract_keys_from_positions_kernel(int* d_keys, int* positions, int* out_keys,
                                                   int num_records, int D);

// Pair matched items with their best-fit slots
__global__ void pair_matches_kernel(const int* item_order, const int* assigned,
                                    const int* slot_order, int match_count,
                                    int* matched_item_indices, int* matched_slot_indices);

// Degree-binned apply reuse (thread / warp / block per matched item); keys
// come from deleted_keys to preserve the BST order
__global__ void apply_reuse_thread_kernel(cbst_node* nodes, int* flat_values, int* avail,
                                          int* positions, int* new_payload, int* new_prefix_sizes,
                                          int* matched_item_indices, int* matched_slot_indices,
                                          int* deleted_keys, int* bin_indices, int bin_count);
__global__ void apply_reuse_warp_kernel(cbst_node* nodes, int* flat_values, int* avail,
                                        int* positions, int* new_payload, int* new_prefix_sizes,
                                        int* matched_item_indices, int* matched_slot_indices,
                                        int* deleted_keys, int* bin_indices, int bin_count);
__global__ void apply_reuse_block_kernel(cbst_node* nodes, int* flat_values, int* avail,
                                         int* positions, int* new_payload, int* new_prefix_sizes,
                                         int* matched_item_indices, int* matched_slot_indices,
                                         int* deleted_keys, int* bin_indices, int bin_count);

// Degree-binned insert_node (fill)
__global__ void insert_node_thread_kernel(cbst_node* nodes, int* flat_values, int* insert_indices,
                                          int* insert_values, int* insert_sizes,
                                          int* partial_solution, int* bin_indices, int bin_count,
                                          const int* avail, std::uint32_t* control);
__global__ void insert_node_warp_kernel(cbst_node* nodes, int* flat_values, int* insert_indices,
                                        int* insert_values, int* insert_sizes,
                                        int* partial_solution, int* bin_indices, int bin_count,
                                        const int* avail, std::uint32_t* control);
__global__ void insert_node_block_kernel(cbst_node* nodes, int* flat_values, int* insert_indices,
                                         int* insert_values, int* insert_sizes,
                                         int* partial_solution, int* bin_indices, int bin_count,
                                         const int* avail, std::uint32_t* control);
__global__ void fixup_overflow_metadata_kernel(cbst_node* nodes, int* insert_indices,
                                               int* insert_sizes, int* partial_solution,
                                               int space_available_from, int insert_size);

// Unfill (degree-binned)
__global__ void unfill_thread_kernel(cbst_node* nodes, int* flat_values, int* keys,
                                     int* values_to_remove, int* remove_prefix_sizes,
                                     int* bin_indices, int bin_count, int used,
                                     std::uint32_t* control, const int* avail);
__global__ void unfill_warp_kernel(cbst_node* nodes, int* flat_values, int* keys,
                                   int* values_to_remove, int* remove_prefix_sizes,
                                   int* bin_indices, int bin_count, int used,
                                   std::uint32_t* control, const int* avail);
__global__ void unfill_block_kernel(cbst_node* nodes, int* flat_values, int* keys,
                                    int* values_to_remove, int* remove_prefix_sizes,
                                    int* bin_indices, int bin_count, int used,
                                    std::uint32_t* control, const int* avail);

// Payload compaction (two passes, one warp per node; MOSP_ESCHER's compactCBST)
__global__ void compact_sizes_kernel(const cbst_node* nodes, const int* avail, int n,
                                     const int* flat, long long used, int* live_count,
                                     long long* slots_by_rank, std::uint32_t* control);
__global__ void compact_copy_kernel(cbst_node* nodes, const int* avail, int n, const int* flat,
                                    long long used, const int* live_count,
                                    const long long* offset_by_rank, int* dst, int* start_offsets);

// Host launchers of the construction kernels and of the inverse layout formula, for the layout
// test of ESCHER-GPU's tests/cbst_test.cu (which launched the kernels directly; the host stubs of
// the library's kernels are not visible outside it). Ordered on `stream`.
void launch_build_layout(cudaStream_t stream, cbst_node* nodes, int* keys, int* starts, int n,
                         int total_size);
void launch_extract_keys_from_positions(cudaStream_t stream, int* keys, int* positions,
                                        int* out_keys, int n, int d);

}  // namespace dyng::detail::escher
