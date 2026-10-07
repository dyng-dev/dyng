// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:include/structure.hpp
/**
 * @file cbst_store.hpp
 * @brief The CBST store: node and context records, the host API (construct, rebuild, insert, append
 *        = fill, erase, remove_values = unfill) and the store object (ESCHER-GPU's CBSTOperations).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#pragma once

#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

namespace dyng::detail::escher {

// CBST node definition shared by kernels across translation units
struct cbst_node {
  int index;
  int value;
  int length;
  int size;
  int occupancy;      // number of data elements in the active (tail) segment
  int tail_base;      // base offset of the last segment in flat_payload
  int tail_capacity;  // usable capacity of the tail segment (excludes sentinel)
  cbst_node* left;
  cbst_node* right;
  cbst_node* parent;
};

// Number of batches of fill, best-fit reuse (insert) and unfill work that
// launched the thread- / warp- / block-level kernel ([0] / [1] / [2]) since
// the store was created (ESCHER-GPU counted them per process). Reported by
// --check so that the tests can verify that every degree-binned path ran.
struct degree_bin_launches {
  long long fill[3] = {0, 0, 0};
  long long reuse[3] = {0, 0, 0};
  long long unfill[3] = {0, 0, 0};
};

// Device-side context is opaque; host context holds device buffers
struct cbst_context {
  cbst_node* d_nodes;
  int* d_keys;
  int* d_start_offsets;
  int* d_flat_payload;
  int* d_insert_keys;
  int* d_insert_payload;
  int* d_insert_prefix_sizes;
  int* d_relocation_plan;
  int* d_avail;
  int* d_subtree_avail;
  int fixed_size;
  int num_records;
  int initial_payload_size;
  std::string dataset_name;
  int alignment;
  // Capacities of the per-operation input buffers (d_insert_keys /
  // d_insert_prefix_sizes / d_relocation_plan and d_insert_payload); grown on
  // demand by the operations.
  int scratch_item_capacity;
  long long scratch_payload_capacity;
  // The resources of the store and the buffers the device pointers above view (allocated from
  // the resources' memory resource, ordered on their stream).
  resources res;
  // Host synchronizations with the stream so far (the read points of ADR 0038).
  long long host_syncs = 0;
  // The degree-bin launches of this store's operations.
  degree_bin_launches bin_launches;
  buffer<cbst_node> nodes_storage;
  buffer<int> keys_storage, start_offsets_storage, flat_payload_storage, insert_keys_storage,
      insert_payload_storage, insert_prefix_sizes_storage, relocation_plan_storage, avail_storage,
      subtree_avail_storage;
};

// Mapping returned by insert_cbst: for each input item i, item_to_key[i] is the
// actual CBST key the item was stored under (may differ from new_keys[i] due to
// best-fit matching reusing a deleted slot's key).
struct insert_mapping {
  std::vector<int> item_to_key;
};

// Host API for CBST operations (free functions). The payload buffer grows on
// demand when fill overflow or surplus inserts need more room.
void construct_cbst(int* keys, int* start_offsets, int num_records, int* flat_payload,
                    int flat_payload_size, int payload_capacity, const std::string& dataset_name,
                    cbst_context& ctx);
// Rebuilds the tree over a payload that is already on the device: rows of
// keys 1..num_records start at d_start_offsets (flattened layout) in d_flat_payload
// (flat_payload_size used entries, payload_capacity allocated). Takes ownership of
// both buffers and releases the previous tree and payload.
void rebuild_cbst_from_device(buffer<int> start_offsets, int num_records, buffer<int> flat_payload,
                              int flat_payload_size, int payload_capacity, cbst_context& ctx);
void fill_cbst(const std::vector<int>& insert_keys, const std::vector<int>& insert_payload,
               const std::vector<int>& insert_prefix_sizes, cbst_context& ctx);
void delete_cbst(const std::vector<int>& delete_keys, cbst_context& ctx);
insert_mapping insert_cbst(const std::vector<int>& new_keys, const std::vector<int>& new_payload,
                           const std::vector<int>& new_prefix_sizes, cbst_context& ctx);
void unfill_cbst(const std::vector<int>& keys_to_unfill, const std::vector<int>& values_to_remove,
                 const std::vector<int>& remove_prefix_sizes, cbst_context& ctx);

// OO wrapper to manage CBST lifecycle and operations
struct cbst_store {
  cbst_store(const resources& res, std::string dataset_name, int payload_capacity,
             int alignment = 4);
  ~cbst_store();
  cbst_store(const cbst_store&) = delete;
  cbst_store& operator=(const cbst_store&) = delete;
  cbst_store(cbst_store&& other) noexcept;
  cbst_store& operator=(cbst_store&& other) noexcept;

  void construct(int* keys, int* start_offsets, int num_records, int* flat_payload,
                 int flat_payload_size);
  void rebuild_from_device(buffer<int> start_offsets, int num_records, buffer<int> flat_payload,
                           int flat_payload_size, int payload_capacity);
  insert_mapping insert(const std::vector<int>& insert_keys, const std::vector<int>& insert_payload,
                        const std::vector<int>& insert_prefix_sizes);
  void append(const std::vector<int>& insert_keys, const std::vector<int>& insert_payload,
              const std::vector<int>& insert_prefix_sizes);
  void erase(const std::vector<int>& delete_keys);
  void remove_values(const std::vector<int>& keys_to_unfill,
                     const std::vector<int>& values_to_remove,
                     const std::vector<int>& remove_prefix_sizes);
  // A deep copy of the store on `res` (dynG addition): every array copied, the node pointers
  // re-wired to the copy; it is independent of its source. The copy waits for the work queued
  // on the source's stream; its counters start at zero. Throws invalid_argument_error for a
  // poisoned store.
  cbst_store clone(const resources& res) const;
  // The same on the store's own resources.
  cbst_store clone() const {
    return clone(ctx_.res);
  }
  // The heap position of the node of every key (-1: absent).
  std::vector<int> find(const std::vector<int>& ids) const;
  // The degree-bin launches of the store's operations so far.
  const degree_bin_launches& bin_launches() const {
    return ctx_.bin_launches;
  }
  // Host synchronizations of the store's operations so far (ADR 0038, the read points).
  long long host_syncs() const {
    return ctx_.host_syncs;
  }
  // Whether a failed operation left the store poisoned (construct or rebuild it).
  bool poisoned() const {
    return poisoned_;
  }

  // Accessor to underlying device-resident context (read-only)
  const cbst_context& context() const;
  // Device memory held by the tree (nodes, index arrays, payload capacity,
  // operation buffers).
  size_t device_bytes() const;

 private:
  void check_not_poisoned(const char* op) const;
  template <typename op_t>
  auto guarded(const char* op, op_t&& body);

  cbst_context ctx_{};
  bool constructed_ = false;
  bool poisoned_ = false;
};

}  // namespace dyng::detail::escher
