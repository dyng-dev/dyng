// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/delta_merge.cu
// Derived from ESCHER-GPU@abcf9b2:structure/graph_maintenance.cu
/**
 * @file sorted_merge.cu
 * @brief The sorted delta merge of the line graph on the device: the rows of the inserted
 *        hyperedges, the removals and mirrored additions, the per-key ranges, the merged rows and
 *        the rebuild of the H2H tree over them (ESCHER-GPU's updateLineGraph).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_line_graph.toml
 * (M8b, ADR 0042; the store's encoding is described in cpp/src/hypergraph/escher/README.md).
 */
// Kernels of the GPU maintenance of the derived adjacency trees (H2H line
// graph, V2V vertex graph); see structure/graph_maintenance.cu.
//
// Pairs are packed as (row key << 32) | value, so a sorted pair array groups
// the entries of each row in ascending value order.
#include "hypergraph/line_graph/device_frontier.cuh"
#include "hypergraph/line_graph/sorted_merge.cuh"

#include <dyng/core/buffer.hpp>

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace dyng::detail::escher {

namespace {

__device__ __forceinline__ int pair_value(unsigned long long p) {
  return static_cast<int>(p & 0xffffffffULL);
}

__device__ __forceinline__ unsigned long long pack_pair(int key, int value) {
  return (static_cast<unsigned long long>(key) << 32) | static_cast<unsigned int>(value);
}

// Number of entries of `pairs[0..n)` whose value is smaller than x.
__device__ __forceinline__ int pairs_below(const unsigned long long* pairs, int n, int x) {
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (pair_value(pairs[mid]) < x)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

// Number of entries of the sorted row `row[0..n)` smaller than x.
__device__ __forceinline__ int values_below(const int* row, int n, int x) {
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (row[mid] < x)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

// Calls f(value) for every entry of a (possibly chained) CBST row.
template <class F>
__device__ __forceinline__ void for_each_chained_entry(const int* flat, int loc, F&& f) {
  if (loc < 0) return;
  while (true) {
    int v = flat[loc];
    if (v < 0 && v != INT_MIN) {
      loc = -v;
      continue;
    }
    if (v == 0 || v == INT_MIN) return;
    f(v);
    ++loc;
  }
}
}  // namespace
__global__ void neighbourhood_count_kernel(const int* keys, const int* row_offsets,
                                           const int* vertices, int num_rows, const int* v2h_head,
                                           const int* v2h_flat, long long* count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_rows) return;
  int e = keys[i];
  long long n = 0;
  for (int p = row_offsets[i]; p < row_offsets[i + 1]; ++p)
    for_each_chained_entry(v2h_flat, v2h_head[vertices[p]], [&](int h) { n += (h != e); });
  count[i] = n;
}

__global__ void neighbourhood_gather_kernel(const int* keys, const int* row_offsets,
                                            const int* vertices, int num_rows, const int* v2h_head,
                                            const int* v2h_flat, const long long* offsets,
                                            unsigned long long* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_rows) return;
  int e = keys[i];
  long long o = offsets[i];
  for (int p = row_offsets[i]; p < row_offsets[i + 1]; ++p)
    for_each_chained_entry(v2h_flat, v2h_head[vertices[p]], [&](int h) {
      if (h != e) out[o++] = pack_pair(e, h);
    });
}

__global__ void removal_count_kernel(row_view old, const int* deleted, int num_deleted,
                                     const unsigned char* is_deleted, long long* count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_deleted) return;
  int d = deleted[i];
  long long n = 0;
  if (d <= old.max_key)
    for (int p = old.loc[d]; p < old.loc[d] + old.len[d]; ++p) n += !is_deleted[old.flat[p]];
  count[i] = n;
}

__global__ void removal_gather_kernel(row_view old, const int* deleted, int num_deleted,
                                      const unsigned char* is_deleted, const long long* offsets,
                                      unsigned long long* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_deleted) return;
  int d = deleted[i];
  long long o = offsets[i];
  if (d <= old.max_key)
    for (int p = old.loc[d]; p < old.loc[d] + old.len[d]; ++p) {
      int u = old.flat[p];
      if (!is_deleted[u]) out[o++] = pack_pair(u, d);
    }
}

__global__ void mirror_additions_kernel(const unsigned long long* rows, long long n,
                                        const unsigned char* is_inserted, unsigned long long* out,
                                        unsigned long long* num_out) {
  long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  int e = static_cast<int>(rows[i] >> 32), h = pair_value(rows[i]);
  if (!is_inserted[h]) out[atomicAdd(num_out, 1ULL)] = pack_pair(h, e);
}
__global__ void pair_histogram_kernel(const unsigned long long* pairs, long long n, int* count) {
  long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i < n) atomicAdd(&count[pairs[i] >> 32], 1);
}

__global__ void merged_row_slots_kernel(row_view old, row_delta d, int num_keys, int* len,
                                        long long* slot) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u > num_keys) return;
  int n = 0;
  if (u >= 1) {
    if (d.replaced != nullptr && d.replaced[u])
      n = d.rep_count[u];
    else
      n = (u <= old.max_key ? old.len[u] : 0) - d.rem_count[u] + d.add_count[u];
  }
  len[u] = n;
  // row padded to a multiple of 4 with at least one terminator slot
  slot[u] = u >= 1 ? ((n + 4) / 4) * 4 : 0;
}

// Warp per row. Rows without changes are copied; changed rows are merged
// in parallel: a kept old entry x lands at (its rank among the kept
// entries) + #additions < x, an addition y at its index + #kept entries < y.
// Requires: removals of a row are entries of its old row, additions are
// disjoint from the kept entries.
__global__ void write_merged_rows_kernel(row_view old, row_delta d, int num_keys,
                                         const long long* start, const int* len, int* out) {
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int num_warps = (gridDim.x * blockDim.x) >> 5;
  for (int u = warp + 1; u <= num_keys; u += num_warps) {
    int* row = out + start[u];
    const int n = len[u];
    if (d.replaced != nullptr && d.replaced[u]) {
      const unsigned long long* rep = d.rep + d.rep_off[u];
      for (int i = lane; i < n; i += 32) row[i] = pair_value(rep[i]);
    } else if (u <= old.max_key) {
      const int* src = old.flat + old.loc[u];
      const int old_len = old.len[u];
      const int rc = d.rem_count[u], ac = d.add_count[u];
      const unsigned long long* rem = d.rem + d.rem_off[u];
      const unsigned long long* add = d.add + d.add_off[u];
      if (rc == 0 && ac == 0) {
        for (int i = lane; i < old_len; i += 32) row[i] = src[i];
      } else {
        for (int i = lane; i < old_len; i += 32) {
          int x = src[i];
          int r = pairs_below(rem, rc, x);
          if (r < rc && pair_value(rem[r]) == x) continue;  // removed
          row[i - r + pairs_below(add, ac, x)] = x;
        }
        for (int k = lane; k < ac; k += 32) {
          int y = pair_value(add[k]);
          row[k + values_below(src, old_len, y) - pairs_below(rem, rc, y)] = y;
        }
      }
    }
    const int padded = ((n + 4) / 4) * 4;
    for (int i = n + lane; i < padded; i += 32) row[i] = (i == padded - 1) ? INT_MIN : 0;
  }
}
namespace {

constexpr int block_size = 256;

void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
    std::exit(-1);
  }
}

int blocks_for(long long n) {
  return static_cast<int>((n + block_size - 1) / block_size);
}

template <class T>
class device_array {
 public:
  explicit device_array(long long n) {
    check(cudaMalloc(&ptr_, sizeof(T) * std::max(1LL, n)));
  }
  ~device_array() {
    cudaFree(ptr_);
  }
  device_array(const device_array&) = delete;
  device_array& operator=(const device_array&) = delete;
  device_array(device_array&& o) noexcept : ptr_(o.ptr_) {
    o.ptr_ = nullptr;
  }
  T* get() const {
    return ptr_;
  }
  T* release() {
    T* p = ptr_;
    ptr_ = nullptr;
    return p;
  }

 private:
  T* ptr_ = nullptr;
};

template <class T>
T read_back(const T* device_ptr) {
  T value;
  check(cudaMemcpy(&value, device_ptr, sizeof(T), cudaMemcpyDeviceToHost));
  return value;
}

// Rows (vertex lists) of the batch in CSR form on the device.
struct device_rows {
  int num_rows;
  device_array<int> offsets, vertices;
  explicit device_rows(const std::vector<std::vector<int>>& rows)
      : num_rows(static_cast<int>(rows.size())), offsets(rows.size() + 1), vertices(total(rows)) {
    std::vector<int> off(rows.size() + 1, 0), flat;
    flat.reserve(total(rows));
    for (size_t i = 0; i < rows.size(); ++i) {
      flat.insert(flat.end(), rows[i].begin(), rows[i].end());
      off[i + 1] = static_cast<int>(flat.size());
    }
    check(cudaMemcpy(offsets.get(), off.data(), sizeof(int) * off.size(), cudaMemcpyHostToDevice));
    if (!flat.empty())
      check(cudaMemcpy(vertices.get(), flat.data(), sizeof(int) * flat.size(),
                       cudaMemcpyHostToDevice));
  }
  static long long total(const std::vector<std::vector<int>>& rows) {
    long long n = 0;
    for (const auto& r : rows) n += r.size();
    return n;
  }
};

// Count / exclusive scan / gather of the pairs produced by num_items items.
struct pair_list {
  device_array<unsigned long long> data;
  long long size;
};
template <class Count, class Gather>
pair_list collect_pairs(int num_items, Count count, Gather gather) {
  device_array<long long> offsets(num_items + 1);
  check(cudaMemset(offsets.get() + num_items, 0, sizeof(long long)));
  if (num_items > 0) count(offsets.get());
  thrust::device_ptr<long long> o(offsets.get());
  thrust::exclusive_scan(o, o + num_items + 1, o);
  long long n = read_back(offsets.get() + num_items);
  pair_list pairs{device_array<unsigned long long>(n), n};
  if (n > 0) gather(offsets.get(), pairs.data.get());
  return pairs;
}

void sort_pairs(pair_list& pairs, bool unique) {
  thrust::device_ptr<unsigned long long> p(pairs.data.get());
  thrust::sort(p, p + pairs.size);
  if (unique) pairs.size = thrust::unique(p, p + pairs.size) - p;
}

// Per-key counts and offsets of a sorted pair array.
struct pair_ranges {
  device_array<int> count;
  device_array<long long> offset;
  pair_ranges(const pair_list& pairs, int num_keys) : count(num_keys + 1), offset(num_keys + 1) {
    check(cudaMemset(count.get(), 0, sizeof(int) * (num_keys + 1)));
    if (pairs.size > 0)
      pair_histogram_kernel<<<blocks_for(pairs.size), block_size>>>(pairs.data.get(), pairs.size,
                                                                    count.get());
    thrust::device_ptr<int> c(count.get());
    thrust::exclusive_scan(c, c + num_keys + 1, thrust::device_ptr<long long>(offset.get()), 0LL);
  }
};

__global__ void narrow_starts_kernel(const long long* start, int num_keys, int* out) {
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (k <= num_keys) out[k - 1] = static_cast<int>(start[k]);
}

// Writes the merged rows into a new payload and rebuilds the tree over it.
void merge_and_rebuild(cbst_store& tree, const row_view& old, const row_delta& delta,
                       int num_keys) {
  device_array<int> len(num_keys + 1);
  device_array<long long> start(num_keys + 2);
  check(cudaMemset(start.get() + num_keys + 1, 0, sizeof(long long)));
  merged_row_slots_kernel<<<blocks_for(num_keys + 1), block_size>>>(old, delta, num_keys, len.get(),
                                                                    start.get());
  thrust::device_ptr<long long> s(start.get());
  thrust::exclusive_scan(s, s + num_keys + 2, s);
  long long total = read_back(start.get() + num_keys + 1);
  if (total > INT_MAX) {
    std::cerr << "[" << tree.context().dataset_name << "] ERROR: " << total
              << " payload entries exceed the int32 offsets of the CBST" << std::endl;
    std::exit(2);
  }
  // dynG (transplant hunk): the merged payload and the row offsets are the store's buffers (the
  // tree takes them over in rebuild_from_device), allocated from the tree's resources.
  buffer<int> flat(tree.context().res, static_cast<size_t>(std::max(1LL, total)));
  int grid =
      static_cast<int>(std::min<long long>((num_keys * 32LL + block_size - 1) / block_size, 65535));
  check(cudaDeviceSynchronize());  // dynG (transplant hunk): the buffers' stream-ordered allocation
  write_merged_rows_kernel<<<grid, block_size>>>(old, delta, num_keys, start.get(), len.get(),
                                                 flat.data());
  buffer<int> starts(tree.context().res, static_cast<size_t>(std::max(1, num_keys)));
  check(cudaDeviceSynchronize());
  narrow_starts_kernel<<<blocks_for(num_keys), block_size>>>(start.get(), num_keys, starts.data());
  check(cudaGetLastError());
  // dynG (transplant hunk): the kernels above run on the legacy default stream, the tree on its
  // resources' stream.
  check(cudaDeviceSynchronize());
  tree.rebuild_from_device(std::move(starts), num_keys, std::move(flat), static_cast<int>(total),
                           static_cast<int>(total));
}

struct or_bytes {
  __host__ __device__ unsigned char operator()(unsigned char a, unsigned char b) const {
    return a | b;
  }
};

}  // namespace

void update_line_graph(cbst_store& h2h, const row_view& old_rows, const cbst_context& v2h,
                       const line_graph_batch& batch, int num_keys) {
  const int num_deleted = static_cast<int>(batch.deleted_keys.size());
  const int num_inserted = static_cast<int>(batch.inserted_keys.size());
  device_frontier deleted(batch.deleted_keys, num_keys);
  device_frontier inserted(batch.inserted_keys, num_keys);  // same order as inserted_rows
  device_rows inserted_rows(batch.inserted_rows);

  // First payload segment of every V2H row (V2H rows may be chained).
  const int numVertices = v2h.num_records;
  device_array<int> head(numVertices + 1);
  check(cudaMemset(head.get(), 0xff, sizeof(int) * (numVertices + 1)));  // -1: absent
  row_heads_kernel<<<blocks_for(v2h.num_records), block_size>>>(
      v2h.d_nodes, v2h.d_avail, v2h.num_records, numVertices, head.get());

  // Rows of the inserted hyperedges: union of their vertices' V2H rows.
  pair_list rows = collect_pairs(
      num_inserted,
      [&](long long* count) {
        neighbourhood_count_kernel<<<blocks_for(num_inserted), block_size>>>(
            inserted.ids(), inserted_rows.offsets.get(), inserted_rows.vertices.get(), num_inserted,
            head.get(), v2h.d_flat_payload, count);
      },
      [&](const long long* offsets, unsigned long long* out) {
        neighbourhood_gather_kernel<<<blocks_for(num_inserted), block_size>>>(
            inserted.ids(), inserted_rows.offsets.get(), inserted_rows.vertices.get(), num_inserted,
            head.get(), v2h.d_flat_payload, offsets, out);
      });
  sort_pairs(rows, true);

  // Deleted hyperedges leave the rows of their surviving neighbours.
  pair_list removals = collect_pairs(
      num_deleted,
      [&](long long* count) {
        removal_count_kernel<<<blocks_for(num_deleted), block_size>>>(
            old_rows, deleted.ids(), num_deleted, deleted.flags(), count);
      },
      [&](const long long* offsets, unsigned long long* out) {
        removal_gather_kernel<<<blocks_for(num_deleted), block_size>>>(
            old_rows, deleted.ids(), num_deleted, deleted.flags(), offsets, out);
      });
  sort_pairs(removals, false);

  // Inserted hyperedges join the rows of their non-inserted neighbours.
  pair_list additions{device_array<unsigned long long>(rows.size), 0};
  {
    device_array<unsigned long long> count(1);
    check(cudaMemset(count.get(), 0, sizeof(unsigned long long)));
    if (rows.size > 0)
      mirror_additions_kernel<<<blocks_for(rows.size), block_size>>>(
          rows.data.get(), rows.size, inserted.flags(), additions.data.get(), count.get());
    additions.size = static_cast<long long>(read_back(count.get()));
  }
  sort_pairs(additions, false);

  // Deleted and inserted keys get a replacement row (empty if not reinserted).
  device_array<unsigned char> replaced(num_keys + 1);
  thrust::transform(thrust::device_ptr<const unsigned char>(deleted.flags()),
                    thrust::device_ptr<const unsigned char>(deleted.flags() + num_keys + 1),
                    thrust::device_ptr<const unsigned char>(inserted.flags()),
                    thrust::device_ptr<unsigned char>(replaced.get()), or_bytes());

  pair_ranges row_ranges(rows, num_keys), removal_ranges(removals, num_keys),
      addition_ranges(additions, num_keys);
  row_delta delta{replaced.get(),      row_ranges.count.get(),      row_ranges.offset.get(),
                  rows.data.get(),     removal_ranges.count.get(),  removal_ranges.offset.get(),
                  removals.data.get(), addition_ranges.count.get(), addition_ranges.offset.get(),
                  additions.data.get()};
  merge_and_rebuild(h2h, old_rows, delta, num_keys);
}

}  // namespace dyng::detail::escher
