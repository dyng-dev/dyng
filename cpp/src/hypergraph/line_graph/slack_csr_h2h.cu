// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDevice.cu
/**
 * @file line_graph/slack_csr_h2h.cu
 * @brief The slack CSR line graph on the device: its build from the host line graph (also the
 *        rebuild after a tail overflow) and the batch's pair delta applied one warp per row, rows
 *        that outgrow their slot moved to the tail (MOSP_ESCHER's buildDeviceH2H, buildCsr,
 *        applyDeltaPairs, applyRowsWarp, scatterWeightsKernel and downloadRows).
 *
 * Ported from MOSP_ESCHER@4b86159 with the renames of parity/escher_port/rename_map_hsosp.toml
 * (M11a, ADR 0089; the layout rules are described in
 * cpp/src/hypergraph/slack_csr/slack_layout.hpp).
 */
#include "core/budget_counters.hpp"
#include "hypergraph/slack_csr/slack_csr_store.hpp"
#include "util/device_error_flags.cuh"

#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cstdio>
#include <limits>

namespace dyng::detail::hyper::slack {

namespace {

inline int align_up_32(int x) {
  return (x + 31) & ~31;
}

inline int row_capacity_for(int deg) {
  // Slack: at least 4 spare slots, at least 25% headroom, 32-aligned
  // (matching ESCHER's warp-aligned block philosophy).
  return align_up_32(deg + std::max(4, deg / 4));
}

// Directed row key of a delta pair entry: row << 33 | isInsert << 32 | col.
// Sorting the keys groups every row's changes, deletions first.
__global__ void expand_delta_keys_kernel(const int2* pairs, int n_del, int n_tot,
                                         unsigned long long* keys) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_tot) return;
  const unsigned long long ins = (i >= n_del) ? 1ull : 0ull;
  const unsigned a = static_cast<unsigned>(pairs[i].x);
  const unsigned b = static_cast<unsigned>(pairs[i].y);
  keys[2LL * i] = (static_cast<unsigned long long>(a) << 33) | (ins << 32) | b;
  keys[2LL * i + 1] = (static_cast<unsigned long long>(b) << 33) | (ins << 32) | a;
}

struct key_row {
  __host__ __device__ int operator()(unsigned long long k) const {
    return static_cast<int>(k >> 33);
  }
};

/**
 * One warp per touched row (rows are unique after the sort, so the rows
 * are race-free): deletions by ballot search + swap-remove, then a
 * coalesced append of the insertions, relocating the row to the tail
 * region (one atomicAdd) when it outgrows its capacity. (The original
 * grouped the rows in a host unordered_map, uploaded 7 arrays and ran one
 * thread per row with an O(deletions x degree) serial search.)
 */
__global__ void apply_rows_warp_kernel(int n_rows, const int* rows, const int* row_off,
                                       const int* row_len, const unsigned long long* keys,
                                       long long* row_start, int* deg, int* cap, int* col_ind,
                                       unsigned long long* tail_cursor, long long cap_entries,
                                       int* overflow_flag, std::uint32_t* error) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n_rows) return;
  const int r = rows[warp];
  const int off = row_off[warp];
  const int len = row_len[warp];
  long long base = row_start[r];
  int d = deg[r];

  // Deletions sort before insertions (isInsert bit = 0).
  int nd = 0;
  for (int k = lane; k < len; k += 32) nd += ((keys[off + k] >> 32) & 1ull) ? 0 : 1;
  for (int o = 16; o > 0; o >>= 1) nd += __shfl_xor_sync(0xffffffffu, nd, o);
  for (int k = 0; k < nd; ++k) {
    const int v = static_cast<int>(keys[off + k] & 0xffffffffu);
    int pos = -1;
    for (int e0 = 0; e0 < d && pos < 0; e0 += 32) {
      const int e = e0 + lane;
      const unsigned m = __ballot_sync(0xffffffffu, e < d && col_ind[base + e] == v);
      if (m) pos = e0 + __ffs(m) - 1;
    }
    if (pos >= 0) {
      if (lane == 0) col_ind[base + pos] = col_ind[base + d - 1];
      --d;
    } else if (lane == 0) {
      // dynG: a deleted pair the row does not hold (a broken invariant of the device graph).
      raise_device_error(error, device_error::internal);
    }
    __syncwarp();
  }

  const int ni = len - nd;
  if (ni > 0) {
    if (d + ni > cap[r]) {
      const int need = d + ni;
      const int new_cap = (need + max(4, need / 4) + 31) & ~31;
      unsigned long long pos = 0;
      if (lane == 0) pos = atomicAdd(tail_cursor, static_cast<unsigned long long>(new_cap));
      pos = __shfl_sync(0xffffffffu, pos, 0);
      if (static_cast<long long>(pos) + new_cap > cap_entries) {
        if (lane == 0) {
          *overflow_flag = 1;
          deg[r] = d;
        }
        return;
      }
      for (int e = lane; e < d; e += 32) col_ind[pos + e] = col_ind[base + e];
      if (lane == 0) {
        row_start[r] = static_cast<long long>(pos);
        cap[r] = new_cap;
      }
      base = static_cast<long long>(pos);
    }
    for (int k = lane; k < ni; k += 32)
      col_ind[base + d + k] = static_cast<int>(keys[off + nd + k] & 0xffffffffu);
    d += ni;
  }
  if (lane == 0) deg[r] = d;
}

__global__ void scatter_weights_kernel(const int* ids, const long long* w, int n,
                                       long long* node_w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) node_w[ids[i]] = w[i];
}

}  // namespace

void build_device_h2h(device_h2h& dev, const host_hypergraph& hg, const line_graph_csr& lg,
                      int max_nodes, double entry_headroom) {
  build_csr(dev, hg, lg, max_nodes, entry_headroom);
  dev.inc.build(hg, max_nodes, entry_headroom);
}

void build_csr(device_h2h& dev, const host_hypergraph& hg, const line_graph_csr& lg, int max_nodes,
               double entry_headroom) {
  const int m = hg.max_id();
  if (m > max_nodes) {
    throw capacity_error("dyng: hypergraph: " + std::to_string(m) +
                         " ids exceed the node capacity " + std::to_string(max_nodes) +
                         " of the slack_csr line graph");
  }
  // !(x >= 1) also rejects NaN. Below 1 the col_ind capacity is smaller
  // than the row layout and the host fill below overran its vector.
  if (!(entry_headroom >= 1.0)) {
    throw invalid_argument_error(
        "dyng: hypergraph: the slack_csr line graph's entry headroom must be at least 1");
  }
  dev.free();
  dev.max_nodes = max_nodes;
  dev.num_nodes = m;

  std::vector<long long> row_start(max_nodes, 0);
  std::vector<int> deg(max_nodes, 0);
  std::vector<int> cap(max_nodes, 0);
  std::vector<long long> node_w(max_nodes, 0);

  long long cursor = 0;
  dev.num_entries = 0;
  for (int id = 1; id <= m; ++id) {
    const int d = hg.alive[id - 1] ? static_cast<int>(lg.degree(id)) : 0;
    dev.num_entries += d;
    const int c = row_capacity_for(d);
    row_start[id - 1] = cursor;
    deg[id - 1] = d;
    cap[id - 1] = c;
    node_w[id - 1] = hg.he_w[id - 1];
    cursor += c;
  }
  dev.used_entries = cursor;
  dev.cap_entries =
      std::max(cursor, static_cast<long long>(static_cast<double>(cursor) * entry_headroom)) + 4096;

  std::vector<int> col_ind(static_cast<std::size_t>(dev.cap_entries), 0);
  for (int id = 1; id <= m; ++id) {
    if (!hg.alive[id - 1]) continue;
    long long base = row_start[id - 1];
    const int* nbs = lg.row(id);
    for (long long e = 0; e < lg.degree(id); ++e) {
      col_ind[base + e] = nbs[e];  // device nodes are 0-based (dynG: so is the host line graph)
    }
  }

  const resources& res = *dev.res;
  const auto nodes = static_cast<std::size_t>(max_nodes);
  dev.d_row_start = buffer<long long>(res, nodes);
  dev.d_deg = buffer<int>(res, nodes);
  dev.d_cap = buffer<int>(res, nodes);
  dev.d_col_ind = buffer<int>(res, static_cast<std::size_t>(dev.cap_entries));
  dev.d_node_w = buffer<long long>(res, nodes);
  dev.d_tail_cursor = buffer<unsigned long long>(res, 1);
  dev.d_overflow_flag = buffer<int>(res, 1);
  dev.d_error = buffer<std::uint32_t>(res, 1);
  DYNG_CUDA_TRY(cudaMemsetAsync(dev.d_error.data(), 0, sizeof(std::uint32_t), dev.stream));

  DYNG_CUDA_TRY(cudaMemcpyAsync(dev.d_row_start.data(), row_start.data(),
                                sizeof(long long) * max_nodes, cudaMemcpyHostToDevice, dev.stream));
  DYNG_CUDA_TRY(cudaMemcpyAsync(dev.d_deg.data(), deg.data(), sizeof(int) * max_nodes,
                                cudaMemcpyHostToDevice, dev.stream));
  DYNG_CUDA_TRY(cudaMemcpyAsync(dev.d_cap.data(), cap.data(), sizeof(int) * max_nodes,
                                cudaMemcpyHostToDevice, dev.stream));
  DYNG_CUDA_TRY(cudaMemcpyAsync(dev.d_col_ind.data(), col_ind.data(), sizeof(int) * dev.cap_entries,
                                cudaMemcpyHostToDevice, dev.stream));
  DYNG_CUDA_TRY(cudaMemcpyAsync(dev.d_node_w.data(), node_w.data(), sizeof(long long) * max_nodes,
                                cudaMemcpyHostToDevice, dev.stream));
  unsigned long long tc = static_cast<unsigned long long>(cursor);
  DYNG_CUDA_TRY(cudaMemcpyAsync(dev.d_tail_cursor.data(), &tc, sizeof(unsigned long long),
                                cudaMemcpyHostToDevice, dev.stream));
  DYNG_CUDA_TRY(cudaMemsetAsync(dev.d_overflow_flag.data(), 0, sizeof(int), dev.stream));
}

/**
 * Applies the pairs in dev.delta to the CSR on the device: directed keys
 * (row << 33 | isInsert << 32 | col) are radix-sorted, run-length encoded
 * by row and applied one warp per row; the weights of new nodes are
 * scattered. Returns false on a tail overflow.
 */
bool apply_delta_pairs(device_h2h& dev) {
  device_delta& dd = dev.delta;
  const int block = 256;
  const long long n_pairs = static_cast<long long>(dd.num_del) + dd.num_ins;
  dd.d_sorted_keys = nullptr;
  dd.num_sorted_keys = 0;
  if (n_pairs > 0) {
    const long long n_keys = 2 * n_pairs;
    if (n_keys > std::numeric_limits<int>::max()) {
      throw capacity_error("dyng: hypergraph: a batch of " + std::to_string(n_pairs) +
                           " line-graph pair changes exceeds the int32 counts of the CSR apply");
    }
    dd.reserve_keys(*dev.res, n_keys);
    expand_delta_keys_kernel<<<grid_for(n_pairs, block), block, 0, dev.stream>>>(
        dd.d_pairs.data(), dd.num_del, static_cast<int>(n_pairs), dd.d_keys.data());
    DYNG_CUDA_TRY(cudaGetLastError());

    // Only the bits up to the row index take part in the sort.
    int row_bits = 1;
    while ((1LL << row_bits) < dev.max_nodes) ++row_bits;
    const int end_bit = 33 + row_bits;
    const int n = static_cast<int>(n_keys);
    cub::DoubleBuffer<unsigned long long> keys(dd.d_keys.data(), dd.d_keys_alt.data());
    auto rows_in =
        thrust::make_transform_iterator(static_cast<const unsigned long long*>(nullptr), key_row());
    std::size_t sort_bytes = 0, rle_bytes = 0, scan_bytes = 0;
    DYNG_CUDA_TRY(
        cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, keys, n, 0, end_bit, dev.stream));
    DYNG_CUDA_TRY(cub::DeviceRunLengthEncode::Encode(nullptr, rle_bytes, rows_in, dd.d_rows.data(),
                                                     dd.d_row_len.data(), dd.d_num_rows.data(), n,
                                                     dev.stream));
    DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, dd.d_row_len.data(),
                                                dd.d_row_off.data(), n, dev.stream));
    dd.reserve_temp(*dev.res, std::max({sort_bytes, rle_bytes, scan_bytes}));
    DYNG_CUDA_TRY(cub::DeviceRadixSort::SortKeys(dd.d_temp.data(), dd.temp_bytes, keys, n, 0,
                                                 end_bit, dev.stream));
    const unsigned long long* sorted = keys.Current();
    dd.d_sorted_keys = sorted;
    dd.num_sorted_keys = n_keys;
    auto rows = thrust::make_transform_iterator(sorted, key_row());
    DYNG_CUDA_TRY(cub::DeviceRunLengthEncode::Encode(dd.d_temp.data(), dd.temp_bytes, rows,
                                                     dd.d_rows.data(), dd.d_row_len.data(),
                                                     dd.d_num_rows.data(), n, dev.stream));
    int n_rows = 0;
    DYNG_CUDA_TRY(cudaMemcpyAsync(&n_rows, dd.d_num_rows.data(), sizeof(int),
                                  cudaMemcpyDeviceToHost, dev.stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(dev.stream));
    note_host_sync();
    DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(dd.d_temp.data(), dd.temp_bytes,
                                                dd.d_row_len.data(), dd.d_row_off.data(), n_rows,
                                                dev.stream));
    apply_rows_warp_kernel<<<grid_for(32LL * n_rows, block), block, 0, dev.stream>>>(
        n_rows, dd.d_rows.data(), dd.d_row_off.data(), dd.d_row_len.data(), sorted,
        dev.d_row_start.data(), dev.d_deg.data(), dev.d_cap.data(), dev.d_col_ind.data(),
        dev.d_tail_cursor.data(), dev.cap_entries, dev.d_overflow_flag.data(), dev.d_error.data());
    DYNG_CUDA_TRY(cudaGetLastError());
  }
  if (dd.num_new > 0) {
    scatter_weights_kernel<<<grid_for(dd.num_new, block), block, 0, dev.stream>>>(
        dd.d_ids.data(), dd.d_new_w.data(), dd.num_new, dev.d_node_w.data());
    DYNG_CUDA_TRY(cudaGetLastError());
  }

  int overflow = 0;
  DYNG_CUDA_TRY(cudaMemcpyAsync(&overflow, dev.d_overflow_flag.data(), sizeof(int),
                                cudaMemcpyDeviceToHost, dev.stream));
  unsigned long long tc = 0;
  DYNG_CUDA_TRY(cudaMemcpyAsync(&tc, dev.d_tail_cursor.data(), sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost, dev.stream));
  std::uint32_t error = 0;
  DYNG_CUDA_TRY(cudaMemcpyAsync(&error, dev.d_error.data(), sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost, dev.stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(dev.stream));  // MOSP_ESCHER's read point (its copies)
  note_host_sync();
  dev.used_entries = static_cast<long long>(tc);
  if (error != 0)
    throw_device_errors(
        error, "hypergraph::apply (slack_csr)",
        "a deleted pair the device line graph or its incidence mirror does not hold");
  return overflow == 0;
}

std::vector<std::vector<int>> download_rows(const device_h2h& dev, int m) {
  std::vector<long long> row_start(m);
  std::vector<int> deg(m);
  std::vector<int> col_ind(static_cast<std::size_t>(dev.cap_entries));
  if (m > 0) {
    DYNG_CUDA_TRY(cudaMemcpyAsync(row_start.data(), dev.d_row_start.data(), sizeof(long long) * m,
                                  cudaMemcpyDeviceToHost, dev.stream));
    DYNG_CUDA_TRY(cudaMemcpyAsync(deg.data(), dev.d_deg.data(), sizeof(int) * m,
                                  cudaMemcpyDeviceToHost, dev.stream));
    DYNG_CUDA_TRY(cudaMemcpyAsync(col_ind.data(), dev.d_col_ind.data(),
                                  sizeof(int) * dev.cap_entries, cudaMemcpyDeviceToHost,
                                  dev.stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(dev.stream));
    note_host_sync();
  }
  std::vector<std::vector<int>> rows(m);
  for (int i = 0; i < m; ++i) {
    rows[i].reserve(deg[i]);
    for (int e = 0; e < deg[i]; ++e) rows[i].push_back(col_ind[row_start[i] + e] + 1);
    std::sort(rows[i].begin(), rows[i].end());
  }
  return rows;
}

}  // namespace dyng::detail::hyper::slack
