// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDelta.cu
/**
 * @file line_graph/incidence_delta.cu
 * @brief incidence_delta on the device: the net sign per incidence, the candidate pairs of the
 *        vertex lists before and after the batch (the vertex mirror patched between the two
 *        passes), their classification by pre / post overlap, the post-batch hyperedge rows written
 *        into the mirror (MOSP_ESCHER's deriveDelta).
 *
 * Ported from MOSP_ESCHER@4b86159 with the renames of parity/escher_port/rename_map_hsosp.toml
 * (M11a, ADR 0089; the layout rules are described in
 * cpp/src/hypergraph/slack_csr/slack_layout.hpp).
 */
/*
 * The host ships the changed incidences (vertex, hyperedge, +/-1) and the
 * vertex lists of the touched hyperedges before and after the batch. On the
 * device:
 *   1. the incidence changes are sorted and reduced to a net sign per
 *      (vertex, hyperedge);
 *   2. candidates: for every changed incidence (v, h) and every o in v's
 *      hyperedge list before the batch (pass 0) and after it (pass 1), the
 *      pair (min(h,o), max(h,o)); the vertex -> hyperedge mirror is updated
 *      between the passes;
 *   3. the candidates are sorted and deduplicated, and each pair (a, b) is
 *      classified by whether the vertex lists of a and b overlap before and
 *      after the batch: inserted if only after, deleted if only before.
 * The classification looks only at the state before and after the batch,
 * so it does not depend on the order of the batch's ops, and it is exact:
 * a pair can only change if a vertex was added to or removed from one of
 * the two hyperedges while it belonged to the other, and that incidence
 * change lists the pair as a candidate. (The original ran this step
 * single-threaded on a host copy of the whole line graph, emulating every
 * op with +/-1 counts in a hash map.)
 */
#include "hypergraph/slack_csr/slack_csr_store.hpp"

#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

namespace dyng::detail::hyper::slack {

namespace {

constexpr u64 sentinel = ~0ull;

/** Grows a device buffer to hold at least @p n elements (contents lost). */
template <class T>
void grow(T*& ptr, long long& cap, long long n) {
  if (n <= cap) return;
  if (ptr) cudaFree(ptr);
  cap = n + n / 2 + 1024;
  HSOSP_CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * cap));
}

template <class T>
void upload(T* dst, const std::vector<T>& src) {
  if (!src.empty())
    HSOSP_CUDA_CHECK(cudaMemcpy(dst, src.data(), sizeof(T) * src.size(), cudaMemcpyHostToDevice));
}

int bits_for(long long n) {
  int b = 1;
  while ((1LL << b) < n) ++b;
  return b;
}

struct high_word {
  __host__ __device__ int operator()(u64 k) const {
    return static_cast<int>(k >> 32);
  }
};
struct is_deleted {
  __host__ __device__ bool operator()(signed char c) const {
    return c < 0;
  }
};
struct is_inserted {
  __host__ __device__ bool operator()(signed char c) const {
    return c > 0;
  }
};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void set_index_kernel(const int* nodes, int n, int* idx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) idx[nodes[i]] = i;
}

__global__ void reset_index_kernel(const int* nodes, int n, int* idx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) idx[nodes[i]] = -1;
}

__global__ void scatter_int_kernel(const int* idx, const int* vals, int n, int* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[idx[i]] = vals[i];
}

// Candidates of each changed incidence: pass 0 counts the vertex's list
// before the batch (the current mirror), pass 1 after it (post_len).
__global__ void count_candidates_kernel(const u64* inc, const int* net, int n, const int* v_len,
                                        const int* post_len, long long* cnt0, long long* cnt1) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int v = static_cast<int>(inc[i] >> 32);
  const bool changed = net[i] != 0;
  cnt0[i] = changed ? v_len[v] : 0;
  cnt1[i] = changed ? post_len[v] : 0;
}

__global__ void totals_kernel(const long long* off0, const long long* cnt0, const long long* off1,
                              const long long* cnt1, int n, long long* totals) {
  if (n == 0) {
    totals[0] = totals[1] = 0;
    return;
  }
  totals[0] = off0[n - 1] + cnt0[n - 1];
  totals[1] = off1[n - 1] + cnt1[n - 1];
}

// One warp per changed incidence (v, h): the pairs (h, o) for o in v's
// current hyperedge list.
__global__ void emit_candidates_kernel(const u64* inc, const int* net, int n,
                                       const long long* v_off, const int* v_len, const int* v_val,
                                       const long long* off, long long base, u64* cand) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n || net[warp] == 0) return;
  const int v = static_cast<int>(inc[warp] >> 32);
  const unsigned h = static_cast<unsigned>(inc[warp] & 0xffffffffu);
  const long long row_start = v_off[v];
  const int len = v_len[v];
  u64* out = cand + base + off[warp];
  for (int k = lane; k < len; k += 32) {
    const unsigned o = static_cast<unsigned>(v_val[row_start + k]);
    const unsigned a = min(h, o), b = max(h, o);
    out[k] = (o == h) ? sentinel : ((static_cast<u64>(a) << 32) | b);
  }
}

// Moves a vertex row to a new (larger) slot before it grows.
__global__ void relocate_rows_kernel(const int* verts, const long long* new_off, int n,
                                     long long* v_off, const int* v_len, int* v_val) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n) return;
  const int v = verts[warp];
  const long long from = v_off[v], to = new_off[warp];
  const int len = v_len[v];
  for (int k = lane; k < len; k += 32) v_val[to + k] = v_val[from + k];
  __syncwarp();
  if (lane == 0) v_off[v] = to;
}

// One warp per vertex segment of the net incidence changes (sorted by
// vertex, then hyperedge): removals by ballot search + swap-remove, then
// appends. Capacities were checked on the host.
__global__ void apply_vertex_rows_kernel(const int* seg_off, const int* seg_len, int n_seg,
                                         const u64* inc, const int* net, const long long* v_off,
                                         int* v_len, int* v_val) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n_seg) return;
  const int s = seg_off[warp];
  const int len = seg_len[warp];
  const int v = static_cast<int>(inc[s] >> 32);
  const long long base = v_off[v];
  int d = v_len[v];
  for (int k = 0; k < len; ++k) {
    if (net[s + k] >= 0) continue;
    const int h = static_cast<int>(inc[s + k] & 0xffffffffu);
    int pos = -1;
    for (int e0 = 0; e0 < d && pos < 0; e0 += 32) {
      const int e = e0 + lane;
      const unsigned m = __ballot_sync(0xffffffffu, e < d && v_val[base + e] == h);
      if (m) pos = e0 + __ffs(m) - 1;
    }
    if (pos >= 0) {
      if (lane == 0) v_val[base + pos] = v_val[base + d - 1];
      --d;
    }
    __syncwarp();
  }
  int appended = 0;
  for (int k = 0; k < len; ++k) {
    if (net[s + k] <= 0) continue;
    if (lane == 0) v_val[base + d + appended] = static_cast<int>(inc[s + k] & 0xffffffffu);
    ++appended;
  }
  __syncwarp();
  if (lane == 0) v_len[v] = d + appended;
}

__device__ bool sorted_overlap(const int* a, int la, const int* b, int lb) {
  int i = 0, j = 0;
  while (i < la && j < lb) {
    if (a[i] == b[j]) return true;
    if (a[i] < b[j])
      ++i;
    else
      ++j;
  }
  return false;
}

// Vertex list of node x before (pre) or after the batch: from the shipped
// rows if x was touched, otherwise from the mirror (unchanged).
__device__ void row_of(int x, bool pre, const int* touched_idx, const int* pre_off,
                       const int* pre_vals, const int* post_off, const int* post_vals,
                       const long long* he_off, const int* he_len, const int* he_val,
                       const int*& row, int& len) {
  const int t = touched_idx[x];
  if (t >= 0) {
    const int* off = pre ? pre_off : post_off;
    row = (pre ? pre_vals : post_vals) + off[t];
    len = off[t + 1] - off[t];
  } else {
    row = he_val + he_off[x];
    len = he_len[x];
  }
}

__global__ void classify_pairs_kernel(const u64* pairs, long long n, const int* touched_idx,
                                      const int* pre_off, const int* pre_vals, const int* post_off,
                                      const int* post_vals, const long long* he_off,
                                      const int* he_len, const int* he_val, signed char* cls) {
  const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const u64 k = pairs[i];
  if (k == sentinel) {
    cls[i] = 0;
    return;
  }
  const int a = static_cast<int>(k >> 32);
  const int b = static_cast<int>(k & 0xffffffffu);
  const int *ra, *rb;
  int la, lb;
  row_of(a, true, touched_idx, pre_off, pre_vals, post_off, post_vals, he_off, he_len, he_val, ra,
         la);
  row_of(b, true, touched_idx, pre_off, pre_vals, post_off, post_vals, he_off, he_len, he_val, rb,
         lb);
  const bool pre = sorted_overlap(ra, la, rb, lb);
  row_of(a, false, touched_idx, pre_off, pre_vals, post_off, post_vals, he_off, he_len, he_val, ra,
         la);
  row_of(b, false, touched_idx, pre_off, pre_vals, post_off, post_vals, he_off, he_len, he_val, rb,
         lb);
  const bool post = sorted_overlap(ra, la, rb, lb);
  cls[i] = static_cast<signed char>(post ? (pre ? 0 : 1) : (pre ? -1 : 0));
}

__global__ void keys_to_pairs_kernel(const u64* keys, long long n, int2* out) {
  const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  out[i] = make_int2(static_cast<int>(keys[i] >> 32), static_cast<int>(keys[i] & 0xffffffffu));
}

// Writes the post-batch vertex lists of the touched hyperedges into the
// mirror (slots chosen by the host).
__global__ void write_he_rows_kernel(const int* touched, const long long* off, int n_t,
                                     const int* post_off, const int* post_vals, long long* he_off,
                                     int* he_len, int* he_val) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n_t) return;
  const int node = touched[warp];
  const int b = post_off[warp];
  const int len = post_off[warp + 1] - b;
  const long long o = off[warp];
  for (int k = lane; k < len; k += 32) he_val[o + k] = post_vals[b + k];
  if (lane == 0) {
    he_off[node] = o;
    he_len[node] = len;
  }
}

}  // namespace

long long derive_delta(device_h2h& dev, const host_hypergraph& hg, const incidence_batch& inc,
                       int& reuploads, long long max_candidates) {
  device_incidence& di = dev.inc;
  device_incidence::scratch_arrays& sc = *di.scratch;
  device_delta& dd = dev.delta;
  const int block = 256;
  const int n_inc = static_cast<int>(inc.inc_key.size());
  const int n_t = static_cast<int>(inc.touched.size());
  const int n_tv = static_cast<int>(inc.touched_vertices.size());

  // ---- upload the batch (0-based node indices) ------------------------
  std::vector<u64> keys(n_inc);
  for (int i = 0; i < n_inc; ++i) keys[i] = inc.inc_key[i] - 1;  // h -> h-1
  std::vector<int> touched0(n_t);
  for (int t = 0; t < n_t; ++t) touched0[t] = inc.touched[t] - 1;
  grow(sc.inc_keys, sc.inc_cap, n_inc);
  grow(sc.inc_keys_alt, sc.inc_cap2, n_inc);
  grow(sc.signs, sc.inc_cap3, n_inc);
  grow(sc.signs_alt, sc.inc_cap4, n_inc);
  grow(sc.uniq, sc.inc_cap5, n_inc);
  grow(sc.net, sc.inc_cap6, n_inc);
  grow(sc.cnt0, sc.inc_cap7, n_inc);
  grow(sc.cnt1, sc.inc_cap8, n_inc);
  grow(sc.off0, sc.inc_cap9, n_inc);
  grow(sc.off1, sc.inc_cap10, n_inc);
  grow(sc.seg_len, sc.inc_cap11, n_inc);
  grow(sc.seg_off, sc.inc_cap12, n_inc);
  grow(sc.seg_vert, sc.inc_cap13, n_inc);
  grow(sc.touched, sc.t_cap, n_t);
  grow(sc.pre_off, sc.t_cap2, n_t + 1);
  grow(sc.post_off, sc.t_cap3, n_t + 1);
  grow(sc.he_write_off, sc.t_cap4, n_t);
  grow(sc.pre_vals, sc.val_cap, static_cast<long long>(inc.pre_vals.size()));
  grow(sc.post_vals, sc.val_cap2, static_cast<long long>(inc.post_vals.size()));
  grow(sc.t_verts, sc.tv_cap, n_tv);
  grow(sc.t_vert_len, sc.tv_cap2, n_tv);
  upload(sc.inc_keys, keys);
  upload(sc.signs, inc.inc_sign);
  upload(sc.touched, touched0);
  upload(sc.pre_off, inc.pre_off);
  upload(sc.post_off, inc.post_off);
  upload(sc.pre_vals, inc.pre_vals);
  upload(sc.post_vals, inc.post_vals);
  upload(sc.t_verts, inc.touched_vertices);
  upload(sc.t_vert_len, inc.touched_vertex_len);

  // ---- 1. net sign per (vertex, hyperedge) -------------------------------
  int n_uniq = 0;
  if (n_inc > 0) {
    const int end_bit = 32 + bits_for(std::max(2, di.num_vertices));
    cub::DoubleBuffer<u64> k(sc.inc_keys, sc.inc_keys_alt);
    cub::DoubleBuffer<int> s(sc.signs, sc.signs_alt);
    std::size_t b1 = 0, b2 = 0;
    HSOSP_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, b1, k, s, n_inc, 0, end_bit));
    HSOSP_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(nullptr, b2, sc.inc_keys, sc.uniq, sc.signs,
                                                    sc.net, sc.counts, ::cuda::std::plus<int>(),
                                                    n_inc));
    sc.reserve_temp(std::max(b1, b2));
    HSOSP_CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(sc.temp, sc.temp_cap, k, s, n_inc, 0, end_bit));
    HSOSP_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(sc.temp, sc.temp_cap, k.Current(), sc.uniq,
                                                    s.Current(), sc.net, sc.counts,
                                                    ::cuda::std::plus<int>(), n_inc));
    HSOSP_CUDA_CHECK(cudaMemcpy(sc.host_counts, sc.counts, sizeof(int), cudaMemcpyDeviceToHost));
    n_uniq = sc.host_counts[0];
  }

  // ---- 2. candidate counts (lists before / after the batch) --------------
  long long total0 = 0, total1 = 0;
  if (n_uniq > 0) {
    if (n_tv > 0)
      scatter_int_kernel<<<grid_for(n_tv, block), block>>>(sc.t_verts, sc.t_vert_len, n_tv,
                                                           di.d_post_len);
    count_candidates_kernel<<<grid_for(n_uniq, block), block>>>(sc.uniq, sc.net, n_uniq, di.d_v_len,
                                                                di.d_post_len, sc.cnt0, sc.cnt1);
    std::size_t b = 0;
    HSOSP_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, b, sc.cnt0, sc.off0, n_uniq));
    sc.reserve_temp(b);
    HSOSP_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(sc.temp, sc.temp_cap, sc.cnt0, sc.off0, n_uniq));
    HSOSP_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(sc.temp, sc.temp_cap, sc.cnt1, sc.off1, n_uniq));
    totals_kernel<<<1, 1>>>(sc.off0, sc.cnt0, sc.off1, sc.cnt1, n_uniq, sc.totals);
    HSOSP_CUDA_CHECK(
        cudaMemcpy(sc.host_totals, sc.totals, sizeof(long long) * 2, cudaMemcpyDeviceToHost));
    total0 = sc.host_totals[0];
    total1 = sc.host_totals[1];
  }
  const long long n_cand = total0 + total1;
  grow(sc.cand, sc.cand_cap, n_cand);
  grow(sc.cand_alt, sc.cand_cap2, n_cand);

  // ---- 3. pass 0, update the vertex -> hyperedge mirror, pass 1 ----------
  if (n_uniq > 0) {
    emit_candidates_kernel<<<grid_for(32LL * n_uniq, block), block>>>(
        sc.uniq, sc.net, n_uniq, di.d_v_off, di.d_v_len, di.d_v_val, sc.off0, 0, sc.cand);
    HSOSP_CUDA_CHECK(cudaGetLastError());

    // Rows that outgrow their slot move to the tail (host layout); a
    // full tail re-uploads the mirror from the host (post-batch state).
    std::vector<int> reloc_v;
    std::vector<long long> reloc_off;
    bool reupload = false;
    for (int i = 0; i < n_tv; ++i) {
      const int v = inc.touched_vertices[i];
      const int l = inc.touched_vertex_len[i];
      if (l <= di.v_cap[v]) continue;
      const int c = v_slot_for(l);
      if (di.v_tail + c > di.v_capacity) {
        reupload = true;
        break;
      }
      di.v_off[v] = di.v_tail;
      di.v_cap[v] = c;
      di.v_tail += c;
      reloc_v.push_back(v);
      reloc_off.push_back(di.v_off[v]);
    }
    if (reupload) {
      upload_vertex_rows(di, hg, 1.5);
      ++reuploads;
    } else {
      const int n_r = static_cast<int>(reloc_v.size());
      if (n_r > 0) {
        grow(sc.reloc_verts, sc.r_cap, n_r);
        grow(sc.reloc_off, sc.r_cap2, n_r);
        upload(sc.reloc_verts, reloc_v);
        upload(sc.reloc_off, reloc_off);
        relocate_rows_kernel<<<grid_for(32LL * n_r, block), block>>>(
            sc.reloc_verts, sc.reloc_off, n_r, di.d_v_off, di.d_v_len, di.d_v_val);
      }
      // Vertex segments of the sorted net changes.
      auto verts = thrust::make_transform_iterator(sc.uniq, high_word());
      std::size_t b = 0;
      HSOSP_CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(nullptr, b, verts, sc.seg_vert,
                                                          sc.seg_len, sc.counts + 1, n_uniq));
      sc.reserve_temp(b);
      HSOSP_CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(sc.temp, sc.temp_cap, verts, sc.seg_vert,
                                                          sc.seg_len, sc.counts + 1, n_uniq));
      HSOSP_CUDA_CHECK(
          cudaMemcpy(sc.host_counts + 1, sc.counts + 1, sizeof(int), cudaMemcpyDeviceToHost));
      const int n_seg = sc.host_counts[1];
      b = 0;
      HSOSP_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, b, sc.seg_len, sc.seg_off, n_seg));
      sc.reserve_temp(b);
      HSOSP_CUDA_CHECK(
          cub::DeviceScan::ExclusiveSum(sc.temp, sc.temp_cap, sc.seg_len, sc.seg_off, n_seg));
      apply_vertex_rows_kernel<<<grid_for(32LL * n_seg, block), block>>>(
          sc.seg_off, sc.seg_len, n_seg, sc.uniq, sc.net, di.d_v_off, di.d_v_len, di.d_v_val);
    }
    emit_candidates_kernel<<<grid_for(32LL * n_uniq, block), block>>>(
        sc.uniq, sc.net, n_uniq, di.d_v_off, di.d_v_len, di.d_v_val, sc.off1, total0, sc.cand);
    HSOSP_CUDA_CHECK(cudaGetLastError());
  }

  // ---- 4. distinct candidates, classified by pre / post overlap ----------
  int num_del = 0, num_ins = 0;
  if (n_cand > 0) {
    if (n_cand > max_candidates) throw std::runtime_error("applyBatch: too many candidate pairs");
    const int n = static_cast<int>(n_cand);
    cub::DoubleBuffer<u64> k(sc.cand, sc.cand_alt);
    std::size_t b1 = 0, b2 = 0, b3 = 0;
    HSOSP_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, b1, k, n));
    HSOSP_CUDA_CHECK(
        cub::DeviceSelect::Unique(nullptr, b2, sc.cand, sc.cand_alt, sc.counts + 2, n));
    grow(sc.cls, sc.cls_cap, n_cand);
    grow(sc.selected, sc.sel_cap, n_cand);
    HSOSP_CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, b3, sc.cand, thrust::make_transform_iterator(sc.cls, is_deleted()), sc.selected,
        sc.counts + 3, n));
    sc.reserve_temp(std::max({b1, b2, b3}));
    HSOSP_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(sc.temp, sc.temp_cap, k, n));
    u64* sorted = k.Current();
    u64* uniq = k.Alternate();
    HSOSP_CUDA_CHECK(
        cub::DeviceSelect::Unique(sc.temp, sc.temp_cap, sorted, uniq, sc.counts + 2, n));
    HSOSP_CUDA_CHECK(
        cudaMemcpy(sc.host_counts + 2, sc.counts + 2, sizeof(int), cudaMemcpyDeviceToHost));
    const int n_pairs = sc.host_counts[2];

    if (n_t > 0)
      set_index_kernel<<<grid_for(n_t, block), block>>>(sc.touched, n_t, di.d_touched_idx);
    classify_pairs_kernel<<<grid_for(n_pairs, block), block>>>(
        uniq, n_pairs, di.d_touched_idx, sc.pre_off, sc.pre_vals, sc.post_off, sc.post_vals,
        di.d_he_off, di.d_he_len, di.d_he_val, sc.cls);
    HSOSP_CUDA_CHECK(cudaGetLastError());
    if (n_t > 0)
      reset_index_kernel<<<grid_for(n_t, block), block>>>(sc.touched, n_t, di.d_touched_idx);

    // Deleted pairs first, then inserted ones.
    dd.reserve(n_pairs, 0);
    HSOSP_CUDA_CHECK(cub::DeviceSelect::Flagged(
        sc.temp, sc.temp_cap, uniq, thrust::make_transform_iterator(sc.cls, is_deleted()),
        sc.selected, sc.counts + 3, n_pairs));
    HSOSP_CUDA_CHECK(
        cudaMemcpy(sc.host_counts + 3, sc.counts + 3, sizeof(int), cudaMemcpyDeviceToHost));
    num_del = sc.host_counts[3];
    if (num_del > 0)
      keys_to_pairs_kernel<<<grid_for(num_del, block), block>>>(sc.selected, num_del, dd.d_pairs);
    HSOSP_CUDA_CHECK(cub::DeviceSelect::Flagged(
        sc.temp, sc.temp_cap, uniq, thrust::make_transform_iterator(sc.cls, is_inserted()),
        sc.selected, sc.counts + 3, n_pairs));
    HSOSP_CUDA_CHECK(
        cudaMemcpy(sc.host_counts + 3, sc.counts + 3, sizeof(int), cudaMemcpyDeviceToHost));
    num_ins = sc.host_counts[3];
    if (num_ins > 0)
      keys_to_pairs_kernel<<<grid_for(num_ins, block), block>>>(sc.selected, num_ins,
                                                                dd.d_pairs + num_del);
    HSOSP_CUDA_CHECK(cudaGetLastError());
  }

  // ---- 5. post-batch vertex lists of the touched hyperedges --------------
  if (n_t > 0) {
    std::vector<long long> write_off(n_t);
    bool reupload = false;
    for (int t = 0; t < n_t && !reupload; ++t) {
      const int node = inc.touched[t] - 1;
      const int l = inc.post_off[t + 1] - inc.post_off[t];
      if (l > di.he_cap[node]) {
        const int c = he_slot_for(l);
        if (di.he_tail + c > di.he_capacity) {
          reupload = true;
          break;
        }
        di.he_off[node] = di.he_tail;
        di.he_cap[node] = c;
        di.he_tail += c;
      }
      write_off[t] = di.he_off[node];
    }
    if (reupload) {
      upload_he_rows(di, hg, 1.5);
      ++reuploads;
    } else {
      upload(sc.he_write_off, write_off);
      write_he_rows_kernel<<<grid_for(32LL * n_t, block), block>>>(
          sc.touched, sc.he_write_off, n_t, sc.post_off, sc.post_vals, di.d_he_off, di.d_he_len,
          di.d_he_val);
      HSOSP_CUDA_CHECK(cudaGetLastError());
    }
  }

  // ---- 6. ids of new and dead nodes, weights -----------------------------
  std::vector<int> ids;
  ids.reserve(inc.new_he.size() + inc.dead_he.size());
  for (int id : inc.new_he) ids.push_back(id - 1);
  for (int id : inc.dead_he) ids.push_back(id - 1);
  dd.reserve(0, static_cast<long long>(ids.size()));
  upload(dd.d_ids, ids);
  upload(dd.d_new_w, inc.new_w);
  dd.num_del = num_del;
  dd.num_ins = num_ins;
  dd.num_new = static_cast<int>(inc.new_he.size());
  dd.num_dead = static_cast<int>(inc.dead_he.size());
  return n_cand;
}

}  // namespace dyng::detail::hyper::slack
