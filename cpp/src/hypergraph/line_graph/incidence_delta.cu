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
#include "core/budget_counters.hpp"
#include "hypergraph/slack_csr/slack_csr_store.hpp"
#include "util/device_error_flags.cuh"
#include "util/kernel_registry.hpp"

#include <dyng/core/profiler.hpp>

#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <limits>
#include <string>
#include <vector>

namespace dyng::detail::hyper::slack {

namespace {

constexpr u64 sentinel = ~0ull;

template <class T>
void upload(T* dst, const std::vector<T>& src, cudaStream_t stream) {
  if (!src.empty())
    DYNG_CUDA_TRY(
        cudaMemcpyAsync(dst, src.data(), sizeof(T) * src.size(), cudaMemcpyHostToDevice, stream));
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
                                         int* v_len, int* v_val, std::uint32_t* error) {
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
    } else if (lane == 0) {
      // dynG: a removed incidence the mirror does not hold (a broken invariant).
      raise_device_error(error, device_error::internal);
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

// dynG: the kernels in the registry of resources::warm_up() (kernel_registry.hpp).
DYNG_REGISTER_KERNEL(set_index_kernel);
DYNG_REGISTER_KERNEL(reset_index_kernel);
DYNG_REGISTER_KERNEL(scatter_int_kernel);
DYNG_REGISTER_KERNEL(count_candidates_kernel);
DYNG_REGISTER_KERNEL(totals_kernel);
DYNG_REGISTER_KERNEL(emit_candidates_kernel);
DYNG_REGISTER_KERNEL(relocate_rows_kernel);
DYNG_REGISTER_KERNEL(apply_vertex_rows_kernel);
DYNG_REGISTER_KERNEL(classify_pairs_kernel);
DYNG_REGISTER_KERNEL(keys_to_pairs_kernel);
DYNG_REGISTER_KERNEL(write_he_rows_kernel);

long long derive_delta(device_h2h& dev, const host_hypergraph& hg, const incidence_batch& inc,
                       int& reuploads, long long max_candidates) {
  device_incidence& di = dev.inc;
  device_incidence::scratch_arrays& sc = *di.scratch;
  device_delta& dd = dev.delta;
  const int block = 256;
  DYNG_CUDA_TRY(cudaMemsetAsync(dev.d_error.data(), 0, sizeof(std::uint32_t),
                                dev.stream));  // dynG: the error word
  // dynG: the batch's counts are int32 on the device.
  if (inc.inc_key.size() > static_cast<std::size_t>(INT_MAX) ||
      inc.touched.size() >= static_cast<std::size_t>(INT_MAX) ||
      inc.pre_vals.size() > static_cast<std::size_t>(INT_MAX) ||
      inc.post_vals.size() > static_cast<std::size_t>(INT_MAX)) {
    throw capacity_error("dyng: hypergraph: a batch of " + std::to_string(inc.inc_key.size()) +
                         " incidence changes and " + std::to_string(inc.touched.size()) +
                         " touched hyperedges exceeds the int32 counts of the device delta");
  }
  const int n_inc = static_cast<int>(inc.inc_key.size());
  const int n_t = static_cast<int>(inc.touched.size());
  const int n_tv = static_cast<int>(inc.touched_vertices.size());

  // ---- upload the batch (0-based node indices) ------------------------
  std::vector<u64> keys(n_inc);
  for (int i = 0; i < n_inc; ++i) keys[i] = inc.inc_key[i] - 1;  // h -> h-1
  std::vector<int> touched0(n_t);
  for (int t = 0; t < n_t; ++t) touched0[t] = inc.touched[t] - 1;
  (void)sc.inc_keys.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_inc));
  (void)sc.inc_keys_alt.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_inc));
  (void)sc.signs.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_inc));
  (void)sc.signs_alt.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_inc));
  (void)sc.uniq.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_inc));
  (void)sc.net.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_inc));
  (void)sc.cnt0.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_inc));
  (void)sc.cnt1.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_inc));
  (void)sc.off0.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_inc));
  (void)sc.off1.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_inc));
  (void)sc.seg_len.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_inc));
  (void)sc.seg_off.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_inc));
  (void)sc.seg_vert.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_inc));
  (void)sc.touched.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_t));
  (void)sc.pre_off.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_t + 1));
  (void)sc.post_off.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_t + 1));
  (void)sc.he_write_off.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_t));
  (void)sc.pre_vals.reserve_with_headroom(
      *dev.res, static_cast<std::size_t>(static_cast<long long>(inc.pre_vals.size())));
  (void)sc.post_vals.reserve_with_headroom(
      *dev.res, static_cast<std::size_t>(static_cast<long long>(inc.post_vals.size())));
  (void)sc.t_verts.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_tv));
  (void)sc.t_vert_len.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_tv));
  upload(sc.inc_keys.data(), keys, dev.stream);
  upload(sc.signs.data(), inc.inc_sign, dev.stream);
  upload(sc.touched.data(), touched0, dev.stream);
  upload(sc.pre_off.data(), inc.pre_off, dev.stream);
  upload(sc.post_off.data(), inc.post_off, dev.stream);
  upload(sc.pre_vals.data(), inc.pre_vals, dev.stream);
  upload(sc.post_vals.data(), inc.post_vals, dev.stream);
  upload(sc.t_verts.data(), inc.touched_vertices, dev.stream);
  upload(sc.t_vert_len.data(), inc.touched_vertex_len, dev.stream);

  // ---- 1. net sign per (vertex, hyperedge) -------------------------------
  int n_uniq = 0;
  if (n_inc > 0) {
    const int end_bit = 32 + bits_for(std::max(2, di.num_vertices));
    cub::DoubleBuffer<u64> k(sc.inc_keys.data(), sc.inc_keys_alt.data());
    cub::DoubleBuffer<int> s(sc.signs.data(), sc.signs_alt.data());
    std::size_t b1 = 0, b2 = 0;
    DYNG_CUDA_TRY(
        cub::DeviceRadixSort::SortPairs(nullptr, b1, k, s, n_inc, 0, end_bit, dev.stream));
    DYNG_CUDA_TRY(cub::DeviceReduce::ReduceByKey(nullptr, b2, sc.inc_keys.data(), sc.uniq.data(),
                                                 sc.signs.data(), sc.net.data(), sc.counts.data(),
                                                 ::cuda::std::plus<int>(), n_inc, dev.stream));
    sc.reserve_temp(std::max(b1, b2));
    DYNG_CUDA_TRY(cub::DeviceRadixSort::SortPairs(sc.temp.data(), sc.temp_cap, k, s, n_inc, 0,
                                                  end_bit, dev.stream));
    DYNG_CUDA_TRY(cub::DeviceReduce::ReduceByKey(
        sc.temp.data(), sc.temp_cap, k.Current(), sc.uniq.data(), s.Current(), sc.net.data(),
        sc.counts.data(), ::cuda::std::plus<int>(), n_inc, dev.stream));
    DYNG_CUDA_TRY(cudaMemcpyAsync(sc.host_counts.data(), sc.counts.data(), sizeof(int),
                                  cudaMemcpyDeviceToHost, dev.stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(dev.stream));
    note_host_sync();
    n_uniq = sc.host_counts.data()[0];
  }

  // ---- 2. candidate counts (lists before / after the batch) --------------
  long long total0 = 0, total1 = 0;
  if (n_uniq > 0) {
    if (n_tv > 0)
      scatter_int_kernel<<<grid_for(n_tv, block), block, 0, dev.stream>>>(
          sc.t_verts.data(), sc.t_vert_len.data(), n_tv, di.d_post_len.data());
    count_candidates_kernel<<<grid_for(n_uniq, block), block, 0, dev.stream>>>(
        sc.uniq.data(), sc.net.data(), n_uniq, di.d_v_len.data(), di.d_post_len.data(),
        sc.cnt0.data(), sc.cnt1.data());
    std::size_t b = 0;
    DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr, b, sc.cnt0.data(), sc.off0.data(), n_uniq,
                                                dev.stream));
    sc.reserve_temp(b);
    DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(sc.temp.data(), sc.temp_cap, sc.cnt0.data(),
                                                sc.off0.data(), n_uniq, dev.stream));
    DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(sc.temp.data(), sc.temp_cap, sc.cnt1.data(),
                                                sc.off1.data(), n_uniq, dev.stream));
    totals_kernel<<<1, 1, 0, dev.stream>>>(sc.off0.data(), sc.cnt0.data(), sc.off1.data(),
                                           sc.cnt1.data(), n_uniq, sc.totals.data());
    DYNG_CUDA_TRY(cudaMemcpyAsync(sc.host_totals.data(), sc.totals.data(), sizeof(long long) * 2,
                                  cudaMemcpyDeviceToHost, dev.stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(dev.stream));
    note_host_sync();
    total0 = sc.host_totals.data()[0];
    total1 = sc.host_totals.data()[1];
  }
  const long long n_cand = total0 + total1;
  (void)sc.cand.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_cand));
  (void)sc.cand_alt.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_cand));

  // ---- 3. pass 0, update the vertex -> hyperedge mirror, pass 1 ----------
  if (n_uniq > 0) {
    emit_candidates_kernel<<<grid_for(32LL * n_uniq, block), block, 0, dev.stream>>>(
        sc.uniq.data(), sc.net.data(), n_uniq, di.d_v_off.data(), di.d_v_len.data(),
        di.d_v_val.data(), sc.off0.data(), 0, sc.cand.data());
    DYNG_CUDA_TRY(cudaGetLastError());

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
      // dynG: the stage of a re-upload (its host layout and copy)
      const scoped_stage stage(*dev.res, "hypergraph.apply.translate.reupload");
      upload_vertex_rows(di, hg, 1.5);
      ++reuploads;
    } else {
      const int n_r = static_cast<int>(reloc_v.size());
      if (n_r > 0) {
        (void)sc.reloc_verts.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_r));
        (void)sc.reloc_off.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_r));
        upload(sc.reloc_verts.data(), reloc_v, dev.stream);
        upload(sc.reloc_off.data(), reloc_off, dev.stream);
        relocate_rows_kernel<<<grid_for(32LL * n_r, block), block, 0, dev.stream>>>(
            sc.reloc_verts.data(), sc.reloc_off.data(), n_r, di.d_v_off.data(), di.d_v_len.data(),
            di.d_v_val.data());
      }
      // Vertex segments of the sorted net changes.
      auto verts = thrust::make_transform_iterator(sc.uniq.data(), high_word());
      std::size_t b = 0;
      DYNG_CUDA_TRY(cub::DeviceRunLengthEncode::Encode(nullptr, b, verts, sc.seg_vert.data(),
                                                       sc.seg_len.data(), sc.counts.data() + 1,
                                                       n_uniq, dev.stream));
      sc.reserve_temp(b);
      DYNG_CUDA_TRY(cub::DeviceRunLengthEncode::Encode(sc.temp.data(), sc.temp_cap, verts,
                                                       sc.seg_vert.data(), sc.seg_len.data(),
                                                       sc.counts.data() + 1, n_uniq, dev.stream));
      DYNG_CUDA_TRY(cudaMemcpyAsync(sc.host_counts.data() + 1, sc.counts.data() + 1, sizeof(int),
                                    cudaMemcpyDeviceToHost, dev.stream));
      DYNG_CUDA_TRY(cudaStreamSynchronize(dev.stream));
      note_host_sync();
      const int n_seg = sc.host_counts.data()[1];
      b = 0;
      DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr, b, sc.seg_len.data(), sc.seg_off.data(),
                                                  n_seg, dev.stream));
      sc.reserve_temp(b);
      DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(sc.temp.data(), sc.temp_cap, sc.seg_len.data(),
                                                  sc.seg_off.data(), n_seg, dev.stream));
      apply_vertex_rows_kernel<<<grid_for(32LL * n_seg, block), block, 0, dev.stream>>>(
          sc.seg_off.data(), sc.seg_len.data(), n_seg, sc.uniq.data(), sc.net.data(),
          di.d_v_off.data(), di.d_v_len.data(), di.d_v_val.data(), dev.d_error.data());
    }
    emit_candidates_kernel<<<grid_for(32LL * n_uniq, block), block, 0, dev.stream>>>(
        sc.uniq.data(), sc.net.data(), n_uniq, di.d_v_off.data(), di.d_v_len.data(),
        di.d_v_val.data(), sc.off1.data(), total0, sc.cand.data());
    DYNG_CUDA_TRY(cudaGetLastError());
  }

  // ---- 4. distinct candidates, classified by pre / post overlap ----------
  int num_del = 0, num_ins = 0;
  if (n_cand > 0) {
    if (n_cand > max_candidates) {
      throw capacity_error("dyng: hypergraph: the line-graph delta of the batch has " +
                           std::to_string(n_cand) + " candidate pairs, more than " +
                           std::to_string(max_candidates) +
                           " (the int32 counts of the delta); split the batch");
    }
    const int n = static_cast<int>(n_cand);
    cub::DoubleBuffer<u64> k(sc.cand.data(), sc.cand_alt.data());
    std::size_t b1 = 0, b2 = 0, b3 = 0;
    DYNG_CUDA_TRY(cub::DeviceRadixSort::SortKeys(nullptr, b1, k, n, 0,
                                                 static_cast<int>(sizeof(u64) * 8), dev.stream));
    DYNG_CUDA_TRY(cub::DeviceSelect::Unique(nullptr, b2, sc.cand.data(), sc.cand_alt.data(),
                                            sc.counts.data() + 2, n, dev.stream));
    (void)sc.cls.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_cand));
    (void)sc.selected.reserve_with_headroom(*dev.res, static_cast<std::size_t>(n_cand));
    DYNG_CUDA_TRY(cub::DeviceSelect::Flagged(
        nullptr, b3, sc.cand.data(), thrust::make_transform_iterator(sc.cls.data(), is_deleted()),
        sc.selected.data(), sc.counts.data() + 3, n, dev.stream));
    sc.reserve_temp(std::max({b1, b2, b3}));
    DYNG_CUDA_TRY(cub::DeviceRadixSort::SortKeys(sc.temp.data(), sc.temp_cap, k, n, 0,
                                                 static_cast<int>(sizeof(u64) * 8), dev.stream));
    u64* sorted = k.Current();
    u64* uniq = k.Alternate();
    DYNG_CUDA_TRY(cub::DeviceSelect::Unique(sc.temp.data(), sc.temp_cap, sorted, uniq,
                                            sc.counts.data() + 2, n, dev.stream));
    DYNG_CUDA_TRY(cudaMemcpyAsync(sc.host_counts.data() + 2, sc.counts.data() + 2, sizeof(int),
                                  cudaMemcpyDeviceToHost, dev.stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(dev.stream));
    note_host_sync();
    const int n_pairs = sc.host_counts.data()[2];

    if (n_t > 0)
      set_index_kernel<<<grid_for(n_t, block), block, 0, dev.stream>>>(sc.touched.data(), n_t,
                                                                       di.d_touched_idx.data());
    classify_pairs_kernel<<<grid_for(n_pairs, block), block, 0, dev.stream>>>(
        uniq, n_pairs, di.d_touched_idx.data(), sc.pre_off.data(), sc.pre_vals.data(),
        sc.post_off.data(), sc.post_vals.data(), di.d_he_off.data(), di.d_he_len.data(),
        di.d_he_val.data(), sc.cls.data());
    DYNG_CUDA_TRY(cudaGetLastError());
    if (n_t > 0)
      reset_index_kernel<<<grid_for(n_t, block), block, 0, dev.stream>>>(sc.touched.data(), n_t,
                                                                         di.d_touched_idx.data());

    // Deleted pairs first, then inserted ones.
    dd.reserve(*dev.res, n_pairs, 0);
    DYNG_CUDA_TRY(
        cub::DeviceSelect::Flagged(sc.temp.data(), sc.temp_cap, uniq,
                                   thrust::make_transform_iterator(sc.cls.data(), is_deleted()),
                                   sc.selected.data(), sc.counts.data() + 3, n_pairs, dev.stream));
    DYNG_CUDA_TRY(cudaMemcpyAsync(sc.host_counts.data() + 3, sc.counts.data() + 3, sizeof(int),
                                  cudaMemcpyDeviceToHost, dev.stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(dev.stream));
    note_host_sync();
    num_del = sc.host_counts.data()[3];
    if (num_del > 0)
      keys_to_pairs_kernel<<<grid_for(num_del, block), block, 0, dev.stream>>>(
          sc.selected.data(), num_del, dd.d_pairs.data());
    DYNG_CUDA_TRY(
        cub::DeviceSelect::Flagged(sc.temp.data(), sc.temp_cap, uniq,
                                   thrust::make_transform_iterator(sc.cls.data(), is_inserted()),
                                   sc.selected.data(), sc.counts.data() + 3, n_pairs, dev.stream));
    DYNG_CUDA_TRY(cudaMemcpyAsync(sc.host_counts.data() + 3, sc.counts.data() + 3, sizeof(int),
                                  cudaMemcpyDeviceToHost, dev.stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(dev.stream));
    note_host_sync();
    num_ins = sc.host_counts.data()[3];
    if (num_ins > 0)
      keys_to_pairs_kernel<<<grid_for(num_ins, block), block, 0, dev.stream>>>(
          sc.selected.data(), num_ins, dd.d_pairs.data() + num_del);
    DYNG_CUDA_TRY(cudaGetLastError());
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
      // dynG: the stage of a re-upload (its host layout and copy)
      const scoped_stage stage(*dev.res, "hypergraph.apply.translate.reupload");
      upload_he_rows(di, hg, 1.5);
      ++reuploads;
    } else {
      upload(sc.he_write_off.data(), write_off, dev.stream);
      write_he_rows_kernel<<<grid_for(32LL * n_t, block), block, 0, dev.stream>>>(
          sc.touched.data(), sc.he_write_off.data(), n_t, sc.post_off.data(), sc.post_vals.data(),
          di.d_he_off.data(), di.d_he_len.data(), di.d_he_val.data());
      DYNG_CUDA_TRY(cudaGetLastError());
    }
  }

  // ---- 6. ids of new and dead nodes, weights -----------------------------
  std::vector<int> ids;
  ids.reserve(inc.new_he.size() + inc.dead_he.size());
  for (int id : inc.new_he) ids.push_back(id - 1);
  for (int id : inc.dead_he) ids.push_back(id - 1);
  dd.reserve(*dev.res, 0, static_cast<long long>(ids.size()));
  upload(dd.d_ids.data(), ids, dev.stream);
  upload(dd.d_new_w.data(), inc.new_w, dev.stream);
  dd.num_del = num_del;
  dd.num_ins = num_ins;
  dd.num_new = static_cast<int>(inc.new_he.size());
  dd.num_dead = static_cast<int>(inc.dead_he.size());
  return n_cand;
}

}  // namespace dyng::detail::hyper::slack
