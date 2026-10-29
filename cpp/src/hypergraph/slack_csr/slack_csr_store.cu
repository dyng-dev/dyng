// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDelta.cu
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDevice.cu
/**
 * @file slack_csr/slack_csr_store.cu
 * @brief The incidence mirror of slack_csr storage on the device (its slot rules, uploads with tail
 *        room, build and the mirror check) and the lifecycle of the device line graph and its delta
 *        buffers (MOSP_ESCHER's DeviceIncidence, DeviceH2H and DeviceDelta).
 *
 * Ported from MOSP_ESCHER@4b86159 with the renames of parity/escher_port/rename_map_hsosp.toml
 * (M11a, ADR 0089; the layout rules are described in
 * cpp/src/hypergraph/slack_csr/slack_layout.hpp).
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

}  // namespace

// ---------------------------------------------------------------------------
// device_incidence
// ---------------------------------------------------------------------------

device_incidence::~device_incidence() {
  free();
}

device_incidence::device_incidence(device_incidence&& o) noexcept {
  *this = std::move(o);
}

device_incidence& device_incidence::operator=(device_incidence&& o) noexcept {
  if (this != &o) {
    free();
    max_nodes = o.max_nodes;
    num_vertices = o.num_vertices;
    d_he_off = o.d_he_off;
    d_he_len = o.d_he_len;
    d_he_val = o.d_he_val;
    he_capacity = o.he_capacity;
    he_tail = o.he_tail;
    he_off = std::move(o.he_off);
    he_cap = std::move(o.he_cap);
    d_v_off = o.d_v_off;
    d_v_len = o.d_v_len;
    d_v_val = o.d_v_val;
    v_capacity = o.v_capacity;
    v_tail = o.v_tail;
    v_off = std::move(o.v_off);
    v_cap = std::move(o.v_cap);
    d_touched_idx = o.d_touched_idx;
    d_post_len = o.d_post_len;
    scratch = o.scratch;
    o.d_he_off = nullptr;
    o.d_he_len = nullptr;
    o.d_he_val = nullptr;
    o.d_v_off = nullptr;
    o.d_v_len = nullptr;
    o.d_v_val = nullptr;
    o.d_touched_idx = nullptr;
    o.d_post_len = nullptr;
    o.scratch = nullptr;
  }
  return *this;
}

void device_incidence::free() {
  for (void* p :
       {static_cast<void*>(d_he_off), static_cast<void*>(d_he_len), static_cast<void*>(d_he_val),
        static_cast<void*>(d_v_off), static_cast<void*>(d_v_len), static_cast<void*>(d_v_val),
        static_cast<void*>(d_touched_idx), static_cast<void*>(d_post_len)})
    if (p) cudaFree(p);
  delete scratch;
  d_he_off = nullptr;
  d_he_len = nullptr;
  d_he_val = nullptr;
  d_v_off = nullptr;
  d_v_len = nullptr;
  d_v_val = nullptr;
  d_touched_idx = nullptr;
  d_post_len = nullptr;
  scratch = nullptr;
  he_capacity = v_capacity = he_tail = v_tail = 0;
  he_off.clear();
  he_cap.clear();
  v_off.clear();
  v_cap.clear();
}

long long device_incidence::device_bytes() const {
  return static_cast<long long>(max_nodes) * (sizeof(long long) + 2 * sizeof(int)) +
         he_capacity * sizeof(int) +
         static_cast<long long>(num_vertices) * (sizeof(long long) + 2 * sizeof(int)) +
         v_capacity * sizeof(int);
}

namespace {

// Slack of a row: hyperedges gain vertices in vertex batches, vertices
// gain hyperedges with every insertion.
int he_slot_for(int len) {
  return len + std::max(2, len / 2);
}
int v_slot_for(int len) {
  return len + std::max(4, len / 2);
}

}  // namespace

// (Re)uploads the hyperedge rows from the host, with free tail space.
void upload_he_rows(device_incidence& di, const host_hypergraph& hg, double growth) {
  const int m = hg.max_id();
  di.he_off.assign(di.max_nodes, 0);
  di.he_cap.assign(di.max_nodes, 0);
  std::vector<int> len(di.max_nodes, 0);
  long long cursor = 0;
  for (int id = 1; id <= m; ++id) {
    const int l = static_cast<int>(hg.he_verts[id - 1].size());
    len[id - 1] = l;
    di.he_off[id - 1] = cursor;
    di.he_cap[id - 1] = he_slot_for(l);
    cursor += di.he_cap[id - 1];
  }
  di.he_tail = cursor;
  const long long capacity = std::max(cursor, static_cast<long long>(cursor * growth)) + 64;
  std::vector<int> val(static_cast<std::size_t>(cursor), 0);
  for (int id = 1; id <= m; ++id)
    std::copy(hg.he_verts[id - 1].begin(), hg.he_verts[id - 1].end(),
              val.begin() + di.he_off[id - 1]);
  if (di.d_he_val) cudaFree(di.d_he_val);
  HSOSP_CUDA_CHECK(cudaMalloc(&di.d_he_val, sizeof(int) * capacity));
  di.he_capacity = capacity;
  upload(di.d_he_val, val);
  // The free tail is zeroed too (rows move there with slack).
  HSOSP_CUDA_CHECK(cudaMemset(di.d_he_val + cursor, 0, sizeof(int) * (capacity - cursor)));
  upload(di.d_he_off, di.he_off);
  upload(di.d_he_len, len);
}

// (Re)uploads the vertex rows (0-based node indices) from the host.
void upload_vertex_rows(device_incidence& di, const host_hypergraph& hg, double growth) {
  const int n = di.num_vertices;
  di.v_off.assign(n, 0);
  di.v_cap.assign(n, 0);
  std::vector<int> len(n, 0);
  long long cursor = 0;
  for (int v = 0; v < n; ++v) {
    const int l = static_cast<int>(hg.v2h[v].size());
    len[v] = l;
    di.v_off[v] = cursor;
    di.v_cap[v] = v_slot_for(l);
    cursor += di.v_cap[v];
  }
  di.v_tail = cursor;
  const long long capacity = std::max(cursor, static_cast<long long>(cursor * growth)) + 64;
  std::vector<int> val(static_cast<std::size_t>(cursor), 0);
  for (int v = 0; v < n; ++v)
    for (std::size_t k = 0; k < hg.v2h[v].size(); ++k)
      val[di.v_off[v] + k] = static_cast<int>(hg.v2h[v][k]);  // dynG: the ids are 0-based
  if (di.d_v_val) cudaFree(di.d_v_val);
  HSOSP_CUDA_CHECK(cudaMalloc(&di.d_v_val, sizeof(int) * capacity));
  di.v_capacity = capacity;
  upload(di.d_v_val, val);
  // The free tail is zeroed too (rows move there with slack).
  HSOSP_CUDA_CHECK(cudaMemset(di.d_v_val + cursor, 0, sizeof(int) * (capacity - cursor)));
  upload(di.d_v_off, di.v_off);
  upload(di.d_v_len, len);
}

void device_incidence::build(const host_hypergraph& hg, int nodes, double headroom) {
  free();
  max_nodes = nodes;
  num_vertices = hg.num_vertices;
  HSOSP_CUDA_CHECK(cudaMalloc(&d_he_off, sizeof(long long) * max_nodes));
  HSOSP_CUDA_CHECK(cudaMalloc(&d_he_len, sizeof(int) * max_nodes));
  HSOSP_CUDA_CHECK(cudaMalloc(&d_touched_idx, sizeof(int) * max_nodes));
  HSOSP_CUDA_CHECK(cudaMemset(d_touched_idx, 0xff, sizeof(int) * max_nodes));
  HSOSP_CUDA_CHECK(cudaMalloc(&d_v_off, sizeof(long long) * std::max(1, num_vertices)));
  HSOSP_CUDA_CHECK(cudaMalloc(&d_v_len, sizeof(int) * std::max(1, num_vertices)));
  HSOSP_CUDA_CHECK(cudaMalloc(&d_post_len, sizeof(int) * std::max(1, num_vertices)));
  upload_he_rows(*this, hg, headroom);
  upload_vertex_rows(*this, hg, headroom);
  scratch = new scratch_arrays();
}

long long incidence_mirror_mismatches(const device_incidence& di, const host_hypergraph& hg) {
  auto download = [](auto* src, long long n, auto& dst) {
    dst.resize(static_cast<std::size_t>(n));
    if (n > 0)
      HSOSP_CUDA_CHECK(cudaMemcpy(dst.data(), src, sizeof(dst[0]) * n, cudaMemcpyDeviceToHost));
  };
  std::vector<long long> off;
  std::vector<int> len, val;
  long long bad = 0;
  // Hyperedge rows (dead hyperedges: empty).
  const int m = hg.max_id();
  download(di.d_he_off, m, off);
  download(di.d_he_len, m, len);
  download(di.d_he_val, di.he_tail, val);  // rows lie below the tail
  for (int id = 1; id <= m; ++id) {
    std::vector<int> got(val.begin() + off[id - 1], val.begin() + off[id - 1] + len[id - 1]);
    std::vector<int> want = hg.alive[id - 1] ? hg.he_verts[id - 1] : std::vector<int>{};
    std::sort(got.begin(), got.end());
    if (got != want) ++bad;
  }
  // Vertex rows (0-based node indices).
  download(di.d_v_off, di.num_vertices, off);
  download(di.d_v_len, di.num_vertices, len);
  download(di.d_v_val, di.v_tail, val);
  for (int v = 0; v < di.num_vertices; ++v) {
    std::vector<int> got(val.begin() + off[v], val.begin() + off[v] + len[v]);
    std::vector<int> want;
    for (const std::int64_t id : hg.v2h[v]) want.push_back(static_cast<int>(id));  // dynG: 0-based
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    if (got != want) ++bad;
  }
  return bad;
}

// ---------------------------------------------------------------------------
// device_h2h
// ---------------------------------------------------------------------------

void device_h2h::free() {
  if (d_row_start) cudaFree(d_row_start);
  if (d_deg) cudaFree(d_deg);
  if (d_cap) cudaFree(d_cap);
  if (d_col_ind) cudaFree(d_col_ind);
  if (d_node_w) cudaFree(d_node_w);
  if (d_tail_cursor) cudaFree(d_tail_cursor);
  if (d_overflow_flag) cudaFree(d_overflow_flag);
  d_row_start = nullptr;
  d_deg = nullptr;
  d_cap = nullptr;
  d_col_ind = nullptr;
  d_node_w = nullptr;
  d_tail_cursor = nullptr;
  d_overflow_flag = nullptr;
}

void device_delta::reserve(long long pairs, long long ids) {
  if (pairs > pair_capacity) {
    if (d_pairs) cudaFree(d_pairs);
    pair_capacity = pairs + pairs / 2 + 1024;
    HSOSP_CUDA_CHECK(cudaMalloc(&d_pairs, sizeof(int2) * pair_capacity));
  }
  if (ids > id_capacity) {
    if (d_ids) cudaFree(d_ids);
    if (d_new_w) cudaFree(d_new_w);
    id_capacity = ids + ids / 2 + 1024;
    HSOSP_CUDA_CHECK(cudaMalloc(&d_ids, sizeof(int) * id_capacity));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_new_w, sizeof(long long) * id_capacity));
  }
}

void device_delta::reserve_keys(long long keys) {
  if (keys <= key_capacity) return;
  if (d_keys) cudaFree(d_keys);
  if (d_keys_alt) cudaFree(d_keys_alt);
  if (d_rows) cudaFree(d_rows);
  if (d_row_len) cudaFree(d_row_len);
  if (d_row_off) cudaFree(d_row_off);
  if (!d_num_rows) HSOSP_CUDA_CHECK(cudaMalloc(&d_num_rows, sizeof(int)));
  key_capacity = keys + keys / 2 + 4096;
  HSOSP_CUDA_CHECK(cudaMalloc(&d_keys, sizeof(unsigned long long) * key_capacity));
  HSOSP_CUDA_CHECK(cudaMalloc(&d_keys_alt, sizeof(unsigned long long) * key_capacity));
  HSOSP_CUDA_CHECK(cudaMalloc(&d_rows, sizeof(int) * key_capacity));
  HSOSP_CUDA_CHECK(cudaMalloc(&d_row_len, sizeof(int) * key_capacity));
  HSOSP_CUDA_CHECK(cudaMalloc(&d_row_off, sizeof(int) * key_capacity));
}

void device_delta::reserve_temp(std::size_t bytes) {
  if (bytes <= temp_bytes) return;
  if (d_temp) cudaFree(d_temp);
  temp_bytes = bytes + bytes / 2 + 4096;
  HSOSP_CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
}

void device_delta::free() {
  for (void* p :
       {static_cast<void*>(d_pairs), static_cast<void*>(d_ids), static_cast<void*>(d_new_w),
        static_cast<void*>(d_keys), static_cast<void*>(d_keys_alt), static_cast<void*>(d_rows),
        static_cast<void*>(d_row_len), static_cast<void*>(d_row_off),
        static_cast<void*>(d_num_rows), d_temp})
    if (p) cudaFree(p);
  if (h_sorted_keys) cudaFreeHost(h_sorted_keys);
  *this = device_delta{};
}

// free() releases the graph arrays only: build_csr (also used to
// rebuild after a tail overflow) keeps the last delta for hsospUpdate and
// the incidence mirror.
device_h2h::~device_h2h() {
  free();
  delta.free();
  inc.free();
}

device_h2h::device_h2h(device_h2h&& o) noexcept {
  *this = std::move(o);
}

device_h2h& device_h2h::operator=(device_h2h&& o) noexcept {
  if (this != &o) {
    free();
    max_nodes = o.max_nodes;
    num_nodes = o.num_nodes;
    cap_entries = o.cap_entries;
    used_entries = o.used_entries;
    d_row_start = o.d_row_start;
    d_deg = o.d_deg;
    d_cap = o.d_cap;
    d_col_ind = o.d_col_ind;
    d_node_w = o.d_node_w;
    d_tail_cursor = o.d_tail_cursor;
    d_overflow_flag = o.d_overflow_flag;
    num_entries = o.num_entries;
    delta.free();
    delta = o.delta;
    o.delta = device_delta{};
    inc = std::move(o.inc);
    o.d_row_start = nullptr;
    o.d_deg = nullptr;
    o.d_cap = nullptr;
    o.d_col_ind = nullptr;
    o.d_node_w = nullptr;
    o.d_tail_cursor = nullptr;
    o.d_overflow_flag = nullptr;
  }
  return *this;
}

long long device_h2h::device_bytes() const {
  return static_cast<long long>(max_nodes) *
             (sizeof(long long) + 2 * sizeof(int) + sizeof(long long)) +
         cap_entries * sizeof(int) + sizeof(unsigned long long) + sizeof(int) + inc.device_bytes();
}

}  // namespace dyng::detail::hyper::slack
