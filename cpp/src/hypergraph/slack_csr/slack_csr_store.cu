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
#include "core/budget_counters.hpp"
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
void grow(buffer<T>& b, long long& cap, long long n, const resources& res) {
  if (n <= cap) return;
  b = buffer<T>();  // release first: the peak holds one array, not two
  cap = n + n / 2 + 1024;
  b = buffer<T>(res, static_cast<std::size_t>(cap));
}

template <class T>
void upload(T* dst, const std::vector<T>& src, cudaStream_t stream) {
  if (!src.empty())
    DYNG_CUDA_TRY(
        cudaMemcpyAsync(dst, src.data(), sizeof(T) * src.size(), cudaMemcpyHostToDevice, stream));
}

}  // namespace

// ---------------------------------------------------------------------------
// device_incidence
// ---------------------------------------------------------------------------

device_incidence::device_incidence() = default;
device_incidence::~device_incidence() = default;
device_incidence::device_incidence(device_incidence&&) noexcept = default;
device_incidence& device_incidence::operator=(device_incidence&&) noexcept = default;

void device_incidence::free() {
  d_he_off = {};
  d_he_len = {};
  d_he_val = {};
  d_v_off = {};
  d_v_len = {};
  d_v_val = {};
  d_touched_idx = {};
  d_post_len = {};
  scratch.reset();
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
  di.d_he_val = buffer<int>();
  di.d_he_val = buffer<int>(*di.res, static_cast<std::size_t>(capacity));
  di.he_capacity = capacity;
  upload(di.d_he_val.data(), val, di.stream);
  // The free tail is zeroed too (rows move there with slack).
  DYNG_CUDA_TRY(cudaMemsetAsync(di.d_he_val.data() + cursor, 0, sizeof(int) * (capacity - cursor),
                                di.stream));
  upload(di.d_he_off.data(), di.he_off, di.stream);
  upload(di.d_he_len.data(), len, di.stream);
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
  di.d_v_val = buffer<int>();
  di.d_v_val = buffer<int>(*di.res, static_cast<std::size_t>(capacity));
  di.v_capacity = capacity;
  upload(di.d_v_val.data(), val, di.stream);
  // The free tail is zeroed too (rows move there with slack).
  DYNG_CUDA_TRY(
      cudaMemsetAsync(di.d_v_val.data() + cursor, 0, sizeof(int) * (capacity - cursor), di.stream));
  upload(di.d_v_off.data(), di.v_off, di.stream);
  upload(di.d_v_len.data(), len, di.stream);
}

void device_incidence::build(const host_hypergraph& hg, int nodes, double headroom) {
  free();
  max_nodes = nodes;
  num_vertices = hg.num_vertices;
  d_he_off = buffer<long long>(*res, static_cast<std::size_t>(max_nodes));
  d_he_len = buffer<int>(*res, static_cast<std::size_t>(max_nodes));
  d_touched_idx = buffer<int>(*res, static_cast<std::size_t>(max_nodes));
  DYNG_CUDA_TRY(cudaMemsetAsync(d_touched_idx.data(), 0xff, sizeof(int) * max_nodes, stream));
  d_v_off = buffer<long long>(*res, static_cast<std::size_t>(std::max(1, num_vertices)));
  d_v_len = buffer<int>(*res, static_cast<std::size_t>(std::max(1, num_vertices)));
  d_post_len = buffer<int>(*res, static_cast<std::size_t>(std::max(1, num_vertices)));
  upload_he_rows(*this, hg, headroom);
  upload_vertex_rows(*this, hg, headroom);
  scratch = std::make_unique<scratch_arrays>(*res);
}

long long incidence_mirror_mismatches(const device_incidence& di, const host_hypergraph& hg) {
  auto download = [&di](auto* src, long long n, auto& dst) {
    dst.resize(static_cast<std::size_t>(n));
    if (n > 0)
      DYNG_CUDA_TRY(
          cudaMemcpyAsync(dst.data(), src, sizeof(dst[0]) * n, cudaMemcpyDeviceToHost, di.stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(di.stream));
    note_host_sync();
  };
  std::vector<long long> off;
  std::vector<int> len, val;
  long long bad = 0;
  // Hyperedge rows (dead hyperedges: empty).
  const int m = hg.max_id();
  download(di.d_he_off.data(), m, off);
  download(di.d_he_len.data(), m, len);
  download(di.d_he_val.data(), di.he_tail, val);  // rows lie below the tail
  for (int id = 1; id <= m; ++id) {
    std::vector<int> got(val.begin() + off[id - 1], val.begin() + off[id - 1] + len[id - 1]);
    std::vector<int> want = hg.alive[id - 1] ? hg.he_verts[id - 1] : std::vector<int>{};
    std::sort(got.begin(), got.end());
    if (got != want) ++bad;
  }
  // Vertex rows (0-based node indices).
  download(di.d_v_off.data(), di.num_vertices, off);
  download(di.d_v_len.data(), di.num_vertices, len);
  download(di.d_v_val.data(), di.v_tail, val);
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
  d_row_start = {};
  d_deg = {};
  d_cap = {};
  d_col_ind = {};
  d_node_w = {};
  d_tail_cursor = {};
  d_overflow_flag = {};
  d_error = {};
}

void device_delta::reserve(const resources& res, long long pairs, long long ids) {
  if (pairs > pair_capacity) {
    d_pairs = buffer<int2>();
    pair_capacity = pairs + pairs / 2 + 1024;
    d_pairs = buffer<int2>(res, static_cast<std::size_t>(pair_capacity));
  }
  if (ids > id_capacity) {
    d_ids = buffer<int>();
    d_new_w = buffer<long long>();
    id_capacity = ids + ids / 2 + 1024;
    d_ids = buffer<int>(res, static_cast<std::size_t>(id_capacity));
    d_new_w = buffer<long long>(res, static_cast<std::size_t>(id_capacity));
  }
}

void device_delta::reserve_keys(const resources& res, long long keys) {
  if (keys <= key_capacity) return;
  d_keys = {};
  d_keys_alt = {};
  d_rows = {};
  d_row_len = {};
  d_row_off = {};
  if (d_num_rows.size() == 0) d_num_rows = buffer<int>(res, 1);
  key_capacity = keys + keys / 2 + 4096;
  const auto n = static_cast<std::size_t>(key_capacity);
  d_keys = buffer<unsigned long long>(res, n);
  d_keys_alt = buffer<unsigned long long>(res, n);
  d_rows = buffer<int>(res, n);
  d_row_len = buffer<int>(res, n);
  d_row_off = buffer<int>(res, n);
}

void device_delta::reserve_temp(const resources& res, std::size_t bytes) {
  if (bytes <= temp_bytes) return;
  d_temp = {};
  temp_bytes = bytes + bytes / 2 + 4096;
  d_temp = buffer<char>(res, temp_bytes);
}

void device_delta::free() {
  *this = device_delta{};
}

// free() releases the graph arrays only: build_csr (also used to
// rebuild after a tail overflow) keeps the last delta for hsospUpdate and
// the incidence mirror. (dynG: the arrays are buffers; the destructor and
// the moves are the defaults.)

long long device_h2h::device_bytes() const {
  return static_cast<long long>(max_nodes) *
             (sizeof(long long) + 2 * sizeof(int) + sizeof(long long)) +
         cap_entries * sizeof(int) + sizeof(unsigned long long) + sizeof(int) + inc.device_bytes();
  // (dynG's error word, 4 bytes, is not in MOSP_ESCHER's graph_mb scope.)
}

}  // namespace dyng::detail::hyper::slack
