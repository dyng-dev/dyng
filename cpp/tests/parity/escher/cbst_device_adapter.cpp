// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_device_adapter.cpp
 * @brief The dynG device store as a store of the CBST trace replay (M8a step 3, ADR 0038).
 */
#include "parity/escher/cbst_device_adapter.hpp"

#include "hypergraph/escher/cbst_store.hpp"
#include "hypergraph/escher/integrity.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>

#include <cuda_runtime_api.h>

#include <cxxabi.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace dyng::test::cbst_compare {

namespace esc = dyng::detail::escher;
namespace tr = dyng::test::cbst_trace;

namespace {

void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("device adapter: ") + what + ": " + cudaGetErrorString(e));
}

std::string demangle(const char* name) {
  int status = 0;
  char* d = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  std::string out = (status == 0 && d != nullptr) ? d : name;
  std::free(d);
  return out;
}

template <typename value_t>
std::vector<value_t> download(const value_t* device, std::int64_t n) {
  std::vector<value_t> out(static_cast<std::size_t>(std::max<std::int64_t>(n, 0)));
  if (!out.empty())
    cuda_check(cudaMemcpy(out.data(), device, out.size() * sizeof(value_t), cudaMemcpyDeviceToHost),
               "download");
  return out;
}

/// The state of one store on the host (the arrays the digest reads).
struct host_state {
  std::int64_t num_records = 0, fixed_size = 0, bump = 0, alignment = 0;
  std::int64_t scratch_items = 0, scratch_payload = 0, device_bytes = 0;
  std::vector<esc::cbst_host_node> nodes;  ///< children as heap positions (download())
  std::vector<int> avail, subtree_avail, flat;
};

/// The twin constructs of every adapter of the process.
std::atomic<std::int64_t> all_twin_constructs{0};

}  // namespace

std::int64_t device_store_adapter::total_twin_constructs() noexcept {
  return all_twin_constructs.load();
}

struct device_store_adapter::tree {
  std::string name;
  std::unique_ptr<esc::cbst_store> store;
};

namespace {

host_state read_state(const esc::cbst_store& s, bool full) {
  const esc::cbst_context& c = s.context();
  host_state h;
  h.device_bytes = static_cast<std::int64_t>(s.device_bytes());
  if (!full) {
    cuda_check(cudaDeviceSynchronize(), "synchronize");
    h.num_records = c.num_records;
    h.fixed_size = c.fixed_size;
    h.bump = c.initial_payload_size;
    h.alignment = c.alignment;
    h.scratch_items = c.scratch_item_capacity;
    h.scratch_payload = c.scratch_payload_capacity;
    // The scalars only: D is the root's subtree_avail.
    const std::int64_t n = c.d_nodes != nullptr ? std::max<std::int64_t>(c.num_records, 0) : 0;
    if (n > 0 && c.d_subtree_avail != nullptr) h.subtree_avail = download(c.d_subtree_avail, 1);
    return h;
  }
  // The whole state through the store's own download (dynG addition).
  esc::cbst_host_state d = s.download();
  h.num_records = d.num_records;
  h.fixed_size = d.fixed_size;
  h.bump = d.initial_payload_size;
  h.alignment = d.alignment;
  h.scratch_items = d.scratch_item_capacity;
  h.scratch_payload = d.scratch_payload_capacity;
  h.nodes = std::move(d.nodes);
  const std::size_t n = h.nodes.size();
  h.avail = d.avail.size() == n ? std::move(d.avail) : std::vector<int>(n, 0);
  h.subtree_avail =
      d.subtree_avail.size() == n ? std::move(d.subtree_avail) : std::vector<int>(n, 0);
  h.flat = std::move(d.payload);
  return h;
}

/// The digest of a state, read exactly as the trace recorder reads the originals' (ADR 0036).
tree_digest digest_of(const host_state& h, bool with_rows) {
  tree_digest d;
  d.num_records = h.num_records;
  d.fixed_size = h.fixed_size;
  d.bump = h.bump;
  d.alignment = h.alignment;
  d.scratch_items = h.scratch_items;
  d.scratch_payload = h.scratch_payload;
  d.device_bytes = h.device_bytes;
  d.d = (h.num_records > 0 && !h.subtree_avail.empty()) ? h.subtree_avail[0] : -1;
  if (!with_rows) return d;
  const auto n = static_cast<std::int64_t>(h.nodes.size());
  auto child = [&](std::int64_t t) -> std::int64_t { return t < n ? t : -2; };
  std::vector<std::int64_t> order;
  order.reserve(static_cast<std::size_t>(n));
  std::vector<char> seen(static_cast<std::size_t>(n), 0);
  if (n > 0) {
    std::vector<std::int64_t> stack;
    std::int64_t t = 0;
    while (t >= 0 || !stack.empty()) {
      while (t >= 0 && !seen[static_cast<std::size_t>(t)]) {
        seen[static_cast<std::size_t>(t)] = 1;
        stack.push_back(t);
        t = child(h.nodes[static_cast<std::size_t>(t)].left);
        if (t == -2) t = -1;
      }
      if (t >= 0) t = -1;
      if (stack.empty()) break;
      t = stack.back();
      stack.pop_back();
      order.push_back(t);
      t = child(h.nodes[static_cast<std::size_t>(t)].right);
      if (t == -2) t = -1;
    }
    std::vector<std::int64_t> rest;
    for (std::int64_t u = 0; u < n; ++u)
      if (!seen[static_cast<std::size_t>(u)]) rest.push_back(u);
    std::stable_sort(rest.begin(), rest.end(), [&](std::int64_t a, std::int64_t b) {
      return h.nodes[static_cast<std::size_t>(a)].index <
             h.nodes[static_cast<std::size_t>(b)].index;
    });
    order.insert(order.end(), rest.begin(), rest.end());
  }
  const auto limit = static_cast<std::int64_t>(h.flat.size());
  d.subtree_avail = h.subtree_avail;
  d.keys.reserve(order.size());
  d.records.reserve(order.size());
  for (std::int64_t t : order) {
    const esc::cbst_host_node& nd = h.nodes[static_cast<std::size_t>(t)];
    key_digest k;
    k.key = nd.index;
    k.avail = h.avail[static_cast<std::size_t>(t)];
    k.length = nd.length;
    k.size = nd.size;
    k.occupancy = nd.occupancy;
    k.tail_capacity = nd.tail_capacity;
    k.value = nd.value;
    k.tail_base = nd.tail_base;
    std::int64_t loc = nd.value;
    int segments = 1;
    bool terminated = false;
    for (std::int64_t steps = 0; steps <= limit; ++steps) {
      if (loc < 0 || loc >= limit) break;
      const int x = h.flat[static_cast<std::size_t>(loc)];
      if (x < 0 && x != INT_MIN) {
        loc = -static_cast<std::int64_t>(x);
        ++segments;
        continue;
      }
      if (x == 0 || x == INT_MIN) {
        terminated = true;
        break;
      }
      k.row.push_back(x);
      ++loc;
    }
    k.segments = segments;
    k.terminated = terminated ? 1 : 0;
    k.row_len = static_cast<std::int32_t>(k.row.size());
    k.row_hash = tr::fnv1a64(k.row.data(), k.row.size() * sizeof(std::int32_t));
    d.keys.push_back(k.key);
    d.records.push_back(std::move(k));
  }
  return d;
}

/// The store's integrity checker (ESCHER-GPU's checkTreeRows, ported) on the rows as read.
std::int64_t integrity_of(const esc::cbst_store& s, const tree_digest& d) {
  std::int64_t max_key = 0;
  for (const key_digest& k : d.records)
    if (k.avail == 0 && k.key >= 1) max_key = std::max<std::int64_t>(max_key, k.key);
  if (max_key > (std::int64_t{1} << 28)) return -1;
  std::vector<std::vector<int>> expected(static_cast<std::size_t>(max_key));
  for (const key_digest& k : d.records)
    if (k.avail == 0 && k.key >= 1) expected[static_cast<std::size_t>(k.key - 1)] = k.row;
  std::ostringstream log;
  return esc::check_tree_rows(s.context(), expected, false, s.context().dataset_name.c_str(), log);
}

/// The scanned tail occupancy of every row of a construct layout, computed on the host as
/// init_tail_metadata_kernel does (the values before the first 0 or terminator, at most the row's
/// tail capacity): row counts for the fast path of construct where a trace has none.
std::vector<int> scanned_counts(const std::vector<int>& starts, const std::vector<int>& flat,
                                std::int64_t flat_size) {
  std::vector<int> counts(starts.size(), 0);
  for (std::size_t r = 0; r < starts.size(); ++r) {
    const std::int64_t end = r + 1 < starts.size() ? starts[r + 1] : flat_size;
    const std::int64_t capacity = std::max<std::int64_t>(0, end - starts[r] - 1);
    int c = 0;
    while (c < capacity) {
      const std::int64_t p = starts[r] + c;
      if (p < 0 || p >= static_cast<std::int64_t>(flat.size())) break;
      const int v = flat[static_cast<std::size_t>(p)];
      if (v == 0 || v == INT_MIN) break;
      ++c;
    }
    counts[r] = c;
  }
  return counts;
}

/// The node array of a store as bytes, its child and parent pointers replaced by their offsets
/// from the array (two stores' arrays are then comparable byte for byte).
std::vector<unsigned char> node_bytes(const esc::cbst_store& s) {
  const esc::cbst_context& c = s.context();
  const std::int64_t n = c.d_nodes != nullptr ? std::max<std::int64_t>(c.num_records, 0) : 0;
  std::vector<esc::cbst_node> nodes = download(c.d_nodes, n);
  const auto base = reinterpret_cast<std::uintptr_t>(c.d_nodes);
  auto rebase = [&](esc::cbst_node*& p) {
    std::uintptr_t a = p == nullptr ? 0 : reinterpret_cast<std::uintptr_t>(p) - base + 1;
    std::memcpy(&p, &a, sizeof p);
  };
  for (esc::cbst_node& nd : nodes) {
    rebase(nd.left);
    rebase(nd.right);
    rebase(nd.parent);
  }
  std::vector<unsigned char> out(nodes.size() * sizeof(esc::cbst_node));
  if (!out.empty()) std::memcpy(out.data(), nodes.data(), out.size());
  return out;
}

std::array<std::int64_t, 9> bins_now(const esc::cbst_store& s) {
  const esc::degree_bin_launches& b = s.bin_launches();
  std::array<std::int64_t, 9> out{};
  for (int l = 0; l < 3; ++l) {
    out[static_cast<std::size_t>(l)] = b.fill[l];
    out[static_cast<std::size_t>(3 + l)] = b.reuse[l];
    out[static_cast<std::size_t>(6 + l)] = b.unfill[l];
  }
  return out;
}

}  // namespace

device_store_adapter::device_store_adapter(std::string policy)
    : policy_(std::move(policy)), res_(resources::cuda()) {
  esc::store_policy p;
  if (!esc::store_policy::from_name(policy_, p))
    throw tr::trace_error("device store: no policy '" + policy_ +
                          "' (the store has escher_gpu and mosp_escher_staged)");
}

device_store_adapter::~device_store_adapter() = default;

std::string device_store_adapter::name() const {
  return "device:" + policy_;
}

void device_store_adapter::create_tree(std::uint32_t id, const std::string& dataset,
                                       std::int64_t capacity, std::int64_t alignment) {
  auto t = std::make_unique<tree>();
  t->name = dataset;
  trees_[id] = std::move(t);
  tree& r = *trees_[id];
  esc::store_policy policy;
  (void)esc::store_policy::from_name(policy_, policy);  // checked by the constructor
  r.store = std::make_unique<esc::cbst_store>(res_, r.name, static_cast<int>(capacity),
                                              static_cast<int>(alignment), std::move(policy));
}

void device_store_adapter::destroy_tree(std::uint32_t id) {
  trees_.erase(id);
}

op_outcome device_store_adapter::run(std::uint32_t id, const op_request& op) {
  op_outcome out;
  const auto it = trees_.find(id);
  if (it == trees_.end()) {
    out.status = op_status::unsupported;
    out.message = "no such tree";
    return out;
  }
  esc::cbst_store& s = *it->second->store;
  const std::array<std::int64_t, 9> before = bins_now(s);
  try {
    if (op.op == "construct") {
      std::vector<int> keys(op.keys.begin(), op.keys.end());
      std::vector<int> starts(op.starts.begin(), op.starts.end());
      std::vector<int> flat(op.flat.begin(), op.flat.end());
      const int n = static_cast<int>(op.num_records);
      const int size = static_cast<int>(op.flat_size);
      // MOSP_ESCHER's caller row counts take the fast path of construct under a preset of its
      // behaviours (ESCHER-GPU had no counts and scanned); its nullptr (occupancy 0) is never
      // passed on: without counts the store scans.
      std::vector<int> counts(op.row_counts.begin(), op.row_counts.end());
      const bool with_counts = op.has_row_counts && policy_ != esc::store_policy::escher_gpu().name;
      if (with_counts)
        s.construct(keys.data(), starts.data(), n, flat.data(), size, counts.data());
      else
        s.construct(keys.data(), starts.data(), n, flat.data(), size);
      // Every construct is also built the other way (the counts when the store scanned: the
      // trace's, or the scan computed on the host; the scan when it took the counts) and the
      // two node arrays must be equal byte for byte (M8a step 4, behaviour (c)).
      if (n > 0) {
        esc::store_policy policy;
        (void)esc::store_policy::from_name(policy_, policy);
        esc::cbst_store twin(res_, it->second->name, s.context().fixed_size, s.context().alignment,
                             std::move(policy));
        if (with_counts) {
          twin.construct(keys.data(), starts.data(), n, flat.data(), size);
        } else {
          if (!op.has_row_counts) counts = scanned_counts(starts, flat, op.flat_size);
          twin.construct(keys.data(), starts.data(), n, flat.data(), size, counts.data());
        }
        ++twin_constructs_;
        all_twin_constructs.fetch_add(1);
        cuda_check(cudaDeviceSynchronize(), "after the twin construct");
        if (node_bytes(twin) != node_bytes(s)) {
          out.status = op_status::error;
          out.error_type = "device adapter: construct";
          out.message = "the node arrays built from the row counts and by the scan differ";
          return out;
        }
      }
    } else if (op.op == "rebuild") {
      const std::size_t n = static_cast<std::size_t>(std::max<std::int64_t>(op.num_records, 0));
      const std::size_t cap = static_cast<std::size_t>(std::max<std::int64_t>(op.capacity, 0));
      buffer<int> starts(res_, std::max<std::size_t>(n, 1));
      buffer<int> payload(res_, std::max<std::size_t>(cap, 1));
      int* d_starts = starts.data();
      int* d_flat = payload.data();
      if (n > 0)
        cuda_check(cudaMemcpy(d_starts, op.starts.data(), n * 4, cudaMemcpyHostToDevice),
                   "rebuild starts");
      cuda_check(cudaMemset(d_flat, 0, std::max<std::size_t>(cap, 1) * 4), "rebuild payload");
      if (!op.flat.empty())
        cuda_check(cudaMemcpy(d_flat, op.flat.data(), op.flat.size() * 4, cudaMemcpyHostToDevice),
                   "rebuild payload");
      s.rebuild_from_device(std::move(starts), static_cast<int>(op.num_records), std::move(payload),
                            static_cast<int>(op.flat_size), static_cast<int>(op.capacity));
    } else if (op.op == "insert") {
      const esc::insert_mapping m = s.insert(std::vector<int>(op.keys.begin(), op.keys.end()),
                                             std::vector<int>(op.payload.begin(), op.payload.end()),
                                             std::vector<int>(op.prefix.begin(), op.prefix.end()));
      out.mapping.assign(m.item_to_key.begin(), m.item_to_key.end());
    } else if (op.op == "fill") {
      s.append(std::vector<int>(op.keys.begin(), op.keys.end()),
               std::vector<int>(op.payload.begin(), op.payload.end()),
               std::vector<int>(op.prefix.begin(), op.prefix.end()));
    } else if (op.op == "erase") {
      s.erase(std::vector<int>(op.keys.begin(), op.keys.end()));
    } else if (op.op == "unfill") {
      s.remove_values(std::vector<int>(op.keys.begin(), op.keys.end()),
                      std::vector<int>(op.payload.begin(), op.payload.end()),
                      std::vector<int>(op.prefix.begin(), op.prefix.end()));
    } else {
      out.status = op_status::unsupported;
      out.message = "the store has no operation '" + op.op + "' under " + policy_;
      return out;
    }
    // The store's device error word (a broken chain, a dead or absent key; M8a step 4) after
    // every operation: an error is the operation's exception.
    s.check_errors();
    cuda_check(cudaDeviceSynchronize(), "after the operation");
  } catch (const std::exception& e) {
    out.status = op_status::error;
    out.error_type = demangle(typeid(e).name());
    out.message = e.what();
  }
  const std::array<std::int64_t, 9> after = bins_now(s);
  out.has_bins = true;
  for (std::size_t i = 0; i < 9; ++i) {
    out.bins[i] = after[i] - before[i];
    total_bins_[i] += out.bins[i];
  }
  return out;
}

tree_digest device_store_adapter::scalars(std::uint32_t id) {
  const auto it = trees_.find(id);
  if (it == trees_.end()) return {};
  return digest_of(read_state(*it->second->store, false), false);
}

tree_digest device_store_adapter::digest(std::uint32_t id) {
  const auto it = trees_.find(id);
  if (it == trees_.end()) return {};
  tree_digest d = digest_of(read_state(*it->second->store, true), true);
  d.integrity = integrity_of(*it->second->store, d);
  {
    std::ostringstream log;
    const esc::cbst_context& c = it->second->store->context();
    d.integrity_subtree = esc::check_subtree_avail(c, c.dataset_name.c_str(), log);
    d.integrity_index = esc::check_tree_index(c, c.dataset_name.c_str(), log);
  }
  auto fold = [this](const void* p, std::size_t n) {
    digest_hash_ = tr::fnv1a64(p, n, digest_hash_);
  };
  const std::int64_t scalars[] = {
      d.num_records,   d.fixed_size,        d.bump,           d.alignment,
      d.scratch_items, d.scratch_payload,   d.device_bytes,   d.d,
      d.integrity,     d.integrity_subtree, d.integrity_index};
  fold(scalars, sizeof scalars);
  fold(d.keys.data(), d.keys.size() * sizeof(std::int32_t));
  fold(d.subtree_avail.data(), d.subtree_avail.size() * sizeof(std::int32_t));
  for (const key_digest& k : d.records) {
    const std::int64_t f[] = {k.key,     k.avail,      k.length,
                              k.size,    k.occupancy,  k.tail_capacity,
                              k.value,   k.tail_base,  k.segments,
                              k.row_len, k.terminated, static_cast<std::int64_t>(k.row_hash)};
    fold(f, sizeof f);
  }
  return d;
}

store_factory device_store_factory(const std::string& policy) {
  return [policy](const std::string& repo) -> std::unique_ptr<store_adapter> {
    (void)repo;
    return std::make_unique<device_store_adapter>(policy.empty() ? "escher_gpu" : policy);
  };
}

bool device_available() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return n > 0;
}

}  // namespace dyng::test::cbst_compare
