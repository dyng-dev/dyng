// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file graph.cpp
 * @brief graph<V,E,W> member functions and their explicit instantiations.
 */
#include "core/budget_counters.hpp"
#include "core/resources_access.hpp"
#include "core/staging.hpp"
#include "graph/apply_common.hpp"
#include "graph/apply_host.hpp"
#include "graph/graph_impl.hpp"
#include "graph/instantiate.hpp"
#include "util/allocation.hpp"

#include <dyng/config.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/graph/graph.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>

namespace dyng {

namespace {

/// Threads for the host-side work (transposition, apply, CSR checks).
int host_threads(const resources& res) {
  return detail::resources_access::host_threads(res);
}

/// Record the placement of a new graph (PLAN Section 4.6 rule 5).
template <typename impl_t>
void set_home(impl_t& impl, const resources& res) {
  impl.home = res.get_backend();
  impl.home_device = res.get_backend() == backend::cuda ? res.device() : -1;
}

void expect_supported_layout(const graph_properties& props) {
  if (props.layout != row_layout::compact) {
    throw not_supported_error(
        "dyng: only row_layout::compact is implemented; slotted and slack arrive in 0.4");
  }
  detail::expect_supported_semantics(props);  // e.g. as_sets with upsert: fail here, not at apply
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
graph<vertex_t, edge_t, weight_t>::graph(const graph_properties& props) try
    : impl_(std::make_unique<impl_type>()) {
  expect_supported_layout(props);
  DYNG_EXPECTS(props.num_weights >= 0, "graph_properties::num_weights must be >= 0, got ",
               props.num_weights);
  impl_->props = props;
  impl_->host_edges_for_write().row_ptr.assign(1, edge_t{0});
  impl_->host_edges_for_write().num_weights = props.num_weights;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("graph (empty graph)")

template <typename vertex_t, typename edge_t, typename weight_t>
graph<vertex_t, edge_t, weight_t>::graph(std::unique_ptr<impl_type> impl) noexcept
    : impl_(std::move(impl)) {}

template <typename vertex_t, typename edge_t, typename weight_t>
graph<vertex_t, edge_t, weight_t>::graph(graph&& other) noexcept = default;

template <typename vertex_t, typename edge_t, typename weight_t>
graph<vertex_t, edge_t, weight_t>& graph<vertex_t, edge_t, weight_t>::operator=(
    graph&& other) noexcept = default;

template <typename vertex_t, typename edge_t, typename weight_t>
graph<vertex_t, edge_t, weight_t>::~graph() = default;

template <typename vertex_t, typename edge_t, typename weight_t>
typename graph<vertex_t, edge_t, weight_t>::impl_type& graph<vertex_t, edge_t, weight_t>::impl() {
  DYNG_EXPECTS(impl_ != nullptr, "use of a moved-from graph");
  return *impl_;
}

template <typename vertex_t, typename edge_t, typename weight_t>
const typename graph<vertex_t, edge_t, weight_t>::impl_type&
graph<vertex_t, edge_t, weight_t>::impl() const {
  DYNG_EXPECTS(impl_ != nullptr, "use of a moved-from graph");
  return *impl_;
}

template <typename vertex_t, typename edge_t, typename weight_t>
graph<vertex_t, edge_t, weight_t> graph<vertex_t, edge_t, weight_t>::from_edges(
    const resources& res, edge_list_view<vertex_t, weight_t> edges,
    const graph_properties& props) try {
  expect_supported_layout(props);
  scoped_stage stage(res, "graph.build");
  auto impl = std::make_unique<impl_type>();
  impl->props = props;
  impl->build_threads = host_threads(res);
  set_home(*impl, res);
  // Built on the host on every backend: arrays in device memory are copied once (copy policy).
  const detail::host_input<vertex_t> src(res, edges.src, "graph::from_edges: edge_list_view::src");
  const detail::host_input<vertex_t> dst(res, edges.dst, "graph::from_edges: edge_list_view::dst");
  const detail::host_input<weight_t> weights(res, edges.weights,
                                             "graph::from_edges: edge_list_view::weights");
  edge_list_view<vertex_t, weight_t> host_edges = edges;
  host_edges.src = src.view();
  host_edges.dst = dst.view();
  host_edges.weights = weights.view();
  detail::build_from_edges_host(host_edges, props, impl->host_edges_for_write());
  impl->props.num_weights = impl->num_weights();
  return graph(std::move(impl));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("graph::from_edges (", edges.num_vertices, " vertices, ",
                                  edges.num_edges(), " edges)")

template <typename vertex_t, typename edge_t, typename weight_t>
graph<vertex_t, edge_t, weight_t> graph<vertex_t, edge_t, weight_t>::from_csr(
    const resources& res, csr_view<vertex_t, edge_t, weight_t> csr,
    const graph_properties& props) try {
  expect_supported_layout(props);
  scoped_stage stage(res, "graph.build");
  auto impl = std::make_unique<impl_type>();
  impl->props = props;
  impl->build_threads = host_threads(res);
  set_home(*impl, res);
  // Built on the host on every backend: arrays in device memory are copied once (copy policy).
  const detail::host_input<edge_t> row_ptr(res, csr.row_ptr, "graph::from_csr: csr_view::row_ptr");
  const detail::host_input<vertex_t> col_ind(res, csr.col_ind,
                                             "graph::from_csr: csr_view::col_ind");
  const detail::host_input<weight_t> weights(res, csr.weights,
                                             "graph::from_csr: csr_view::weights");
  csr_view<vertex_t, edge_t, weight_t> host_csr = csr;
  host_csr.row_ptr = row_ptr.view();
  host_csr.col_ind = col_ind.view();
  host_csr.weights = weights.view();
  detail::build_from_csr_host(host_csr, props, impl->host_edges_for_write(), impl->build_threads);
  impl->props.num_weights = impl->num_weights();
  return graph(std::move(impl));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("graph::from_csr (", csr.num_vertices(), " vertices, ",
                                  csr.num_edges(), " edges)")

template <typename vertex_t, typename edge_t, typename weight_t>
graph<vertex_t, edge_t, weight_t> graph<vertex_t, edge_t, weight_t>::from_csr(
    const resources& res, csr_type&& csr, const graph_properties& props) try {
  expect_supported_layout(props);
  scoped_stage stage(res, "graph.build");
  auto impl = std::make_unique<impl_type>();
  impl->props = props;
  impl->build_threads = host_threads(res);
  set_home(*impl, res);
  detail::build_from_csr_host(csr.view(), props, impl->host_edges_for_write(), impl->build_threads,
                              &csr);
  impl->props.num_weights = impl->num_weights();
  return graph(std::move(impl));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("graph::from_csr (", csr.num_vertices(), " vertices, ",
                                  csr.num_edges(), " edges)")

template <typename vertex_t, typename edge_t, typename weight_t>
graph<vertex_t, edge_t, weight_t> graph<vertex_t, edge_t, weight_t>::clone(
    const resources& res) const try {
  auto copy = std::make_unique<impl_type>(impl());
  copy->build_threads = host_threads(res);
  set_home(*copy, res);  // the copy belongs to the backend of `res` (its device copy is rebuilt)
  return graph(std::move(copy));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("graph::clone (", num_vertices(), " vertices, ", num_edges(),
                                  " edges)")

template <typename vertex_t, typename edge_t, typename weight_t>
graph<vertex_t, edge_t, weight_t> graph<vertex_t, edge_t, weight_t>::to_backend(
    const resources& res) const {
  return clone(res);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void graph<vertex_t, edge_t, weight_t>::reserve(const resources& res, edge_t edge_capacity) try {
  (void)res;  // the host storage (also of a CUDA graph, whose device copy is rebuilt per state)
  DYNG_EXPECTS(edge_capacity >= 0, "graph::reserve: negative capacity ", edge_capacity);
  auto& state = impl();
  const auto m = static_cast<std::size_t>(edge_capacity);
  auto& out = state.host_edges_for_write();
  const auto k = static_cast<std::size_t>(out.num_weights);
  out.col_ind.reserve(m);
  out.weights.reserve(m * k);
  if (state.props.store_transposed) {
    state.in_storage().col_ind.reserve(m);
    state.in_storage().weights.reserve(m * k);
  }
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("graph::reserve (", edge_capacity, " edges)")

template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t graph<vertex_t, edge_t, weight_t>::num_vertices() const noexcept {
  return impl_ ? impl_->num_vertices() : vertex_t{0};
}

template <typename vertex_t, typename edge_t, typename weight_t>
edge_t graph<vertex_t, edge_t, weight_t>::num_edges() const noexcept {
  return impl_ ? impl_->num_edges() : edge_t{0};
}

template <typename vertex_t, typename edge_t, typename weight_t>
int graph<vertex_t, edge_t, weight_t>::num_weights() const noexcept {
  return impl_ ? impl_->num_weights() : 0;
}

template <typename vertex_t, typename edge_t, typename weight_t>
const graph_properties& graph<vertex_t, edge_t, weight_t>::properties() const noexcept {
  static const graph_properties moved_from{};
  return impl_ ? impl_->props : moved_from;
}

template <typename vertex_t, typename edge_t, typename weight_t>
bool graph<vertex_t, edge_t, weight_t>::is_directed() const noexcept {
  return properties().directed;
}

template <typename vertex_t, typename edge_t, typename weight_t>
bool graph<vertex_t, edge_t, weight_t>::has_transposed() const noexcept {
  return properties().store_transposed;
}

template <typename vertex_t, typename edge_t, typename weight_t>
memory_space graph<vertex_t, edge_t, weight_t>::space() const noexcept {
  return impl_ && impl_->home == backend::cuda ? memory_space::device : memory_space::host;
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::uint64_t graph<vertex_t, edge_t, weight_t>::version() const noexcept {
  return impl_ ? impl_->version : 0;
}

template <typename vertex_t, typename edge_t, typename weight_t>
typename graph<vertex_t, edge_t, weight_t>::view_type graph<vertex_t, edge_t, weight_t>::view()
    const try {
  const auto& state = impl();
  view_type out;
  out.out = state.host_edges().view();
  if (state.props.store_transposed) {
    out.in = state.in_edges(state.build_threads).view();
  }
  out.has_transposed = state.props.store_transposed;
  out.directed = state.props.directed;
  out.layout = state.props.layout;
  out.version = state.version;
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("graph::view (the in-edges of ", num_edges(), " edges)")

template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary graph<vertex_t, edge_t, weight_t>::apply(
    const resources& res, const edge_batch_view<vertex_t, weight_t>& batch) {
  // Applied on the host on every backend in this release: device arrays are copied once.
  const detail::host_batch<vertex_t, weight_t> staged(res, batch, "graph::apply");
  return detail::graph_access::apply(res, *this, staged.view(),
                                     static_cast<detail::apply_delta<vertex_t>*>(nullptr));
}

template <typename vertex_t, typename edge_t, typename weight_t>
typename graph<vertex_t, edge_t, weight_t>::csr_type graph<vertex_t, edge_t, weight_t>::to_csr(
    const resources& res) const try {
  (void)res;  // the host CSR (downloaded first if a device apply left it stale)
  return impl().host_edges();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("graph::to_csr (", num_edges(), " edges)")

template <typename vertex_t, typename edge_t, typename weight_t>
void graph<vertex_t, edge_t, weight_t>::check_integrity(const resources& res) const try {
  (void)res;  // checks the host storage (downloaded first if a device apply left it stale)
  (void)impl().host_edges();
  const std::string violation = detail::integrity_violation(impl());
  if (!violation.empty()) {
    DYNG_FAIL("graph integrity: ", violation);
  }
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("graph::check_integrity")

namespace detail {

#if !DYNG_HAS_CUDA
namespace {
[[noreturn]] void throw_no_cuda() {
  throw not_supported_error(
      "dyng: the cuda backend is not built (configure with "
      "DYNG_ENABLE_CUDA=ON)");
}
}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
void build_device_graph(const resources& /*res*/, const csr<vertex_t, edge_t, weight_t>& /*host*/,
                        device_graph<vertex_t, edge_t, weight_t>& /*out*/) {
  throw_no_cuda();
}

template <typename vertex_t, typename edge_t, typename weight_t>
void build_device_in_edges(const resources& /*res*/,
                           device_graph<vertex_t, edge_t, weight_t>& /*g*/) {
  throw_no_cuda();
}

template <typename vertex_t, typename edge_t, typename weight_t>
void download_device_graph(const device_graph<vertex_t, edge_t, weight_t>& /*g*/,
                           csr<vertex_t, edge_t, weight_t>& /*host*/) {
  throw_no_cuda();
}

template <typename vertex_t, typename edge_t, typename weight_t>
void apply_set_batch_device(const resources& /*res*/,
                            const device_graph<vertex_t, edge_t, weight_t>& /*base*/,
                            const normalized_batch<vertex_t>& /*normalized*/,
                            device_graph<vertex_t, edge_t, weight_t>& /*next*/) {
  throw_no_cuda();
}

template <typename vertex_t>
const std::uint32_t* upload_normalized_batch(const resources& /*res*/,
                                             const normalized_batch<vertex_t>& /*nb*/) {
  throw_no_cuda();
}
template const std::uint32_t* upload_normalized_batch<std::int32_t>(
    const resources&, const normalized_batch<std::int32_t>&);
#endif

std::uint64_t next_graph_state_id() noexcept {
  // Starts at 1, so the 0 of a moved-from graph or result never matches.
  static std::atomic<std::uint64_t> counter{0};
  return counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary graph_access::apply(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                                  const edge_batch_view<vertex_t, weight_t>& batch,
                                  apply_delta<vertex_t>* delta,
                                  const normalized_batch<vertex_t>* normalized) try {
  auto& state = g.impl();
  scoped_stage stage(res, "graph.apply");
  if constexpr (device_set_apply_supported_v<vertex_t>) {
    // A resident CUDA graph under set semantics without weight columns: merge the batch on the
    // device and keep the graph there (CycleEnumeration-GPU's build_next_rows_kernel; ADR 0020).
    if (state.home == backend::cuda && res.get_backend() == backend::cuda &&
        res.device() == state.home_device && state.has_device_edges() &&
        state.props.semantics.as_sets && state.num_weights() == 0) {
      normalized_batch<vertex_t> local;
      if (normalized == nullptr) {
        normalize(res, g, batch, local);
        normalized = &local;
      }
      auto next = std::make_unique<device_graph<vertex_t, edge_t, weight_t>>();
      apply_set_batch_device(res, state.device_edges(res, false), *normalized, *next);
      if (delta != nullptr) {
        fill_set_delta(*normalized, *delta);
      }
      state.replace_device_edges(std::move(next));
      ++state.version;
      state.state_id = next_graph_state_id();
      return normalized->summary;
    }
  }
  csr<vertex_t, edge_t, weight_t> updated;
  const apply_summary summary = apply_batch_host(state.host_edges(), batch, state.props, updated,
                                                 delta, host_threads(res), normalized);
  state.host_edges_for_write() = std::move(updated);
  state.drop_in_edges();      // rebuilt on first use (graph_access::view, graph::view)
  state.drop_device_edges();  // uploaded again on first use (graph_access::device)
  ++state.version;
  state.state_id = next_graph_state_id();
  return summary;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("graph::apply (", g.num_vertices(), " vertices, ", g.num_edges(),
                                  " edges; batch of ", batch.num_insertions(), " insertions, ",
                                  batch.num_deletions(), " deletions)")

template <typename vertex_t, typename edge_t, typename weight_t>
void graph_access::normalize(const resources& res, const graph<vertex_t, edge_t, weight_t>& g,
                             const edge_batch_view<vertex_t, weight_t>& batch,
                             normalized_batch<vertex_t>& out) {
  const auto& state = g.impl();
  if constexpr (device_set_apply_supported_v<vertex_t>) {
    if (!state.host_current() && state.home == backend::cuda &&
        res.get_backend() == backend::cuda && res.device() == state.home_device &&
        state.has_device_edges()) {
      normalize_set_batch_device(res, state.device_edges(res, false), batch, state.props, out);
      out.state_id = state.state_id;
      return;
    }
  }
  normalize_set_batch(state.host_edges(), batch, state.props, out);
  out.state_id = state.state_id;
}

template <typename vertex_t, typename edge_t, typename weight_t>
graph_impl<vertex_t, edge_t, weight_t>::graph_impl(const graph_impl& other)
    : props(other.props),
      version(other.version),
      state_id(other.state_id),
      build_threads(other.build_threads),
      home(other.home),
      home_device(other.home_device),
      out_(other.host_edges()) {
  const std::lock_guard<std::mutex> lock(other.in_mutex_);
  if (other.in_built_.load(std::memory_order_acquire)) {
    in_ = other.in_;
    in_built_.store(true, std::memory_order_release);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
const typename graph_impl<vertex_t, edge_t, weight_t>::csr_type&
graph_impl<vertex_t, edge_t, weight_t>::in_edges(int threads) const {
  if (!in_built_.load(std::memory_order_acquire)) {
    const std::lock_guard<std::mutex> lock(in_mutex_);
    if (!in_built_.load(std::memory_order_relaxed)) {
      transpose_host(host_edges(), in_, threads);
      in_built_.store(true, std::memory_order_release);
    }
  }
  return in_;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void graph_impl<vertex_t, edge_t, weight_t>::drop_in_edges() noexcept {
  in_built_.store(false, std::memory_order_release);
  in_ = csr_type();
}

template <typename vertex_t, typename edge_t, typename weight_t>
const typename graph_impl<vertex_t, edge_t, weight_t>::device_type&
graph_impl<vertex_t, edge_t, weight_t>::device_edges(const resources& res, bool in_edges) const {
  if (!device_built_.load(std::memory_order_acquire) ||
      (in_edges && !device_in_built_.load(std::memory_order_acquire))) {
    const std::lock_guard<std::mutex> lock(device_mutex_);
    const container_scope container;  // the graph's own work (budgets, I9)
    if (!device_built_.load(std::memory_order_relaxed)) {
      scoped_stage stage(res, "graph.upload");
      auto built = std::make_unique<device_type>();
      build_device_graph(res, host_edges(), *built);
      // Complete before any stream reads it (read-only calls may run concurrently on other
      // streams; MOSP-CUDA's uploadDeviceGraph ends with a synchronization as well).
      res.synchronize();
      device_ = std::move(built);
      device_built_.store(true, std::memory_order_release);
    }
    if (in_edges && !device_in_built_.load(std::memory_order_relaxed)) {
      scoped_stage stage(res, "graph.transpose_device");
      build_device_in_edges(res, *device_);
      res.synchronize();
      device_in_built_.store(true, std::memory_order_release);
    }
  }
  return *device_;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void graph_impl<vertex_t, edge_t, weight_t>::drop_device_edges() noexcept {
  device_built_.store(false, std::memory_order_release);
  device_in_built_.store(false, std::memory_order_release);
  device_.reset();
}

template <typename vertex_t, typename edge_t, typename weight_t>
void graph_impl<vertex_t, edge_t, weight_t>::replace_device_edges(
    std::unique_ptr<device_type> next) noexcept {
  host_current_.store(false, std::memory_order_release);
  drop_in_edges();
  device_ = std::move(next);
  device_in_built_.store(device_->has_in_edges, std::memory_order_release);
  device_built_.store(true, std::memory_order_release);
}

template <typename vertex_t, typename edge_t, typename weight_t>
const typename graph_impl<vertex_t, edge_t, weight_t>::csr_type&
graph_impl<vertex_t, edge_t, weight_t>::host_edges() const {
  if (!host_current_.load(std::memory_order_acquire)) {
    const std::lock_guard<std::mutex> lock(host_mutex_);
    if (!host_current_.load(std::memory_order_relaxed)) {
      const container_scope container;  // the graph's own work (budgets, I9)
      download_device_graph(*device_, out_);
      host_current_.store(true, std::memory_order_release);
    }
  }
  return out_;
}

template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t graph_impl<vertex_t, edge_t, weight_t>::num_vertices() const noexcept {
  return host_current_.load(std::memory_order_acquire) ? out_.num_vertices()
                                                       : device_->num_vertices;
}

template <typename vertex_t, typename edge_t, typename weight_t>
edge_t graph_impl<vertex_t, edge_t, weight_t>::num_edges() const noexcept {
  return host_current_.load(std::memory_order_acquire) ? out_.num_edges() : device_->num_edges;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void graph_access::expect_placement(const resources& res,
                                    const graph<vertex_t, edge_t, weight_t>& g, const char* what) {
  const auto& state = g.impl();
  const backend b = res.get_backend();
  if (b == backend::cuda) {
    DYNG_EXPECTS(state.home == backend::cuda && state.home_device == res.device(), what,
                 ": the graph was built for ", to_string(state.home),
                 state.home == backend::cuda ? " on CUDA device " : "",
                 state.home == backend::cuda ? std::to_string(state.home_device) : std::string(),
                 " but the resources are cuda on device ", res.device(),
                 "; build the graph with these resources or move it with g.to_backend(res)");
  } else {
    DYNG_EXPECTS(state.home != backend::cuda, what, ": the graph is resident on CUDA device ",
                 state.home_device, " but the resources are ", to_string(b),
                 "; move it with g.to_backend(res)");
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
const device_graph<vertex_t, edge_t, weight_t>& graph_access::device(
    const resources& res, const graph<vertex_t, edge_t, weight_t>& g) {
  expect_placement(res, g, "graph (device copy)");
  return g.impl().device_edges(res);
}

template <typename vertex_t, typename edge_t, typename weight_t>
const device_graph<vertex_t, edge_t, weight_t>& graph_access::device_out(
    const resources& res, const graph<vertex_t, edge_t, weight_t>& g) {
  expect_placement(res, g, "graph (device copy)");
  return g.impl().device_edges(res, false);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void graph_access::prepare(const resources& res, const graph<vertex_t, edge_t, weight_t>& g) {
  if (res.get_backend() == backend::cuda) {
    (void)device(res, g);
  } else {
    (void)view(res, g);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
graph_view<vertex_t, edge_t, weight_t> graph_access::view(
    const resources& res, const graph<vertex_t, edge_t, weight_t>& g) {
  const auto& state = g.impl();
  if (state.props.store_transposed && !state.has_in_edges()) {
    scoped_stage stage(res, "graph.transpose");
    (void)state.in_edges(host_threads(res));
  }
  return g.view();
}

}  // namespace detail

#define DYNG_INSTANTIATE_GRAPH(V, E, W)                                                           \
  template class detail::graph_impl<V, E, W>;                                                     \
  template graph_view<V, E, W> detail::graph_access::view<V, E, W>(const resources&,              \
                                                                   const graph<V, E, W>&);        \
  template const detail::device_graph<V, E, W>& detail::graph_access::device<V, E, W>(            \
      const resources&, const graph<V, E, W>&);                                                   \
  template const detail::device_graph<V, E, W>& detail::graph_access::device_out<V, E, W>(        \
      const resources&, const graph<V, E, W>&);                                                   \
  template void detail::graph_access::prepare<V, E, W>(const resources&, const graph<V, E, W>&);  \
  template void detail::graph_access::normalize<V, E, W>(const resources&, const graph<V, E, W>&, \
                                                         const edge_batch_view<V, W>&,            \
                                                         detail::normalized_batch<V>&);           \
  template void detail::graph_access::expect_placement<V, E, W>(                                  \
      const resources&, const graph<V, E, W>&, const char*);                                      \
  template class graph<V, E, W>;                                                                  \
  template apply_summary detail::graph_access::apply<V, E, W>(                                    \
      const resources&, graph<V, E, W>&, const edge_batch_view<V, W>&, detail::apply_delta<V>*,   \
      const detail::normalized_batch<V>*);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_GRAPH)
DYNG_FOR_EACH_UNWEIGHTED_GRAPH_TYPE(DYNG_INSTANTIATE_GRAPH)
#undef DYNG_INSTANTIATE_GRAPH

}  // namespace dyng
