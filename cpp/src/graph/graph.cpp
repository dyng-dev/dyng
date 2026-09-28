// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file graph.cpp
 * @brief graph<V,E,W> member functions and their explicit instantiations.
 */
#include "core/resources_access.hpp"
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
        "dyng: only row_layout::compact is implemented; slotted and slack arrive in 0.3");
  }
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
graph<vertex_t, edge_t, weight_t>::graph(const graph_properties& props) try
    : impl_(std::make_unique<impl_type>()) {
  expect_supported_layout(props);
  DYNG_EXPECTS(props.num_weights >= 0, "graph_properties::num_weights must be >= 0, got ",
               props.num_weights);
  impl_->props = props;
  impl_->out.row_ptr.assign(1, edge_t{0});
  impl_->out.num_weights = props.num_weights;
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
  detail::build_from_edges_host(edges, props, impl->out);
  impl->props.num_weights = impl->out.num_weights;
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
  detail::build_from_csr_host(csr, props, impl->out, impl->build_threads);
  impl->props.num_weights = impl->out.num_weights;
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
  detail::build_from_csr_host(csr.view(), props, impl->out, impl->build_threads, &csr);
  impl->props.num_weights = impl->out.num_weights;
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
void graph<vertex_t, edge_t, weight_t>::reserve(const resources& res, edge_t edge_capacity) try {
  (void)res;  // the host storage (also of a CUDA graph, whose device copy is rebuilt per state)
  DYNG_EXPECTS(edge_capacity >= 0, "graph::reserve: negative capacity ", edge_capacity);
  auto& state = impl();
  const auto m = static_cast<std::size_t>(edge_capacity);
  const auto k = static_cast<std::size_t>(state.out.num_weights);
  state.out.col_ind.reserve(m);
  state.out.weights.reserve(m * k);
  if (state.props.store_transposed) {
    state.in_storage().col_ind.reserve(m);
    state.in_storage().weights.reserve(m * k);
  }
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("graph::reserve (", edge_capacity, " edges)")

template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t graph<vertex_t, edge_t, weight_t>::num_vertices() const noexcept {
  return impl_ ? impl_->out.num_vertices() : vertex_t{0};
}

template <typename vertex_t, typename edge_t, typename weight_t>
edge_t graph<vertex_t, edge_t, weight_t>::num_edges() const noexcept {
  return impl_ ? impl_->out.num_edges() : edge_t{0};
}

template <typename vertex_t, typename edge_t, typename weight_t>
int graph<vertex_t, edge_t, weight_t>::num_weights() const noexcept {
  return impl_ ? impl_->out.num_weights : 0;
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
  out.out = state.out.view();
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
  return detail::graph_access::apply(res, *this, batch,
                                     static_cast<detail::apply_delta<vertex_t>*>(nullptr));
}

template <typename vertex_t, typename edge_t, typename weight_t>
typename graph<vertex_t, edge_t, weight_t>::csr_type graph<vertex_t, edge_t, weight_t>::to_csr(
    const resources& res) const try {
  (void)res;  // the host CSR is the authoritative copy on every backend in this release
  return impl().out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("graph::to_csr (", num_edges(), " edges)")

template <typename vertex_t, typename edge_t, typename weight_t>
void graph<vertex_t, edge_t, weight_t>::check_integrity(const resources& res) const try {
  (void)res;  // checks the host storage
  const std::string violation = detail::integrity_violation(impl());
  if (!violation.empty()) {
    DYNG_FAIL("graph integrity: ", violation);
  }
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("graph::check_integrity")

namespace detail {

#if !DYNG_HAS_CUDA
template <typename vertex_t, typename edge_t, typename weight_t>
void build_device_graph(const resources& /*res*/, const csr<vertex_t, edge_t, weight_t>& /*host*/,
                        device_graph<vertex_t, edge_t, weight_t>& /*out*/) {
  throw not_supported_error(
      "dyng: the cuda backend is not built (configure with "
      "DYNG_ENABLE_CUDA=ON)");
}
#endif

std::uint64_t next_graph_state_id() noexcept {
  // Starts at 1, so the 0 of a moved-from graph or result never matches.
  static std::atomic<std::uint64_t> counter{0};
  return counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary graph_access::apply(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                                  const edge_batch_view<vertex_t, weight_t>& batch,
                                  apply_delta<vertex_t>* delta) try {
  auto& state = g.impl();
  scoped_stage stage(res, "graph.apply");
  csr<vertex_t, edge_t, weight_t> updated;
  const apply_summary summary =
      apply_batch_host(state.out, batch, state.props, updated, delta, host_threads(res));
  state.out = std::move(updated);
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
graph_impl<vertex_t, edge_t, weight_t>::graph_impl(const graph_impl& other)
    : props(other.props),
      version(other.version),
      state_id(other.state_id),
      out(other.out),
      build_threads(other.build_threads),
      home(other.home),
      home_device(other.home_device) {
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
      transpose_host(out, in_, threads);
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
graph_impl<vertex_t, edge_t, weight_t>::device_edges(const resources& res) const {
  if (!device_built_.load(std::memory_order_acquire)) {
    const std::lock_guard<std::mutex> lock(device_mutex_);
    if (!device_built_.load(std::memory_order_relaxed)) {
      scoped_stage stage(res, "graph.upload");
      auto built = std::make_unique<device_type>();
      build_device_graph(res, out, *built);
      device_ = std::move(built);
      device_built_.store(true, std::memory_order_release);
    }
  }
  return *device_;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void graph_impl<vertex_t, edge_t, weight_t>::drop_device_edges() noexcept {
  device_built_.store(false, std::memory_order_release);
  device_.reset();
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
                 "; build the graph with these resources or copy it with g.clone(res)");
  } else {
    DYNG_EXPECTS(state.home != backend::cuda, what, ": the graph is resident on CUDA device ",
                 state.home_device, " but the resources are ", to_string(b),
                 "; copy it with g.clone(res)");
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
const device_graph<vertex_t, edge_t, weight_t>& graph_access::device(
    const resources& res, const graph<vertex_t, edge_t, weight_t>& g) {
  expect_placement(res, g, "graph (device copy)");
  return g.impl().device_edges(res);
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

#define DYNG_INSTANTIATE_GRAPH(V, E, W)                                                          \
  template class detail::graph_impl<V, E, W>;                                                    \
  template graph_view<V, E, W> detail::graph_access::view<V, E, W>(const resources&,             \
                                                                   const graph<V, E, W>&);       \
  template const detail::device_graph<V, E, W>& detail::graph_access::device<V, E, W>(           \
      const resources&, const graph<V, E, W>&);                                                  \
  template void detail::graph_access::prepare<V, E, W>(const resources&, const graph<V, E, W>&); \
  template void detail::graph_access::expect_placement<V, E, W>(                                 \
      const resources&, const graph<V, E, W>&, const char*);                                     \
  template class graph<V, E, W>;                                                                 \
  template apply_summary detail::graph_access::apply<V, E, W>(                                   \
      const resources&, graph<V, E, W>&, const edge_batch_view<V, W>&, detail::apply_delta<V>*);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_GRAPH)
DYNG_FOR_EACH_UNWEIGHTED_GRAPH_TYPE(DYNG_INSTANTIATE_GRAPH)
#undef DYNG_INSTANTIATE_GRAPH

}  // namespace dyng
