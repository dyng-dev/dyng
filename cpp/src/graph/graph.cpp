// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file graph.cpp
 * @brief graph<V,E,W> member functions and their explicit instantiations.
 */
#include "graph/apply_host.hpp"
#include "graph/graph_impl.hpp"
#include "graph/instantiate.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/graph/graph.hpp>

#include <memory>
#include <string>
#include <utility>

namespace dyng {

namespace {

void expect_host_backend(const resources& res, const char* what) {
  if (res.get_backend() == backend::cuda) {
    throw not_supported_error(std::string("dyng: ") + what +
                              ": the device-resident graph arrives with the CUDA backend (M1b); "
                              "use resources::sequential() or resources::openmp()");
  }
}

void expect_supported_layout(const graph_properties& props) {
  if (props.layout != row_layout::compact) {
    throw not_supported_error(
        "dyng: only row_layout::compact is implemented; slotted and slack arrive in 0.3");
  }
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
graph<vertex_t, edge_t, weight_t>::graph(const graph_properties& props)
    : impl_(std::make_unique<impl_type>()) {
  expect_supported_layout(props);
  DYNG_EXPECTS(props.num_weights >= 0, "graph_properties::num_weights must be >= 0, got ",
               props.num_weights);
  impl_->props = props;
  impl_->out.row_ptr.assign(1, edge_t{0});
  impl_->out.num_weights = props.num_weights;
  if (props.store_transposed) {
    impl_->in = impl_->out;
  }
}

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
    const resources& res, edge_list_view<vertex_t, weight_t> edges, const graph_properties& props) {
  expect_host_backend(res, "graph::from_edges");
  expect_supported_layout(props);
  auto impl = std::make_unique<impl_type>();
  impl->props = props;
  detail::build_from_edges_host(edges, props, impl->out);
  impl->props.num_weights = impl->out.num_weights;
  if (props.store_transposed) {
    detail::transpose_host(impl->out, impl->in);
  }
  return graph(std::move(impl));
}

template <typename vertex_t, typename edge_t, typename weight_t>
graph<vertex_t, edge_t, weight_t> graph<vertex_t, edge_t, weight_t>::from_csr(
    const resources& res, csr_view<vertex_t, edge_t, weight_t> csr, const graph_properties& props) {
  expect_host_backend(res, "graph::from_csr");
  expect_supported_layout(props);
  auto impl = std::make_unique<impl_type>();
  impl->props = props;
  detail::build_from_csr_host(csr, props, impl->out);
  impl->props.num_weights = impl->out.num_weights;
  if (props.store_transposed) {
    detail::transpose_host(impl->out, impl->in);
  }
  return graph(std::move(impl));
}

template <typename vertex_t, typename edge_t, typename weight_t>
graph<vertex_t, edge_t, weight_t> graph<vertex_t, edge_t, weight_t>::clone(
    const resources& res) const {
  expect_host_backend(res, "graph::clone");
  return graph(std::make_unique<impl_type>(impl()));
}

template <typename vertex_t, typename edge_t, typename weight_t>
void graph<vertex_t, edge_t, weight_t>::reserve(const resources& res, edge_t edge_capacity) {
  expect_host_backend(res, "graph::reserve");
  DYNG_EXPECTS(edge_capacity >= 0, "graph::reserve: negative capacity ", edge_capacity);
  auto& state = impl();
  const auto m = static_cast<std::size_t>(edge_capacity);
  const auto k = static_cast<std::size_t>(state.out.num_weights);
  state.out.col_ind.reserve(m);
  state.out.weights.reserve(m * k);
  if (state.props.store_transposed) {
    state.in.col_ind.reserve(m);
    state.in.weights.reserve(m * k);
  }
}

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
  return memory_space::host;
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::uint64_t graph<vertex_t, edge_t, weight_t>::version() const noexcept {
  return impl_ ? impl_->version : 0;
}

template <typename vertex_t, typename edge_t, typename weight_t>
typename graph<vertex_t, edge_t, weight_t>::view_type graph<vertex_t, edge_t, weight_t>::view()
    const {
  const auto& state = impl();
  view_type out;
  out.out = state.out.view();
  if (state.props.store_transposed) {
    out.in = state.in.view();
  }
  out.has_transposed = state.props.store_transposed;
  out.directed = state.props.directed;
  out.layout = state.props.layout;
  out.version = state.version;
  return out;
}

template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary graph<vertex_t, edge_t, weight_t>::apply(
    const resources& res, const edge_batch_view<vertex_t, weight_t>& batch) {
  return detail::graph_access::apply(res, *this, batch,
                                     static_cast<detail::apply_delta<vertex_t>*>(nullptr));
}

template <typename vertex_t, typename edge_t, typename weight_t>
typename graph<vertex_t, edge_t, weight_t>::csr_type graph<vertex_t, edge_t, weight_t>::to_csr(
    const resources& res) const {
  expect_host_backend(res, "graph::to_csr");
  return impl().out;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void graph<vertex_t, edge_t, weight_t>::check_integrity(const resources& res) const {
  expect_host_backend(res, "graph::check_integrity");
  const std::string violation = detail::integrity_violation(impl());
  if (!violation.empty()) {
    DYNG_FAIL("graph integrity: ", violation);
  }
}

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary graph_access::apply(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                                  const edge_batch_view<vertex_t, weight_t>& batch,
                                  apply_delta<vertex_t>* delta) {
  expect_host_backend(res, "graph::apply");
  auto& state = g.impl();
  scoped_stage stage(res, "graph.apply");
  csr<vertex_t, edge_t, weight_t> updated;
  const apply_summary summary = apply_batch_host(state.out, batch, state.props, updated, delta);
  state.out = std::move(updated);
  if (state.props.store_transposed) {
    transpose_host(state.out, state.in);
  }
  ++state.version;
  return summary;
}

}  // namespace detail

#define DYNG_INSTANTIATE_GRAPH(V, E, W)                        \
  template class graph<V, E, W>;                               \
  template apply_summary detail::graph_access::apply<V, E, W>( \
      const resources&, graph<V, E, W>&, const edge_batch_view<V, W>&, detail::apply_delta<V>*);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_GRAPH)
#undef DYNG_INSTANTIATE_GRAPH

}  // namespace dyng
