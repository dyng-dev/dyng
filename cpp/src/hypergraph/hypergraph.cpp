// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/dynamic_hypergraph.cu (update: the erase_first order)
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/DynamicHypergraph.cpp (beginBatch,
// finishBatch: the insert_first order)
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/HostHypergraph.cpp (applyBatch: the skip rules,
// the phases of the incidence edits)
/**
 * @file hypergraph.cpp
 * @brief hypergraph<V,W> member functions, hypergraph_access, and their explicit instantiations
 *        (ADR 0041).
 */
#include "core/budget_counters.hpp"
#include "core/resources_access.hpp"
#include "core/staging.hpp"
#include "hypergraph/capacity.hpp"
#include "hypergraph/escher_store.hpp"
#include "hypergraph/host_team.hpp"
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/incidence_model.hpp"
#include "hypergraph/instantiate.hpp"
#include "hypergraph/radix_sort.hpp"
#include "hypergraph/slack_csr/device_store.hpp"
#include "hypergraph/slack_csr/host_store.hpp"
#include "hypergraph/transaction.hpp"
#include "util/allocation.hpp"
#include "util/region_exceptions.hpp"

#include <dyng/config.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dyng {

namespace {

/// The largest weight (PLAN 4.4.2: integer weights in [1, 2^31 - 1]).
template <typename weight_t>
void check_weight(weight_t w, std::int64_t index, const char* what) {
  if constexpr (!is_unweighted_v<weight_t>) {
    DYNG_EXPECTS(static_cast<std::int64_t>(w) >= 1 && static_cast<std::int64_t>(w) <= INT_MAX, what,
                 ": the weight of hyperedge ", index, " is ", static_cast<std::int64_t>(w),
                 "; weights must lie in [1, 2^31 - 1]");
  } else {
    (void)w;
    (void)index;
    (void)what;
  }
}

/// The rows of a CSR of member lists, checked and normalized: offsets from 0, non-decreasing,
/// ending at the member count; every row non-empty; members in [0, n); each row sorted and its
/// repeated members dropped (counted in `dropped`).
template <typename vertex_t>
void normalize_rows(array_view<const std::int64_t> offsets, array_view<const vertex_t> members,
                    vertex_t num_vertices, const char* what, detail::hyper::member_rows& out,
                    std::int64_t& dropped, bool batch_payload = false) {
  // A batch's rows are its insertions: "inserted hyperedge i of the batch", not the id i (R030
  // review), and a member outside the vertex set names where that set is fixed.
  const char* row_name = batch_payload ? "inserted hyperedge " : "hyperedge ";
  const char* of_batch = batch_payload ? " of the batch" : "";
  const char* fixed = batch_payload
                          ? " (the vertex count is fixed when the hypergraph is built: its "
                            "num_vertices)"
                          : "";
  out.clear();
  dropped = 0;
  if (offsets.empty()) {
    DYNG_EXPECTS(members.empty(), what, ": ", members.size(),
                 " members are given without offsets (offsets need one entry more than the "
                 "hyperedges)");
    return;
  }
  const auto num_rows = static_cast<std::int64_t>(offsets.size()) - 1;
  DYNG_EXPECTS(offsets[0] == 0, what, ": the first offset is ", offsets[0], ", not 0");
  DYNG_EXPECTS(offsets[offsets.size() - 1] == static_cast<std::int64_t>(members.size()), what,
               ": the last offset is ", offsets[offsets.size() - 1], ", not the member count ",
               members.size());
  if (members.size() > static_cast<std::size_t>(INT_MAX / 3)) {
    throw capacity_error(detail::concat_message(
        "dyng: ", what, ": ", members.size(),
        " members exceed the int32 payload of the hypergraph store (at most ", INT_MAX / 3, ")"));
  }
  out.offsets.reserve(static_cast<std::size_t>(num_rows) + 1);
  out.members.reserve(members.size());
  std::vector<std::int32_t> row;
  for (std::int64_t i = 0; i < num_rows; ++i) {
    const std::int64_t begin = offsets[static_cast<std::size_t>(i)];
    const std::int64_t end = offsets[static_cast<std::size_t>(i) + 1];
    DYNG_EXPECTS(begin <= end, what, ": the offsets decrease at ", row_name, i, of_batch);
    DYNG_EXPECTS(begin < end, what, ": ", row_name, i, of_batch,
                 " is empty (a hyperedge needs a member)");
    row.clear();
    for (std::int64_t k = begin; k < end; ++k) {
      const vertex_t v = members[static_cast<std::size_t>(k)];
      DYNG_EXPECTS(v >= 0 && v < num_vertices, what, ": ", row_name, i, of_batch,
                   " has the vertex ", static_cast<std::int64_t>(v), ", outside [0, ",
                   static_cast<std::int64_t>(num_vertices), ")", fixed);
      row.push_back(static_cast<std::int32_t>(v));
    }
    std::sort(row.begin(), row.end());
#if defined(DYNG_MUTATION_HYPERGRAPH_H2V_NOT_DEDUPLICATED)
    // MUTATION (PLAN 8.4, MOSP_ESCHER #6; M8b criterion 20): the h2v insert payload of a batch is
    // not deduplicated (repeated members stored).
    const auto last = batch_payload ? row.end() : std::unique(row.begin(), row.end());
#else
    const auto last = std::unique(row.begin(), row.end());
#endif
    dropped += static_cast<std::int64_t>(row.end() - last);
    out.members.insert(out.members.end(), row.begin(), last);
    out.offsets.push_back(static_cast<std::int64_t>(out.members.size()));
  }
}

/// The weights of `num` hyperedges, checked (empty for unweighted).
template <typename weight_t>
std::vector<weight_t> checked_weights(array_view<const weight_t> weights, std::int64_t num,
                                      const char* what) {
  std::vector<weight_t> out;
  if constexpr (is_unweighted_v<weight_t>) {
    DYNG_EXPECTS(weights.empty(), what, ": an unweighted hypergraph takes no weights, got ",
                 weights.size());
  } else {
    DYNG_EXPECTS(static_cast<std::int64_t>(weights.size()) == num, what, ": ", weights.size(),
                 " weights for ", num, " hyperedges (one weight per hyperedge)");
    out.assign(weights.begin(), weights.end());
    for (std::int64_t i = 0; i < num; ++i) check_weight(out[static_cast<std::size_t>(i)], i, what);
  }
  return out;
}

/// Record the placement of a new hypergraph (PLAN 4.6 rule 5).
template <typename impl_t>
void set_home(impl_t& impl, const resources& res) {
  impl.home = res.get_backend();
  impl.home_device = res.get_backend() == backend::cuda ? res.device() : -1;
}

/// The weights by id as the slack line graph keeps them (int64; 1 for an unweighted hypergraph).
template <typename weight_t>
std::vector<std::int64_t> node_weights_of(const std::vector<weight_t>& weights, std::int64_t n) {
  std::vector<std::int64_t> out(static_cast<std::size_t>(n), 1);
  if constexpr (!is_unweighted_v<weight_t>) {
    for (std::size_t i = 0; i < out.size() && i < weights.size(); ++i)
      out[i] = static_cast<std::int64_t>(weights[i]);
  } else {
    (void)weights;
  }
  return out;
}

/// A store for the backend of `res` from a snapshot of any store.
std::unique_ptr<detail::hyper::incidence_store> store_from_snapshot(
    const resources& res, const detail::hyper::store_snapshot& snapshot,
    const hypergraph_properties& props, const detail::hyper::slack::knobs& knobs) {
  if (props.storage == hypergraph_storage::slack_csr) {
    if (res.get_backend() == backend::cuda) {
      return detail::hyper::make_slack_device_store(res, snapshot, props.capacity, knobs);
    }
    return std::make_unique<detail::hyper::slack_host_store>(res, snapshot, props.capacity, knobs);
  }
  if (res.get_backend() == backend::cuda) {
    return detail::hyper::make_escher_store(res, snapshot, props.capacity);
  }
  return std::make_unique<detail::hyper::incidence_model>(res, snapshot, props.capacity);
}

}  // namespace

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t>::hypergraph(std::unique_ptr<impl_type> impl) noexcept
    : impl_(std::move(impl)) {}

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t>::hypergraph(hypergraph&& other) noexcept = default;

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t>& hypergraph<vertex_t, weight_t>::operator=(
    hypergraph&& other) noexcept = default;

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t>::~hypergraph() = default;

template <typename vertex_t, typename weight_t>
typename hypergraph<vertex_t, weight_t>::impl_type& hypergraph<vertex_t, weight_t>::impl() {
  DYNG_EXPECTS(impl_ != nullptr, "use of a moved-from hypergraph");
  return *impl_;
}

template <typename vertex_t, typename weight_t>
const typename hypergraph<vertex_t, weight_t>::impl_type& hypergraph<vertex_t, weight_t>::impl()
    const {
  DYNG_EXPECTS(impl_ != nullptr, "use of a moved-from hypergraph");
  return *impl_;
}

namespace {

/// The state of a hypergraph for a call that reads it: not moved from, not poisoned.
template <typename impl_t>
const impl_t& readable(const impl_t& impl, const char* what) {
  DYNG_EXPECTS(!impl.poisoned, what,
               ": the hypergraph was poisoned by a failed apply(); rebuild it with "
               "from_hyperedges()");
  return impl;
}

}  // namespace

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t> hypergraph<vertex_t, weight_t>::from_hyperedges(
    const resources& res, hyperedge_list_view<vertex_t, weight_t> hyperedges,
    const hypergraph_properties& props) {
  return detail::hypergraph_access::build(res, hyperedges, props, detail::hyper::slack::knobs{});
}

namespace detail {

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t> hypergraph_access::build(
    const resources& res, hyperedge_list_view<vertex_t, weight_t> hyperedges,
    const hypergraph_properties& props, const hyper::slack::knobs& knobs) try {
  using impl_type = hypergraph_impl<vertex_t, weight_t>;
  static_assert(std::is_same_v<vertex_t, std::int32_t>,
                "hypergraph supports vertex_t = std::int32_t in this release");
  detail::hyper::validate_properties(props);
  const bool slack = props.storage == hypergraph_storage::slack_csr;
  if (slack) detail::hyper::validate_knobs(knobs);
  DYNG_EXPECTS(hyperedges.num_vertices >= 0, "hypergraph::from_hyperedges: num_vertices ",
               static_cast<std::int64_t>(hyperedges.num_vertices), " is negative");
  auto impl = std::make_unique<impl_type>();
  impl->props = props;
  impl->knobs = knobs;
  set_home(*impl, res);
  detail::hyper::member_rows rows;
  {
    // The host side of the construction (ESCHER-GPU's init.v2h_build / flatten region).
    const scoped_stage stage(res, "hypergraph.build");
    const detail::host_input<std::int64_t> offsets(
        res, hyperedges.offsets, "hypergraph::from_hyperedges: hyperedge_list_view::offsets");
    const detail::host_input<vertex_t> members(
        res, hyperedges.members, "hypergraph::from_hyperedges: hyperedge_list_view::members");
    const detail::host_input<weight_t> weights(
        res, hyperedges.weights, "hypergraph::from_hyperedges: hyperedge_list_view::weights");
    std::int64_t dropped = 0;
    normalize_rows(offsets.view(), members.view(), hyperedges.num_vertices,
                   "hypergraph::from_hyperedges", rows, dropped);
    impl->weights = checked_weights(weights.view(), rows.size(), "hypergraph::from_hyperedges");
    if (slack) {
      // slack_csr: MOSP_ESCHER's host incidence model, its ids and its slack CSR (ADR 0086); on
      // cuda the device graph over the same host model (ADR 0089).
      if (res.get_backend() == backend::cuda) {
        impl->store = detail::hyper::make_slack_device_store(
            res, static_cast<std::int32_t>(hyperedges.num_vertices), rows, props.capacity, knobs);
      } else {
        impl->store = std::make_unique<detail::hyper::slack_host_store>(
            res, static_cast<std::int32_t>(hyperedges.num_vertices), rows, props.capacity, knobs);
      }
      impl->store->set_node_weights(node_weights_of(impl->weights, rows.size()));
    } else if (res.get_backend() != backend::cuda) {
      impl->store = std::make_unique<detail::hyper::incidence_model>(
          res, static_cast<std::int32_t>(hyperedges.num_vertices), rows, props.capacity,
          props.store_v2h);
    }
  }
  const bool line_graph = props.line_graph == line_graph_policy::sorted_merge;
  if (slack) {
    if (props.line_graph == line_graph_policy::slack_csr) {
      // The slack CSR from the start (MOSP_ESCHER's buildDeviceH2H after bulkLoad).
      const scoped_stage stage(res, "hypergraph.build");
      impl->store->enable_line_graph();
    }
  } else if (res.get_backend() == backend::cuda) {
    const detail::container_scope container;  // the container's own work (budgets, I9)
    impl->store =
        detail::hyper::make_escher_store(res, static_cast<std::int32_t>(hyperedges.num_vertices),
                                         rows, props.capacity, props.store_v2h, line_graph);
  } else if (line_graph) {
    // The line graph of sorted_merge from the start (ESCHER-GPU's init.h2h_build).
    const scoped_stage stage(res, "hypergraph.build");
    impl->store->enable_line_graph();
  }
  return hypergraph<vertex_t, weight_t>(std::move(impl));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::from_hyperedges (", hyperedges.num_hyperedges(),
                                  " hyperedges, ", hyperedges.members.size(), " members)")

}  // namespace detail

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t> hypergraph<vertex_t, weight_t>::clone(const resources& res) const
    try {
  const auto& self = readable(impl(), "hypergraph::clone");
  auto copy = std::make_unique<impl_type>();
  copy->props = self.props;
  copy->version = self.version;
  copy->state_id = self.state_id;
  copy->weights = self.weights;
  set_home(*copy, res);
  const bool from_device = self.home == backend::cuda;
  const bool to_device = res.get_backend() == backend::cuda;
  copy->knobs = self.knobs;
  if (from_device == to_device) {
    copy->store = self.store->clone(res);
    copy->store->set_host_threads(detail::resources_access::host_threads(res));
  } else {
    copy->store = store_from_snapshot(res, self.store->snapshot(), self.props, self.knobs);
    if (self.props.storage == hypergraph_storage::slack_csr) {
      copy->store->set_node_weights(node_weights_of(copy->weights, copy->store->id_bound()));
    }
  }
  return hypergraph(std::move(copy));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::clone (", hyperedge_id_bound(), " ids)")

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t> hypergraph<vertex_t, weight_t>::to_backend(
    const resources& res) const {
  return clone(res);
}

template <typename vertex_t, typename weight_t>
vertex_t hypergraph<vertex_t, weight_t>::num_vertices() const noexcept {
  return impl_ && impl_->store ? static_cast<vertex_t>(impl_->store->num_vertices()) : vertex_t{0};
}

template <typename vertex_t, typename weight_t>
std::int64_t hypergraph<vertex_t, weight_t>::num_hyperedges() const noexcept {
  return impl_ && impl_->store ? impl_->store->num_live() : 0;
}

template <typename vertex_t, typename weight_t>
std::int64_t hypergraph<vertex_t, weight_t>::hyperedge_id_bound() const noexcept {
  return impl_ && impl_->store ? impl_->store->id_bound() : 0;
}

template <typename vertex_t, typename weight_t>
const hypergraph_properties& hypergraph<vertex_t, weight_t>::properties() const noexcept {
  static const hypergraph_properties moved_from{};
  return impl_ ? impl_->props : moved_from;
}

template <typename vertex_t, typename weight_t>
memory_space hypergraph<vertex_t, weight_t>::space() const noexcept {
  return impl_ && impl_->home == backend::cuda ? memory_space::device : memory_space::host;
}

template <typename vertex_t, typename weight_t>
std::uint64_t hypergraph<vertex_t, weight_t>::version() const noexcept {
  return impl_ ? impl_->version : 0;
}

template <typename vertex_t, typename weight_t>
typename hypergraph<vertex_t, weight_t>::h2v_view_type hypergraph<vertex_t, weight_t>::h2v() const
    try {
  const auto& self = readable(impl(), "hypergraph::h2v");
  h2v_view_type out;
  array_view<const std::int32_t> cols;
  self.store->h2v_view(out.row_ptr, cols);
  out.col_ind = array_view<const vertex_t>(cols.data(), cols.size(), cols.space(), cols.device());
  out.version = self.version;
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::h2v (", hyperedge_id_bound(), " ids)")

template <typename vertex_t, typename weight_t>
typename hypergraph<vertex_t, weight_t>::v2h_view_type hypergraph<vertex_t, weight_t>::v2h() const
    try {
  const auto& self = readable(impl(), "hypergraph::v2h");
  DYNG_EXPECTS(self.props.store_v2h,
               "hypergraph::v2h: the hypergraph was built with store_v2h = false");
  v2h_view_type out;
  self.store->v2h_view(out.row_ptr, out.col_ind);
  out.version = self.version;
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::v2h (", static_cast<std::int64_t>(num_vertices()),
                                  " vertices)")

template <typename vertex_t, typename weight_t>
typename hypergraph<vertex_t, weight_t>::line_graph_type hypergraph<vertex_t, weight_t>::h2h() const
    try {
  const auto& self = readable(impl(), "hypergraph::h2h");
  detail::hypergraph_access::require_line_graph(*this, "hypergraph::h2h");
  line_graph_type out;
  self.store->h2h_view(out.row_ptr, out.col_ind, out.out);
  if constexpr (!is_unweighted_v<weight_t>) out.weights = host_view(self.weights);
  out.version = self.version;
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::h2h (", hyperedge_id_bound(), " ids)")

template <typename vertex_t, typename weight_t>
array_view<const weight_t> hypergraph<vertex_t, weight_t>::weights() const {
  return host_view(readable(impl(), "hypergraph::weights").weights);
}

template <typename vertex_t, typename weight_t>
hyper_apply_summary hypergraph<vertex_t, weight_t>::apply(const resources& res,
                                                          const batch_view_type& batch) {
  return detail::hypergraph_access::apply(res, *this, batch, false);
}

template <typename vertex_t, typename weight_t>
typename hypergraph<vertex_t, weight_t>::list_type
hypergraph<vertex_t, weight_t>::to_hyperedge_list(const resources& res) const try {
  detail::hypergraph_access::expect_placement(res, *this, "hypergraph::to_hyperedge_list");
  const auto& self = readable(impl(), "hypergraph::to_hyperedge_list");
  array_view<const std::int64_t> row_ptr;
  array_view<const std::int32_t> col_ind;
  self.store->h2v_view(row_ptr, col_ind);
  const std::vector<std::int64_t> ptr = to_vector(res, row_ptr);
  const std::vector<std::int32_t> cols = to_vector(res, col_ind);
  list_type out;
  out.num_vertices = num_vertices();
  out.offsets.assign(1, 0);
  for (std::size_t id = 0; id + 1 < ptr.size(); ++id) {
    if (ptr[id + 1] == ptr[id]) continue;  // a dead id
    out.members.insert(out.members.end(), cols.begin() + ptr[id], cols.begin() + ptr[id + 1]);
    out.offsets.push_back(static_cast<std::int64_t>(out.members.size()));
    if constexpr (!is_unweighted_v<weight_t>) out.weights.push_back(self.weights[id]);
  }
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::to_hyperedge_list (", num_hyperedges(),
                                  " hyperedges)")

template <typename vertex_t, typename weight_t>
storage_stats hypergraph<vertex_t, weight_t>::store_stats() const noexcept {
  if (!impl_ || !impl_->store) return {};
  try {
    return impl_->store->stats();
  } catch (...) {
    return {};
  }
}

template <typename vertex_t, typename weight_t>
void hypergraph<vertex_t, weight_t>::check_integrity(const resources& res) const try {
  detail::hypergraph_access::expect_placement(res, *this, "hypergraph::check_integrity");
  const auto& self = readable(impl(), "hypergraph::check_integrity");
  self.store->check_integrity();
  if constexpr (!is_unweighted_v<weight_t>) {
    const std::int64_t bound = self.store->id_bound();
    if (static_cast<std::int64_t>(self.weights.size()) != bound) {
      DYNG_FAIL("hypergraph integrity: ", self.weights.size(), " weights for an id bound of ",
                bound);
    }
    for (std::int64_t id = 0; id < bound; ++id) {
      const auto w = static_cast<std::int64_t>(self.weights[static_cast<std::size_t>(id)]);
      const bool alive = self.store->is_live(id);
      if (alive ? w < 1 : w != 0) {
        DYNG_FAIL("hypergraph integrity: id ", id, (alive ? " is live" : " is dead"),
                  " with the weight ", w);
      }
    }
  }
  if (const auto* slack = self.store->slack();
      slack != nullptr && self.store->line_graph_enabled()) {
    // The slack line graph's node weights are the container's (1 for an unweighted hypergraph).
    const detail::hyper::slack_row_view rows = slack->rows();
    if (rows.space == memory_space::host) {
      for (std::int64_t id = 0; id < self.store->id_bound(); ++id) {
        if (!self.store->is_live(id)) continue;
        std::int64_t want = 1;
        if constexpr (!is_unweighted_v<weight_t>) {
          want = static_cast<std::int64_t>(self.weights[static_cast<std::size_t>(id)]);
        }
        if (rows.node_weight[id] != want) {
          DYNG_FAIL("hypergraph integrity: the line graph's weight of id ", id, " is ",
                    rows.node_weight[id], ", the hypergraph's ", want);
        }
      }
    }
  }
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::check_integrity")

namespace detail {

#if !DYNG_HAS_CUDA
namespace hyper {
std::unique_ptr<incidence_store> make_escher_store(const resources& /*res*/,
                                                   std::int32_t /*num_vertices*/,
                                                   const member_rows& /*rows*/,
                                                   const capacity_policy& /*capacity*/,
                                                   bool /*has_v2h*/, bool /*line_graph*/) {
  throw not_supported_error(
      "dyng: the cuda backend is not built (configure with "
      "DYNG_ENABLE_CUDA=ON)");
}
std::unique_ptr<incidence_store> make_escher_store(const resources& /*res*/,
                                                   const store_snapshot& /*snapshot*/,
                                                   const capacity_policy& /*capacity*/) {
  throw not_supported_error(
      "dyng: the cuda backend is not built (configure with "
      "DYNG_ENABLE_CUDA=ON)");
}
std::unique_ptr<incidence_store> make_slack_device_store(const resources& /*res*/,
                                                         std::int32_t /*num_vertices*/,
                                                         const member_rows& /*rows*/,
                                                         const capacity_policy& /*capacity*/,
                                                         const slack::knobs& /*knobs*/) {
  throw not_supported_error(
      "dyng: the cuda backend is not built (configure with "
      "DYNG_ENABLE_CUDA=ON)");
}
std::unique_ptr<incidence_store> make_slack_device_store(const resources& /*res*/,
                                                         const store_snapshot& /*snapshot*/,
                                                         const capacity_policy& /*capacity*/,
                                                         const slack::knobs& /*knobs*/) {
  throw not_supported_error(
      "dyng: the cuda backend is not built (configure with "
      "DYNG_ENABLE_CUDA=ON)");
}
}  // namespace hyper
#endif

template <typename vertex_t, typename weight_t>
void hypergraph_access::expect_placement(const resources& res,
                                         const hypergraph<vertex_t, weight_t>& h,
                                         const char* what) {
  const auto& state = h.impl();
  const backend b = res.get_backend();
  if (b == backend::cuda) {
    DYNG_EXPECTS(state.home == backend::cuda && state.home_device == res.device(), what,
                 ": the hypergraph was built for ", to_string(state.home),
                 state.home == backend::cuda ? " on CUDA device " : "",
                 state.home == backend::cuda ? std::to_string(state.home_device) : std::string(),
                 " but the resources are cuda on device ", res.device(),
                 "; build the hypergraph with these resources or move it with h.to_backend(res)");
  } else {
    DYNG_EXPECTS(state.home != backend::cuda, what, ": the hypergraph is resident on CUDA device ",
                 state.home_device, " but the resources are ", to_string(b),
                 "; move it with h.to_backend(res)");
  }
}

template <typename vertex_t, typename weight_t>
void hypergraph_access::require_line_graph(const hypergraph<vertex_t, weight_t>& h,
                                           const char* what) {
  const auto& state = h.impl();
  DYNG_EXPECTS(!state.poisoned, what,
               ": the hypergraph was poisoned by a failed apply(); rebuild it with "
               "from_hyperedges()");
  if (state.props.storage == hypergraph_storage::slack_csr) {
    DYNG_EXPECTS(state.props.line_graph != line_graph_policy::none, what,
                 ": the hypergraph was built with line_graph_policy::none (no line graph); build "
                 "it with line_graph_policy::automatic or slack_csr");
  }
  DYNG_EXPECTS(state.props.line_graph != line_graph_policy::none, what,
               ": the hypergraph was built with line_graph_policy::none (no line graph); build it "
               "with line_graph_policy::automatic or sorted_merge");
  DYNG_EXPECTS(state.props.store_v2h, what,
               ": the line graph needs v2h; build the hypergraph with store_v2h = true");
  // line_graph_policy::automatic builds it here, once, and maintains it from then on.
  state.store->enable_line_graph();
}

template <typename vertex_t, typename weight_t>
hyper_apply_summary hypergraph_access::apply(const resources& res,
                                             hypergraph<vertex_t, weight_t>& h,
                                             const hyperedge_batch_view<vertex_t, weight_t>& batch,
                                             bool record_incidence_delta) {
  // A transaction without participants (ADR 0043): begin (Step 0), then the commit.
  const scoped_stage stage(res, "hypergraph.apply");
  std::optional<hyper_transaction<vertex_t, weight_t>> txn;
  {
    // Step 0 (the region map of hyper_sssp names it: parity/timed_regions/hyper_sssp.toml).
    const scoped_stage normalize(res, "hypergraph.apply.normalize");
    txn.emplace(res, h, batch, "hypergraph::apply");
  }
  return txn->commit(record_incidence_delta);
}

namespace {

/// begin's checks before the batch is staged: placement, a live hypergraph, no incidence edits
/// under escher storage (they need slack_csr storage, HD9).
template <typename vertex_t, typename weight_t>
const char* begin_checks(const resources& res, hypergraph<vertex_t, weight_t>& h,
                         const hyperedge_batch_view<vertex_t, weight_t>& batch,
                         const char* function) {
  hypergraph_access::expect_placement(res, h, function);
  DYNG_EXPECTS(!hypergraph_access::impl(h).poisoned, function,
               ": the hypergraph was poisoned by a failed apply(); rebuild it with "
               "from_hyperedges()");
  if ((!batch.delete_incidence_hyperedge.empty() || !batch.delete_incidence_vertex.empty() ||
       !batch.insert_incidence_hyperedge.empty() || !batch.insert_incidence_vertex.empty()) &&
      hypergraph_access::impl(h).props.storage != hypergraph_storage::slack_csr) {
    throw not_supported_error(concat_message(
        "dyng: ", function,
        ": incidence edits (delete_incidence / insert_incidence) need hypergraph_storage::"
        "slack_csr in 0.4 (escher storage keeps its rows sorted for ESCHER-GPU's line-graph "
        "merge); build the hypergraph with hypergraph_storage::slack_csr, or delete the "
        "hyperedge and insert it with its new members"));
  }
  return function;
}

/// Step 0 of the incidence edits (slack_csr storage; ADR 0085): MOSP_ESCHER's applyBatch phases
/// 2 and 3 (HostHypergraph.cpp:197-235) evaluated in batch order against G_t, the removals after
/// the hyperedge deletions, the additions after the removals. An edit is skipped (and counted)
/// when its hyperedge is dead on G_t, out of range or deleted by the batch (an id the batch
/// inserts is one of these: it is not live on G_t, or it is deleted by the batch), its vertex is
/// out of range, a removal names a non-member or would empty its hyperedge, or an addition names
/// a member. Nothing of G_t changes.
///
/// An edit's outcome depends only on G_t's row of its hyperedge and on the earlier edits of the
/// same hyperedge, so the edits are grouped by hyperedge (a stable radix sort: a group keeps its
/// removals, then its additions, in batch order) and the groups are evaluated in parallel on the
/// host threads of the resources (M11a step 3, ADR 0089: the host costs of the early maintain
/// reading); the outcome equals the batch-order evaluation for any thread count. The groups come
/// out in increasing id: the edited ids, their rows on G_t and after the edits, and the applied
/// edits grouped by id for the allocator's H2V unfill and fill (MOSP_ESCHER's flushGrouped), so
/// the commit groups and sorts nothing of them.
template <typename vertex_t, typename weight_t>
void stage_incidence_edits(hypergraph_impl<vertex_t, weight_t>& state, int threads,
                           array_view<const std::int64_t> removal_ids,
                           array_view<const vertex_t> removal_vertices,
                           array_view<const std::int64_t> addition_ids,
                           array_view<const vertex_t> addition_vertices,
                           hyper_staged_batch<weight_t>& staged) {
  hyper::incidence_store& store = *state.store;
  const std::vector<std::vector<std::int32_t>>& rows = store.slack()->host_rows();
  const std::int32_t num_vertices = store.num_vertices();
  const std::int64_t bound = store.id_bound();
  const std::size_t num_removals = removal_ids.size();
  const std::size_t num_edits = num_removals + addition_ids.size();
  const auto id_of = [&](std::size_t j) {
    return j < num_removals ? removal_ids[j] : addition_ids[j - num_removals];
  };
  const auto vertex_of = [&](std::size_t j) {
    return j < num_removals ? removal_vertices[j] : addition_vertices[j - num_removals];
  };
  if (num_edits > static_cast<std::size_t>(UINT32_MAX)) {
    throw capacity_error(concat_message("dyng: hypergraph::apply: ", num_edits,
                                        " incidence edits exceed the 2^32 - 1 of one batch"));
  }
  // The edits whose id and vertex are in range, as (id << 32 | edit), grouped by id (stable).
  std::vector<std::uint64_t>& words = state.sort_words;
  words.clear();
  words.reserve(num_edits);
  for (std::size_t j = 0; j < num_edits; ++j) {
    const std::int64_t h = id_of(j);
    const vertex_t v = vertex_of(j);
    if (h >= 0 && h < bound && v >= 0 && static_cast<std::int64_t>(v) < num_vertices)
      words.push_back(static_cast<std::uint64_t>(h) << 32 | j);
  }
  hyper::radix_sort_by_key(words, static_cast<std::uint64_t>(bound), state.sort_scratch);
  // The groups: [start[g], start[g + 1]) of `words`.
  std::vector<std::size_t> start;
  for (std::size_t i = 0; i < words.size(); ++i)
    if (i == 0 || (words[i] >> 32) != (words[i - 1] >> 32)) start.push_back(i);
  const std::size_t num_groups = start.size();
  start.push_back(words.size());
  std::vector<std::int64_t> deleted = staged.deletions;
  std::sort(deleted.begin(), deleted.end());
  std::vector<std::uint8_t> applied(num_edits, 0);
  std::vector<std::int64_t> post_size(num_groups, -1);  // -1: no applied edit
  std::vector<std::int64_t> pre_size(num_groups, 0);
  const auto group_id = [&](std::size_t g) {
    return static_cast<std::int64_t>(words[start[g]] >> 32);
  };
  // Evaluate group g's edits in order on a copy of its row (into `row`); true if one applied.
  const auto evaluate = [&](std::size_t g, std::vector<std::int32_t>& row, bool mark) {
    const std::int64_t h = group_id(g);
    const auto& before = rows[static_cast<std::size_t>(h)];
    // Live = a non-empty row (a slack_csr store keeps no live row empty and clears a dead one;
    // check_integrity() checks it).
    if (before.empty() || std::binary_search(deleted.begin(), deleted.end(), h)) return false;
    row.assign(before.begin(), before.end());
    bool any = false;
    for (std::size_t i = start[g]; i < start[g + 1]; ++i) {
      const auto j = static_cast<std::size_t>(words[i] & 0xffffffffULL);
      const auto v = static_cast<std::int32_t>(vertex_of(j));
      const auto it = std::lower_bound(row.begin(), row.end(), v);
      const bool member = it != row.end() && *it == v;
      if (j < num_removals) {
        // A non-member, or the last member (that would be an implicit deletion: heDelete is for
        // it).
        if (!member || row.size() == 1) continue;
        row.erase(it);
      } else {
        if (member) continue;  // already a member
        row.insert(it, v);
      }
      if (mark) applied[j] = 1;
      any = true;
    }
    return any;
  };
  const auto groups = static_cast<std::int64_t>(num_groups);
  // The host team (host_team.hpp: short iterations, one fixed team size).
  const int team = hyper::host_team(threads, groups);
  const bool parallel = team > 1;
  (void)parallel;
  (void)team;
  {
    // Each group writes its own entries of `applied` and `post_size` (distinct edits, distinct
    // groups); a copy of a row allocates: its exception is carried out of the region (ADR 0061).
    region_exceptions errors;
#if DYNG_HAS_OPENMP
#pragma omp parallel num_threads(team) if (parallel)
#endif
    {
      std::vector<std::int32_t> row;
#if DYNG_HAS_OPENMP
#pragma omp for schedule(static, 256)
#endif
      for (std::int64_t g = 0; g < groups; ++g) {
        errors.run([&] {
          if (evaluate(static_cast<std::size_t>(g), row, true)) {
            post_size[static_cast<std::size_t>(g)] = static_cast<std::int64_t>(row.size());
            pre_size[static_cast<std::size_t>(g)] = static_cast<std::int64_t>(
                rows[static_cast<std::size_t>(group_id(static_cast<std::size_t>(g)))].size());
          }
        });
      }
    }
    errors.rethrow();
  }
  // The applied edits in batch order, the skipped count.
  staged.removal_ids.reserve(num_removals);
  staged.removal_vertices.reserve(num_removals);
  staged.addition_ids.reserve(num_edits - num_removals);
  staged.addition_vertices.reserve(num_edits - num_removals);
  std::int64_t applied_count = 0;
  for (std::size_t j = 0; j < num_edits; ++j) {
    if (applied[j] == 0) continue;
    ++applied_count;
    if (j < num_removals) {
      staged.removal_ids.push_back(removal_ids[j]);
      staged.removal_vertices.push_back(static_cast<std::int32_t>(removal_vertices[j]));
    } else {
      staged.addition_ids.push_back(addition_ids[j - num_removals]);
      staged.addition_vertices.push_back(
          static_cast<std::int32_t>(addition_vertices[j - num_removals]));
    }
  }
  staged.skipped_incidence_edits = static_cast<std::int64_t>(num_edits) - applied_count;
  // The edited ids (increasing: the groups' order), the offsets of their rows, the applied edits
  // grouped by id.
  std::vector<std::size_t> edited;  // the groups with an applied edit
  auto& pre = staged.edited_pre_rows;
  auto& post = staged.edited_post_rows;
  for (std::size_t g = 0; g < num_groups; ++g) {
    if (post_size[g] < 0) continue;
    edited.push_back(g);
    const std::int64_t h = group_id(g);
    staged.edited_ids.push_back(h);
    pre.offsets.push_back(pre.offsets.back() + pre_size[g]);
    post.offsets.push_back(post.offsets.back() + post_size[g]);
    bool removal = false, addition = false;
    for (std::size_t i = start[g]; i < start[g + 1]; ++i) {
      const auto j = static_cast<std::size_t>(words[i] & 0xffffffffULL);
      if (applied[j] == 0) continue;
      hyper::grouped_members& to =
          j < num_removals ? staged.removal_groups : staged.addition_groups;
      bool& opened = j < num_removals ? removal : addition;
      if (!opened) {
        to.ids.push_back(h);
        to.prefix.push_back(to.prefix.empty() ? 0 : to.prefix.back());
        opened = true;
      }
      to.members.push_back(static_cast<std::int32_t>(vertex_of(j)));
      ++to.prefix.back();
    }
  }
  pre.members.resize(static_cast<std::size_t>(pre.offsets.back()));
  post.members.resize(static_cast<std::size_t>(post.offsets.back()));
  {
    // The rows of the edited ids on G_t and after the edits (evaluated once more), each into its
    // own range of the flat members.
    const auto num_edited = static_cast<std::int64_t>(edited.size());
    const int rows_team = hyper::host_team(threads, num_edited);
    const bool parallel_rows = rows_team > 1;
    (void)parallel_rows;
    (void)rows_team;
    region_exceptions errors;
#if DYNG_HAS_OPENMP
#pragma omp parallel num_threads(rows_team) if (parallel_rows)
#endif
    {
      std::vector<std::int32_t> row;
#if DYNG_HAS_OPENMP
#pragma omp for schedule(static, 256)
#endif
      for (std::int64_t e = 0; e < num_edited; ++e) {
        errors.run([&] {
          const std::size_t g = edited[static_cast<std::size_t>(e)];
          const auto& before = rows[static_cast<std::size_t>(group_id(g))];
          std::copy(before.begin(), before.end(),
                    pre.members.begin() + pre.offsets[static_cast<std::size_t>(e)]);
          (void)evaluate(g, row, false);
          std::copy(row.begin(), row.end(),
                    post.members.begin() + post.offsets[static_cast<std::size_t>(e)]);
        });
      }
    }
    errors.rethrow();
  }
}

}  // namespace

template <typename vertex_t, typename weight_t>
hyper_transaction<vertex_t, weight_t>::hyper_transaction(const resources& res, container_type& h,
                                                         const batch_view_type& batch,
                                                         const char* function) try
    : res_(res),
      h_(h),
      function_(begin_checks(res, h, batch, function)),
      deletions_(res, batch.delete_hyperedges, concat_message(function, ": delete_hyperedges")),
      offsets_(res, batch.insert_offsets, concat_message(function, ": insert_offsets")),
      members_(res, batch.insert_members, concat_message(function, ": insert_members")),
      weights_(res, batch.insert_weights, concat_message(function, ": insert_weights")),
      removal_ids_(res, batch.delete_incidence_hyperedge,
                   concat_message(function, ": delete_incidence_hyperedge")),
      removal_vertices_(res, batch.delete_incidence_vertex,
                        concat_message(function, ": delete_incidence_vertex")),
      addition_ids_(res, batch.insert_incidence_hyperedge,
                    concat_message(function, ": insert_incidence_hyperedge")),
      addition_vertices_(res, batch.insert_incidence_vertex,
                         concat_message(function, ": insert_incidence_vertex")) {
  // ---- Step 0, once: everything that can fail before the store changes (strong guarantee).
  host_.delete_hyperedges = deletions_.view();
  host_.insert_offsets = offsets_.view();
  host_.insert_members = members_.view();
  host_.insert_weights = weights_.view();
  host_.delete_incidence_hyperedge = removal_ids_.view();
  host_.delete_incidence_vertex = removal_vertices_.view();
  host_.insert_incidence_hyperedge = addition_ids_.view();
  host_.insert_incidence_vertex = addition_vertices_.view();
  const hyper::incidence_store& store = *hypergraph_access::impl(h).store;
  staged_.version = h.version();
  staged_.id_bound = store.id_bound();
  staged_.requested_insertions = static_cast<std::int64_t>(host_.num_insertions());
  staged_.requested_deletions = static_cast<std::int64_t>(host_.num_deletions());
  normalize_rows(offsets_.view(), members_.view(), static_cast<vertex_t>(store.num_vertices()),
                 function, staged_.rows, staged_.dropped_members, /*batch_payload=*/true);
  staged_.weights = checked_weights(weights_.view(), staged_.rows.size(), function);
  // Deletions are checked against G_t (MOSP_ESCHER's applyBatch, phase 1): a negative,
  // out-of-range, dead or repeated id is skipped and counted.
  std::vector<std::int64_t>& live = staged_.deletions;
  std::int64_t ignored = 0;
  for (const std::int64_t id : deletions_.view()) {
    if (id < 0 || id >= store.id_bound() || !store.is_live(id)) {
      ++ignored;
      continue;
    }
    live.push_back(id);
  }
  // Repeated ids: the first deletion counts, the others are skipped. Only a batch with a repeated
  // id pays for more than the sort: its (id, position) pairs are sorted, and the first position
  // of every id is kept (O(k log k) in the batch, independent of the id bound).
  std::vector<std::int64_t> sorted = live;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    std::vector<std::pair<std::int64_t, std::size_t>> at(live.size());
    for (std::size_t i = 0; i < live.size(); ++i) at[i] = {live[i], i};
    std::sort(at.begin(), at.end());
    std::vector<std::uint8_t> keep(live.size(), 0);
    for (std::size_t i = 0; i < at.size(); ++i) {
      if (i == 0 || at[i].first != at[i - 1].first) {
        keep[at[i].second] = 1;
      } else {
        ++ignored;
      }
    }
    std::vector<std::int64_t> unique_order;
    unique_order.reserve(live.size());
    for (std::size_t i = 0; i < live.size(); ++i)
      if (keep[i] != 0) unique_order.push_back(live[i]);
    live = std::move(unique_order);
  }
  staged_.ignored_deletions = ignored;
  // The incidence edits (slack_csr storage only: begin_checks refused them otherwise).
  DYNG_EXPECTS(removal_ids_.view().size() == removal_vertices_.view().size(), function, ": ",
               removal_ids_.view().size(), " incidence removals name a hyperedge but ",
               removal_vertices_.view().size(),
               " name a vertex (delete_incidence_hyperedge and delete_incidence_vertex need one "
               "entry per removal)");
  DYNG_EXPECTS(addition_ids_.view().size() == addition_vertices_.view().size(), function, ": ",
               addition_ids_.view().size(), " incidence additions name a hyperedge but ",
               addition_vertices_.view().size(),
               " name a vertex (insert_incidence_hyperedge and insert_incidence_vertex need one "
               "entry per addition)");
  staged_.requested_incidence_removals = static_cast<std::int64_t>(removal_ids_.view().size());
  staged_.requested_incidence_additions = static_cast<std::int64_t>(addition_ids_.view().size());
  if (staged_.requested_incidence_removals + staged_.requested_incidence_additions > 0) {
    stage_incidence_edits<vertex_t, weight_t>(
        hypergraph_access::state(h), resources_access::host_threads(res), removal_ids_.view(),
        removal_vertices_.view(), addition_ids_.view(), addition_vertices_.view(), staged_);
  }
}
DYNG_TRANSLATE_ALLOCATION_FAILURE(function, " (", h.hyperedge_id_bound(), " ids; batch of ",
                                  batch.num_insertions(), " insertions, ", batch.num_deletions(),
                                  " deletions)")

template <typename vertex_t, typename weight_t>
hyper_transaction<vertex_t, weight_t>::~hyper_transaction() = default;

template <typename vertex_t, typename weight_t>
framework::old_view<hypergraph<vertex_t, weight_t>> hyper_transaction<vertex_t, weight_t>::before()
    const {
  if (committed_) {
    DYNG_FAIL(function_, ": the transaction's G_t was asked for after its commit (invariant I1)");
  }
  return framework::old_view<container_type>(h_);
}

template <typename vertex_t, typename weight_t>
framework::new_view<hypergraph<vertex_t, weight_t>> hyper_transaction<vertex_t, weight_t>::after()
    const {
  if (!committed_) {
    DYNG_FAIL(function_, ": the transaction's G_{t+1} was asked for before its commit");
  }
  return framework::new_view<container_type>(h_);
}

template <typename vertex_t, typename weight_t>
const hyper_applied& hyper_transaction<vertex_t, weight_t>::applied() const {
  if (!committed_) {
    DYNG_FAIL(function_, ": the transaction's applied record was asked for before its commit");
  }
  return hypergraph_access::applied(h_);
}

template <typename vertex_t, typename weight_t>
void hyper_transaction<vertex_t, weight_t>::commit_slack_csr(hyper_applied& applied,
                                                             std::vector<std::int64_t>& ids) {
  // The store phase of slack_csr storage (ADR 0085, 0086, 0089): MOSP_ESCHER's beginBatch /
  // finishBatch order under insert_first (H2V insert; then H2V erase, H2V unfill, V2H unfill,
  // H2V fill, V2H fill), erase_first's documented order otherwise (erase, unfill, fill, insert);
  // the applied record; the pair delta inside the commit (the stage hypergraph.apply.translate);
  // the line graph's CSR apply. Every list is built once from Step 0's staging and the batch, in
  // one pass each, without a grouping or a sort beyond the radix sorts of the touched ids and
  // vertices (M11a step 3: the host costs of the early maintain reading, ADR 0089).
  auto& state = hypergraph_access::state(h_);
  hyper::incidence_store& store = *state.store;
  const resources& res = res_;
  const hyper::member_rows& rows = staged_.rows;
  const std::vector<std::int64_t>& live_deletions = staged_.deletions;
  // The (vertex, id, sign) of every incidence the batch changes, in MOSP_ESCHER's order: the
  // deleted rows' members and the removed incidences (its v2hUnfill order, sign -1), then the
  // added incidences and the inserted rows' members (its v2hFill order, +1; the inserted ids are
  // filled in once the H2V insert has given them). One list serves the v2h edits, the pair delta
  // and the record.
  applied.has_incidence_delta = true;
  auto& delta = applied.incidence_delta;
  std::size_t removed_end = 0;
  std::vector<std::int32_t> pre_vertices;
  std::vector<std::uint64_t> by_vertex;  // vertex << 32 | its place in the incidence list
  std::vector<std::size_t> vertex_start;
  std::vector<std::uint64_t> sorted_deletions;  // id << 32 | its place in live_deletions
  std::vector<std::int64_t> pre_ids;
  {
    const scoped_stage stage(res, "hypergraph.apply.incidence");
    const std::size_t total = applied.deleted_members.members.size() + staged_.removal_ids.size() +
                              staged_.addition_ids.size() + rows.members.size();
    delta.vertices.reserve(total);
    delta.ids.reserve(total);
    delta.signs.reserve(total);
    for (std::int64_t i = 0; i < applied.deleted_members.size(); ++i) {
      for (std::int64_t k = applied.deleted_members.offsets[static_cast<std::size_t>(i)];
           k < applied.deleted_members.offsets[static_cast<std::size_t>(i) + 1]; ++k) {
        delta.vertices.push_back(applied.deleted_members.members[static_cast<std::size_t>(k)]);
        delta.ids.push_back(live_deletions[static_cast<std::size_t>(i)]);
      }
    }
    delta.vertices.insert(delta.vertices.end(), staged_.removal_vertices.begin(),
                          staged_.removal_vertices.end());
    delta.ids.insert(delta.ids.end(), staged_.removal_ids.begin(), staged_.removal_ids.end());
    removed_end = delta.vertices.size();
    delta.vertices.insert(delta.vertices.end(), staged_.addition_vertices.begin(),
                          staged_.addition_vertices.end());
    delta.ids.insert(delta.ids.end(), staged_.addition_ids.begin(), staged_.addition_ids.end());
    delta.vertices.insert(delta.vertices.end(), rows.members.begin(), rows.members.end());
    delta.ids.resize(delta.vertices.size(), -1);  // the inserted rows' ids: after the insert
    delta.signs.assign(removed_end, -1);
    delta.signs.resize(delta.vertices.size(), +1);
    // The incidences grouped by vertex (a stable radix sort: a vertex keeps the batch's order of
    // its changes), which gives the touched vertices (sorted, unique) and the v2h edits' groups.
    if (delta.vertices.size() > static_cast<std::size_t>(UINT32_MAX)) {
      throw capacity_error(concat_message("dyng: hypergraph::apply: ", delta.vertices.size(),
                                          " incidence changes exceed the 2^32 - 1 of one batch"));
    }
    by_vertex.resize(delta.vertices.size());
    for (std::size_t i = 0; i < delta.vertices.size(); ++i)
      by_vertex[i] = static_cast<std::uint64_t>(delta.vertices[i]) << 32 | i;
    hyper::radix_sort_by_key(by_vertex, static_cast<std::uint64_t>(store.num_vertices()),
                             state.sort_scratch);
    for (std::size_t i = 0; i < by_vertex.size(); ++i) {
      if (i == 0 || (by_vertex[i] >> 32) != (by_vertex[i - 1] >> 32)) {
        vertex_start.push_back(i);
        pre_vertices.push_back(static_cast<std::int32_t>(by_vertex[i] >> 32));
      }
    }
    vertex_start.push_back(by_vertex.size());
    sorted_deletions.resize(live_deletions.size());
    for (std::size_t i = 0; i < live_deletions.size(); ++i)
      sorted_deletions[i] = static_cast<std::uint64_t>(live_deletions[i]) << 32 | i;
    hyper::radix_sort_by_key(sorted_deletions, static_cast<std::uint64_t>(staged_.id_bound),
                             state.sort_scratch);
    pre_ids.resize(sorted_deletions.size() + staged_.edited_ids.size());
    std::size_t a = 0, b = 0, o = 0;
    while (a < sorted_deletions.size() || b < staged_.edited_ids.size()) {
      const auto da = a < sorted_deletions.size()
                          ? static_cast<std::int64_t>(sorted_deletions[a] >> 32)
                          : INT64_MAX;
      const std::int64_t eb = b < staged_.edited_ids.size() ? staged_.edited_ids[b] : INT64_MAX;
      if (da < eb) {
        pre_ids[o++] = da;
        ++a;
      } else {
        pre_ids[o++] = eb;
        ++b;
      }
    }
    store.begin_line_graph_delta(pre_vertices, pre_ids);
  }
  // H2V erase, unfill and fill; then (the inserted rows' ids filled in) every v2h edit, grouped by
  // vertex (nothing of the H2V side reads v2h: one pass at the end is MOSP_ESCHER's order per
  // vertex).
  const auto remove_phase = [&] {
    store.erase(live_deletions);
    store.h2v_edit(staged_.removal_groups, staged_.addition_groups, staged_.edited_ids,
                   staged_.edited_post_rows);
  };
  const auto append_phase = [&] {
    std::size_t at = removed_end + staged_.addition_ids.size();
    for (std::int64_t i = 0; i < rows.size(); ++i) {
      for (std::int64_t k = rows.offsets[static_cast<std::size_t>(i)];
           k < rows.offsets[static_cast<std::size_t>(i) + 1]; ++k)
        delta.ids[at++] = ids[static_cast<std::size_t>(i)];
    }
    store.v2h_apply(by_vertex, vertex_start, delta.ids.data(), delta.signs.data(),
                    applied.touched_vertex_lengths);
  };
  if (state.props.reuse == id_reuse::insert_first) {
    {
      const scoped_stage stage(res, "hypergraph.apply.ids");
      store.insert(rows, ids);
    }
    const scoped_stage stage(res, "hypergraph.apply.incidence");
    remove_phase();
    append_phase();
  } else {
    {
      const scoped_stage stage(res, "hypergraph.apply.incidence");
      remove_phase();
    }
    {
      const scoped_stage stage(res, "hypergraph.apply.ids");
      store.insert(rows, ids);
    }
    const scoped_stage stage(res, "hypergraph.apply.incidence");
    append_phase();
  }
  // The rest of the applied record (what hyper_sssp reads; no new virtual on the participant),
  // before the pair delta, which ships its touched rows to a device store.
  {
    const scoped_stage stage(res, "hypergraph.apply.incidence");
    applied.inserted_ids = ids;
    if (state.home == backend::cuda && !ids.empty()) {
      applied.inserted_ids_device = buffer<std::int64_t>(res, ids.size());
      copy<std::int64_t>(res, host_view(applied.inserted_ids), applied.inserted_ids_device.view());
    }
    applied.deleted_ids = live_deletions;
    applied.has_slack_record = true;
    applied.edited_ids = staged_.edited_ids;
    // The touched ids (sorted, unique) with their rows before (G_t: the deleted and edited rows;
    // an inserted id was not live) and after the batch (the inserted and edited rows): one merge
    // of the sorted deletions, insertions and edits (the edited ids are live and kept; an inserted
    // id may be one the batch deleted, under erase_first).
    std::int64_t id_end = std::max(store.id_bound(), staged_.id_bound);
    for (const std::int64_t id : ids) id_end = std::max(id_end, id + 1);
    std::vector<std::uint64_t> sorted_insertions(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i)
      sorted_insertions[i] = static_cast<std::uint64_t>(ids[i]) << 32 | i;
    hyper::radix_sort_by_key(sorted_insertions, static_cast<std::uint64_t>(id_end),
                             state.sort_scratch);
    const auto& edited = staged_.edited_ids;
    auto& touched = applied.touched_hyperedges;
    auto& pre = applied.touched_pre_rows;
    auto& post = applied.touched_post_rows;
    touched.clear();
    pre.clear();
    post.clear();
    touched.reserve(sorted_deletions.size() + sorted_insertions.size() + edited.size());
    pre.offsets.reserve(touched.capacity() + 1);
    post.offsets.reserve(touched.capacity() + 1);
    pre.members.reserve(applied.deleted_members.members.size() +
                        staged_.edited_pre_rows.members.size());
    post.members.reserve(rows.members.size() + staged_.edited_post_rows.members.size());
    std::vector<std::uint8_t> reused(live_deletions.size(), 0);
    const auto append_row = [](hyper::member_rows& to, const hyper::member_rows& from,
                               std::size_t r) {
      to.members.insert(to.members.end(), from.members.begin() + from.offsets[r],
                        from.members.begin() + from.offsets[r + 1]);
    };
    std::size_t d = 0, n = 0, e = 0;
    const auto key = [](std::uint64_t w) { return static_cast<std::int64_t>(w >> 32); };
    while (d < sorted_deletions.size() || n < sorted_insertions.size() || e < edited.size()) {
      const std::int64_t kd = d < sorted_deletions.size() ? key(sorted_deletions[d]) : INT64_MAX;
      const std::int64_t kn = n < sorted_insertions.size() ? key(sorted_insertions[n]) : INT64_MAX;
      const std::int64_t ke = e < edited.size() ? edited[e] : INT64_MAX;
      const std::int64_t id = std::min({kd, kn, ke});
      touched.push_back(id);
      const bool del = kd == id, ins = kn == id;
      if (del) {
        append_row(pre, applied.deleted_members,
                   static_cast<std::size_t>(sorted_deletions[d] & 0xffffffffULL));
        if (ins) reused[static_cast<std::size_t>(sorted_deletions[d] & 0xffffffffULL)] = 1;
        ++d;
      }
      if (ins) {
        append_row(post, rows, static_cast<std::size_t>(sorted_insertions[n] & 0xffffffffULL));
        ++n;
      }
      if (ke == id) {
        append_row(pre, staged_.edited_pre_rows, e);
        append_row(post, staged_.edited_post_rows, e);
        ++e;
      }
      pre.offsets.push_back(static_cast<std::int64_t>(pre.members.size()));
      post.offsets.push_back(static_cast<std::int64_t>(post.members.size()));
    }
    applied.touched_vertices = std::move(pre_vertices);  // sorted, unique
    for (const std::int64_t id : ids) {
      (id >= staged_.id_bound ? applied.new_ids : applied.recreated_ids).push_back(id);
    }
    for (std::size_t i = 0; i < live_deletions.size(); ++i)
      if (reused[i] == 0) applied.dead_ids.push_back(live_deletions[i]);
  }
  std::vector<std::int64_t> new_weights(ids.size(), 1);
  if constexpr (!is_unweighted_v<weight_t>) {
    for (std::size_t i = 0; i < ids.size(); ++i)
      new_weights[i] = static_cast<std::int64_t>(staged_.weights[i]);
  }
  {
    // The pair delta of the line graph (a device store ships the touched rows of the record).
    const scoped_stage stage(res, "hypergraph.apply.translate");
    applied.has_line_graph_delta = store.line_graph_enabled();
    hyper::touched_rows touched;
    touched.ids = &applied.touched_hyperedges;
    touched.pre = &applied.touched_pre_rows;
    touched.post = &applied.touched_post_rows;
    touched.vertices = &applied.touched_vertices;
    touched.vertex_lengths = &applied.touched_vertex_lengths;
    touched.dead_ids = &applied.dead_ids;
    touched.new_weights = &new_weights;
    store.line_graph_delta_of(applied.incidence_delta.vertices, applied.incidence_delta.ids,
                              applied.incidence_delta.signs, ids, touched, applied.pairs);
    applied.pairs_on_device = applied.has_line_graph_delta && store.pairs_on_device();
  }
  {
    const scoped_stage stage(res, "hypergraph.apply.line_graph");
    store.finish_line_graph(applied.pairs, ids, new_weights);
  }
  const scoped_stage stage(res, "hypergraph.apply.incidence");
  applied.inserted_incidences = static_cast<std::int64_t>(staged_.addition_ids.size());
  applied.deleted_incidences = static_cast<std::int64_t>(staged_.removal_ids.size());
  applied.skipped_incidence_edits = staged_.skipped_incidence_edits;
  applied.ignored_deletions = staged_.ignored_deletions;
  applied.dropped_members = staged_.dropped_members;
}

template <typename vertex_t, typename weight_t>
hyper_apply_summary hyper_transaction<vertex_t, weight_t>::commit(bool record_incidence_delta) try {
  if (committed_) {
    DYNG_FAIL(function_, ": a transaction commits once");
  }
  auto& state = hypergraph_access::state(h_);
  DYNG_EXPECTS(!state.poisoned, function_,
               ": the hypergraph was poisoned by a failed apply(); rebuild it with "
               "from_hyperedges()");
  DYNG_EXPECTS(state.version == staged_.version, function_, ": the hypergraph changed (version ",
               staged_.version, " -> ", state.version,
               ") between the transaction's begin and its commit");
  const resources& res = res_;
  const hyper::member_rows& rows = staged_.rows;
  const std::vector<std::int64_t>& live_deletions = staged_.deletions;
  const std::int64_t ignored = staged_.ignored_deletions;
  const std::int64_t dropped = staged_.dropped_members;
  hyper::incidence_store& store = *state.store;
  // ---- The store phase (basic guarantee: a failure poisons the hypergraph).
  hyper_applied applied;
  hyper_apply_summary summary;
  try {
    const container_scope container;  // the container's own work (budgets, I9)
    store.set_host_threads(resources_access::host_threads(res));
    // The store works on the stream it was built with: its work is ordered after the work queued
    // so far on the stream of `res` (a reader of G_t's views there, count(-) of dyng::update),
    // and the work later queued on that stream after the store's (ADR 0041 section 4).
    store.wait_for(res);
    hyper::grouped_incidences groups;
    std::vector<std::int64_t> ids;
    // The members of the deleted ids, read from the store (the callers do not supply them).
    store.read_rows(live_deletions, applied.deleted_members);
    if (!store.supports_incidence_edits()) {
      if (state.props.reuse == id_reuse::erase_first) {
        // ESCHER-GPU's DynamicHypergraph::update: H2V erase, V2H unfill, H2V insert, V2H fill with
        // the keys the insert assigned.
        {
          const scoped_stage delete_stage(res, "hypergraph.apply.delete");
          store.erase(live_deletions);
          hyper::group_by_vertex(applied.deleted_members, live_deletions, groups);
          store.v2h_remove(groups);
        }
        {
          const scoped_stage insert_stage(res, "hypergraph.apply.insert");
          store.insert(rows, ids);
          hyper::group_by_vertex(rows, ids, groups);
          store.v2h_append(groups);
        }
      } else {
        // MOSP_ESCHER's beginBatch / finishBatch: the H2V insert decides the ids first; then H2V
        // erase, V2H unfill, V2H fill.
        {
          const scoped_stage insert_stage(res, "hypergraph.apply.insert");
          store.insert(rows, ids);
        }
        {
          const scoped_stage delete_stage(res, "hypergraph.apply.delete");
          store.erase(live_deletions);
          hyper::group_by_vertex(applied.deleted_members, live_deletions, groups);
          store.v2h_remove(groups);
        }
        {
          const scoped_stage insert_stage(res, "hypergraph.apply.insert");
          hyper::group_by_vertex(rows, ids, groups);
          store.v2h_append(groups);
        }
      }
      if (store.line_graph_enabled()) {
        // ESCHER-GPU's maintain.h2h_merge: the sorted delta merge of the line graph.
        const scoped_stage line_graph_stage(res, "hypergraph.apply.line_graph");
        store.update_line_graph(live_deletions, ids, rows);
      }
      // The applied record.
      applied.inserted_ids = ids;
      if (state.home == backend::cuda && !ids.empty()) {
        applied.inserted_ids_device = buffer<std::int64_t>(res, ids.size());
        copy<std::int64_t>(res, host_view(applied.inserted_ids),
                           applied.inserted_ids_device.view());
      }
      applied.deleted_ids = live_deletions;
      // The touched hyperedges and vertices, sorted and unique, from the batch alone (reusable
      // marks: no O(|E| + |V|) pass per batch; sorting all incidences took about 16 ms of a 50K
      // batch on synthetic 1M, parity/results/M8b.md section 2.4, and a scan of every id and vertex
      // about 9 ms of every coauth batch, the M8b review).
      {
        std::int64_t id_end = store.id_bound();
        for (const std::int64_t id : live_deletions) id_end = std::max(id_end, id + 1);
        state.touched_ids.begin(static_cast<std::size_t>(id_end));
        for (const std::int64_t id : live_deletions) state.touched_ids.mark(id);
        for (const std::int64_t id : ids) state.touched_ids.mark(id);
        state.touched_ids.take_sorted(applied.touched_hyperedges);
        state.touched_vertices.begin(static_cast<std::size_t>(store.num_vertices()));
        for (const std::int32_t v : applied.deleted_members.members) state.touched_vertices.mark(v);
        for (const std::int32_t v : rows.members) state.touched_vertices.mark(v);
        state.touched_vertices.take_sorted(applied.touched_vertices);
      }
      applied.ignored_deletions = ignored;
      applied.dropped_members = dropped;
      if (record_incidence_delta) {
        applied.has_incidence_delta = true;
        auto& delta = applied.incidence_delta;
        const auto add = [&](const hyper::member_rows& r, const std::vector<std::int64_t>& rid,
                             std::int8_t sign) {
          for (std::int64_t i = 0; i < r.size(); ++i) {
            for (std::int64_t k = r.offsets[static_cast<std::size_t>(i)];
                 k < r.offsets[static_cast<std::size_t>(i) + 1]; ++k) {
              delta.vertices.push_back(r.members[static_cast<std::size_t>(k)]);
              delta.ids.push_back(rid[static_cast<std::size_t>(i)]);
              delta.signs.push_back(sign);
            }
          }
        };
        add(applied.deleted_members, live_deletions, -1);
        add(rows, ids, +1);
      }
    } else {
      commit_slack_csr(applied, ids);
    }
    if constexpr (!is_unweighted_v<weight_t>) {
      // The weights by id are updated in place after the commit point (ADR 0088): only the room
      // for a larger bound is allocated here, inside the store phase (a failure poisons).
      state.weights.reserve(static_cast<std::size_t>(store.id_bound()));
    }
    // One read of the device error words, then the row views of the new state.
    store.finish_apply();
    store.signal_to(res);
    // The summary, built here: everything that allocates comes before the commit point.
    summary.inserted_hyperedges = static_cast<std::int64_t>(applied.inserted_ids.size());
    summary.deleted_hyperedges = static_cast<std::int64_t>(live_deletions.size());
    summary.ignored_deletions = ignored;
    summary.dropped_members = dropped;
    summary.inserted_incidences = static_cast<std::int64_t>(staged_.addition_ids.size());
    summary.deleted_incidences = static_cast<std::int64_t>(staged_.removal_ids.size());
    summary.skipped_ops = ignored + dropped + staged_.skipped_incidence_edits;
    summary.num_hyperedges_after = store.num_live();
    summary.id_bound_after = store.id_bound();
    summary.inserted_ids = applied.inserted_ids;
  } catch (...) {
    state.poisoned = true;
    throw;
  }
  // ---- The commit (no-throw: integer assignments, the weights written into reserved room, and
  // moves only).
  applied.version_before = state.version;
  applied.state_before = state.state_id;
  ++state.version;
  state.state_id = next_graph_state_id();
  applied.version_after = state.version;
  applied.state_after = state.state_id;
  if constexpr (!is_unweighted_v<weight_t>) {
    // In place, O(batch) (ADR 0088; the copy of every weight per batch is gone): the deleted ids
    // weigh 0, the bound moves (a smaller bound drops the dead tail, a larger one fits the
    // reserved room: no allocation), then the inserted ids get theirs (one may be an id the batch
    // deleted, under erase_first).
    auto& w = state.weights;
    for (const std::int64_t id : live_deletions) w[static_cast<std::size_t>(id)] = weight_t{0};
    w.resize(static_cast<std::size_t>(summary.id_bound_after), weight_t{0});
    for (std::size_t i = 0; i < summary.inserted_ids.size(); ++i) {
      w[static_cast<std::size_t>(summary.inserted_ids[i])] = staged_.weights[i];
    }
  }
  static_assert(std::is_nothrow_move_assignable_v<hyper_applied>,
                "the commit moves the applied record after the commit point");
  state.applied = std::move(applied);
  committed_ = true;
  return summary;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE(function_, " (", h_.hyperedge_id_bound(), " ids; batch of ",
                                  staged_.requested_insertions, " insertions, ",
                                  staged_.requested_deletions, " deletions)")

}  // namespace detail

#define DYNG_INSTANTIATE_HYPERGRAPH(V, W)                                                    \
  template class hypergraph<V, W>;                                                           \
  template class detail::hyper_transaction<V, W>;                                            \
  template hypergraph<V, W> detail::hypergraph_access::build<V, W>(                          \
      const resources&, hyperedge_list_view<V, W>, const hypergraph_properties&,             \
      const detail::hyper::slack::knobs&);                                                   \
  template hyper_apply_summary detail::hypergraph_access::apply<V, W>(                       \
      const resources&, hypergraph<V, W>&, const hyperedge_batch_view<V, W>&, bool);         \
  template void detail::hypergraph_access::expect_placement<V, W>(                           \
      const resources&, const hypergraph<V, W>&, const char*);                               \
  template void detail::hypergraph_access::require_line_graph<V, W>(const hypergraph<V, W>&, \
                                                                    const char*);
DYNG_FOR_EACH_HYPERGRAPH_TYPE(DYNG_INSTANTIATE_HYPERGRAPH)
#undef DYNG_INSTANTIATE_HYPERGRAPH

}  // namespace dyng
