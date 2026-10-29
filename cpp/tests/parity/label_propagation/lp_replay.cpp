// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file lp_replay.cpp
 * @brief The original's layouts of the generator's arrays and of lp_host_model's predictions
 *        (see lp_replay.hpp).
 */
#include "parity/label_propagation/lp_replay.hpp"

#include "support/sha256.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace dyng::test::lp {

namespace {

template <typename value_t>
named_array array_of(const std::vector<value_t>& v) {
  named_array a;
  a.count = v.size();
  a.bytes.resize(v.size() * sizeof(value_t));
  if (!v.empty()) {
    std::memcpy(a.bytes.data(), v.data(), a.bytes.size());
  }
  return a;
}

template <typename value_t>
named_array view_of(const std::vector<value_t>& v) {
  named_array a;
  a.count = v.size();
  a.view = reinterpret_cast<const std::uint8_t*>(v.data());
  a.view_size = v.size() * sizeof(value_t);
  if (a.view == nullptr) {
    static const std::uint8_t empty = 0;
    a.view = &empty;  // an empty vector: a non-null view of 0 bytes
  }
  return a;
}

template <typename value_t>
named_array scalar_of(value_t v) {
  return array_of(std::vector<value_t>{v});
}

std::vector<std::int32_t> narrow(const std::vector<std::int64_t>& v) {
  return std::vector<std::int32_t>(v.begin(), v.end());
}

}  // namespace

std::vector<std::int32_t> original_statuses(const testing::lp_host_model& model) {
  using state = testing::lp_host_model::vertex_state;
  const auto& states = model.states();
  const auto& seeds = model.seed_labels();
  std::vector<std::int32_t> out(states.size());
  for (std::size_t v = 0; v < states.size(); ++v) {
    switch (states[v]) {
      case state::inactive:
        out[v] = original_inactive;
        break;
      case state::deleted:
        out[v] = original_deleted;
        break;
      case state::active:
        out[v] = seeds[v] >= 0 ? original_ground_truth : original_active;
        break;
    }
  }
  return out;
}

testing::lp_model_batch model_batch(const detail::dynlp_legacy_batch& batch,
                                    std::int32_t num_vertices) {
  testing::lp_model_batch m;
  m.insert_vertices = batch.inserted;
  m.insert_labels.assign(batch.inserted.size(), -1);
  std::vector<std::int64_t> position(static_cast<std::size_t>(num_vertices), -1);
  for (std::size_t i = 0; i < batch.inserted.size(); ++i) {
    position[static_cast<std::size_t>(batch.inserted[i])] = static_cast<std::int64_t>(i);
  }
  for (std::size_t i = 0; i < batch.gt_ids.size(); ++i) {
    const std::int64_t p = position[static_cast<std::size_t>(batch.gt_ids[i])];
    if (p < 0) {
      throw std::runtime_error("model_batch: ground-truth vertex " +
                               std::to_string(batch.gt_ids[i]) + " does not arrive");
    }
    m.insert_labels[static_cast<std::size_t>(p)] = batch.gt_labels[i] < 0.5F ? 0 : 1;
  }
  m.delete_vertices = batch.deleted;
  m.edge_sources = batch.edge_src;
  m.edge_destinations = batch.edge_dst;
  m.edge_weights = batch.edge_weight;
  return m;
}

std::vector<std::int64_t> row_capacities(const std::vector<std::int32_t>& row_ptr) {
  std::vector<std::int64_t> cap(row_ptr.empty() ? 0 : row_ptr.size() - 1);
  for (std::size_t v = 0; v < cap.size(); ++v) {
    cap[v] = static_cast<std::int64_t>(row_ptr[v + 1]) - row_ptr[v];
  }
  return cap;
}

std::map<std::string, named_array> generator_arrays(const detail::dynlp_legacy_output& out) {
  std::map<std::string, named_array> a;
  a["row_ptr"] = view_of(out.row_ptr);
  a["col_ind"] = view_of(out.col_ind);
  a["values"] = view_of(out.values);
  a["true_labels"] = view_of(out.true_labels);
  for (std::size_t b = 0; b < out.batches.size(); ++b) {
    const std::string p = "b" + std::to_string(b) + ".";
    const detail::dynlp_legacy_batch& x = out.batches[b];
    a[p + "inserted"] = view_of(x.inserted);
    a[p + "gt_ids"] = view_of(x.gt_ids);
    a[p + "gt_labels"] = view_of(x.gt_labels);
    a[p + "deleted"] = view_of(x.deleted);
    a[p + "edge_src"] = view_of(x.edge_src);
    a[p + "edge_dst"] = view_of(x.edge_dst);
    a[p + "edge_weight"] = view_of(x.edge_weight);
  }
  return a;
}

std::map<std::string, named_array> model_fields(const testing::lp_host_model& model,
                                                const testing::lp_model_report& report, float tau,
                                                std::int32_t batch) {
  std::map<std::string, named_array> f;
  f["apply_vertex_changes.status"] = array_of(original_statuses(model));
  std::vector<std::int32_t> gt_class(model.seed_labels().begin(), model.seed_labels().end());
  f["apply_vertex_changes.gt_class"] = array_of(gt_class);
  f["apply_vertex_changes.inserted"] = array_of(report.inserted);
  f["apply_vertex_changes.num_inserted"] =
      scalar_of(static_cast<std::int32_t>(report.inserted.size()));
  if (batch == 0) {
    f["append_edges.row_ptr"] = array_of(narrow(model.row_ptr()));
  }
  f["append_edges.row_end"] = array_of(narrow(model.row_end()));
  f["append_edges.slot_col_ind"] = array_of(model.slot_columns());
  f["append_edges.slot_values"] = array_of(model.slot_weights());
  f["append_edges.error_flags"] =
      array_of(std::vector<std::int32_t>{model.row_overflow_flag(), model.reinserted_flag()});
  if (!report.inserted.empty()) {
    f["sparsify.tau"] = scalar_of(tau);
    f["sparsify.num_sub_edges"] =
        scalar_of(static_cast<std::int32_t>(report.sparsified.size() / 2));
    f["sparsify.sub_edges_sorted"] = array_of(report.sparsified);
    f["connected_components.num_components"] = scalar_of(report.num_components);
    f["connected_components.component_of_inserted"] = array_of(report.component_of_inserted);
    f["label_init.num_components"] = scalar_of(report.num_components);
    f["label_init.w0"] = array_of(report.w0);
    f["label_init.w1"] = array_of(report.w1);
    f["label_init.initial_labels_of_inserted"] = array_of(report.initial_labels);
  }
  f["propagation_begin.frontier_size"] =
      scalar_of(static_cast<std::int32_t>(report.affected.size()));
  f["propagation_begin.initial_frontier"] = array_of(report.affected);
  return f;
}

std::string digest(const named_array& a) {
  sha256 h;
  h.update(a.data(), a.size());
  return h.hex();
}

}  // namespace dyng::test::lp
