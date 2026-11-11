// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/checker.cu (DynLPChecker: the host model)
/**
 * @file lp_host_model.cpp
 * @brief The exact host model of label_propagation's slot rows and deterministic stages (see
 *        lp_host_model.hpp). Standard library only.
 */
#include <dyng/testing/lp_host_model.hpp>
#include <dyng/testing/union_find.hpp>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dyng::testing {

namespace {

/// O's fixed-point scale of the supernode weights (labelInitialization.cu: WEIGHT_SCALE).
constexpr double weight_scale = 4294967296.0;  // 2^32

/// __double2ull_rn(x) for 0 <= x < 2^64: round to nearest, ties to even.
std::uint64_t round_to_u64(double x) {
  const int mode = std::fegetround();
  if (mode != FE_TONEAREST) {
    std::fesetround(FE_TONEAREST);
  }
  const double r = std::nearbyint(x);
  if (mode != FE_TONEAREST) {
    std::fesetround(mode);
  }
  return static_cast<std::uint64_t>(r);
}

}  // namespace

lp_host_model::lp_host_model(const std::vector<std::int64_t>& row_capacity) {
  const std::size_t n = row_capacity.size();
  state_.assign(n, vertex_state::inactive);
  present_.assign(n, 0);
  previous_.assign(n, 0);
  seed_.assign(n, -1);
  row_ptr_.assign(n + 1, 0);
  for (std::size_t v = 0; v < n; ++v) {
    if (row_capacity[v] < 0) {
      throw std::invalid_argument("lp_host_model: negative row capacity of vertex " +
                                  std::to_string(v));
    }
    row_ptr_[v + 1] = row_ptr_[v] + row_capacity[v];
  }
  row_end_.assign(row_ptr_.begin(), row_ptr_.end() - 1);
  col_.assign(static_cast<std::size_t>(row_ptr_[n]), -1);
  weight_.assign(static_cast<std::size_t>(row_ptr_[n]), 0.0F);
}

float lp_host_model::average_weight(const std::vector<std::int32_t>& col_ind,
                                    const std::vector<float>& weights) {
  if (col_ind.size() != weights.size()) {
    throw std::invalid_argument("lp_host_model::average_weight: sizes differ");
  }
  if (col_ind.empty()) {
    return 0.5F;
  }
  double sum = 0.0;
  std::int64_t count = 0;
  for (std::size_t i = 0; i < col_ind.size(); ++i) {
    if (col_ind[i] >= 0) {
      sum += weights[i];
      ++count;
    }
  }
  return count > 0 ? static_cast<float>(sum / static_cast<double>(count)) : 0.5F;
}

lp_model_report lp_host_model::apply(const lp_model_batch& batch, float tau) {
  const std::size_t n = state_.size();
  const auto in_range = [n](std::int32_t v) { return v >= 0 && static_cast<std::size_t>(v) < n; };
  // --- Validation, before any change ------------------------------------------------------------
  if (batch.insert_labels.size() != batch.insert_vertices.size()) {
    throw std::invalid_argument("lp_host_model::apply: one label per inserted vertex is needed");
  }
  if (batch.edge_destinations.size() != batch.edge_sources.size() ||
      batch.edge_weights.size() != batch.edge_sources.size()) {
    throw std::invalid_argument("lp_host_model::apply: edge arrays of different sizes");
  }
  for (const auto* ids : {&batch.insert_vertices, &batch.delete_vertices, &batch.edge_sources,
                          &batch.edge_destinations}) {
    for (const std::int32_t v : *ids) {
      if (!in_range(v)) {
        throw std::invalid_argument("lp_host_model::apply: vertex id " + std::to_string(v) +
                                    " out of range [0, " + std::to_string(n) + ")");
      }
    }
  }
  std::vector<std::uint8_t> in_del(n, 0);
  for (const std::int32_t v : batch.delete_vertices) {
    in_del[static_cast<std::size_t>(v)] = 1;
  }
  for (std::size_t i = 0; i < batch.insert_vertices.size(); ++i) {
    const std::int8_t label = batch.insert_labels[i];
    if (label < -1 || label > 1) {
      throw std::invalid_argument("lp_host_model::apply: a seed label is not -1, 0 or 1");
    }
    if (label >= 0 && in_del[static_cast<std::size_t>(batch.insert_vertices[i])] != 0) {
      throw std::invalid_argument("lp_host_model::apply: seeded vertex " +
                                  std::to_string(batch.insert_vertices[i]) +
                                  " is also deleted in the batch");
    }
  }

  lp_model_report out;
  previous_ = present_;
  // Ins_eff = Ins \ Del (Ins order), Del_eff = Del ∩ V_t (Del order).
  for (const std::int32_t v : batch.insert_vertices) {
    if (in_del[static_cast<std::size_t>(v)] == 0) {
      out.inserted.push_back(v);
    }
  }
  for (const std::int32_t v : batch.delete_vertices) {
    if (previous_[static_cast<std::size_t>(v)] != 0) {
      out.deleted.push_back(v);
    }
  }

  // N_t(Del_eff): the deleted vertices' rows before this batch's appends.
  std::vector<std::uint8_t> affected(n, 0);
  for (const std::int32_t u : out.deleted) {
    const auto uu = static_cast<std::size_t>(u);
    for (std::int64_t k = row_ptr_[uu]; k < row_end_[uu]; ++k) {
      const std::int32_t v = col_[static_cast<std::size_t>(k)];
      if (v >= 0) {
        affected[static_cast<std::size_t>(v)] = 1;
      }
    }
  }

  // Statuses: every listed deletion retires its id, then the effective insertions become active
  // (an id that is not inactive raises the reinserted flag), then the seeds.
  for (const std::int32_t v : batch.delete_vertices) {
    state_[static_cast<std::size_t>(v)] = vertex_state::deleted;
  }
  for (const std::int32_t v : out.inserted) {
    const auto vv = static_cast<std::size_t>(v);
    if (state_[vv] != vertex_state::inactive) {
      reinserted_ = std::max(reinserted_, v + 1);
    }
    state_[vv] = vertex_state::active;
  }
  for (std::size_t i = 0; i < batch.insert_vertices.size(); ++i) {
    if (batch.insert_labels[i] >= 0) {
      seed_[static_cast<std::size_t>(batch.insert_vertices[i])] = batch.insert_labels[i];
    }
  }
  for (std::size_t v = 0; v < n; ++v) {
    present_[v] = state_[v] == vertex_state::active ? 1 : 0;
  }
  for (const std::int32_t v : out.inserted) {
    affected[static_cast<std::size_t>(v)] = 1;
  }

  // Appends in half-edge-id order: per edge, v into u's row, then u into v's row.
  const auto append = [&](std::int32_t row, std::int32_t neighbor, float w) {
    const auto r = static_cast<std::size_t>(row);
    if (row_end_[r] >= row_ptr_[r + 1]) {
      overflow_ = 1;
      ++out.dropped_overflow;
      return;
    }
    const auto slot = static_cast<std::size_t>(row_end_[r]++);
    col_[slot] = neighbor;
    weight_[slot] = w;
    ++out.appended;
  };
  for (std::size_t e = 0; e < batch.edge_sources.size(); ++e) {
    const std::int32_t u = batch.edge_sources[e];
    const std::int32_t v = batch.edge_destinations[e];
    if (present_[static_cast<std::size_t>(u)] == 0 || present_[static_cast<std::size_t>(v)] == 0) {
      ++out.dropped_absent;
      continue;
    }
    append(u, v, batch.edge_weights[e]);
    append(v, u, batch.edge_weights[e]);
  }

  // N_{t+1}(Ins_eff): the inserted vertices' rows after the appends.
  for (const std::int32_t u : out.inserted) {
    const auto uu = static_cast<std::size_t>(u);
    for (std::int64_t k = row_ptr_[uu]; k < row_end_[uu]; ++k) {
      const std::int32_t v = col_[static_cast<std::size_t>(k)];
      if (v >= 0) {
        affected[static_cast<std::size_t>(v)] = 1;
      }
    }
  }
  for (std::size_t v = 0; v < n; ++v) {
    if (affected[v] != 0 && present_[v] != 0 && seed_[v] < 0) {
      out.affected.push_back(static_cast<std::int32_t>(v));
    }
  }

  // G': new edges with w > tau between two vertices of Ins_eff.
  std::vector<std::uint8_t> is_new(n, 0);
  for (const std::int32_t v : out.inserted) {
    is_new[static_cast<std::size_t>(v)] = 1;
  }
  std::vector<std::pair<std::int32_t, std::int32_t>> sub;
  for (std::size_t e = 0; e < batch.edge_sources.size(); ++e) {
    const std::int32_t u = batch.edge_sources[e];
    const std::int32_t v = batch.edge_destinations[e];
    if (batch.edge_weights[e] > tau && is_new[static_cast<std::size_t>(u)] != 0 &&
        is_new[static_cast<std::size_t>(v)] != 0) {
      sub.emplace_back(std::min(u, v), std::max(u, v));
    }
  }
  std::sort(sub.begin(), sub.end());
  out.sparsified.reserve(2 * sub.size());
  for (const auto& [a, b] : sub) {
    out.sparsified.push_back(a);
    out.sparsified.push_back(b);
  }

  // Components of G' on Ins_eff: root = smallest id, id = the root's rank in Ins_eff order.
  if (!out.inserted.empty()) {
    union_find sets(static_cast<std::int64_t>(n));
    for (const auto& [a, b] : sub) {
      sets.unite(a, b);
    }
    std::vector<std::int32_t> root_id(n, -1);
    for (const std::int32_t v : out.inserted) {
      if (sets.find(v) == v) {
        root_id[static_cast<std::size_t>(v)] = out.num_components++;
      }
    }
    out.component_of_inserted.reserve(out.inserted.size());
    for (const std::int32_t v : out.inserted) {
      out.component_of_inserted.push_back(root_id[static_cast<std::size_t>(sets.find(v))]);
    }

    // Supernode weights and initial labels.
    out.w0.assign(static_cast<std::size_t>(out.num_components), 0);
    out.w1.assign(static_cast<std::size_t>(out.num_components), 0);
    for (std::size_t i = 0; i < out.inserted.size(); ++i) {
      const auto u = static_cast<std::size_t>(out.inserted[i]);
      const auto c = static_cast<std::size_t>(out.component_of_inserted[i]);
      for (std::int64_t k = row_ptr_[u]; k < row_end_[u]; ++k) {
        const std::int32_t v = col_[static_cast<std::size_t>(k)];
        if (v < 0 || seed_[static_cast<std::size_t>(v)] < 0) {
          continue;
        }
        const std::uint64_t w =
            round_to_u64(static_cast<double>(weight_[static_cast<std::size_t>(k)]) * weight_scale);
        (seed_[static_cast<std::size_t>(v)] == 0 ? out.w0[c] : out.w1[c]) += w;
      }
    }
    out.initial_labels.reserve(out.inserted.size());
    for (std::size_t i = 0; i < out.inserted.size(); ++i) {
      const std::int8_t s = seed_[static_cast<std::size_t>(out.inserted[i])];
      if (s == 0 || s == 1) {
        out.initial_labels.push_back(s == 0 ? 0.0F : 1.0F);
        continue;
      }
      const auto c = static_cast<std::size_t>(out.component_of_inserted[i]);
      const double l0 = static_cast<double>(out.w0[c]) / weight_scale;
      const double l1 = static_cast<double>(out.w1[c]) / weight_scale;
      const double denominator = l0 + l1;
      out.initial_labels.push_back(denominator > 1e-10 ? static_cast<float>(l1 / denominator)
                                                       : 0.5F);
    }
  }
  return out;
}

}  // namespace dyng::testing
