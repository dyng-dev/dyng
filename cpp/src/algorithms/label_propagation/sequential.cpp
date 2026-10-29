// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/iterativePropagation.cu (halfSweep,
// accumulateNeighbors, updateLabel, appendNextFrontier, launchIterativePropagation) and
// src/validation.cu (irLPKernel, launchIrLP, runIrLPSnapshot)
/**
 * @file sequential.cpp
 * @brief label_propagation's sequential backend (ADR 0072): the host propagation and IrLP.
 *
 * There is no host DynLP in the original; this backend is new, written to the original's rules so
 * that its deterministic stages are the original's exactly and its labels meet the same oracle:
 * the propagation runs the queues of the original's launchIterativePropagation with its run-wide
 * tags and its hashed half-sweeps, and each half in ascending vertex order (in_place: Gauss-
 * Seidel, a vertex reads the labels written earlier in the sweep; damped: the side array committed
 * after both halves); updateLabel's 1e-10 guard, the damping and the clamp are the original's
 * arithmetic in single precision (this file is compiled with -ffp-contract=off, so the result is
 * the same on every host build). IrLP is runIrLPSnapshot on the host: the column-sorted snapshot
 * of the present vertices, double-buffered Jacobi, the ground truth restored every iteration.
 */
#include "algorithms/label_propagation/problem.hpp"
#include "graph/graph_impl.hpp"

#include <dyng/graph/graph_view.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace dyng::detail {

namespace {

/// The original's halfSweep(v, tag): the half-sweep (0 or 1) of vertex v in the iteration of
/// the queue with tag `tag` (a hash that separates neighbours with probability 1/2).
inline int half_sweep(std::int64_t v, std::int64_t tag) noexcept {
  std::uint32_t x = static_cast<std::uint32_t>(v) ^ (static_cast<std::uint32_t>(tag) * 0x9E3779B9U);
  x ^= x >> 16;
  x *= 0x7FEB352DU;
  x ^= x >> 15;
  x *= 0x846CA68BU;
  x ^= x >> 16;
  return static_cast<int>(x & 1U);
}

}  // namespace

template <typename edge_t>
lp_rows<edge_t> host_rows(const graph<std::int32_t, edge_t, float>& g) {
  const auto& state = graph_access::impl(g);
  const csr<std::int32_t, edge_t, float>& slots = state.host_edges();
  lp_rows<edge_t> rows;
  rows.n = slots.num_vertices();
  rows.row_ptr = slots.row_ptr.data();
  rows.col_ind = slots.col_ind.data();
  rows.weights = slots.weights.data();
  if (state.is_slotted()) {
    rows.row_end = state.slotted.row_end.data();
    rows.status = state.slotted.status.data();
  }
  return rows;
}

template <typename edge_t>
lp_propagation_report propagate_sequential(const lp_rows<edge_t>& rows,
                                           const std::vector<std::int8_t>& seeds,
                                           std::vector<float>& labels,
                                           std::vector<std::int64_t>& stamp, std::int64_t& next_tag,
                                           const std::vector<std::uint8_t>& affected, bool damped,
                                           float delta, std::int64_t max_iterations,
                                           label_propagation_workspace& ws) {
  const std::int64_t n = rows.n;
  lp_propagation_report report;
  std::vector<std::int32_t>& frontier = ws.frontier;
  std::vector<std::int32_t>& next = ws.next;
  std::vector<std::int32_t>& halves = ws.halves;
  std::vector<float>& side = ws.side;
  frontier.clear();
  // The first queue (initFrontierKernel): affected, active, without a seed label; tag = the
  // result's next tag (the original's ws.nextTag++).
  const std::int64_t first = next_tag;
  for (std::int64_t v = 0; v < n; ++v) {
    const auto vv = static_cast<std::size_t>(v);
    if (affected[vv] != 0 && rows.active(v) && seeds[vv] < 0) {
      stamp[vv] = first;
      frontier.push_back(static_cast<std::int32_t>(v));
    }
  }
  report.initial_frontier = static_cast<std::int64_t>(frontier.size());
  std::int64_t tag = first;
  while (!frontier.empty() && report.iterations < max_iterations) {
    const std::int64_t out_tag = tag + 1;
    next.clear();
    report.frontier_visits += static_cast<std::int64_t>(frontier.size());
    // The two half-sweeps of this queue, each in ascending vertex order.
    halves.clear();
    for (const std::int32_t v : frontier) {
      if (half_sweep(v, tag) == 0) {
        halves.push_back(v);
      }
    }
    const std::size_t middle = halves.size();
    for (const std::int32_t v : frontier) {
      if (half_sweep(v, tag) != 0) {
        halves.push_back(v);
      }
    }
    std::sort(halves.begin(), halves.begin() + static_cast<std::ptrdiff_t>(middle));
    std::sort(halves.begin() + static_cast<std::ptrdiff_t>(middle), halves.end());
    if (damped) {
      side.resize(halves.size());
    }
    const auto append = [&](std::int32_t v) {
      std::int64_t& s = stamp[static_cast<std::size_t>(v)];
      if (s == out_tag) {
        return;  // once per iteration (appendToQueue)
      }
      s = out_tag;
      next.push_back(v);
    };
    for (std::size_t i = 0; i < halves.size(); ++i) {
      const std::int32_t u = halves[i];
      const auto uu = static_cast<std::size_t>(u);
      const auto begin = static_cast<std::size_t>(rows.row_ptr[uu]);
      const auto end = static_cast<std::size_t>(rows.end(u));
      // accumulateNeighbors: the entries to vertices that are not deleted.
      float w_all = 0.0F;
      float weighted = 0.0F;
      for (std::size_t e = begin; e < end; ++e) {
        const std::int32_t v = rows.col_ind[e];
        if (v < 0 || !rows.active(v)) {
          continue;
        }
        const float w = rows.weights[e];
        w_all += w;
        weighted += w * labels[static_cast<std::size_t>(v)];
      }
      // updateLabel.
      const float old = labels[uu];
      float updated = (w_all > 1e-10F) ? weighted / w_all : old;
      if (damped) {
        updated = 0.5F * (old + updated);
      }
      updated = std::fmin(std::fmax(updated, 0.0F), 1.0F);
      if (damped) {
        side[i] = updated;
      } else {
        labels[uu] = updated;
      }
      if (std::fabs(updated - old) > delta) {
        // appendNextFrontier: the active unlabelled neighbours, then u.
        for (std::size_t e = begin; e < end; ++e) {
          const std::int32_t v = rows.col_ind[e];
          if (v >= 0 && rows.active(v) && seeds[static_cast<std::size_t>(v)] < 0) {
            append(v);
          }
        }
        append(u);
      }
    }
    if (damped) {
      for (std::size_t i = 0; i < halves.size(); ++i) {
        labels[static_cast<std::size_t>(halves[i])] = side[i];
      }
    }
    frontier.swap(next);
    tag = out_tag;
    ++report.iterations;
  }
  report.capped = !frontier.empty();
  next_tag = tag + 1;  // the original's ws.nextTag = tag + 1
  return report;
}

template <typename edge_t>
lp_irlp_report irlp_sequential(const lp_rows<edge_t>& rows, const std::vector<std::int8_t>& seeds,
                               std::vector<float>& labels, float delta, std::int64_t max_iterations,
                               label_propagation_workspace& ws) {
  const std::int64_t n = rows.n;
  const auto nn = static_cast<std::size_t>(n);
  // The snapshot (runIrLPSnapshot): the entries between present vertices, each row sorted by
  // column (stable: parallel entries keep their order), as the original's full graph's rows are.
  csr<std::int32_t, std::int64_t, float>& snap = ws.snapshot;
  snap.num_weights = 1;
  snap.row_ptr.assign(nn + 1, 0);
  for (std::int64_t u = 0; u < n; ++u) {
    std::int64_t count = 0;
    if (rows.active(u)) {
      for (auto e = static_cast<std::size_t>(rows.row_ptr[u]);
           e < static_cast<std::size_t>(rows.end(u)); ++e) {
        const std::int32_t v = rows.col_ind[e];
        count += (v >= 0 && rows.active(v)) ? 1 : 0;
      }
    }
    snap.row_ptr[static_cast<std::size_t>(u) + 1] =
        snap.row_ptr[static_cast<std::size_t>(u)] + count;
  }
  const auto m = static_cast<std::size_t>(snap.row_ptr[nn]);
  snap.col_ind.resize(m);
  snap.weights.resize(m);
  std::vector<std::int64_t>& order = ws.snapshot_count;
  for (std::int64_t u = 0; u < n; ++u) {
    if (!rows.active(u)) {
      continue;
    }
    const auto first = static_cast<std::size_t>(snap.row_ptr[static_cast<std::size_t>(u)]);
    std::size_t at = first;
    for (auto e = static_cast<std::size_t>(rows.row_ptr[u]);
         e < static_cast<std::size_t>(rows.end(u)); ++e) {
      const std::int32_t v = rows.col_ind[e];
      if (v >= 0 && rows.active(v)) {
        snap.col_ind[at] = v;
        snap.weights[at] = rows.weights[e];
        ++at;
      }
    }
    // Sort the row by column, stably (an index sort keeps the weights with their columns).
    const std::size_t len = at - first;
    if (len > 1) {
      order.resize(len);
      std::iota(order.begin(), order.end(), std::int64_t{0});
      std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
        return snap.col_ind[first + static_cast<std::size_t>(a)] <
               snap.col_ind[first + static_cast<std::size_t>(b)];
      });
      std::vector<float>& wbuf = ws.irlp_next;  // free until the iterations
      std::vector<std::int32_t>& cbuf = ws.next;
      wbuf.resize(len);
      cbuf.resize(len);
      for (std::size_t i = 0; i < len; ++i) {
        cbuf[i] = snap.col_ind[first + static_cast<std::size_t>(order[i])];
        wbuf[i] = snap.weights[first + static_cast<std::size_t>(order[i])];
      }
      std::copy(cbuf.begin(), cbuf.end(),
                snap.col_ind.begin() + static_cast<std::ptrdiff_t>(first));
      std::copy(wbuf.begin(), wbuf.end(),
                snap.weights.begin() + static_cast<std::ptrdiff_t>(first));
    }
  }
  // The starting labels: 0.5, a present seed its class (the snapshot's ground truth).
  const auto seeded = [&](std::size_t v) {
    return seeds[v] >= 0 && rows.active(static_cast<std::int64_t>(v));
  };
  labels.assign(nn, 0.5F);
  for (std::size_t v = 0; v < nn; ++v) {
    if (seeded(v)) {
      labels[v] = seeds[v] == 0 ? 0.0F : 1.0F;
    }
  }
  std::vector<float>& other = ws.irlp_next;
  other = labels;
  std::vector<float>* old_labels = &labels;
  std::vector<float>* new_labels = &other;
  lp_irlp_report report;
  bool changed = false;
  for (std::int64_t iteration = 0; iteration < max_iterations; ++iteration) {
    changed = false;
    const std::vector<float>& f = *old_labels;
    std::vector<float>& g = *new_labels;
    for (std::size_t u = 0; u < nn; ++u) {
      if (seeded(u) || !rows.active(static_cast<std::int64_t>(u))) {
        g[u] = f[u];  // ground truth, and the vertices that are not present, keep their label
        continue;
      }
      float w_all = 0.0F;
      float weighted = 0.0F;
      for (auto e = static_cast<std::size_t>(snap.row_ptr[u]);
           e < static_cast<std::size_t>(snap.row_ptr[u + 1]); ++e) {
        const float w = snap.weights[e];
        w_all += w;
        weighted += w * f[static_cast<std::size_t>(snap.col_ind[e])];
      }
      const float old = f[u];
      float updated = (w_all > 1e-10F) ? weighted / w_all : old;
      updated = std::fmin(std::fmax(updated, 0.0F), 1.0F);
      g[u] = updated;
      changed = changed || std::fabs(updated - old) > delta;
    }
    for (std::size_t v = 0; v < nn; ++v) {  // irLPEnforceGTKernel
      if (seeded(v)) {
        g[v] = seeds[v] == 0 ? 0.0F : 1.0F;
      }
    }
    std::swap(old_labels, new_labels);
    report.iterations = iteration + 1;
    if (!changed) {
      break;
    }
  }
  report.capped = changed;
  if (old_labels != &labels) {
    labels.swap(other);
  }
  return report;
}

template <typename edge_t>
float average_live_weight(const lp_rows<edge_t>& rows) {
  double sum = 0.0;
  std::int64_t count = 0;
  for (std::int64_t u = 0; u < rows.n; ++u) {
    if (!rows.active(u)) {
      continue;
    }
    for (auto e = static_cast<std::size_t>(rows.row_ptr[u]);
         e < static_cast<std::size_t>(rows.end(u)); ++e) {
      const std::int32_t v = rows.col_ind[e];
      if (v >= 0 && rows.active(v)) {
        sum += rows.weights[e];
        ++count;
      }
    }
  }
  return count > 0 ? static_cast<float>(sum / static_cast<double>(count)) : 0.5F;
}

#define DYNG_INSTANTIATE_LP_SEQUENTIAL(E)                                                        \
  template lp_rows<E> host_rows<E>(const graph<std::int32_t, E, float>&);                        \
  template lp_propagation_report propagate_sequential<E>(                                        \
      const lp_rows<E>&, const std::vector<std::int8_t>&, std::vector<float>&,                   \
      std::vector<std::int64_t>&, std::int64_t&, const std::vector<std::uint8_t>&, bool, float,  \
      std::int64_t, label_propagation_workspace&);                                               \
  template lp_irlp_report irlp_sequential<E>(const lp_rows<E>&, const std::vector<std::int8_t>&, \
                                             std::vector<float>&, float, std::int64_t,           \
                                             label_propagation_workspace&);                      \
  template float average_live_weight<E>(const lp_rows<E>&);
DYNG_INSTANTIATE_LP_SEQUENTIAL(std::int32_t)
DYNG_INSTANTIATE_LP_SEQUENTIAL(std::int64_t)
#undef DYNG_INSTANTIATE_LP_SEQUENTIAL

}  // namespace dyng::detail
