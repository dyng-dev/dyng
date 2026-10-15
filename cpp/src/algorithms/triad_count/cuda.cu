// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:src/driver.cu (run(): count(), subtract() on the snapshot before
// graph.update(), add() on the snapshot after it, with the keys the CBST insert assigned)
/**
 * @file cuda.cu
 * @brief The CUDA backend of triad_count: ESCHER-GPU's counters (engine/) behind the hooks of the
 *        problem (problem.hpp).
 *
 * The transplant (commit c1 of the port, ADR 0053): the counters run as ESCHER-GPU's driver runs
 * them (src/driver.cu): count() for compute(), subtract() on G_t before the commit and add() on
 * G_{t+1} after it, with the snapshot of the hypergraph's internal row views and the batch's keys
 * (ids + 1). The snapshot's key count is the h2v view's (hyperedge_id_bound()): ESCHER-GPU's
 * numHyperedgeKeys never shrinks, dynG's bound can, and the h2v view covers the keys up to the
 * bound only.
 */
#include "algorithms/triad_count/engine/counters.cuh"
#include "algorithms/triad_count/problem.hpp"
#include "core/cuda_runtime.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace dyng::detail {

/// The counter of one call and the batch it reads.
struct triad_cuda_engine {
  triad_count::pattern kind = triad_count::pattern::h_motif;
  std::unique_ptr<triad::dynamic_counter> counter;
  triad::counter_batch batch;
  std::vector<long long> delta;
};

void triad_cuda_engine_deleter::operator()(triad_cuda_engine* e) const noexcept {
  delete e;
}

namespace {

std::unique_ptr<triad::dynamic_counter> make_counter(triad_count::pattern kind) {
  switch (kind) {
    case triad_count::pattern::h_motif:
      return std::make_unique<triad::motif_counter>();
    case triad_count::pattern::stathyper_type1:
      return std::make_unique<triad::type1_counter>();
    case triad_count::pattern::stathyper_type2:
      return std::make_unique<triad::type2_counter>();
    case triad_count::pattern::stathyper_type3:
      return std::make_unique<triad::type3_counter>();
  }
  DYNG_FAIL("triad_count: no counter for pattern ", static_cast<int>(kind));
}

/// ESCHER-GPU's Snapshot of a state's row views.
triad::counter_snapshot snapshot_of(const hyper::row_views& views) {
  triad::counter_snapshot s;
  s.h2v_rows = views.h2v;
  s.h2h_rows = views.h2h;
  s.num_hyperedge_keys = views.h2v.max_key;
  return s;
}

}  // namespace

triad_cuda_handle triad_cuda_open(const resources& res, triad_count::pattern kind) {
  const scoped_device guard(res.device());
  triad_cuda_handle e(new triad_cuda_engine);
  e->kind = kind;
  e->counter = make_counter(kind);
  return e;
}

void triad_cuda_compute(const resources& res, triad_cuda_engine& e, const hyper::row_views& views,
                        triad_bins& counts) {
  const scoped_device guard(res.device());
  std::vector<long long> values;
  {
    scoped_stage stage(res, "triad_count.count");
    values = e.counter->count(snapshot_of(views));
  }
  scoped_stage stage(res, "triad_count.finalize");
  counts.fill(0);
  for (std::size_t k = 0; k < values.size() && k < counts.size(); ++k) {
    counts[k] = static_cast<std::uint64_t>(values[k]);
  }
}

void triad_cuda_count_minus(const resources& res, triad_cuda_engine& e,
                            const hyper::row_views& views,
                            const std::vector<std::int64_t>& deletions,
                            std::int64_t /*insertions*/) {
  const scoped_device guard(res.device());
  e.batch.deleted_keys.resize(deletions.size());
  for (std::size_t i = 0; i < deletions.size(); ++i) {
    e.batch.deleted_keys[i] = static_cast<int>(deletions[i] + 1);
  }
  e.delta.assign(static_cast<std::size_t>(e.counter->num_values()), 0);
  e.counter->subtract(snapshot_of(views), e.batch, e.delta);
}

void triad_cuda_identify_affected(const resources& /*res*/, triad_cuda_engine& e,
                                  const hyper::row_views& /*views*/, const hyper_applied& record) {
  e.batch.inserted_keys.resize(record.inserted_ids.size());
  for (std::size_t i = 0; i < record.inserted_ids.size(); ++i) {
    e.batch.inserted_keys[i] = static_cast<int>(record.inserted_ids[i] + 1);
  }
}

void triad_cuda_count_plus(const resources& res, triad_cuda_engine& e,
                           const hyper::row_views& views) {
  const scoped_device guard(res.device());
  e.counter->add(snapshot_of(views), e.batch, e.delta);
}

void triad_cuda_finalize(const resources& /*res*/, triad_cuda_engine& e, triad_signed_bins& minus,
                         triad_signed_bins& plus) {
  // ESCHER-GPU's counters accumulate both phases in one device array: the net delta of each bin.
  minus.fill(0);
  plus.fill(0);
  for (std::size_t k = 0; k < e.delta.size() && k < minus.size(); ++k) {
    const std::int64_t d = e.delta[k];
    (d < 0 ? minus : plus)[k] = d;
  }
}

std::int64_t triad_cuda_host_syncs(triad_count::pattern kind) noexcept {
  // The wedge count of each phase that launches wedges and the read of the bins.
  return kind == triad_count::pattern::stathyper_type1 ? 1 : 3;
}

}  // namespace dyng::detail
