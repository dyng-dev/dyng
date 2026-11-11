// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// The recorder of bin/propagation_test of LabelPropagation-CUDA@a276a3a: the hub graph of the
// degree bins (tests/propagation_test.cu, both update rules; the golden set
// label_propagation_hub, ADR 0065). parity/export_patches/LabelPropagation-CUDA/build.sh relinks
// the test's UNCHANGED objects (obj/tests/propagation_test.o and the library objects) with this
// file and dynlp_trace.cpp through -Xlinker --wrap for the one symbol of wrap_hub.txt;
// __wrap_launchIterativePropagation calls the original unchanged. Without DYNG_DYNLP_TRACE_DIR it
// only forwards.
//
// One record per call (tag "hub_propagation", batch = the call's index): before the call the
// graph as the test passes it (row_start, row_end, col_ind and value bits up to the largest
// row end), the status, gt_class, affected flags, the initial label bits, the rule, delta and the
// iteration cap; after it the iterations, the label bits, the prop.frontierVisits and
// prop.persistentIterations deltas and ws.nextTag.
#include "dynlp_trace.hpp"
#include "iterativePropagation.h"
#include "profiler.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace rec = dyng_parity::dynlp_rec;

namespace {

int g_call = -1;

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    rec::fatal(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

template <typename T>
std::vector<T> download(const T* d, std::size_t n) {
  std::vector<T> h(n);
  if (n > 0) {
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
  }
  return h;
}

}  // namespace

extern "C" void
__real__Z26launchIterativePropagationPKiS0_S0_PKfPfS0_S0_S0_ifi10UpdateRuleR20PropagationWorkspaceRiP11CUstream_st(
    const int*, const int*, const int*, const float*, float*, const int*, const int*, const int*,
    int, float, int, UpdateRule, PropagationWorkspace&, int&, cudaStream_t);

extern "C" void
__wrap__Z26launchIterativePropagationPKiS0_S0_PKfPfS0_S0_S0_ifi10UpdateRuleR20PropagationWorkspaceRiP11CUstream_st(
    const int* d_rowStart, const int* d_rowEnd, const int* d_colInd, const float* d_values,
    float* d_labels, const int* d_vertexStatus, const int* d_gtClass, const int* d_isAffected,
    int numNodes, float delta, int maxIterations, UpdateRule updateRule, PropagationWorkspace& ws,
    int& iterationsUsed, cudaStream_t stream) {
  rec::trace* t = rec::trace::get();
  rec::record r;
  long long visits = 0, persistent = 0;
  if (t != nullptr) {
    ++g_call;
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    const std::size_t n = static_cast<std::size_t>(numNodes);
    const std::vector<int> start = download(d_rowStart, n);
    const std::vector<int> end = download(d_rowEnd, n);
    int nnz = 0;
    for (const int e : end) {
      nnz = std::max(nnz, e);
    }
    const std::vector<int> col = download(d_colInd, static_cast<std::size_t>(nnz));
    const std::vector<float> val = download(d_values, static_cast<std::size_t>(nnz));
    const std::vector<int> status = download(d_vertexStatus, n);
    const std::vector<int> cls = download(d_gtClass, n);
    const std::vector<int> aff = download(d_isAffected, n);
    const std::vector<float> labels = download(d_labels, n);
    r = t->begin("hub_propagation", g_call);
    t->add_i32(r, "num_nodes", numNodes);
    t->add_f32(r, "delta", delta);
    t->add_i32(r, "max_iterations", maxIterations);
    t->add_i32(r, "update_rule", static_cast<std::int32_t>(updateRule));
    t->add_i32(r, "next_tag_before", ws.nextTag);
    t->add_array(r, "row_start", rec::i32, start.data(), start.size());
    t->add_array(r, "row_end", rec::i32, end.data(), end.size());
    t->add_array(r, "col_ind", rec::i32, col.data(), col.size());
    t->add_array(r, "values", rec::f32, val.data(), val.size());
    t->add_array(r, "status", rec::i32, status.data(), status.size());
    t->add_array(r, "gt_class", rec::i32, cls.data(), cls.size());
    t->add_array(r, "affected", rec::i32, aff.data(), aff.size());
    t->add_array(r, "labels_before", rec::f32, labels.data(), labels.size());
    visits = prof::counter("prop.frontierVisits");
    persistent = prof::counter("prop.persistentIterations");
  }
  __real__Z26launchIterativePropagationPKiS0_S0_PKfPfS0_S0_S0_ifi10UpdateRuleR20PropagationWorkspaceRiP11CUstream_st(
      d_rowStart, d_rowEnd, d_colInd, d_values, d_labels, d_vertexStatus, d_gtClass, d_isAffected,
      numNodes, delta, maxIterations, updateRule, ws, iterationsUsed, stream);
  if (t == nullptr) {
    return;
  }
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  const std::vector<float> labels = download(d_labels, static_cast<std::size_t>(numNodes));
  t->add_i32(r, "iterations", iterationsUsed);
  t->add_i64(r, "frontier_visits", prof::counter("prop.frontierVisits") - visits);
  t->add_i64(r, "persistent_iterations", prof::counter("prop.persistentIterations") - persistent);
  t->add_i32(r, "next_tag", ws.nextTag);
  t->add_array(r, "labels", rec::f32, labels.data(), labels.size());
  t->end_record(r);
}
