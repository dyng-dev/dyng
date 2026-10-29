// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// The stage recorder of bin/dynLP of LabelPropagation-CUDA@a276a3a (PLAN 6.3 step 1, 8.3;
// ADR 0065). parity/export_patches/LabelPropagation-CUDA/build.sh relinks the PATCHED copy's
// UNCHANGED obj/*.o with this file and dynlp_trace.cpp through -Xlinker --wrap=<symbol> for every
// symbol of wrap_dynlp.txt. Every __wrap_<symbol> below calls __real_<symbol>, the original,
// unchanged, with the same arguments. When DYNG_DYNLP_TRACE_DIR is not set a wrapper only
// forwards (parity/exporters/dynlp/check_twins.py compares the twin with the original then and
// with recording on). When it is set, a wrapper synchronizes the stream it was given and copies
// what it records from the device: that changes timing only (the stream order is unchanged).
//
// The wrapped functions are called across translation units (main.o -> generateSparseGraph,
// generateBatches, runDynLP; dynLP.o -> the stage launchers and runIrLPSnapshot), which is what
// --wrap can redirect.
//
// Records (tag, batch; see dynlp_trace.hpp for the file format):
//   generate_graph      (-1)  generateSparseGraph's arguments and its return value
//   generator           (-1)  DYNG_DYNLP_GENERATOR_ONLY=1 only: the CSR (row_ptr, col_ind,
//                             value bits), the true-label bits and the config; then one
//                             batch_data record per batch, and the process exits 0 before
//                             runDynLP (the generator-only mode; with --noGraphFiles at 5M+)
//   run_begin           (-1)  runDynLP before: DynLPConfig, the full CSR, the true labels
//   batch_data          (b)   (with run_begin) every BatchData of the run
//   apply_vertex_changes (b)  the batch marker (exactly one per batch: the call happens in every
//                             batch) with the device batch (deleted, effective inserted, GT ids and
//                             label bits) and the status and gt_class after the call
//   append_edges        (b)   after: row_end, the slot col_ind and value bits (row_ptr in batch
//                             0), the sticky error flags
//   sparsify            (b)   tau and batch_stamp, the G' count and G' as sorted (min, max) pairs
//   connected_components (b)  num_components, the component id of every vertex of Ins \ Del in
//                             list order, the cc.hookRounds delta (prof::counter)
//   label_init          (b)   W0 / W1 per component (uint64) and the initial labels of Ins \ Del
//   propagation_begin   (b)   the initial frontier (affected and ACTIVE and gtClass < 0, sorted)
//                             and the label bits
//   propagation_end     (b)   iterations, label bits, the prop.frontierVisits and
//                             prop.persistentIterations deltas, ws.nextTag
//   irlp_snapshot       (b)   runIrLPSnapshot (--validate): the label bits, iterations, time_ms
//   run_end             (-1)  runDynLP after: the final and IrLP label bits, the BatchResult
//                             columns, the return value
//   check_labels        (b)   --check only (M10a step 2): the values of O's label oracle line,
//                             taken from the arguments of the snprintf call that formats it in
//                             DynLPChecker::checkLabels (checker.o; --wrap=snprintf, which passes
//                             every call on to vsnprintf unchanged): PCG iterations, relative
//                             residual, vertices compared, mean |F - F*| and its tolerance, max,
//                             class flips, excluded vertices, as full-precision values; with
//                             skipped = 1 the oracle did not converge (iterations, residual only)
// A stage that is not called in a batch has no record there (numInserted == 0 skips sparsify,
// connected_components and label_init).
#include "connectedComponents.h"
#include "dynLP.h"
#include "dynamicGraph.h"
#include "dynlp_trace.hpp"
#include "iterativePropagation.h"
#include "labelInitialization.h"
#include "profiler.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace rec = dyng_parity::dynlp_rec;

namespace {

int g_batch = -1;  // the batch of the current apply_vertex_changes marker

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

void sync(cudaStream_t s) {
  check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
}

void add_config(rec::trace* t, rec::record& r, const DynLPConfig& c) {
  t->add_i32(r, "total_nodes", c.totalNodes);
  t->add_i32(r, "avg_degree", c.avgDegree);
  t->add_i32(r, "num_batches", c.numBatches);
  t->add_i64(r, "seed", static_cast<std::int64_t>(c.seed));
  t->add_f32(r, "tau", c.tau);
  t->add_f32(r, "delta", c.delta);
  t->add_i32(r, "max_iterations", c.maxIterations);
  t->add_i32(r, "update_rule", static_cast<std::int32_t>(c.updateRule));
  t->add_f32(r, "ground_truth_fraction", c.groundTruthFraction);
  t->add_f32(r, "unlabeled_fraction", c.unlabeledFraction);
  t->add_f32(r, "deleted_fraction", c.deletedFraction);
  t->add_i32(r, "validate", c.validate ? 1 : 0);
  t->add_i32(r, "write_graph_files", c.writeGraphFiles ? 1 : 0);
  t->add_i32(r, "check", c.check ? 1 : 0);
  t->add_i32(r, "profile", c.profile ? 1 : 0);
}

void add_graph(rec::trace* t, rec::record& r, const CSRGraph& g, const std::vector<float>& labels) {
  t->add_i32(r, "num_nodes", g.numNodes);
  t->add_i32(r, "num_edges", g.numEdges);
  t->add_array(r, "row_ptr", rec::i32, g.rowPtr.data(), g.rowPtr.size());
  t->add_array(r, "col_ind", rec::i32, g.colInd.data(), g.colInd.size());
  t->add_array(r, "values", rec::f32, g.values.data(), g.values.size());
  t->add_array(r, "true_labels", rec::f32, labels.data(), labels.size());
}

void write_batches(rec::trace* t, const std::vector<BatchData>& batches) {
  for (const BatchData& b : batches) {
    rec::record r = t->begin("batch_data", b.batchId);
    t->add_i32(r, "batch_id", b.batchId);
    t->add_array(r, "inserted", rec::i32, b.insertedNodeIds.data(), b.insertedNodeIds.size());
    t->add_array(r, "gt_ids", rec::i32, b.groundTruthNodeIds.data(), b.groundTruthNodeIds.size());
    t->add_array(r, "gt_labels", rec::f32, b.groundTruthLabels.data(), b.groundTruthLabels.size());
    t->add_array(r, "deleted", rec::i32, b.deletedNodeIds.data(), b.deletedNodeIds.size());
    t->add_array(r, "edge_src", rec::i32, b.newEdgeSrc.data(), b.newEdgeSrc.size());
    t->add_array(r, "edge_dst", rec::i32, b.newEdgeDst.data(), b.newEdgeDst.size());
    t->add_array(r, "edge_weight", rec::f32, b.newEdgeWeight.data(), b.newEdgeWeight.size());
    t->end_record(r);
  }
}

}  // namespace

extern "C" {
bool __real__Z19generateSparseGraphiijR8CSRGraphRSt6vectorIfSaIfEERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(
    int, int, unsigned int, CSRGraph&, std::vector<float>&, const std::string&);
bool __real__Z15generateBatchesRK8CSRGraphRKSt6vectorIfSaIfEERK11DynLPConfigRS2_I9BatchDataSaISA_EE(
    const CSRGraph&, const std::vector<float>&, const DynLPConfig&, std::vector<BatchData>&);
bool __real__Z8runDynLPRK11DynLPConfigRK8CSRGraphRKSt6vectorIfSaIfEERKS5_I9BatchDataSaISA_EERS7_SF_RS5_I11BatchResultSaISG_EE(
    const DynLPConfig&, const CSRGraph&, const std::vector<float>&, const std::vector<BatchData>&,
    std::vector<float>&, std::vector<float>&, std::vector<BatchResult>&);
void __real__Z24launchApplyVertexChangesPKiiS0_iS0_PKfiPiS3_R11DeviceGraphP11CUstream_st(
    const int*, int, const int*, int, const int*, const float*, int, int*, int*, DeviceGraph&,
    cudaStream_t);
void __real__Z17launchAppendEdgesR11DeviceGraphPKiS2_PKfiS2_P11CUstream_st(DeviceGraph&, const int*,
                                                                           const int*, const float*,
                                                                           int, const int*,
                                                                           cudaStream_t);
void __real__Z29launchBuildSparsifiedSubgraphPiPKiS1_PKfiiS1_S1_ifS_S_S_P11CUstream_st(
    int*, const int*, const int*, const float*, int, int, const int*, const int*, int, float, int*,
    int*, int*, cudaStream_t);
void __real__Z25launchConnectedComponentsPKiS0_S0_iS0_iPiRiR11CCWorkspaceP11CUstream_st(
    const int*, const int*, const int*, int, const int*, int, int*, int&, CCWorkspace&,
    cudaStream_t);
void __real__Z25launchLabelInitializationPKiS0_S0_PKfPfS0_S0_iS0_iS0_iR18LabelInitWorkspaceP11CUstream_st(
    const int*, const int*, const int*, const float*, float*, const int*, const int*, int,
    const int*, int, const int*, int, LabelInitWorkspace&, cudaStream_t);
void __real__Z26launchIterativePropagationPKiS0_S0_PKfPfS0_S0_S0_ifi10UpdateRuleR20PropagationWorkspaceRiP11CUstream_st(
    const int*, const int*, const int*, const float*, float*, const int*, const int*, const int*,
    int, float, int, UpdateRule, PropagationWorkspace&, int&, cudaStream_t);
bool __real__Z15runIrLPSnapshotRK8CSRGraphRKSt6vectorIiSaIiEERKS2_IfSaIfEEfiRS8_RiRf(
    const CSRGraph&, const std::vector<int>&, const std::vector<float>&, float, int,
    std::vector<float>&, int&, float&);
}

// --- main.o: the generator and the run ---------------------------------------------------------

extern "C" bool
__wrap__Z19generateSparseGraphiijR8CSRGraphRSt6vectorIfSaIfEERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(
    int totalNodes, int avgDegree, unsigned int seed, CSRGraph& graph,
    std::vector<float>& trueLabels, const std::string& outputPrefix) {
  const bool ok =
      __real__Z19generateSparseGraphiijR8CSRGraphRSt6vectorIfSaIfEERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(
          totalNodes, avgDegree, seed, graph, trueLabels, outputPrefix);
  if (rec::trace* t = rec::trace::get()) {
    rec::record r = t->begin("generate_graph", -1);
    t->add_i32(r, "total_nodes", totalNodes);
    t->add_i32(r, "avg_degree", avgDegree);
    t->add_i64(r, "seed", static_cast<std::int64_t>(seed));
    t->add_i32(r, "writes_graph_files", outputPrefix.empty() ? 0 : 1);
    t->add_i32(r, "returned", ok ? 1 : 0);
    t->end_record(r);
  }
  return ok;
}

extern "C" bool
__wrap__Z15generateBatchesRK8CSRGraphRKSt6vectorIfSaIfEERK11DynLPConfigRS2_I9BatchDataSaISA_EE(
    const CSRGraph& graph, const std::vector<float>& trueLabels, const DynLPConfig& config,
    std::vector<BatchData>& batches) {
  const bool ok =
      __real__Z15generateBatchesRK8CSRGraphRKSt6vectorIfSaIfEERK11DynLPConfigRS2_I9BatchDataSaISA_EE(
          graph, trueLabels, config, batches);
  rec::trace* t = rec::trace::get();
  const char* only = std::getenv("DYNG_DYNLP_GENERATOR_ONLY");
  if (t != nullptr && only != nullptr && std::strcmp(only, "1") == 0) {
    rec::record r = t->begin("generator", -1);
    add_config(t, r, config);
    add_graph(t, r, graph, trueLabels);
    t->add_i32(r, "num_batch_records", static_cast<std::int32_t>(batches.size()));
    t->add_i32(r, "returned", ok ? 1 : 0);
    t->end_record(r);
    write_batches(t, batches);
    t->close();
    std::fflush(stdout);
    std::exit(ok ? 0 : 1);  // the generator-only mode: DynLP does not run
  }
  return ok;
}

extern "C" bool
__wrap__Z8runDynLPRK11DynLPConfigRK8CSRGraphRKSt6vectorIfSaIfEERKS5_I9BatchDataSaISA_EERS7_SF_RS5_I11BatchResultSaISG_EE(
    const DynLPConfig& config, const CSRGraph& graph, const std::vector<float>& trueLabels,
    const std::vector<BatchData>& batches, std::vector<float>& finalLabels,
    std::vector<float>& irlpLabels, std::vector<BatchResult>& results) {
  rec::trace* t = rec::trace::get();
  if (t != nullptr) {
    rec::record r = t->begin("run_begin", -1);
    add_config(t, r, config);
    add_graph(t, r, graph, trueLabels);
    t->add_i32(r, "num_batch_records", static_cast<std::int32_t>(batches.size()));
    t->end_record(r);
    write_batches(t, batches);
  }
  const bool ok =
      __real__Z8runDynLPRK11DynLPConfigRK8CSRGraphRKSt6vectorIfSaIfEERKS5_I9BatchDataSaISA_EERS7_SF_RS5_I11BatchResultSaISG_EE(
          config, graph, trueLabels, batches, finalLabels, irlpLabels, results);
  if (t != nullptr) {
    rec::record r = t->begin("run_end", -1);
    t->add_i32(r, "returned", ok ? 1 : 0);
    t->add_array(r, "final_labels", rec::f32, finalLabels.data(), finalLabels.size());
    t->add_array(r, "irlp_labels", rec::f32, irlpLabels.data(), irlpLabels.size());
    const std::size_t n = results.size();
    std::vector<float> time_ms(n), eval_ms(n), accuracy(n), irlp_time_ms(n), irlp_accuracy(n),
        accuracy_vs_irlp(n);
    std::vector<std::int32_t> iterations(n), components(n), irlp_iterations(n);
    std::vector<std::int64_t> visits(n);
    for (std::size_t i = 0; i < n; ++i) {
      const BatchResult& b = results[i];
      time_ms[i] = b.timeMs;
      eval_ms[i] = b.evalMs;
      iterations[i] = b.iterations;
      visits[i] = b.frontierVisits;
      components[i] = b.components;
      accuracy[i] = b.accuracy;
      irlp_time_ms[i] = b.irlpTimeMs;
      irlp_iterations[i] = b.irlpIterations;
      irlp_accuracy[i] = b.irlpAccuracy;
      accuracy_vs_irlp[i] = b.accuracyVsIrLP;
    }
    // Scalars per batch: stored raw in both modes (they are small).
    t->add(r, "time_ms", rec::f32, time_ms.data(), n);
    t->add(r, "eval_ms", rec::f32, eval_ms.data(), n);
    t->add(r, "iterations", rec::i32, iterations.data(), n);
    t->add(r, "frontier_visits", rec::i64, visits.data(), n);
    t->add(r, "components", rec::i32, components.data(), n);
    t->add(r, "accuracy", rec::f32, accuracy.data(), n);
    t->add(r, "irlp_time_ms", rec::f32, irlp_time_ms.data(), n);
    t->add(r, "irlp_iterations", rec::i32, irlp_iterations.data(), n);
    t->add(r, "irlp_accuracy", rec::f32, irlp_accuracy.data(), n);
    t->add(r, "accuracy_vs_irlp", rec::f32, accuracy_vs_irlp.data(), n);
    t->end_record(r);
  }
  return ok;
}

// --- dynLP.o: the stages of a batch ---------------------------------------------------------------

extern "C" void __wrap__Z24launchApplyVertexChangesPKiiS0_iS0_PKfiPiS3_R11DeviceGraphP11CUstream_st(
    const int* d_deleted, int numDeleted, const int* d_inserted, int numInserted,
    const int* d_gtIds, const float* d_gtLabels, int numGT, int* d_vertexStatus, int* d_gtClass,
    DeviceGraph& g, cudaStream_t stream) {
  rec::trace* t = rec::trace::get();
  if (t != nullptr) {
    ++g_batch;
  }
  __real__Z24launchApplyVertexChangesPKiiS0_iS0_PKfiPiS3_R11DeviceGraphP11CUstream_st(
      d_deleted, numDeleted, d_inserted, numInserted, d_gtIds, d_gtLabels, numGT, d_vertexStatus,
      d_gtClass, g, stream);
  if (t == nullptr) {
    return;
  }
  sync(stream);
  const std::size_t n = static_cast<std::size_t>(g.numNodes);
  rec::record r = t->begin("apply_vertex_changes", g_batch);
  t->add_i32(r, "num_deleted", numDeleted);
  t->add_i32(r, "num_inserted", numInserted);
  t->add_i32(r, "num_gt", numGT);
  const std::vector<int> del = download(d_deleted, static_cast<std::size_t>(numDeleted));
  const std::vector<int> ins = download(d_inserted, static_cast<std::size_t>(numInserted));
  const std::vector<int> gt = download(d_gtIds, static_cast<std::size_t>(numGT));
  const std::vector<float> gtl = download(d_gtLabels, static_cast<std::size_t>(numGT));
  t->add_array(r, "deleted", rec::i32, del.data(), del.size());
  t->add_array(r, "inserted", rec::i32, ins.data(), ins.size());
  t->add_array(r, "gt_ids", rec::i32, gt.data(), gt.size());
  t->add_array(r, "gt_labels", rec::f32, gtl.data(), gtl.size());
  const std::vector<int> status = download(d_vertexStatus, n);
  const std::vector<int> cls = download(d_gtClass, n);
  t->add_array(r, "status", rec::i32, status.data(), status.size());
  t->add_array(r, "gt_class", rec::i32, cls.data(), cls.size());
  t->end_record(r);
}

extern "C" void __wrap__Z17launchAppendEdgesR11DeviceGraphPKiS2_PKfiS2_P11CUstream_st(
    DeviceGraph& g, const int* d_src, const int* d_dst, const float* d_w, int numNewEdges,
    const int* d_vertexStatus, cudaStream_t stream) {
  __real__Z17launchAppendEdgesR11DeviceGraphPKiS2_PKfiS2_P11CUstream_st(
      g, d_src, d_dst, d_w, numNewEdges, d_vertexStatus, stream);
  rec::trace* t = rec::trace::get();
  if (t == nullptr) {
    return;
  }
  sync(stream);
  const std::size_t n = static_cast<std::size_t>(g.numNodes);
  const std::size_t cap = static_cast<std::size_t>(g.capacity);
  rec::record r = t->begin("append_edges", g_batch);
  t->add_i32(r, "num_new_edges", numNewEdges);
  t->add_i64(r, "capacity", g.capacity);
  if (g_batch == 0) {
    const std::vector<int> row_ptr = download(g.d_rowPtr, n + 1);
    t->add_array(r, "row_ptr", rec::i32, row_ptr.data(), row_ptr.size());
  }
  const std::vector<int> row_end = download(g.d_rowEnd, n);
  const std::vector<int> col = download(g.d_colInd, cap);
  const std::vector<float> val = download(g.d_values, cap);
  const std::vector<int> flags =
      download(g.d_errorFlags, static_cast<std::size_t>(NUM_ERROR_FLAGS));
  t->add_array(r, "row_end", rec::i32, row_end.data(), row_end.size());
  t->add_array(r, "slot_col_ind", rec::i32, col.data(), col.size());
  t->add_array(r, "slot_values", rec::f32, val.data(), val.size());
  t->add(r, "error_flags", rec::i32, flags.data(), flags.size());
  t->end_record(r);
}

extern "C" void __wrap__Z29launchBuildSparsifiedSubgraphPiPKiS1_PKfiiS1_S1_ifS_S_S_P11CUstream_st(
    int* d_isAffected, const int* d_src, const int* d_dst, const float* d_w, int numNewEdges,
    int numNodes, const int* d_vertexStatus, const int* d_newStamp, int batchStamp, float tau,
    int* d_subSrc, int* d_subDst, int* d_numSub, cudaStream_t stream) {
  __real__Z29launchBuildSparsifiedSubgraphPiPKiS1_PKfiiS1_S1_ifS_S_S_P11CUstream_st(
      d_isAffected, d_src, d_dst, d_w, numNewEdges, numNodes, d_vertexStatus, d_newStamp,
      batchStamp, tau, d_subSrc, d_subDst, d_numSub, stream);
  rec::trace* t = rec::trace::get();
  if (t == nullptr) {
    return;
  }
  sync(stream);
  const int count = download(d_numSub, 1)[0];
  const std::vector<int> src = download(d_subSrc, static_cast<std::size_t>(count));
  const std::vector<int> dst = download(d_subDst, static_cast<std::size_t>(count));
  std::vector<std::pair<int, int>> pairs(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    pairs[i] = {std::min(src[i], dst[i]), std::max(src[i], dst[i])};
  }
  std::sort(pairs.begin(), pairs.end());
  std::vector<int> flat;
  flat.reserve(2 * pairs.size());
  for (const auto& [a, b] : pairs) {
    flat.push_back(a);
    flat.push_back(b);
  }
  rec::record r = t->begin("sparsify", g_batch);
  t->add_f32(r, "tau", tau);
  t->add_i32(r, "batch_stamp", batchStamp);
  t->add_i32(r, "num_new_edges", numNewEdges);
  t->add_i32(r, "num_sub_edges", count);
  t->add_array(r, "sub_edges_sorted", rec::i32, flat.data(), flat.size());
  t->end_record(r);
}

extern "C" void __wrap__Z25launchConnectedComponentsPKiS0_S0_iS0_iPiRiR11CCWorkspaceP11CUstream_st(
    const int* d_edgeSrc, const int* d_edgeDst, const int* d_numEdges, int maxEdges,
    const int* d_insertedNodes, int numInserted, int* d_componentId, int& numComponents,
    CCWorkspace& ws, cudaStream_t stream) {
  rec::trace* t = rec::trace::get();
  const long long hooks_before = t != nullptr ? prof::counter("cc.hookRounds") : 0;
  __real__Z25launchConnectedComponentsPKiS0_S0_iS0_iPiRiR11CCWorkspaceP11CUstream_st(
      d_edgeSrc, d_edgeDst, d_numEdges, maxEdges, d_insertedNodes, numInserted, d_componentId,
      numComponents, ws, stream);
  if (t == nullptr) {
    return;
  }
  sync(stream);
  const std::vector<int> ins = download(d_insertedNodes, static_cast<std::size_t>(numInserted));
  int num_nodes = 0;
  for (const int v : ins) {
    num_nodes = std::max(num_nodes, v + 1);
  }
  const std::vector<int> all = download(d_componentId, static_cast<std::size_t>(num_nodes));
  std::vector<int> ids(ins.size());
  for (std::size_t i = 0; i < ins.size(); ++i) {
    ids[i] = all[static_cast<std::size_t>(ins[i])];
  }
  rec::record r = t->begin("connected_components", g_batch);
  t->add_i32(r, "num_components", numComponents);
  t->add_i32(r, "num_inserted", numInserted);
  t->add_i64(r, "hook_rounds", prof::counter("cc.hookRounds") - hooks_before);
  t->add_array(r, "component_of_inserted", rec::i32, ids.data(), ids.size());
  t->end_record(r);
}

extern "C" void
__wrap__Z25launchLabelInitializationPKiS0_S0_PKfPfS0_S0_iS0_iS0_iR18LabelInitWorkspaceP11CUstream_st(
    const int* d_rowStart, const int* d_rowEnd, const int* d_colInd, const float* d_values,
    float* d_labels, const int* d_vertexStatus, const int* d_componentId, int numComponents,
    const int* d_insertedNodes, int numInserted, const int* d_gtClass, int numNodes,
    LabelInitWorkspace& ws, cudaStream_t stream) {
  __real__Z25launchLabelInitializationPKiS0_S0_PKfPfS0_S0_iS0_iS0_iR18LabelInitWorkspaceP11CUstream_st(
      d_rowStart, d_rowEnd, d_colInd, d_values, d_labels, d_vertexStatus, d_componentId,
      numComponents, d_insertedNodes, numInserted, d_gtClass, numNodes, ws, stream);
  rec::trace* t = rec::trace::get();
  if (t == nullptr) {
    return;
  }
  sync(stream);
  const std::vector<unsigned long long> w0 =
      download(ws.d_weightL0, static_cast<std::size_t>(numComponents));
  const std::vector<unsigned long long> w1 =
      download(ws.d_weightL1, static_cast<std::size_t>(numComponents));
  const std::vector<int> ins = download(d_insertedNodes, static_cast<std::size_t>(numInserted));
  const std::vector<float> labels = download(d_labels, static_cast<std::size_t>(numNodes));
  std::vector<float> init(ins.size());
  for (std::size_t i = 0; i < ins.size(); ++i) {
    init[i] = labels[static_cast<std::size_t>(ins[i])];
  }
  rec::record r = t->begin("label_init", g_batch);
  t->add_i32(r, "num_components", numComponents);
  t->add_array(r, "w0", rec::u64, w0.data(), w0.size());
  t->add_array(r, "w1", rec::u64, w1.data(), w1.size());
  t->add_array(r, "initial_labels_of_inserted", rec::f32, init.data(), init.size());
  t->end_record(r);
}

extern "C" void
__wrap__Z26launchIterativePropagationPKiS0_S0_PKfPfS0_S0_S0_ifi10UpdateRuleR20PropagationWorkspaceRiP11CUstream_st(
    const int* d_rowStart, const int* d_rowEnd, const int* d_colInd, const float* d_values,
    float* d_labels, const int* d_vertexStatus, const int* d_gtClass, const int* d_isAffected,
    int numNodes, float delta, int maxIterations, UpdateRule updateRule, PropagationWorkspace& ws,
    int& iterationsUsed, cudaStream_t stream) {
  rec::trace* t = rec::trace::get();
  long long visits = 0, persistent = 0;
  if (t != nullptr) {
    sync(stream);
    const std::size_t n = static_cast<std::size_t>(numNodes);
    const std::vector<int> aff = download(d_isAffected, n);
    const std::vector<int> status = download(d_vertexStatus, n);
    const std::vector<int> cls = download(d_gtClass, n);
    const std::vector<float> labels = download(d_labels, n);
    std::vector<int> frontier;
    for (std::size_t v = 0; v < n; ++v) {
      if (aff[v] == 1 && status[v] == VERTEX_ACTIVE && cls[v] < 0) {
        frontier.push_back(static_cast<int>(v));
      }
    }
    rec::record r = t->begin("propagation_begin", g_batch);
    t->add_i32(r, "num_nodes", numNodes);
    t->add_f32(r, "delta", delta);
    t->add_i32(r, "max_iterations", maxIterations);
    t->add_i32(r, "update_rule", static_cast<std::int32_t>(updateRule));
    t->add_i32(r, "next_tag", ws.nextTag);
    t->add_i32(r, "frontier_size", static_cast<std::int32_t>(frontier.size()));
    t->add_array(r, "initial_frontier", rec::i32, frontier.data(), frontier.size());
    t->add_array(r, "labels", rec::f32, labels.data(), labels.size());
    t->end_record(r);
    visits = prof::counter("prop.frontierVisits");
    persistent = prof::counter("prop.persistentIterations");
  }
  __real__Z26launchIterativePropagationPKiS0_S0_PKfPfS0_S0_S0_ifi10UpdateRuleR20PropagationWorkspaceRiP11CUstream_st(
      d_rowStart, d_rowEnd, d_colInd, d_values, d_labels, d_vertexStatus, d_gtClass, d_isAffected,
      numNodes, delta, maxIterations, updateRule, ws, iterationsUsed, stream);
  if (t == nullptr) {
    return;
  }
  sync(stream);
  const std::vector<float> labels = download(d_labels, static_cast<std::size_t>(numNodes));
  rec::record r = t->begin("propagation_end", g_batch);
  t->add_i32(r, "iterations", iterationsUsed);
  t->add_i64(r, "frontier_visits", prof::counter("prop.frontierVisits") - visits);
  t->add_i64(r, "persistent_iterations", prof::counter("prop.persistentIterations") - persistent);
  t->add_i32(r, "next_tag", ws.nextTag);
  t->add_array(r, "labels", rec::f32, labels.data(), labels.size());
  t->end_record(r);
}

extern "C" bool __wrap__Z15runIrLPSnapshotRK8CSRGraphRKSt6vectorIiSaIiEERKS2_IfSaIfEEfiRS8_RiRf(
    const CSRGraph& fullGraph, const std::vector<int>& vertexStatus,
    const std::vector<float>& gtLabel, float delta, int maxIterations, std::vector<float>& labels,
    int& iterationsUsed, float& timeMs) {
  const bool ok = __real__Z15runIrLPSnapshotRK8CSRGraphRKSt6vectorIiSaIiEERKS2_IfSaIfEEfiRS8_RiRf(
      fullGraph, vertexStatus, gtLabel, delta, maxIterations, labels, iterationsUsed, timeMs);
  if (rec::trace* t = rec::trace::get()) {
    rec::record r = t->begin("irlp_snapshot", g_batch);
    t->add_i32(r, "returned", ok ? 1 : 0);
    t->add_i32(r, "iterations", iterationsUsed);
    t->add_f32(r, "time_ms", timeMs);
    t->add_f32(r, "delta", delta);
    t->add_array(r, "labels", rec::f32, labels.data(), labels.size());
    t->end_record(r);
  }
  return ok;
}

// --- checker.o: the label oracle's values (--check) --------------------------------------------

namespace {

bool starts_with(const char* s, const char* prefix) {
  return std::strncmp(s, prefix, std::strlen(prefix)) == 0;
}

// The two formats of DynLPChecker::checkLabels (src/checker.cu at a276a3a), whose arguments are
// recorded; every other call is only passed on.
constexpr const char* k_checked =
    "labels vs harmonic optimum (PCG, %d it, residual %.1e): %lld vertices, mean |F-F*| %.2e "
    "(tol %.1e), max %.2e, class flips %lld (%.3f%%); %lld in GT-free components excluded";
constexpr const char* k_skipped =
    "labels vs harmonic optimum: the oracle did not converge (PCG, %d it, relative residual "
    "%.2e > %.0e); labels not compared";

void record_check(const char* fmt, va_list args) {
  rec::trace* t = rec::trace::get();
  if (t == nullptr || !starts_with(fmt, "labels vs harmonic optimum")) {
    return;
  }
  rec::record r = t->begin("check_labels", g_batch);
  if (std::strcmp(fmt, k_checked) == 0) {
    const std::int32_t it = va_arg(args, int);
    const double residual = va_arg(args, double);
    const std::int64_t count = va_arg(args, long long);
    const double mean = va_arg(args, double);
    const double tol = va_arg(args, double);
    const double max = va_arg(args, double);
    const std::int64_t flips = va_arg(args, long long);
    const double percent = va_arg(args, double);
    const std::int64_t excluded = va_arg(args, long long);
    (void)percent;
    const std::int32_t skipped = 0;
    t->add(r, "skipped", rec::i32, &skipped, 1);
    t->add(r, "pcg_iterations", rec::i32, &it, 1);
    t->add(r, "relative_residual", rec::f64, &residual, 1);
    t->add(r, "compared", rec::i64, &count, 1);
    t->add(r, "mean", rec::f64, &mean, 1);
    t->add(r, "tolerance", rec::f64, &tol, 1);
    t->add(r, "max", rec::f64, &max, 1);
    t->add(r, "flips", rec::i64, &flips, 1);
    t->add(r, "excluded", rec::i64, &excluded, 1);
  } else if (std::strcmp(fmt, k_skipped) == 0) {
    const std::int32_t it = va_arg(args, int);
    const double residual = va_arg(args, double);
    const std::int32_t skipped = 1;
    t->add(r, "skipped", rec::i32, &skipped, 1);
    t->add(r, "pcg_iterations", rec::i32, &it, 1);
    t->add(r, "relative_residual", rec::f64, &residual, 1);
  } else {
    rec::fatal(std::string("an unknown label-oracle format: ") + fmt);
  }
  t->end_record(r);
}

}  // namespace

extern "C" int __wrap_snprintf(char* s, std::size_t n, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  va_list copy;
  va_copy(copy, args);
  const int written = std::vsnprintf(s, n, fmt, args);
  va_end(args);
  record_check(fmt, copy);
  va_end(copy);
  return written;
}
