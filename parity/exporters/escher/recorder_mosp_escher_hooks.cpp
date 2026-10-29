// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// Batch-dump hooks of the CBST trace recorder for MOSP_ESCHER@4b86159 (PLAN 6.4.5, ADR 0036):
// escher_mosp::DynamicHypergraph::bulkLoad, ::beginBatch and ::finishBatch. Linked into a twin
// only when its link command has DynamicHypergraph.cpp.o (see recorder_mosp_escher.cpp for the
// method).
//
// Compiled against the PATCHED scratch copy's unchanged headers (escher/include/structure.hpp,
// escher/include/integrity.hpp, hypergraph/include/DynamicHypergraph.hpp) and linked with the
// copy's unchanged objects and build/lib/libescher_core.a by
// parity/export_patches/MOSP_ESCHER/build.sh with
//   -Xlinker --wrap=<symbol>   for every symbol of wrap_symbols_mosp_escher.txt.
// MOSP_ESCHER compacts the payload inside fillCBST and insertCBST (one translation unit): those
// compactions are not calls of their own; the state digest after the fill / insert shows them.
// Every __wrap_<symbol> below calls __real_<symbol>, the original, unchanged. A member function
// is called through the Itanium C++ ABI as a function whose first parameter is `this` (the
// hidden result pointer of a function returning InsertMapping comes before it, as it does for a
// free function returning one), so each wrapper is declared as such a function.
#include "DynamicHypergraph.hpp"
#include "cbst_recorder.hpp"
#include "integrity.hpp"
#include "structure.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace rec = dyng_parity::cbst_rec;

// The H-SOSP dump (parity/exporters/hsosp/recorder_mosp_escher_hsosp.cpp, ADR 0081; on with
// DYNG_HSOSP_DUMP_DIR): defined only in the twins that link it (relink_twins.sh --extra), so
// they are declared weak and called only when they are there. They record into their own dump,
// never into the CBST trace, whose records stay as they were.
extern "C" {
__attribute__((weak)) bool dyng_hsosp_dump_active();
__attribute__((weak)) void dyng_hsosp_on_bulk_load_enter(const void*,
                                                         const std::vector<std::vector<int>>*,
                                                         const std::vector<long long>*, int, int);
__attribute__((weak)) void dyng_hsosp_on_bulk_load_exit(const void*,
                                                        const escher_mosp::HostHypergraph*,
                                                        const escher_mosp::LineGraphCSR*,
                                                        const char*);
__attribute__((weak)) void dyng_hsosp_on_begin_batch_enter(const void*,
                                                           const escher_mosp::HgBatch*);
__attribute__((weak)) void dyng_hsosp_on_begin_batch_exit(
    const void*, const escher_mosp::DynamicHypergraph::BatchResult*,
    const escher_mosp::HostHypergraph*, const char*);
}

// The wrappers have C linkage (their names are the mangled C++ symbols) and return C++ types
// through the Itanium ABI's hidden result pointer, exactly as the wrapped members do.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

namespace {
rec::recorder* recorder() {
  return rec::original_recorder();
}
// Whether this twin has the H-SOSP dump and it is on.
bool hsosp_dump() {
  return dyng_hsosp_dump_active != nullptr && dyng_hsosp_dump_active();
}
std::string exception_text(std::exception_ptr e) {
  try {
    std::rethrow_exception(e);
  } catch (const std::exception& x) {
    return x.what();
  } catch (...) {
    return "unknown exception";
  }
}
}  // namespace

using escher_mosp::DynamicHypergraph;
using escher_mosp::HgBatch;
using escher_mosp::LineGraphCSR;

extern "C" {
LineGraphCSR
__real__ZN11escher_mosp17DynamicHypergraph8bulkLoadEOSt6vectorIS1_IiSaIiEESaIS3_EEOS1_IxSaIxEEii(
    DynamicHypergraph*, std::vector<std::vector<int>>&&, std::vector<long long>&&, int, int);
DynamicHypergraph::BatchResult
__real__ZN11escher_mosp17DynamicHypergraph10beginBatchERKNS_7HgBatchE(DynamicHypergraph*,
                                                                      const HgBatch&);
double __real__ZN11escher_mosp17DynamicHypergraph11finishBatchERNS0_11BatchResultEPKmm(
    DynamicHypergraph*, DynamicHypergraph::BatchResult&, const std::uint64_t*, std::size_t);
}

// --- Batch hooks: escher_mosp::DynamicHypergraph::bulkLoad, ::beginBatch, ::finishBatch -------
// --- (called from hsospBench.cu, hsospStress.cu, hsospDelta.cu and the unit tests) -------------

extern "C" LineGraphCSR
__wrap__ZN11escher_mosp17DynamicHypergraph8bulkLoadEOSt6vectorIS1_IiSaIiEESaIS3_EEOS1_IxSaIxEEii(
    DynamicHypergraph* self, std::vector<std::vector<int>>&& rows, std::vector<long long>&& weights,
    int source_he, int target_he) {
  rec::recorder* r = recorder();
  const bool h = hsosp_dump();
  if (r == nullptr && !h)
    return __real__ZN11escher_mosp17DynamicHypergraph8bulkLoadEOSt6vectorIS1_IiSaIiEESaIS3_EEOS1_IxSaIxEEii(
        self, std::move(rows), std::move(weights), source_he, target_he);
  if (h) dyng_hsosp_on_bulk_load_enter(self, &rows, &weights, source_he, target_he);
  if (r != nullptr) {
    // The rows as given (0-based vertices; the virtual source and target hyperedges included).
    rec::batch_record b(r, "mosp_escher.bulk_load", "enter");
    b.rows("rows", rows);
    b.longs("weights", weights);
    b.scalar("source_he", source_he);
    b.scalar("target_he", target_he);
    b.write();
  }
  LineGraphCSR lg;
  try {
    lg =
        __real__ZN11escher_mosp17DynamicHypergraph8bulkLoadEOSt6vectorIS1_IiSaIiEESaIS3_EEOS1_IxSaIxEEii(
            self, std::move(rows), std::move(weights), source_he, target_he);
  } catch (...) {
    if (r != nullptr) {
      rec::batch_record b(r, "mosp_escher.bulk_load", "exit");
      b.exception(std::current_exception());
      b.write();
    }
    if (h) {
      const std::string what = exception_text(std::current_exception());
      dyng_hsosp_on_bulk_load_exit(self, nullptr, nullptr, what.c_str());
    }
    throw;
  }
  if (r != nullptr) {
    rec::batch_record b(r, "mosp_escher.bulk_load", "exit");
    b.scalar("max_id", self->host().maxId());
    b.scalar("line_graph_entries", lg.numEntries());
    b.scalar("escher_device_bytes", self->escherDeviceBytes());
    b.write();
  }
  if (h) dyng_hsosp_on_bulk_load_exit(self, &self->host(), &lg, nullptr);
  return lg;
}

extern "C" DynamicHypergraph::BatchResult
__wrap__ZN11escher_mosp17DynamicHypergraph10beginBatchERKNS_7HgBatchE(DynamicHypergraph* self,
                                                                      const HgBatch& batch) {
  rec::recorder* r = recorder();
  const bool h = hsosp_dump();
  if (r == nullptr && !h)
    return __real__ZN11escher_mosp17DynamicHypergraph10beginBatchERKNS_7HgBatchE(self, batch);
  if (h) dyng_hsosp_on_begin_batch_enter(self, &batch);
  if (r != nullptr) {
    rec::batch_record b(r, "mosp_escher.begin_batch", "enter");
    std::vector<int> a, c;
    b.ints("he_delete", batch.heDelete);
    for (const auto& x : batch.vtxDelete) a.push_back(x.heId), c.push_back(x.vertex);
    b.ints("vtx_delete.he", a);
    b.ints("vtx_delete.vertex", c);
    a.clear();
    c.clear();
    for (const auto& x : batch.vtxInsert) a.push_back(x.heId), c.push_back(x.vertex);
    b.ints("vtx_insert.he", a);
    b.ints("vtx_insert.vertex", c);
    std::vector<std::vector<int>> verts;
    std::vector<long long> w;
    for (const auto& x : batch.heInsert) verts.push_back(x.vertices), w.push_back(x.weight);
    b.rows("he_insert", verts);
    b.longs("he_insert.weight", w);
    b.write();
  }
  DynamicHypergraph::BatchResult res;
  try {
    res = __real__ZN11escher_mosp17DynamicHypergraph10beginBatchERKNS_7HgBatchE(self, batch);
  } catch (...) {
    if (r != nullptr) {
      rec::batch_record b(r, "mosp_escher.begin_batch", "exit");
      b.exception(std::current_exception());
      b.write();
    }
    if (h) {
      const std::string what = exception_text(std::current_exception());
      dyng_hsosp_on_begin_batch_exit(self, nullptr, nullptr, what.c_str());
    }
    throw;
  }
  if (r != nullptr) {
    rec::batch_record b(r, "mosp_escher.begin_batch", "exit");
    b.ints("final_ids", res.inc.newHe);
    b.longs("final_weights", res.inc.newW);
    b.ints("dead_he", res.inc.deadHe);
    b.scalar("skipped_ops", res.inc.skippedOps);
    b.scalar("max_id", self->host().maxId());
    b.write();
  }
  if (h) dyng_hsosp_on_begin_batch_exit(self, &res, &self->host(), nullptr);
  return res;
}

extern "C" double __wrap__ZN11escher_mosp17DynamicHypergraph11finishBatchERNS0_11BatchResultEPKmm(
    DynamicHypergraph* self, DynamicHypergraph::BatchResult& res, const std::uint64_t* keys,
    std::size_t num_keys) {
  rec::recorder* r = recorder();
  if (r == nullptr)
    return __real__ZN11escher_mosp17DynamicHypergraph11finishBatchERNS0_11BatchResultEPKmm(
        self, res, keys, num_keys);
  {
    // The line-graph delta that drives the h2h operations (recorded as those operations).
    rec::batch_record b(r, "mosp_escher.finish_batch", "enter");
    b.scalar("num_directed_keys", static_cast<long long>(num_keys));
    b.write();
  }
  double ms = 0;
  try {
    ms = __real__ZN11escher_mosp17DynamicHypergraph11finishBatchERNS0_11BatchResultEPKmm(
        self, res, keys, num_keys);
  } catch (...) {
    rec::batch_record b(r, "mosp_escher.finish_batch", "exit");
    b.exception(std::current_exception());
    b.write();
    throw;
  }
  rec::batch_record b(r, "mosp_escher.finish_batch", "exit");
  b.scalar("escher_device_bytes", self->escherDeviceBytes());
  b.write();
  return ms;
}
