// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// Batch-dump hooks of the CBST trace recorder for ESCHER-GPU@abcf9b2 (PLAN 6.4.5, ADR 0036):
// DynamicHypergraph::build and ::update(Batch&, int), called from driver.o. Linked into a twin
// only when its link command has dynamic_hypergraph.o (see recorder_escher_gpu.cpp for the
// method).
//
// Compiled against the PATCHED scratch copy's unchanged headers (include/structure.hpp,
// include/dynamic_hypergraph.hpp, include/integrity.hpp) and linked with the copy's unchanged
// objects by parity/export_patches/ESCHER-GPU/build.sh with
//   -Xlinker --wrap=<symbol>   for every symbol of wrap_symbols_escher_gpu.txt.
// Every __wrap_<symbol> below calls __real_<symbol>, the original, unchanged. A member function
// is called through the Itanium C++ ABI as a function whose first parameter is `this` (the
// hidden result pointer of a function returning InsertMapping comes before it, as it does for a
// free function returning one), so each wrapper is declared as such a function.
#include "cbst_recorder.hpp"
#include "dynamic_hypergraph.hpp"
#include "integrity.hpp"
#include "structure.hpp"

#include <cstddef>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace rec = dyng_parity::cbst_rec;

// The wrappers have C linkage (their names are the mangled C++ symbols) and return C++ types
// through the Itanium ABI's hidden result pointer, exactly as the wrapped members do.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

namespace {
rec::recorder* recorder() {
  return rec::original_recorder();
}
}  // namespace

extern "C" {
void __real__ZN17DynamicHypergraph5buildEv(DynamicHypergraph*);
void __real__ZN17DynamicHypergraph6updateER5Batchi(DynamicHypergraph*, Batch&, int);
}

// --- Batch hooks: DynamicHypergraph::build and ::update (called from driver.o) ---------------

extern "C" void __wrap__ZN17DynamicHypergraph5buildEv(DynamicHypergraph* self) {
  rec::recorder* r = recorder();
  if (r == nullptr) {
    __real__ZN17DynamicHypergraph5buildEv(self);
    return;
  }
  {
    rec::batch_record b(r, "escher_gpu.build", "enter");
    b.write();
  }
  try {
    __real__ZN17DynamicHypergraph5buildEv(self);
  } catch (...) {
    rec::batch_record b(r, "escher_gpu.build", "exit");
    b.exception(std::current_exception());
    b.write();
    throw;
  }
  rec::batch_record b(r, "escher_gpu.build", "exit");
  // The loaded rows (the host model: row key-1 holds hyperedge `key`, sorted) and the vertex range.
  b.rows("loaded_rows", self->hyperedges());
  b.scalar("max_vertex_id", self->snapshot().numVertexKeys);
  b.scalar("num_hyperedge_keys", self->snapshot().numHyperedgeKeys);
  b.scalar("device_bytes", static_cast<long long>(self->deviceBytes()));
  b.write();
}

extern "C" void __wrap__ZN17DynamicHypergraph6updateER5Batchi(DynamicHypergraph* self, Batch& batch,
                                                              int index) {
  rec::recorder* r = recorder();
  if (r == nullptr) {
    __real__ZN17DynamicHypergraph6updateER5Batchi(self, batch, index);
    return;
  }
  {
    rec::batch_record b(r, "escher_gpu.update", "enter");
    b.scalar("batch", index);
    b.ints("deleted_keys", batch.deletedKeys);
    b.rows("deleted_rows", batch.deletedRows);
    b.rows("inserted_rows", batch.insertedRows);
    b.write();
  }
  try {
    __real__ZN17DynamicHypergraph6updateER5Batchi(self, batch, index);
  } catch (...) {
    rec::batch_record b(r, "escher_gpu.update", "exit");
    b.scalar("batch", index);
    b.exception(std::current_exception());
    b.write();
    throw;
  }
  rec::batch_record b(r, "escher_gpu.update", "exit");
  b.scalar("batch", index);
  b.ints("inserted_keys", batch.insertedKeys);
  b.scalar("num_hyperedge_keys", self->snapshot().numHyperedgeKeys);
  b.scalar("device_bytes", static_cast<long long>(self->deviceBytes()));
  b.write();
}
