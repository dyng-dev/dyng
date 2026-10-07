// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// CBST trace recorder adapter for MOSP_ESCHER@4b86159 (PLAN 6.4.5, ADR 0036).
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

// The wrappers have C linkage (their names are the mangled C++ symbols) and return C++ types
// through the Itanium ABI's hidden result pointer, exactly as the wrapped members do.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

static_assert(sizeof(CBSTNode) == sizeof(rec::node_mirror), "CBSTNode layout");
static_assert(offsetof(CBSTNode, index) == offsetof(rec::node_mirror, index), "CBSTNode::index");
static_assert(offsetof(CBSTNode, value) == offsetof(rec::node_mirror, value), "CBSTNode::value");
static_assert(offsetof(CBSTNode, length) == offsetof(rec::node_mirror, length), "length");
static_assert(offsetof(CBSTNode, size) == offsetof(rec::node_mirror, size), "size");
static_assert(offsetof(CBSTNode, occupancy) == offsetof(rec::node_mirror, occupancy), "occ");
static_assert(offsetof(CBSTNode, tailBase) == offsetof(rec::node_mirror, tail_base), "base");
static_assert(offsetof(CBSTNode, tailCapacity) == offsetof(rec::node_mirror, tail_capacity),
              "tailCapacity");
static_assert(offsetof(CBSTNode, left) == offsetof(rec::node_mirror, left), "left");
static_assert(offsetof(CBSTNode, right) == offsetof(rec::node_mirror, right), "right");

namespace {

rec::ctx_view view(const CBSTContext& c) {
  rec::ctx_view v;
  v.d_nodes = c.d_nodes;
  v.d_avail = c.d_avail;
  v.d_subtree_avail = c.d_subtreeAvail;
  v.d_payload = c.d_flatPayload;
  v.num_records = c.numRecords;
  v.fixed_size = c.fixedSize;
  v.initial_payload_size = c.initialPayloadSize;
  v.alignment = c.alignment;
  v.scratch_items = c.scratchKeysCap;
  v.scratch_payload = c.scratchPayloadCap;
  v.dataset_name = c.datasetName;
  return v;
}

const rec::adapter& mosp_escher() {
  static const rec::adapter a = [] {
    rec::adapter x;
    x.repo = "MOSP_ESCHER";
    x.commit = "4b86159ec0b3bfe6e18c718ff4bef1494d7276a5";
    x.view_of_object = [](const void* o) {
      return view(static_cast<const CBSTOperations*>(o)->context());
    };
    x.view_of_context = [](const void* c) { return view(*static_cast<const CBSTContext*>(c)); };
    x.context_of = [](const void* o) -> const void* {
      return &static_cast<const CBSTOperations*>(o)->context();
    };
    x.check_rows = [](const void* c, const std::vector<std::vector<int>>& expected,
                      const char* name, std::string& log) -> long long {
      std::ostringstream os;
      const long long n =
          checkTreeRows(*static_cast<const CBSTContext*>(c), expected, false, name, os);
      log += os.str();
      return n;
    };
    x.check_subtree = [](const void* c, const char* name, std::string& log) -> long long {
      std::ostringstream os;
      const long long n = checkSubtreeAvail(*static_cast<const CBSTContext*>(c), name, os);
      log += os.str();
      return n;
    };
    return x;
  }();
  return a;
}

rec::recorder* recorder_impl() {
  return rec::get(mosp_escher());
}

rec::recorder* recorder() {
  return recorder_impl();
}

const char* name_of(const CBSTOperations* self) {
  return self->context().datasetName;
}

}  // namespace

namespace dyng_parity::cbst_rec {
recorder* original_recorder() {
  return recorder_impl();
}
}  // namespace dyng_parity::cbst_rec

// --- The originals (resolved by the linker to the unchanged definitions) -------------------

using escher_mosp::DynamicHypergraph;
using escher_mosp::HgBatch;
using escher_mosp::LineGraphCSR;

extern "C" {
void __real__ZN14CBSTOperationsC1EPKcii(CBSTOperations*, const char*, int, int);
void __real__ZN14CBSTOperationsD1Ev(CBSTOperations*);
void __real__ZN14CBSTOperationsC1EOS_(CBSTOperations*, CBSTOperations*);
CBSTOperations* __real__ZN14CBSTOperationsaSEOS_(CBSTOperations*, CBSTOperations*);
void __real__ZN14CBSTOperations9constructEPiS0_iS0_iPKi(CBSTOperations*, int*, int*, int, int*, int,
                                                        const int*);
InsertMapping __real__ZN14CBSTOperations6insertERKSt6vectorIiSaIiEES4_S4_(CBSTOperations*,
                                                                          const std::vector<int>&,
                                                                          const std::vector<int>&,
                                                                          const std::vector<int>&);
void __real__ZN14CBSTOperations4fillERKSt6vectorIiSaIiEES4_S4_(CBSTOperations*,
                                                               const std::vector<int>&,
                                                               const std::vector<int>&,
                                                               const std::vector<int>&);
void __real__ZN14CBSTOperations5eraseERKSt6vectorIiSaIiEE(CBSTOperations*, const std::vector<int>&);
int __real__ZN14CBSTOperations7compactEv(CBSTOperations*);
void __real__Z13constructCBSTPiS_iS_iiPKcR11CBSTContextPKi(int*, int*, int, int*, int, int,
                                                           const char*, CBSTContext&, const int*);
void __real__Z8fillCBSTRKSt6vectorIiSaIiEES3_S3_R11CBSTContext(const std::vector<int>&,
                                                               const std::vector<int>&,
                                                               const std::vector<int>&,
                                                               CBSTContext&);
void __real__Z10deleteCBSTRKSt6vectorIiSaIiEER11CBSTContext(const std::vector<int>&, CBSTContext&);
InsertMapping __real__Z10insertCBSTRKSt6vectorIiSaIiEES3_S3_R11CBSTContext(const std::vector<int>&,
                                                                           const std::vector<int>&,
                                                                           const std::vector<int>&,
                                                                           CBSTContext&);
void __real__Z10unfillCBSTRKSt6vectorIiSaIiEES3_S3_R11CBSTContext(const std::vector<int>&,
                                                                  const std::vector<int>&,
                                                                  const std::vector<int>&,
                                                                  CBSTContext&);
int __real__Z11compactCBSTR11CBSTContext(CBSTContext&);
}

// --- Shared bodies -----------------------------------------------------------------------------

namespace {

// Records `body` (which calls the original) as one op on the tree of `object` / `context`.
template <class Inputs, class Body>
void record_op(rec::op_kind kind, const void* object, const void* context, Inputs inputs,
               Body body) {
  rec::note_call();
  rec::recorder* r = recorder();
  if (r == nullptr) {
    body();
    return;
  }
  rec::op_record op(r, kind, object, context);
  inputs(op, r);
  try {
    body();
  } catch (...) {
    op.exception(std::current_exception());
    op.finish();
    throw;
  }
  op.finish();
}

void construct_inputs(rec::op_record& op, rec::recorder* r, const char* name, int* keys,
                      int* starts, int n, int* flat, int flat_size, int capacity, int alignment,
                      const int* row_occupancy) {
  op.scalar("num_records", n);
  op.scalar("flat_payload_size", flat_size);
  op.scalar("capacity", capacity);
  op.scalar("alignment", alignment);
  op.text("dataset_name", name ? name : "");
  op.ints("keys", keys, n, rec::input_policy(r, name, "construct"));
  op.ints("start_offsets", starts, n, rec::input_policy(r, name, "construct"));
  op.ints("flat_payload", flat, flat_size, rec::input_policy(r, name, "construct"));
  // The caller's row counts (MOSP_ESCHER only); null means the legacy behaviour (occupancy 0).
  op.scalar("row_occupancy_null", row_occupancy == nullptr ? 1 : 0);
  op.ints("row_occupancy", row_occupancy, row_occupancy ? n : 0,
          rec::input_policy(r, name, "construct"));
}

void rows_inputs(rec::op_record& op, const std::vector<int>& keys, const std::vector<int>& values,
                 const std::vector<int>& prefix, bool named) {
  op.ints("keys", keys);
  op.ints("payload", values);
  op.ints("prefix", prefix);
  if (named) op.named_keys(keys);
}

InsertMapping record_insert(const void* object, const void* context, const std::vector<int>& keys,
                            const std::vector<int>& values, const std::vector<int>& prefix,
                            InsertMapping (*call)(const void*, const void*, const std::vector<int>&,
                                                  const std::vector<int>&,
                                                  const std::vector<int>&)) {
  rec::note_call();
  rec::recorder* r = recorder();
  if (r == nullptr) return call(object, context, keys, values, prefix);
  rec::op_record op(r, rec::op_kind::insert, object, context);
  rows_inputs(op, keys, values, prefix, false);
  InsertMapping m;
  try {
    m = call(object, context, keys, values, prefix);
  } catch (...) {
    op.exception(std::current_exception());
    op.finish();
    throw;
  }
  op.mapping(m.itemToKey);
  op.finish();
  return m;
}

}  // namespace

// --- Lifecycle -------------------------------------------------------------------------------

extern "C" void __wrap__ZN14CBSTOperationsC1EPKcii(CBSTOperations* self, const char* name,
                                                   int capacity, int alignment) {
  __real__ZN14CBSTOperationsC1EPKcii(self, name, capacity, alignment);
  rec::on_construct_object(recorder(), self, name, capacity, alignment);
}

extern "C" void __wrap__ZN14CBSTOperationsD1Ev(CBSTOperations* self) {
  rec::on_destroy_object(recorder(), self);
  __real__ZN14CBSTOperationsD1Ev(self);
}

extern "C" void __wrap__ZN14CBSTOperationsC1EOS_(CBSTOperations* self, CBSTOperations* other) {
  __real__ZN14CBSTOperationsC1EOS_(self, other);
  rec::on_move_construct(recorder(), self, other);
}

extern "C" CBSTOperations* __wrap__ZN14CBSTOperationsaSEOS_(CBSTOperations* self,
                                                            CBSTOperations* other) {
  CBSTOperations* out = __real__ZN14CBSTOperationsaSEOS_(self, other);
  if (self != other) rec::on_move_assign(recorder(), self, other);
  return out;
}

// --- Members ---------------------------------------------------------------------------------

extern "C" void __wrap__ZN14CBSTOperations9constructEPiS0_iS0_iPKi(CBSTOperations* self, int* keys,
                                                                   int* starts, int n, int* flat,
                                                                   int flat_size,
                                                                   const int* row_occupancy) {
  record_op(
      rec::op_kind::construct, self, nullptr,
      [&](rec::op_record& op, rec::recorder* r) {
        construct_inputs(op, r, name_of(self), keys, starts, n, flat, flat_size,
                         self->context().fixedSize, self->context().alignment, row_occupancy);
      },
      [&] {
        __real__ZN14CBSTOperations9constructEPiS0_iS0_iPKi(self, keys, starts, n, flat, flat_size,
                                                           row_occupancy);
      });
}

extern "C" InsertMapping __wrap__ZN14CBSTOperations6insertERKSt6vectorIiSaIiEES4_S4_(
    CBSTOperations* self, const std::vector<int>& keys, const std::vector<int>& values,
    const std::vector<int>& prefix) {
  return record_insert(self, nullptr, keys, values, prefix,
                       [](const void* o, const void*, const std::vector<int>& k,
                          const std::vector<int>& v, const std::vector<int>& p) {
                         return __real__ZN14CBSTOperations6insertERKSt6vectorIiSaIiEES4_S4_(
                             const_cast<CBSTOperations*>(static_cast<const CBSTOperations*>(o)), k,
                             v, p);
                       });
}

extern "C" void __wrap__ZN14CBSTOperations4fillERKSt6vectorIiSaIiEES4_S4_(
    CBSTOperations* self, const std::vector<int>& keys, const std::vector<int>& values,
    const std::vector<int>& prefix) {
  record_op(
      rec::op_kind::fill, self, nullptr,
      [&](rec::op_record& op, rec::recorder*) { rows_inputs(op, keys, values, prefix, true); },
      [&] {
        __real__ZN14CBSTOperations4fillERKSt6vectorIiSaIiEES4_S4_(self, keys, values, prefix);
      });
}

extern "C" void __wrap__ZN14CBSTOperations5eraseERKSt6vectorIiSaIiEE(CBSTOperations* self,
                                                                     const std::vector<int>& keys) {
  record_op(
      rec::op_kind::erase, self, nullptr,
      [&](rec::op_record& op, rec::recorder*) {
        op.ints("keys", keys);
        op.named_keys(keys);
      },
      [&] { __real__ZN14CBSTOperations5eraseERKSt6vectorIiSaIiEE(self, keys); });
}

namespace {
int record_compact(const void* object, const void* context, int (*call)(const void*, const void*)) {
  rec::note_call();
  rec::recorder* r = recorder();
  if (r == nullptr) return call(object, context);
  rec::op_record op(r, rec::op_kind::compact, object, context);
  int out = 0;
  try {
    out = call(object, context);
  } catch (...) {
    op.exception(std::current_exception());
    op.finish();
    throw;
  }
  op.output("compact_return", out);
  op.finish();
  return out;
}
}  // namespace

extern "C" int __wrap__ZN14CBSTOperations7compactEv(CBSTOperations* self) {
  return record_compact(self, nullptr, [](const void* o, const void*) {
    return __real__ZN14CBSTOperations7compactEv(
        const_cast<CBSTOperations*>(static_cast<const CBSTOperations*>(o)));
  });
}

// --- Free functions: unfillCBST is called from DynamicHypergraph.cpp, DynamicGraph.cpp and
// --- tests/unit/test_cbst_ops.cu; the others are wrapped so every entry point is seen ----------

extern "C" void __wrap__Z13constructCBSTPiS_iS_iiPKcR11CBSTContextPKi(
    int* keys, int* starts, int n, int* flat, int flat_size, int capacity, const char* name,
    CBSTContext& ctx, const int* row_occupancy) {
  record_op(
      rec::op_kind::construct, nullptr, &ctx,
      [&](rec::op_record& op, rec::recorder* r) {
        construct_inputs(op, r, name, keys, starts, n, flat, flat_size, capacity, ctx.alignment,
                         row_occupancy);
      },
      [&] {
        __real__Z13constructCBSTPiS_iS_iiPKcR11CBSTContextPKi(keys, starts, n, flat, flat_size,
                                                              capacity, name, ctx, row_occupancy);
      });
}

extern "C" void __wrap__Z8fillCBSTRKSt6vectorIiSaIiEES3_S3_R11CBSTContext(
    const std::vector<int>& keys, const std::vector<int>& values, const std::vector<int>& prefix,
    CBSTContext& ctx) {
  record_op(
      rec::op_kind::fill, nullptr, &ctx,
      [&](rec::op_record& op, rec::recorder*) { rows_inputs(op, keys, values, prefix, true); },
      [&] {
        __real__Z8fillCBSTRKSt6vectorIiSaIiEES3_S3_R11CBSTContext(keys, values, prefix, ctx);
      });
}

extern "C" void __wrap__Z10deleteCBSTRKSt6vectorIiSaIiEER11CBSTContext(const std::vector<int>& keys,
                                                                       CBSTContext& ctx) {
  record_op(
      rec::op_kind::erase, nullptr, &ctx,
      [&](rec::op_record& op, rec::recorder*) {
        op.ints("keys", keys);
        op.named_keys(keys);
      },
      [&] { __real__Z10deleteCBSTRKSt6vectorIiSaIiEER11CBSTContext(keys, ctx); });
}

extern "C" InsertMapping __wrap__Z10insertCBSTRKSt6vectorIiSaIiEES3_S3_R11CBSTContext(
    const std::vector<int>& keys, const std::vector<int>& values, const std::vector<int>& prefix,
    CBSTContext& ctx) {
  return record_insert(nullptr, &ctx, keys, values, prefix,
                       [](const void*, const void* c, const std::vector<int>& k,
                          const std::vector<int>& v, const std::vector<int>& p) {
                         return __real__Z10insertCBSTRKSt6vectorIiSaIiEES3_S3_R11CBSTContext(
                             k, v, p,
                             *const_cast<CBSTContext*>(static_cast<const CBSTContext*>(c)));
                       });
}

extern "C" void __wrap__Z10unfillCBSTRKSt6vectorIiSaIiEES3_S3_R11CBSTContext(
    const std::vector<int>& keys, const std::vector<int>& values, const std::vector<int>& prefix,
    CBSTContext& ctx) {
  record_op(
      rec::op_kind::unfill, nullptr, &ctx,
      [&](rec::op_record& op, rec::recorder*) { rows_inputs(op, keys, values, prefix, true); },
      [&] {
        __real__Z10unfillCBSTRKSt6vectorIiSaIiEES3_S3_R11CBSTContext(keys, values, prefix, ctx);
      });
}

extern "C" int __wrap__Z11compactCBSTR11CBSTContext(CBSTContext& ctx) {
  return record_compact(nullptr, &ctx, [](const void*, const void* c) {
    return __real__Z11compactCBSTR11CBSTContext(
        *const_cast<CBSTContext*>(static_cast<const CBSTContext*>(c)));
  });
}
