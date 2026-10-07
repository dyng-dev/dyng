// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// The CBST trace recorder of the parity harness (PLAN Sections 6.4.5 and 8.3; ADR 0036).
//
// parity/export_patches/{ESCHER-GPU,MOSP_ESCHER}/build.sh compile this core and one adapter
// (recorder_escher_gpu.cpp, recorder_mosp_escher.cpp) against the PATCHED scratch copy's
// unchanged headers and relink the copy's unchanged objects with them through
// `-Xlinker --wrap=<symbol>`: every call of a CBST entry point from another object reaches the
// adapter's __wrap_ function, which records the call's inputs, calls __real_ (the original),
// then records its outputs and the state of the tree read back through context(). No original
// file is changed and no original code is copied: the recorder only reads the documented fields
// of CBSTContext / CBSTNode (mirrored below and checked against the original's layout by the
// adapter) and calls the original's own functions.
//
// Recording is on only when DYNG_CBST_TRACE_DIR is set; otherwise every __wrap_ function calls
// __real_ and returns (no extra work, no synchronization). The trace format (version 1) is
// documented in docs/adr/0036-cbst-trace-format-and-recorder.md; readers:
// cpp/tests/parity/escher/cbst_trace.{hpp,cpp} and parity/escher_trace.py.
#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace dyng_parity::cbst_rec {

/// Version of this recorder (written into every trace header).
inline constexpr const char* recorder_version = "1.0.0";
/// Version of the trace format.
inline constexpr std::uint32_t format_version = 1;

/// The fields of the original's CBSTNode, in its order (both originals: 7 ints, 3 pointers).
struct node_mirror {
  int index;
  int value;
  int length;
  int size;
  int occupancy;
  int tail_base;
  int tail_capacity;
  const node_mirror* left;
  const node_mirror* right;
  const node_mirror* parent;
};

/// What the recorder reads of a CBSTContext (device pointers and host scalars).
struct ctx_view {
  const void* d_nodes = nullptr;
  const int* d_avail = nullptr;
  const int* d_subtree_avail = nullptr;
  const int* d_payload = nullptr;
  long long num_records = 0;
  long long fixed_size = 0;
  long long initial_payload_size = 0;
  long long alignment = 0;
  long long scratch_items = 0;    // A: scratchItemCapacity; B: scratchKeysCap
  long long scratch_payload = 0;  // A: scratchPayloadCapacity; B: scratchPayloadCap
  const char* dataset_name = nullptr;
};

/// The calls into the original that an adapter provides.
struct adapter {
  const char* repo = "";
  const char* commit = "";
  /// The context of a tree object (CBSTOperations*), or of a context pointer itself.
  ctx_view (*view_of_object)(const void* object) = nullptr;
  ctx_view (*view_of_context)(const void* context) = nullptr;
  /// The address of the object's context (CBSTOperations::context()).
  const void* (*context_of)(const void* object) = nullptr;
  /// CBSTOperations::deviceBytes() of an object; nullptr when the original has none (B).
  long long (*device_bytes)(const void* object) = nullptr;
  /// The process-wide degree-bin launch counters (fill, reuse, unfill x thread, warp, block);
  /// nullptr when the original has none (B).
  void (*bin_launches)(long long out[9]) = nullptr;
  /// The original's checkTreeRows(ctx, expected, orderInsensitive = false, name, log).
  long long (*check_rows)(const void* context, const std::vector<std::vector<int>>& expected,
                          const char* name, std::string& log) = nullptr;
  /// The original's checkSubtreeAvail(ctx, name, log); nullptr when it has none (A).
  long long (*check_subtree)(const void* context, const char* name, std::string& log) = nullptr;
};

/// Kinds of tree operations (the "op" field of an OP record).
enum class op_kind { construct, rebuild, insert, fill, erase, unfill, compact };

/// Where an array of an OP or BATCH record goes.
enum class array_policy { normal, input, hash_only };

class op_record;

/// The process-wide recorder (nullptr when DYNG_CBST_TRACE_DIR is not set).
class recorder;
recorder* get(const adapter& a);

/// Counts one CBST operation for the invocation log (DYNG_CBST_INVOCATION_LOG; nothing without).
void note_call();

/// Writes a NOTE record (the cross-replay drivers name the source op of the next op with it).
void note(recorder* r, const std::string& text);

/// The recorder of the original this twin is linked with (defined by its adapter).
recorder* original_recorder();

// --- Lifecycle of tree objects -------------------------------------------------------------

void on_construct_object(recorder* r, const void* object, const char* name, int capacity,
                         int alignment);
void on_destroy_object(recorder* r, const void* object);
void on_move_construct(recorder* r, const void* dst, const void* src);
void on_move_assign(recorder* r, const void* dst, const void* src);

// --- Operations ----------------------------------------------------------------------------

/// One operation on a tree, from its inputs to the state after it. Created before the original
/// runs (it reads the state at entry), finished after (it reads the state after the call).
class op_record {
 public:
  /// `object` is the CBSTOperations*, or nullptr when only `context` is known (free functions).
  op_record(recorder* r, op_kind kind, const void* object, const void* context);
  ~op_record();
  op_record(const op_record&) = delete;
  op_record& operator=(const op_record&) = delete;

  void scalar(const char* name, long long value);
  void text(const char* name, const std::string& value);
  /// An output scalar of the op ("out.<name>", e.g. the return value of compact()).
  void output(const char* name, long long value);
  void ints(const char* name, const int* data, long long count,
            array_policy policy = array_policy::normal);
  void ints(const char* name, const std::vector<int>& v,
            array_policy policy = array_policy::normal) {
    ints(name, v.data(), static_cast<long long>(v.size()), policy);
  }
  /// Device ints, copied to the host first.
  void device_ints(const char* name, const int* device, long long count,
                   array_policy policy = array_policy::input);
  /// Keys the op names (they are written in full in digest mode).
  void named_keys(const std::vector<int>& keys);
  void named_keys(const int* keys, long long count);
  /// The exception the original threw (called inside a catch block).
  void exception(std::exception_ptr e);
  /// The insert's mapping, the compact() return value.
  void mapping(const std::vector<int>& item_to_key);
  /// Reads the state after the call and writes the record.
  void finish();

 private:
  struct impl;
  impl* impl_;
};

/// The policy of the large construct / rebuild / batch arrays of a tree (digest mode may store
/// them in the side file or as a hash only, see DYNG_CBST_TRACE_HASH_ONLY).
array_policy input_policy(recorder* r, const char* dataset_name, const char* what);

// --- Batch hooks (the originals' batch entry points) ---------------------------------------

/// A BATCH record: one call of a batch entry point (phase "enter" before the original runs,
/// "exit" after it).
class batch_record {
 public:
  batch_record(recorder* r, const char* hook, const char* phase);
  ~batch_record();
  batch_record(const batch_record&) = delete;
  batch_record& operator=(const batch_record&) = delete;
  void scalar(const char* name, long long value);
  void text(const char* name, const std::string& value);
  void ints(const char* name, const std::vector<int>& v);
  void longs(const char* name, const std::vector<long long>& v);
  /// Rows as values + inclusive prefix of their sizes (the CBST call layout).
  void rows(const char* name, const std::vector<std::vector<int>>& rows);
  void exception(std::exception_ptr e);
  void write();

 private:
  struct impl;
  impl* impl_;
};

}  // namespace dyng_parity::cbst_rec
