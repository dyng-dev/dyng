// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:src/main.cu (the h-motif counter's operation, MotifOp)
// Derived from ESCHER-GPU@abcf9b2:src/type1.cu (Type1Counter's operation, Choose3Op)
// Derived from ESCHER-GPU@abcf9b2:src/type2.cu (Type2Counter's operation, Type2Op)
// Derived from ESCHER-GPU@abcf9b2:src/type3.cu (Type3Counter's operation, Type3Op)
/**
 * @file builtin_ops.cpp
 * @brief triad_count's four patterns as Ops of the local-pattern engine (M9b, ADR 0058): their
 *        tables (num_bins, shape, the CUDA kernel's launch from counters.cu, the host work of a
 *        source instantiated here), their value-initialized Op values.
 *
 * The patterns of triad_count::pattern are the built-in Ops motif_op (h_motif, 30 bins,
 * triangles), choose3_op (stathyper_type1, keys), type2_op (stathyper_type2, pairs) and type3_op
 * (stathyper_type3, triangles); a new pattern is a new Op (experimental::count_local_patterns),
 * not a new enumerator. The built-in tables keep M9a's behaviour: their bins handles test the
 * sign of a bin only (local_patterns::builtin_op), their errors are triad_count's, and the merge
 * limit of the vertex ids applies to the patterns that classify by a three-row merge (h_motif and
 * type 3) only.
 */
#include "algorithms/triad_count/engine/count_ops.hpp"
#include "algorithms/triad_count/engine/motif_ops.hpp"
#include "algorithms/triad_count/engine/stathyper_ops.hpp"
#include "algorithms/triad_count/problem.hpp"

#include <dyng/experimental/detail/local_patterns_engine.hpp>
#include <dyng/experimental/local_patterns.hpp>
#include <dyng/triad_count.hpp>

#include <cstring>
#include <vector>

namespace dyng::detail {

namespace {

/// The table of a built-in Op.
template <class op_t>
local_patterns::kernels builtin_table(triad_count::pattern kind, bool merge_limit) noexcept {
  local_patterns::kernels k;
  k.num_bins = op_t::num_bins;
  k.pattern_shape = op_t::shape;
  k.builtin = true;
  k.merge_limit = merge_limit;
  k.op_size = sizeof(op_t);
  k.launch = triad_device_launch(kind);
  k.source = &local_patterns::host_source<op_t>;
  return k;
}

/// The bytes of a value-initialized Op.
template <class op_t>
void value_of(std::vector<unsigned char>& out) {
  const op_t value{};
  out.assign(sizeof(op_t), 0);
  std::memcpy(out.data(), static_cast<const void*>(&value), sizeof(op_t));
}

}  // namespace

const local_patterns::kernels& triad_kernels(triad_count::pattern kind) noexcept {
  // h_motif and type 3 classify a triad by a merge of its three rows (venn3), which ends a row at
  // INT32_MAX: their compute() refuses a hypergraph whose largest member code could reach it.
  static const local_patterns::kernels tables[4] = {
      builtin_table<triad::motif_op>(triad_count::pattern::h_motif, true),
      builtin_table<triad::choose3_op>(triad_count::pattern::stathyper_type1, false),
      builtin_table<triad::type2_op>(triad_count::pattern::stathyper_type2, false),
      builtin_table<triad::type3_op>(triad_count::pattern::stathyper_type3, true),
  };
  switch (kind) {
    case triad_count::pattern::stathyper_type1:
      return tables[1];
    case triad_count::pattern::stathyper_type2:
      return tables[2];
    case triad_count::pattern::stathyper_type3:
      return tables[3];
    case triad_count::pattern::h_motif:
      break;
  }
  return tables[0];
}

void triad_op_value(triad_count::pattern kind, std::vector<unsigned char>& out) {
  switch (kind) {
    case triad_count::pattern::stathyper_type1:
      value_of<triad::choose3_op>(out);
      return;
    case triad_count::pattern::stathyper_type2:
      value_of<triad::type2_op>(out);
      return;
    case triad_count::pattern::stathyper_type3:
      value_of<triad::type3_op>(out);
      return;
    case triad_count::pattern::h_motif:
      break;
  }
  value_of<triad::motif_op>(out);
}

}  // namespace dyng::detail
