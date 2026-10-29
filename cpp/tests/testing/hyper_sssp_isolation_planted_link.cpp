// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// A PLANTED violation of the hyper_sssp oracle's isolation (M11a criterion 5, ADR 0082): this
// object references a symbol of namespace dyng that it does not define (as an oracle linking
// hyper_sssp or the container would), so the link check (check_model_isolation.cmake, test
// hyper_sssp_reference_isolation_planted_link) must fail. Its object is compiled, never linked.
#include <dyng/testing/hyper_sssp_reference.hpp>

#include <cstdint>

namespace dyng::detail {
std::int64_t planted_isolation_violation(std::int64_t id);
}  // namespace dyng::detail

std::int64_t dyng_hsosp_isolation_planted(std::int64_t id) {
  return dyng::detail::planted_isolation_violation(id);
}
