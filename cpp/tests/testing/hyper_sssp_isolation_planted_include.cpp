// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// A PLANTED violation of the hyper_sssp oracle's isolation (M11a criterion 5, ADR 0082): this
// source includes a header of the hypergraph container's internals, so the include-graph check
// (check_model_isolation.cmake, test hyper_sssp_reference_isolation_planted_include) must fail.
// It is never compiled.
#include "../../src/hypergraph/row_views.hpp"

#include <dyng/testing/hyper_sssp_reference.hpp>
