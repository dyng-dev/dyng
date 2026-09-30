// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry_test.cpp
 * @brief The registry (dyng::algorithms(), generated from the manifests) and the conformance kit
 *        agree: every algorithm built into the library is registered with the kit, and every
 *        entry follows the registration rules (invariant I8).
 */
#include "conformance/conformance.hpp"

#include <dyng/citation.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/registry.hpp>

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// This executable checks the registry only; the kit's suite is instantiated per algorithm in
// dyng_<name>_conformance_tests.
namespace dyng::conformance {
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(conformance);
}  // namespace dyng::conformance

namespace {

using dyng::conformance::registered_algorithms;

std::vector<std::string> kit_names() {
  std::vector<std::string> out;
  dyng::conformance::for_each_type(registered_algorithms{}, [&](auto* tag) {
    using tag_t = std::remove_pointer_t<decltype(tag)>;
    out.emplace_back(dyng::conformance::test_traits<tag_t>::name);
  });
  return out;
}

TEST(Registry, EveryBuiltAlgorithmHasConformanceTraits) {
  std::vector<std::string> library;
  for (const dyng::algorithm_info& info : dyng::algorithms()) {
    library.emplace_back(info.name);
  }
  EXPECT_FALSE(library.empty());
  EXPECT_EQ(library, kit_names()) << "dyng::algorithms() and the kit's registry differ: run "
                                     "scripts/regen.py and check DYNG_ALGORITHMS";
}

TEST(Registry, EntriesFollowTheRegistrationRules) {
  for (const dyng::algorithm_info& info : dyng::algorithms()) {
    SCOPED_TRACE(std::string(info.name));
    ASSERT_FALSE(info.backends.empty());
    EXPECT_EQ(info.backends.front(), dyng::backend::sequential);
    EXPECT_FALSE(info.title.empty());
    EXPECT_EQ(dyng::find_algorithm(info.name), &info);
    EXPECT_FALSE(dyng::citation(info.name).empty()) << "dyng::citation() knows the algorithm";
  }
}

TEST(Registry, UnknownNamesAreNotFound) {
  EXPECT_EQ(dyng::find_algorithm("no_such_algorithm"), nullptr);
  EXPECT_EQ(dyng::find_algorithm(""), nullptr);
}

}  // namespace
