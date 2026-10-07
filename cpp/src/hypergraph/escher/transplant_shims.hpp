// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:include/printUtils.hpp
/**
 * @file transplant_shims.hpp
 * @brief The two helpers of ESCHER-GPU's utils/printUtils.cpp that the transplanted store calls
 *        (M8a step 3, commit c1 only; the next commit replaces them by the log sink).
 *
 * verbose() is ESCHER-GPU's --verbose flag, off unless the user asks: every traced run of the
 * golden sets ran with it off, so the transplant keeps it off. print_vector is the debug print
 * of printVector.
 */
#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace dyng::detail::escher {

/// ESCHER-GPU's verbose(): off.
inline bool verbose() {
  return false;
}

/// ESCHER-GPU's printVector.
inline void print_vector(const std::vector<int>& vec, const std::string& name) {
  std::cout << name << ": [ ";
  for (int val : vec) {
    std::cout << val << " ";
  }
  std::cout << "]" << std::endl;
}

}  // namespace dyng::detail::escher
