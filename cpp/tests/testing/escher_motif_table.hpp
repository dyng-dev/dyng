// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:include/motif_table.hpp (ESCHER_MOTIF_ID_TO_INDEX)
/**
 * @file escher_motif_table.hpp
 * @brief A TEST-ONLY copy of ESCHER-GPU@abcf9b2's table ESCHER_MOTIF_ID_TO_INDEX (Venn id ->
 *        h-motif class 1..30, 0 = no class), the fixture against which the class map that
 *        dyng::testing::triad_reference derives itself is checked (M9a criterion 4). The oracle
 *        never includes this file (triad_reference_isolation).
 *
 * Bits of a Venn id: 0..2 = vertices only in a / b / c, 3 = only in a and b, 4 = only in b and c,
 * 5 = only in c and a, 6 = in all three.
 */
#pragma once

#include <array>
#include <cstdint>

namespace dyng::test::escher_fixture {

/// ESCHER_MOTIF_ID_TO_INDEX of ESCHER-GPU@abcf9b2, its values in order (8 lines of 16).
inline constexpr std::array<std::int8_t, 128> motif_id_to_index = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   //
    0,  0,  0,  0,  0,  0,  0,  0,  21, 23, 22, 24, 23, 25, 24, 26,  //
    0,  0,  0,  0,  0,  0,  0,  0,  21, 22, 23, 24, 23, 24, 25, 26,  //
    21, 23, 23, 25, 22, 24, 24, 26, 27, 28, 28, 29, 28, 29, 29, 30,  //
    1,  2,  2,  3,  2,  3,  3,  4,  5,  6,  6,  8,  7,  9,  9,  10,  //
    5,  7,  6,  9,  6,  9,  8,  10, 11, 13, 12, 14, 13, 15, 14, 16,  //
    5,  6,  7,  9,  6,  8,  9,  10, 11, 12, 13, 14, 13, 14, 15, 16,  //
    11, 13, 13, 15, 12, 14, 14, 16, 17, 18, 18, 19, 18, 19, 19, 20,  //
};

}  // namespace dyng::test::escher_fixture
