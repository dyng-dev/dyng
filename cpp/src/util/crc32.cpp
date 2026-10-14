// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file crc32.cpp
 * @brief CRC-32 with a slice-by-8 table (eight bytes per step).
 */
#include "util/crc32.hpp"

#include <array>
#include <cstring>

namespace dyng::detail {

namespace {

using crc_tables = std::array<std::array<std::uint32_t, 256>, 8>;

constexpr crc_tables make_tables() noexcept {
  crc_tables t{};
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1U) != 0 ? 0xEDB88320U ^ (c >> 1U) : c >> 1U;
    t[0][i] = c;
  }
  for (std::uint32_t i = 0; i < 256; ++i) {
    for (std::size_t s = 1; s < 8; ++s) t[s][i] = (t[s - 1][i] >> 8U) ^ t[0][t[s - 1][i] & 0xffU];
  }
  return t;
}

constexpr crc_tables tables = make_tables();

}  // namespace

std::uint32_t crc32(const void* data, std::size_t n, std::uint32_t crc) noexcept {
  const auto* p = static_cast<const unsigned char*>(data);
  std::uint32_t c = ~crc;
  while (n >= 8) {
    std::uint32_t lo = 0;
    std::uint32_t hi = 0;
    std::memcpy(&lo, p, 4);
    std::memcpy(&hi, p + 4, 4);
    lo ^= c;  // little-endian hosts (static_assert in batch_binary.cpp)
    c = tables[7][lo & 0xffU] ^ tables[6][(lo >> 8U) & 0xffU] ^ tables[5][(lo >> 16U) & 0xffU] ^
        tables[4][lo >> 24U] ^ tables[3][hi & 0xffU] ^ tables[2][(hi >> 8U) & 0xffU] ^
        tables[1][(hi >> 16U) & 0xffU] ^ tables[0][hi >> 24U];
    p += 8;
    n -= 8;
  }
  while (n-- > 0) c = tables[0][(c ^ *p++) & 0xffU] ^ (c >> 8U);
  return ~c;
}

}  // namespace dyng::detail
