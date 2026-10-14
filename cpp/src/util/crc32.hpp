// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file crc32.hpp
 * @brief CRC-32 (IEEE 802.3, the polynomial of zlib, gzip and PNG) for the records of the `.dgb`
 *        batch files.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace dyng::detail {

/**
 * @brief The CRC-32 of `n` bytes (reflected polynomial 0xEDB88320, initial value and final
 *        XOR 0xFFFFFFFF), continuing from `crc` (the CRC-32 of the bytes before; 0 to start).
 *
 * crc32("123456789") == 0xCBF43926 (the standard check value).
 * @param[in] data The bytes.
 * @param[in] n    Their count.
 * @param[in] crc  The CRC-32 of the preceding bytes (0 for none).
 * @return The CRC-32 of the preceding bytes and these.
 */
std::uint32_t crc32(const void* data, std::size_t n, std::uint32_t crc = 0) noexcept;

}  // namespace dyng::detail
