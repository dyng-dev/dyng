// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sha256.hpp
 * @brief SHA-256 (FIPS 180-4) for the parity tests: the digests the DynLP recorder's digest mode
 *        stores (parity/exporters/dynlp, the same implementation), so a test can compare an array
 *        with a recorded digest without the original data.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace dyng::test {

/// Incremental SHA-256.
class sha256 {
 public:
  sha256() {
    std::memcpy(h_, k_init, sizeof(h_));
  }

  void update(const void* data, std::size_t size) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    bytes_ += size;
    while (size > 0) {
      if (used_ == 0 && size >= 64) {
        block(p);
        p += 64;
        size -= 64;
        continue;
      }
      const std::size_t take = (64 - used_) < size ? (64 - used_) : size;
      std::memcpy(buf_ + used_, p, take);
      used_ += take;
      p += take;
      size -= take;
      if (used_ == 64) {
        block(buf_);
        used_ = 0;
      }
    }
  }

  /// The 32-byte digest (the object cannot be updated afterwards).
  std::vector<std::uint8_t> finish() {
    const std::uint64_t bits = bytes_ * 8;
    const std::uint8_t pad = 0x80;
    update(&pad, 1);
    const std::uint8_t zero = 0;
    while (used_ != 56) {
      update(&zero, 1);
    }
    std::uint8_t len[8];
    for (int i = 0; i < 8; ++i) {
      len[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
    }
    update(len, 8);
    std::vector<std::uint8_t> out(32);
    for (int i = 0; i < 8; ++i) {
      out[4 * i] = static_cast<std::uint8_t>(h_[i] >> 24);
      out[4 * i + 1] = static_cast<std::uint8_t>(h_[i] >> 16);
      out[4 * i + 2] = static_cast<std::uint8_t>(h_[i] >> 8);
      out[4 * i + 3] = static_cast<std::uint8_t>(h_[i]);
    }
    return out;
  }

  /// The digest as 64 lowercase hex digits.
  std::string hex() {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    for (const std::uint8_t b : finish()) {
      out.push_back(digits[b >> 4U]);
      out.push_back(digits[b & 15U]);
    }
    return out;
  }

 private:
  static std::uint32_t rotr(std::uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
  }
  void block(const std::uint8_t* p) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16) |
             (std::uint32_t(p[4 * i + 2]) << 8) | std::uint32_t(p[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6],
                  h = h_[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t t1 = h + s1 + ch + k_round[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
  }

  static constexpr std::uint32_t k_init[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  static constexpr std::uint32_t k_round[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
      0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
      0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
      0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
      0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
      0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
      0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
      0xc67178f2u};
  std::uint32_t h_[8];
  std::uint8_t buf_[64] = {};
  std::size_t used_ = 0;
  std::uint64_t bytes_ = 0;
};

/// The hex SHA-256 of the bytes of a vector of trivially copyable values (little-endian hosts).
template <typename value_t>
std::string sha256_hex(const std::vector<value_t>& values) {
  sha256 h;
  h.update(values.data(), values.size() * sizeof(value_t));
  return h.hex();
}

/// The hex SHA-256 of a string's bytes.
inline std::string sha256_hex(const std::string& bytes) {
  sha256 h;
  h.update(bytes.data(), bytes.size());
  return h.hex();
}

}  // namespace dyng::test
