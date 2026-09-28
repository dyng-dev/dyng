// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file device_error_flags.hpp
 * @brief The sticky device error word (PLAN Section 4.7.3): kernels report contract violations
 *        (row overflow, re-insertion, out of space, a parent cycle in an input tree) by setting
 *        bits; the host reads the word once, merged into an existing synchronization point, and
 *        throws a precise exception. Kernels never trap, print or abort.
 *
 * Host part (no CUDA headers). Kernels include device_error_flags.cuh for raise_device_error().
 *
 * Use:
 *
 *     device_error_flags flags(res);            // once, e.g. in a workspace
 *     flags.reset(stream);                      // before the kernels
 *     my_kernel<<<..., stream>>>(..., flags.data());
 *     flags.enqueue_readback(stream);           // after the last kernel
 *     ... the existing synchronization of the call ...
 *     flags.throw_if_raised("sssp::update");    // reads the pinned mirror, no extra sync
 */
#pragma once

#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stream.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace dyng::detail {

/**
 * @brief The bits of the device error word. A kernel may set several.
 */
enum class device_error : std::uint32_t {
  none = 0,             ///< no error
  capacity = 1U << 0U,  ///< a fixed capacity was exceeded (throws capacity_error)
  invalid_input =
      1U << 1U,             ///< the input breaks a documented precondition (invalid_argument_error)
  parent_cycle = 1U << 2U,  ///< a parent array contains a cycle (invalid_argument_error)
  internal = 1U << 31U,     ///< a broken invariant of the library (internal_error)
};

/**
 * @brief The bits of one error kind as an integer.
 * @param[in] e The error kind.
 * @return Its bit mask.
 */
[[nodiscard]] constexpr std::uint32_t bits_of(device_error e) noexcept {
  return static_cast<std::uint32_t>(e);
}

/**
 * @brief A readable list of the error kinds set in a word ("capacity, parent_cycle", "unknown
 *        bit 0x10").
 * @param[in] bits The error word.
 * @return The description ("none" for 0).
 */
[[nodiscard]] std::string describe_device_errors(std::uint32_t bits);

/**
 * @brief Throw the exception for a non-zero error word.
 *
 * The most specific kind wins: internal (internal_error), then capacity (capacity_error), then
 * the input kinds (invalid_argument_error). Unknown bits are internal errors.
 * @param[in] bits    The error word (non-zero).
 * @param[in] context The operation, for the message (e.g. "sssp::update").
 * @param[in] detail  Optional precise description of the violation (appended).
 * @throws capacity_error, invalid_argument_error or internal_error.
 */
[[noreturn]] void throw_device_errors(std::uint32_t bits, std::string_view context,
                                      std::string_view detail = {});

/**
 * @brief One device error word and its pinned host mirror.
 *
 * The word is allocated from the memory resource of the resources, the mirror from its staging
 * resource; both live as long as the object (allocate it once, e.g. in a workspace, and reuse it).
 */
class device_error_flags {
 public:
  /**
   * @brief Allocate the word and the mirror; the word is cleared on the stream of `res`.
   * @param[in] res Resources of the CUDA backend.
   * @throws out_of_memory_error if an allocation fails.
   * @throws cuda_error          if the runtime reports an error.
   */
  explicit device_error_flags(const resources& res);

  /**
   * @brief The device word, to pass to kernels.
   * @return Device pointer to the word.
   */
  [[nodiscard]] std::uint32_t* data() noexcept {
    return word_.data();
  }

  /**
   * @brief Clear the word, ordered on `stream`.
   * @param[in] stream The stream of the kernels that follow.
   * @throws cuda_error if the runtime reports an error.
   * @async
   */
  void reset(stream_ref stream);

  /**
   * @brief Copy the word into the pinned mirror, ordered on `stream` after the kernels.
   * @param[in] stream The stream of the kernels.
   * @throws cuda_error if the runtime reports an error.
   * @async
   */
  void enqueue_readback(stream_ref stream);

  /**
   * @brief The mirror's value; valid once the stream has been synchronized after
   *        enqueue_readback().
   * @return The error word.
   */
  [[nodiscard]] std::uint32_t mirrored() const noexcept;

  /**
   * @brief Read the word now: readback and a synchronization of `stream`.
   * @param[in] stream The stream of the kernels.
   * @return The error word.
   * @throws cuda_error if the runtime reports an error.
   * @sync
   */
  [[nodiscard]] std::uint32_t read(stream_ref stream);

  /**
   * @brief Throw if the mirror (after readback and synchronization) holds an error.
   * @param[in] context The operation, for the message.
   * @param[in] detail  Optional precise description of the violation.
   * @throws capacity_error, invalid_argument_error or internal_error (throw_device_errors()).
   */
  void throw_if_raised(std::string_view context, std::string_view detail = {}) const;

 private:
  buffer<std::uint32_t> word_;
  buffer<std::uint32_t> mirror_;
};

}  // namespace dyng::detail
