// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file staging.hpp
 * @brief Implicit copies of input arrays under the copy policy of `resources` (PLAN Section
 *        4.7.1; ADR 0016 item 11).
 *
 * Every function that consumes arrays accepts any memory space. When an input is not in the space
 * the implementation consumes it in (in this release: host memory for batches, graph builds and
 * imported trees, which are applied, built and checked on the host on every backend), the library
 * copies it once. resources::get_copy_policy() decides what else happens: `allow` logs the copy at
 * debug level, `warn` at warn level, `error` throws invalid_argument_error instead of copying.
 * While a profiler is attached, `allow` acts as `warn` (a misplaced input must not silently
 * distort a timed run).
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_batch.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace dyng::detail {

/**
 * @brief The policy that applies to an implicit copy through `res`: the handle's policy, with
 *        copy_policy::allow raised to copy_policy::warn while a profiler is attached.
 * @param[in] res The resources of the call.
 * @return The effective policy.
 */
[[nodiscard]] copy_policy effective_copy_policy(const resources& res) noexcept;

/**
 * @brief Apply the copy policy to an implicit copy of an input before it is made.
 * @param[in] res   The resources of the call.
 * @param[in] what  The input, for the message (e.g. "dyng::update: edge_batch_view::insert_src").
 * @param[in] from  The input's memory space.
 * @param[in] to    The space the implementation consumes it in.
 * @param[in] bytes Size of the copy.
 * @throws invalid_argument_error if the effective policy is copy_policy::error.
 */
void note_implicit_copy(const resources& res, std::string_view what, memory_space from,
                        memory_space to, std::size_t bytes);

/**
 * @brief Copy bytes of an array in any space into host memory and wait for the copy.
 * @param[in]  res   The resources of the call (stream and device of a CUDA handle).
 * @param[out] dst   Host destination.
 * @param[in]  src   Source pointer.
 * @param[in]  space Memory space of `src`.
 * @param[in]  device Device of `src` (-1: the device of `res`).
 * @param[in]  bytes Byte count.
 * @throws not_supported_error if `src` is device memory and CUDA is not built.
 * @throws cuda_error          if the CUDA runtime reports an error.
 * @sync
 */
void copy_to_host(const resources& res, void* dst, const void* src, memory_space space, int device,
                  std::size_t bytes);

/**
 * @brief An input array readable on the host: a view of the caller's array if it is
 *        host-accessible, otherwise a host copy made once under the copy policy.
 * @tparam value_t Element type (without const).
 */
template <typename value_t>
class host_input {
  static_assert(!std::is_const_v<value_t>, "host_input<T>: T without const");

 public:
  /**
   * @brief Stage `in` for host reads.
   * @param[in] res  The resources of the call.
   * @param[in] in   The input array (any memory space).
   * @param[in] what The input, for messages.
   * @throws invalid_argument_error if a copy is needed and the policy is copy_policy::error.
   * @throws not_supported_error    if a copy from device memory is needed without CUDA.
   * @sync A copy from device memory is complete when the constructor returns.
   */
  host_input(const resources& res, array_view<const value_t> in, std::string_view what) {
    if (in.empty() || is_host_accessible(in.space())) {
      view_ = in;
      return;
    }
    note_implicit_copy(res, what, in.space(), memory_space::host, in.size_bytes());
    copy_.resize(in.size());
    copy_to_host(res, copy_.data(), in.data(), in.space(), in.device(), in.size_bytes());
    view_ = array_view<const value_t>(copy_.data(), copy_.size());
  }

  host_input(const host_input&) = delete;             ///< not copyable (views point into it)
  host_input& operator=(const host_input&) = delete;  ///< not copyable
  host_input(host_input&&) = delete;                  ///< not movable
  host_input& operator=(host_input&&) = delete;       ///< not movable
  ~host_input() = default;                            ///< frees the copy

  /**
   * @brief The host-readable array.
   * @return A host-accessible view with the elements of the input.
   */
  [[nodiscard]] array_view<const value_t> view() const noexcept {
    return view_;
  }

 private:
  std::vector<value_t> copy_;
  array_view<const value_t> view_;
};

/**
 * @brief A batch whose arrays are all host-readable (each staged by host_input).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename weight_t>
class host_batch {
 public:
  /**
   * @brief Stage every array of `batch` for host reads.
   * @param[in] res   The resources of the call.
   * @param[in] batch The batch (arrays in any memory space).
   * @param[in] what  The calling function, for messages.
   * @throws invalid_argument_error if a copy is needed and the policy is copy_policy::error.
   * @throws not_supported_error    if a copy from device memory is needed without CUDA.
   * @sync
   */
  host_batch(const resources& res, const edge_batch_view<vertex_t, weight_t>& batch,
             std::string_view what)
      : insert_src_(res, batch.insert_src, name(what, "insert_src")),
        insert_dst_(res, batch.insert_dst, name(what, "insert_dst")),
        insert_weights_(res, batch.insert_weights, name(what, "insert_weights")),
        delete_src_(res, batch.delete_src, name(what, "delete_src")),
        delete_dst_(res, batch.delete_dst, name(what, "delete_dst")),
        insert_vertices_(res, batch.insert_vertices, name(what, "insert_vertices")),
        insert_vertex_labels_(res, batch.insert_vertex_labels, name(what, "insert_vertex_labels")),
        delete_vertices_(res, batch.delete_vertices, name(what, "delete_vertices")) {
    view_.insert_src = insert_src_.view();
    view_.insert_dst = insert_dst_.view();
    view_.insert_weights = insert_weights_.view();
    view_.delete_src = delete_src_.view();
    view_.delete_dst = delete_dst_.view();
    view_.insert_vertices = insert_vertices_.view();
    view_.insert_vertex_labels = insert_vertex_labels_.view();
    view_.delete_vertices = delete_vertices_.view();
    view_.num_weights = batch.num_weights;
  }

  /**
   * @brief The staged batch.
   * @return A batch view whose arrays are host-accessible.
   */
  [[nodiscard]] const edge_batch_view<vertex_t, weight_t>& view() const noexcept {
    return view_;
  }

 private:
  static std::string name(std::string_view what, std::string_view field) {
    std::string out(what);
    out += ": edge_batch_view::";
    out += field;
    return out;
  }

  host_input<vertex_t> insert_src_;
  host_input<vertex_t> insert_dst_;
  host_input<weight_t> insert_weights_;
  host_input<vertex_t> delete_src_;
  host_input<vertex_t> delete_dst_;
  host_input<vertex_t> insert_vertices_;
  host_input<std::int8_t> insert_vertex_labels_;
  host_input<vertex_t> delete_vertices_;
  edge_batch_view<vertex_t, weight_t> view_;
};

}  // namespace dyng::detail
