// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file entry_check.hpp
 * @brief The entry checks run_update() makes before it reads a batch of a graph whose layout is
 *        not row_layout::compact (ADR 0070, AD11).
 *
 * An algorithm of 0.1-0.3 reads row_layout::compact only, and the compact-only guard
 * (expect_compact_layout) must refuse any other layout with its named error before any work: in
 * particular before the batch is copied to the host and checked against G_t (Step 0 of
 * batch_semantics::net_effect, which would otherwise raise a misleading invalid_argument_error for
 * an invalid vertex batch, or upload it to the device). run_update() therefore asks each
 * participant, for such a graph and before Step 0, to check its entry: the participants of the
 * framework (problem_participant) implement this interface, either with the problem's own
 * check_entry hook (label_propagation: its backend, its result's state, the layout it reads) or
 * with the compact-only guard named after the problem. A participant that does not implement the
 * interface gets the guard named after the update function.
 *
 * This is internal: the public update_participant interface (dyng/update.hpp) is unchanged.
 */
#pragma once

#include <dyng/core/resources.hpp>

namespace dyng::detail::framework {

/**
 * @brief The entry check of a participant (see the file comment).
 * @tparam container_t The container (graph<V,E,W>).
 */
template <typename container_t>
class entry_check {
 public:
  entry_check() = default;                               ///< constructor
  entry_check(const entry_check&) = delete;              ///< not copyable
  entry_check& operator=(const entry_check&) = delete;   ///< not copyable
  entry_check(entry_check&&) = delete;                   ///< not movable
  entry_check& operator=(entry_check&&) = delete;        ///< not movable
  virtual ~entry_check() = default;                      ///< destructor

  /**
   * @brief The checks of the participant's entry that read nothing of the batch, for a graph whose
   *        layout is not row_layout::compact; throws what the participant's update would throw.
   * @param[in] res Execution resources.
   * @param[in] g   The graph before the batch.
   */
  virtual void check_entry(const resources& res, const container_t& g) = 0;
};

}  // namespace dyng::detail::framework
