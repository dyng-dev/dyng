// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file entry_check.hpp
 * @brief The entry checks run_update() makes before it reads a batch of a graph whose layout is
 *        not row_layout::compact (ADR 0070, AD11), and the participant of a problem that makes
 *        its own.
 *
 * An algorithm of 0.1-0.3 reads row_layout::compact only, and the compact-only guard
 * (expect_compact_layout) must refuse any other layout with its named error before any work: in
 * particular before the batch is copied to the host and checked against G_t (Step 0 of
 * batch_semantics::net_effect, which would otherwise raise a misleading invalid_argument_error for
 * an invalid vertex batch, or upload it to the device; M10a review). run_update() therefore,
 * for such a graph and before Step 0, asks each participant that implements entry_check to check
 * its entry (label_propagation: its backend, its result's state, the layout it reads), and
 * applies the compact-only guard, named after the update function, to every other participant.
 *
 * checked_participant<problem_t> is problem_participant<problem_t> with the problem's
 * check_entry(res, g) as its entry check; make_checked_participant() and update_one_checked()
 * are make_participant() and update_one() with it. This header is internal and included only by
 * the code that uses it (composition.cpp, label_propagation): the public update_participant and
 * framework/composition.hpp are unchanged, so no other algorithm's object code changes (AD11:
 * triad_count's object files).
 */
#pragma once

#include "framework/composition.hpp"

#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>

#include <memory>
#include <utility>

namespace dyng::detail::framework {

/**
 * @brief The entry check of a participant (see the file comment).
 * @tparam container_t The container (graph<V,E,W>).
 */
template <typename container_t>
class entry_check {
 public:
  entry_check() = default;                              ///< constructor
  entry_check(const entry_check&) = delete;             ///< not copyable
  entry_check& operator=(const entry_check&) = delete;  ///< not copyable
  entry_check(entry_check&&) = delete;                  ///< not movable
  entry_check& operator=(entry_check&&) = delete;       ///< not movable
  virtual ~entry_check() = default;                     ///< destructor

  /**
   * @brief The checks of the participant's entry that read nothing of the batch, for a graph whose
   *        layout is not row_layout::compact; throws what the participant's update would throw.
   * @param[in] res Execution resources.
   * @param[in] g   The graph before the batch.
   */
  virtual void check_entry(const resources& res, const container_t& g) = 0;
};

/**
 * @brief problem_participant<problem_t> with the problem's `check_entry(res, g)` as its entry
 *        check (see the file comment).
 * @tparam problem_t A graph problem with `void check_entry(const resources&, const
 *                   container_type&)`.
 */
template <typename problem_t>
class checked_participant final
    : public composition_detail::graph_types<typename problem_t::container_type>::participant,
      public entry_check<typename problem_t::container_type> {
  using types = composition_detail::graph_types<typename problem_t::container_type>;

 public:
  using graph_type = typename problem_t::container_type;  ///< the container
  using batch_type = typename types::batch;               ///< the batch view
  using stats_type = typename problem_t::stats_type;      ///< the problem's stats

  /**
   * @brief The participant of a problem built from `args`.
   * @tparam args_t The problem's constructor arguments.
   * @param[out] out  Receives the update's stats when it succeeds (must outlive the participant).
   * @param[in]  args Forwarded to the problem's constructor.
   */
  template <typename... args_t>
  explicit checked_participant(stats_type& out, args_t&&... args)
      : inner_(out, std::forward<args_t>(args)...) {}

  /// The result this participant updates (problem_participant's).
  [[nodiscard]] const void* target() const noexcept override {
    return inner_.target();
  }
  /// Whether the commit prepares the graph for this problem (problem_participant's).
  [[nodiscard]] bool reads_prepared_graph() const noexcept override {
    return inner_.reads_prepared_graph();
  }
  /// Receive the framework's Step 0 (problem_participant's).
  void use_normalized(const typename types::normalized* normalized) noexcept override {
    inner_.use_normalized(normalized);
  }
  /// Steps 0 and 1a on G_t (problem_participant's).
  void before_apply(const resources& res, const graph_type& g, const batch_type& batch) override {
    inner_.before_apply(res, g, batch);
  }
  /// The algorithm phase on G_{t+1} (problem_participant's).
  void after_apply(const resources& res, const graph_type& g,
                   const typename types::summary& summary,
                   const typename types::record& record) override {
    inner_.after_apply(res, g, summary, record);
  }
  /// Mark the result unusable (problem_participant's).
  void poison() noexcept override {
    inner_.poison();
  }

  /**
   * @brief The problem's entry checks (before Step 0; run_update()).
   * @param[in] res Execution resources.
   * @param[in] g   G_t.
   */
  void check_entry(const resources& res, const graph_type& g) override {
    inner_.problem().check_entry(res, g);
  }

 private:
  problem_participant<problem_t> inner_;
};

/**
 * @brief make_participant() with the problem's entry check (dyng::update()'s participant).
 * @tparam problem_t The problem.
 * @tparam args_t    Its constructor arguments.
 * @param[out] out  Receives the stats when the update succeeds.
 * @param[in]  args Forwarded to the problem's constructor.
 * @return The participant.
 */
template <typename problem_t, typename... args_t>
std::unique_ptr<
    typename composition_detail::graph_types<typename problem_t::container_type>::participant>
make_checked_participant(typename problem_t::stats_type& out, args_t&&... args) {
  return std::make_unique<checked_participant<problem_t>>(out, std::forward<args_t>(args)...);
}

/**
 * @brief update_one() with the problem's entry check: `<algo>::update(res, g, batch, r)` inside
 *        the stage "<algo>.update", with the commit stage "<algo>.commit".
 * @tparam problem_t The problem.
 * @tparam args_t    Its constructor arguments.
 * @param[in]     res   Execution resources.
 * @param[in,out] g     The graph (version + 1).
 * @param[in]     batch The batch (any memory space; run_update() stages it to the host).
 * @param[in]     args  Forwarded to the problem's constructor.
 * @return The update's stats.
 * @throws what run_update() and the hooks throw.
 */
template <typename problem_t, typename... args_t>
typename problem_t::stats_type update_one_checked(
    const resources& res, typename problem_t::container_type& g,
    const typename composition_detail::graph_types<typename problem_t::container_type>::batch&
        batch,
    args_t&&... args) {
  const hook_stages& names = stages_of<problem_t>();
  scoped_stage stage(res, names.update);
  typename problem_t::stats_type out;
  checked_participant<problem_t> participant(out, std::forward<args_t>(args)...);
  typename composition_detail::graph_types<typename problem_t::container_type>::participant*
      participants[] = {&participant};
  run_update(res, g, batch, participants, 1, names.commit);
  return out;
}

}  // namespace dyng::detail::framework
