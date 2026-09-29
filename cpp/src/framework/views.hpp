// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file views.hpp
 * @brief old_view / new_view: typed snapshots of the container before and after the commit
 *        (invariant I1, "subtract on G_t, add on G_{t+1}"; PLAN Sections 4.5.1 and 4.5.5).
 *
 * The enactor hands hooks that run before the commit (normalize, prepare, before_apply, the
 * aggregate-delta count(-)) an old_view and hooks that run after it (identify_affected, seed,
 * loop, count(+), finalize, enact_fused) a new_view. The two are distinct types, so a hook written
 * for G_t cannot be handed G_{t+1} by mistake: the call does not compile. A view records the
 * container's version when it is made; in Debug builds (NDEBUG undefined) every access checks that
 * the container is still at that version, so an old_view kept past the commit (or a new_view kept
 * past the next batch) throws internal_error instead of silently reading the wrong graph.
 *
 * PLAN Section 4.2 names this header snapshot.hpp; the task that extracted the framework named it
 * views.hpp (docs/developer/framework.md).
 */
#pragma once

#include <dyng/core/error.hpp>

#include <cstdint>

namespace dyng::detail::framework {

namespace view_detail {

/// The Debug check of a view: the container is at the version the view was made for.
[[noreturn]] inline void fail_stale_view(const char* which, std::uint64_t made_at,
                                         std::uint64_t now) {
  DYNG_FAIL("framework: an ", which, " made at graph version ", made_at, " was used at version ",
            now,
            (which[0] == 'o' ? " (invariant I1: G_t is read only before the commit; count the "
                               "deletions in before_apply / count(-), not after it)"
                             : " (the view belongs to an earlier update)"));
}

}  // namespace view_detail

/**
 * @brief G_t: the container before the commit of the current update (read-only).
 * @tparam container_t The container type (e.g. graph<V,E,W>); needs `std::uint64_t version()`.
 */
template <typename container_t>
class old_view {
 public:
  using container_type = container_t;  ///< the container

  /**
   * @brief View `g` at its current version.
   * @param[in] g The container before the commit (must outlive the view).
   */
  explicit old_view(const container_t& g) noexcept : g_(&g), version_(g.version()) {}

  /**
   * @brief The container.
   * @return G_t.
   * @throws internal_error in Debug builds if the container changed since the view was made.
   */
  [[nodiscard]] const container_t& get() const {
#ifndef NDEBUG
    if (g_->version() != version_) {
      view_detail::fail_stale_view("old_view", version_, g_->version());
    }
#endif
    return *g_;
  }

  /**
   * @brief Member access to the container.
   * @return A pointer to G_t.
   * @throws internal_error in Debug builds if the container changed since the view was made.
   */
  const container_t* operator->() const {
    return &get();
  }

  /**
   * @brief The version the view was made at.
   * @return The container's version of G_t.
   */
  [[nodiscard]] std::uint64_t version() const noexcept {
    return version_;
  }

  /**
   * @brief Whether the container is still at the view's version (false after the commit).
   * @return true while the view is valid.
   */
  [[nodiscard]] bool is_current() const noexcept {
    return g_->version() == version_;
  }

 private:
  const container_t* g_;
  std::uint64_t version_;
};

/**
 * @brief G_{t+1}: the container after the commit of the current update (read-only), or the
 *        container a static compute() runs on.
 * @tparam container_t The container type (e.g. graph<V,E,W>); needs `std::uint64_t version()`.
 */
template <typename container_t>
class new_view {
 public:
  using container_type = container_t;  ///< the container

  /**
   * @brief View `g` at its current version.
   * @param[in] g The container after the commit (must outlive the view).
   */
  explicit new_view(const container_t& g) noexcept : g_(&g), version_(g.version()) {}

  /**
   * @brief The container.
   * @return G_{t+1}.
   * @throws internal_error in Debug builds if the container changed since the view was made.
   */
  [[nodiscard]] const container_t& get() const {
#ifndef NDEBUG
    if (g_->version() != version_) {
      view_detail::fail_stale_view("new_view", version_, g_->version());
    }
#endif
    return *g_;
  }

  /**
   * @brief Member access to the container.
   * @return A pointer to G_{t+1}.
   * @throws internal_error in Debug builds if the container changed since the view was made.
   */
  const container_t* operator->() const {
    return &get();
  }

  /**
   * @brief The version the view was made at.
   * @return The container's version of G_{t+1}.
   */
  [[nodiscard]] std::uint64_t version() const noexcept {
    return version_;
  }

  /**
   * @brief Whether the container is still at the view's version.
   * @return true while the view is valid.
   */
  [[nodiscard]] bool is_current() const noexcept {
    return g_->version() == version_;
  }

 private:
  const container_t* g_;
  std::uint64_t version_;
};

}  // namespace dyng::detail::framework
