// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file update_dispatch_test.cpp
 * @brief dyng::update() and update_each() reach a container through the customization point
 *        detail::participant_of<container_t> (its participant interface, batch view and run()),
 *        so a container declared after <dyng/update.hpp> (the hypergraph of 0.2) takes part
 *        without a change to the frozen header. The fake container below is declared after the
 *        include on purpose.
 */
#include <dyng/core/array_view.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stats.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string_view>
#include <tuple>
#include <vector>

namespace fake {

/// A container of another kind (declared after <dyng/update.hpp>).
struct container {
  int version = 0;  ///< +1 per run()
};
/// Its batch view.
struct batch_view {
  int size = 0;  ///< elements
};
/// Its participant interface.
struct participant {
  participant() = default;
  participant(const participant&) = delete;
  participant& operator=(const participant&) = delete;
  participant(participant&&) = delete;
  participant& operator=(participant&&) = delete;
  virtual ~participant() = default;
  virtual void update(container& c, const batch_view& b) = 0;
};
/// A result on it.
struct result {
  int seen_version = -1;  ///< the version after its last update
  int seen_batch = -1;    ///< the size of its last batch
};
/// Its stats.
struct stats : dyng::update_stats {};

struct result_participant final : participant {
  result_participant(result& r, stats& s) : r_(r), s_(s) {}
  void update(container& c, const batch_view& b) override {
    r_.seen_version = c.version;
    r_.seen_batch = b.size;
    s_.affected = b.size;
  }

 private:
  result& r_;
  stats& s_;
};

}  // namespace fake

namespace dyng::detail {

template <>
struct participant_of<fake::container> {
  static constexpr bool supported = true;
  using type = fake::participant;
  using batch_type = fake::batch_view;
  static apply_summary run(const resources& /*res*/, fake::container& c, const batch_type& b,
                           type* const* participants, std::size_t count,
                           std::string_view /*commit_stage*/) {
    ++c.version;  // one commit for every participant
    for (std::size_t i = 0; i < count; ++i) {
      participants[i]->update(c, b);
    }
    return {};
  }
};

template <>
struct update_traits<fake::result> {
  using stats_type = fake::stats;
  template <typename container_t>
  static std::unique_ptr<fake::participant> make_participant(fake::result& r, fake::stats& s) {
    return std::make_unique<fake::result_participant>(r, s);
  }
};

}  // namespace dyng::detail

namespace {

TEST(UpdateDispatch, AContainerDeclaredAfterTheHeaderTakesPart) {
  const dyng::resources res = dyng::resources::sequential();
  fake::container c;
  fake::result a;
  fake::result b;
  const auto [sa, sb] = dyng::update(res, c, fake::batch_view{3}, a, b);
  EXPECT_EQ(c.version, 1);  // one commit
  EXPECT_EQ(a.seen_version, 1);
  EXPECT_EQ(b.seen_version, 1);
  EXPECT_EQ(sa.affected, 3);
  EXPECT_EQ(sb.affected, 3);
  std::vector<fake::result*> list{&a, &b};
  const auto each = dyng::update_each(res, c, fake::batch_view{5}, dyng::host_view(list));
  EXPECT_EQ(c.version, 2);
  ASSERT_EQ(each.size(), 2U);
  EXPECT_EQ(a.seen_batch, 5);
}

TEST(UpdateDispatch, UpdateEachRejectsAnEmptyListBeforeAnythingChanges) {
  const dyng::resources res = dyng::resources::sequential();
  fake::container c;
  std::vector<fake::result*> none;
  EXPECT_THROW((void)dyng::update_each(res, c, fake::batch_view{1}, dyng::host_view(none)),
               dyng::invalid_argument_error);
  EXPECT_EQ(c.version, 0);
}

TEST(UpdateDispatch, UpdateEachRejectsAListInDeviceMemory) {
  const dyng::resources res = dyng::resources::sequential();
  fake::container c;
  fake::result a;
  fake::result* pointers[] = {&a};
  const dyng::array_view<fake::result* const> device_list(pointers, 1, dyng::memory_space::device,
                                                          0);
  EXPECT_THROW((void)dyng::update_each(res, c, fake::batch_view{1}, device_list),
               dyng::invalid_argument_error);
  EXPECT_EQ(c.version, 0);
}

}  // namespace
