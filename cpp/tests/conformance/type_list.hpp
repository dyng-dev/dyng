// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file type_list.hpp
 * @brief A small type list for the conformance kit (the registered algorithms, the graph types of
 *        an algorithm, the typed cases of a suite).
 */
#pragma once

#include <gtest/gtest.h>

#include <type_traits>

namespace dyng::conformance {

/// A list of types.
template <typename... types_t>
struct type_list {
  static constexpr int size = static_cast<int>(sizeof...(types_t));  ///< the number of types
};

/// Whether `type_t` is one of the list's types.
template <typename list_t, typename type_t>
struct contains;
/// @copydoc contains
template <typename... types_t, typename type_t>
struct contains<type_list<types_t...>, type_t>
    : std::bool_constant<(std::is_same_v<types_t, type_t> || ...)> {};
/// @copydoc contains
template <typename list_t, typename type_t>
inline constexpr bool contains_v = contains<list_t, type_t>::value;

/// The concatenation of two lists.
template <typename a_t, typename b_t>
struct concat;
/// @copydoc concat
template <typename... a_t, typename... b_t>
struct concat<type_list<a_t...>, type_list<b_t...>> {
  using type = type_list<a_t..., b_t...>;  ///< the result
};

/// The list with `type_t` appended.
template <typename list_t, typename type_t>
using append_t = typename concat<list_t, type_list<type_t>>::type;

/// The types of a list for which `predicate_t<type>::value` holds.
template <typename list_t, template <typename> class predicate_t>
struct filter;
/// @copydoc filter
template <template <typename> class predicate_t>
struct filter<type_list<>, predicate_t> {
  using type = type_list<>;  ///< the result
};
/// @copydoc filter
template <typename first_t, typename... rest_t, template <typename> class predicate_t>
struct filter<type_list<first_t, rest_t...>, predicate_t> {
  using rest = typename filter<type_list<rest_t...>, predicate_t>::type;  ///< the filtered tail
  /// the result
  using type = std::conditional_t<predicate_t<first_t>::value,
                                  typename concat<type_list<first_t>, rest>::type, rest>;
};

/// The list of `wrap_t<type>` for every type of the list.
template <typename list_t, template <typename> class wrap_t>
struct transform;
/// @copydoc transform
template <typename... types_t, template <typename> class wrap_t>
struct transform<type_list<types_t...>, wrap_t> {
  using type = type_list<wrap_t<types_t>...>;  ///< the result
};

/// The list as GoogleTest's ::testing::Types.
template <typename list_t>
struct as_gtest_types;
/// @copydoc as_gtest_types
template <typename... types_t>
struct as_gtest_types<type_list<types_t...>> {
  using type = ::testing::Types<types_t...>;  ///< the result
};

/// Call `f(type_t{})`-style visitors once per type: f.template operator()<type_t>() is not C++17,
/// so `f` receives a value-initialized pointer `static_cast<type_t*>(nullptr)`.
template <typename... types_t, typename function_t>
void for_each_type(type_list<types_t...> /*list*/, function_t&& f) {
  (f(static_cast<types_t*>(nullptr)), ...);
}

}  // namespace dyng::conformance
