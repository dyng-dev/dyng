// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file host_only.cpp
 * @brief The host-only consumer of ci/cmake_consumer.sh: a CXX-only program against the installed
 *        package that uses the experimental host header (never the extension header
 *        local_patterns.cuh): the rows of a hypergraph by access_rows(), and triad_count.
 */
#include <dyng/dyng.hpp>
#include <dyng/experimental/local_patterns.hpp>
#include <dyng/triad_count.hpp>

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
  const dyng::resources res = dyng::resources::sequential();
  dyng::hyperedge_list<std::int32_t, dyng::unweighted> list;
  list.num_vertices = 6;
  for (const std::vector<std::int32_t>& row :
       std::vector<std::vector<std::int32_t>>{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {0, 4, 5}}) {
    list.add_hyperedge(dyng::host_view(row));
  }
  const auto h =
      dyng::hypergraph<std::int32_t, dyng::unweighted>::from_hyperedges(res, list.view());
  const dyng::experimental::row_access rows = dyng::experimental::access_rows(h);
  std::int64_t members = 0;
  for (std::int64_t id = 0; id < rows.id_bound(); ++id) {
    for (const dyng::experimental::member_code c : rows.hyperedges().row(static_cast<int>(id))) {
      members += dyng::experimental::hyperedge_rows::decode(c) >= 0 ? 1 : 0;
    }
  }
  const std::uint64_t triads = dyng::triad_count::compute(res, h).total();
  std::printf("host_only: %lld members, %llu triads\n", static_cast<long long>(members),
              static_cast<unsigned long long>(triads));
  return members == 12 && triads == 2 ? 0 : 1;
}
