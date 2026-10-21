// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file op.cu
 * @brief The nvcc consumer of ci/cmake_consumer.sh: a user Op (the triangles of the line graph)
 *        compiled by nvcc against the installed package, counted with
 *        experimental::count_local_patterns on every backend of the build (cuda when a GPU is
 *        visible), updated under a batch together with triad_count in one dyng::update(), and
 *        compared with triad_count's total and with a fresh compute().
 */
#include <dyng/dyng.hpp>
#include <dyng/experimental/local_patterns.cuh>
#include <dyng/triad_count.hpp>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

namespace dx = dyng::experimental;

struct triangles {
  static constexpr int num_bins = 1;
  static constexpr dx::shape shape = dx::shape::triangles;
  std::uint64_t count = 0;
  DYNG_HD void begin_wedge(const dx::hyperedge_rows&, int, int) {}
  template <class bins_t>
  DYNG_HD void triangle(const dx::hyperedge_rows&, int, bins_t&) {
    ++count;
  }
  template <class bins_t>
  DYNG_HD void flush(bins_t& bins) {
    if (count != 0) bins.add(0, count);
  }
};

using graph_t = dyng::hypergraph<std::int32_t, dyng::unweighted>;

bool run(const dyng::resources& res) {
  dyng::hyperedge_list<std::int32_t, dyng::unweighted> list;
  list.num_vertices = 8;
  for (const std::vector<std::int32_t>& row : std::vector<std::vector<std::int32_t>>{
           {0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {0, 4, 5}, {5, 6}, {2, 5, 7}}) {
    list.add_hyperedge(dyng::host_view(row));
  }
  graph_t h = graph_t::from_hyperedges(res, list.view());
  using count_t = dx::count_local_patterns<triangles>;
  count_t::result r = count_t::compute(res, h);
  dyng::triad_count::result motifs = dyng::triad_count::compute(res, h);
  bool ok = r.count(0) == motifs.total();
  dyng::hyperedge_batch<std::int32_t, dyng::unweighted> batch;
  batch.delete_hyperedge(1);
  batch.insert_hyperedge({1, 3, 5});
  const auto [st, mt] = dyng::update(res, h, batch.view(), r, motifs);
  ok = ok && r.count(0) == motifs.total() && r.count(0) == count_t::compute(res, h).count(0);
  ok = ok && st.delta[0] == static_cast<std::int64_t>(st.added - st.removed);
  (void)mt;
  std::printf("op: %s: %llu triangles after the batch: %s\n",
              std::string(dyng::to_string(res.get_backend())).c_str(),
              static_cast<unsigned long long>(r.count(0)), ok ? "ok" : "WRONG");
  return ok;
}

}  // namespace

int main() {
  bool ok = run(dyng::resources::sequential());
  if (dyng::backend_available(dyng::backend::openmp)) ok = run(dyng::resources::openmp(2)) && ok;
  if (dyng::backend_available(dyng::backend::cuda)) ok = run(dyng::resources::cuda()) && ok;
  return ok ? 0 : 1;
}
