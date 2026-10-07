// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fuzz_legacy_batch.cpp
 * @brief libFuzzer target of dyng::io::read_legacy_batch(), MOSP's `insert.txt` / `delete.txt`
 *        (cpp/fuzz/fuzz_input.hpp).
 *
 * Options: 0 the weight count K (0..3), 1 the vertex count (unchecked -1, 4, 1000, 2^31 - 1),
 * 2 mosp_lenient, 3 the vertex id type (int32, int64). Two files: the insertions, the deletions.
 * An accepted input must give a batch with ids and weights in range, and write_legacy_batch()
 * followed by the strict read_legacy_batch() must give the same batch.
 */
#include "fuzz_input.hpp"

#include <dyng/io/batch_io.hpp>

#include <cstddef>
#include <cstdint>

namespace {

template <typename vertex_t>
void run(const dyng::fuzz::fuzz_input& in, const std::string& ins, const std::string& del) {
  constexpr std::int64_t counts[] = {-1, 4, 1000, 2147483647};
  dyng::io::legacy_batch_options opt;
  opt.num_weights = in.option(0, 4);
  opt.num_vertices = counts[in.option(1, 4)];
  opt.mosp_lenient = in.option(2, 2) == 1;
  const auto read = dyng::fuzz::read_or_reject(
      [&] { return dyng::io::read_legacy_batch<vertex_t, std::int32_t>(ins, del, opt); });
  if (!read) {
    return;  // rejected with a documented exception
  }
  const auto& b = *read;
  DYNG_FUZZ_CHECK(b.num_weights() == opt.num_weights);
  DYNG_FUZZ_CHECK(b.insert_dst().size() == b.num_insertions());
  DYNG_FUZZ_CHECK(b.delete_dst().size() == b.num_deletions());
  DYNG_FUZZ_CHECK(b.insert_weights().size() ==
                  b.num_insertions() * static_cast<std::size_t>(opt.num_weights));
  const auto in_range = [&](vertex_t v) {
    return v >= 0 && (opt.num_vertices < 0 || static_cast<std::int64_t>(v) < opt.num_vertices);
  };
  for (std::size_t i = 0; i < b.num_insertions(); ++i) {
    DYNG_FUZZ_CHECK(in_range(b.insert_src()[i]) && in_range(b.insert_dst()[i]));
  }
  for (std::size_t i = 0; i < b.num_deletions(); ++i) {
    DYNG_FUZZ_CHECK(in_range(b.delete_src()[i]) && in_range(b.delete_dst()[i]));
  }
  for (const std::int32_t w : b.insert_weights()) {
    DYNG_FUZZ_CHECK(w >= 1);
  }
  dyng::fuzz::round_trip([&] {
    // write -> strict read
    auto& files = dyng::fuzz::files();
    const std::string ins_copy = files.path("insert_copy.txt");
    const std::string del_copy = files.path("delete_copy.txt");
    dyng::io::write_legacy_batch(ins_copy, del_copy, b.view());
    dyng::fuzz::written(ins_copy);
    dyng::fuzz::written(del_copy);
    dyng::io::legacy_batch_options strict = opt;
    strict.mosp_lenient = false;
    const auto back =
        dyng::io::read_legacy_batch<vertex_t, std::int32_t>(ins_copy, del_copy, strict);
    DYNG_FUZZ_CHECK(back.insert_src() == b.insert_src());
    DYNG_FUZZ_CHECK(back.insert_dst() == b.insert_dst());
    DYNG_FUZZ_CHECK(back.insert_weights() == b.insert_weights());
    DYNG_FUZZ_CHECK(back.delete_src() == b.delete_src());
    DYNG_FUZZ_CHECK(back.delete_dst() == b.delete_dst());
  });
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const dyng::fuzz::fuzz_input in(data, size, 2);
  auto& files = dyng::fuzz::files();
  const std::string ins = files.write("insert.txt", in.file(0));
  const std::string del = files.write("delete.txt", in.file(1));
  if (in.option(3, 2) == 0) {
    run<std::int32_t>(in, ins, del);
  } else {
    run<std::int64_t>(in, ins, del);
  }
  return 0;
}
