// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_replay_main.cpp
 * @brief dyng_cbst_replay: replays CBST traces (ADR 0036) through a store and compares every
 *        operation (ADR 0037). parity/compare.py cbst_trace drives it over the golden sets.
 *
 *     dyng_cbst_replay [--store host_model|device] [--policy escher_gpu|mosp_escher|dyng_default]
 *                      [--fields semantic|offsets|exact] [--json OUT] [--tmp DIR]
 *                      [--max-mismatches N] <trace>...
 *
 * A trace is a `.cbt` file (its side file `<name>.blob` next to it is used) or a `.cbt.zst`
 * (decompressed through `zstd -dc --long=31`; its side file `trace.blob.zst` is decompressed into
 * --tmp, default $TMPDIR, and removed afterwards). Without --policy each trace runs under the
 * preset of its original. `--store device` (CUDA builds) replays through the dynG store
 * (cpp/src/hypergraph/escher; M8a step 3, ADR 0038) on the current CUDA device, under --policy
 * (default escher_gpu). Prints one line per trace; --json writes a JSON array of the reports.
 * Exit status: 0 every trace matched, 1 a mismatch, 2 a usage or input error.
 */
#include "parity/escher/cbst_compare.hpp"
#if defined(DYNG_CBST_DEVICE_STORE) && DYNG_CBST_DEVICE_STORE
#include "parity/escher/cbst_device_adapter.hpp"
#endif

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace cmp = dyng::test::cbst_compare;

namespace {

int usage() {
  std::fprintf(
      stderr,
      "usage: dyng_cbst_replay [--store host_model|device] [--policy NAME] [--fields "
      "semantic|offsets|exact] [--json OUT] [--tmp DIR] [--max-mismatches N] <trace>...\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string store = "host_model", policy_name, json_path, tmp_dir;
  cmp::options opt;
  std::vector<std::string> traces;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--store") {
      store = value();
    } else if (a == "--policy") {
      policy_name = value();
    } else if (a == "--fields") {
      if (!cmp::parse_field_set(value(), opt.fields)) return usage();
    } else if (a == "--json") {
      json_path = value();
    } else if (a == "--tmp") {
      tmp_dir = value();
    } else if (a == "--max-mismatches") {
      opt.max_mismatches = static_cast<std::size_t>(std::strtoull(value().c_str(), nullptr, 10));
    } else if (!a.empty() && a[0] == '-') {
      return usage();
    } else {
      traces.push_back(a);
    }
  }
  if (traces.empty()) return usage();
  cmp::store_factory factory;
  if (store == "host_model") {
    if (!policy_name.empty()) {
      dyng::testing::cbst_host_policy p;
      if (!cmp::resolve_host_policy(policy_name, p)) {
        std::fprintf(stderr, "unknown policy %s\n", policy_name.c_str());
        return 2;
      }
    }
    factory = cmp::host_model_factory(policy_name);
  } else if (store == "device") {
#if defined(DYNG_CBST_DEVICE_STORE) && DYNG_CBST_DEVICE_STORE
    if (!cmp::device_available()) {
      std::fprintf(stderr, "store device: no CUDA device is visible\n");
      return 2;
    }
    try {
      (void)cmp::device_store_adapter(policy_name.empty() ? "escher_gpu" : policy_name);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 2;
    }
    factory = cmp::device_store_factory(policy_name);
#else
    std::fprintf(stderr, "store device: this build has no CUDA (configure a CUDA preset)\n");
    return 2;
#endif
  } else {
    std::fprintf(stderr, "unknown store %s (host_model, device)\n", store.c_str());
    return 2;
  }
  if (tmp_dir.empty()) {
    const char* t = std::getenv("TMPDIR");
    tmp_dir = t ? t : "/tmp";
  }
  std::vector<std::string> reports;
  bool all_ok = true;
  for (const std::string& path : traces) {
    try {
      cmp::trace_source source(path, tmp_dir);
      const cmp::replay_report rep = cmp::replay(source.reader(), factory, opt, path);
      all_ok &= rep.ok();
      std::printf(
          "%s: %s, %lld ops (%lld compared, %lld unreplayable), %lld mappings, %lld keys, "
          "%lld hash-only rows, %lld growths, %lld compactions, %lld exceptions, "
          "%lld mismatches\n",
          path.c_str(), rep.ok() ? "OK" : "MISMATCH", static_cast<long long>(rep.ops),
          static_cast<long long>(rep.compared), static_cast<long long>(rep.unreplayable),
          static_cast<long long>(rep.mappings), static_cast<long long>(rep.keys_compared),
          static_cast<long long>(rep.hash_only_rows), static_cast<long long>(rep.growths),
          static_cast<long long>(rep.compactions), static_cast<long long>(rep.exceptions),
          static_cast<long long>(rep.mismatch_count));
      for (const auto& d : rep.documented)
        std::printf("  documented divergence at op %lld (tree %u %s, its op %lld, %s): %s; %s\n",
                    static_cast<long long>(d.op_index), d.tree, d.dataset.c_str(),
                    static_cast<long long>(d.tree_op), d.op.c_str(), d.rule.c_str(),
                    d.detail.c_str());
      for (const auto& m : rep.mismatches)
        std::printf(
            "  op %lld (seq %lld, tree %u %s, its op %lld, %s) %s key %lld: trace %s, "
            "store %s\n",
            static_cast<long long>(m.op_index), static_cast<long long>(m.seq), m.tree,
            m.dataset.c_str(), static_cast<long long>(m.tree_op), m.op.c_str(), m.field.c_str(),
            static_cast<long long>(m.key), m.expected.c_str(), m.actual.c_str());
      reports.push_back(rep.to_json());
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s: %s\n", path.c_str(), e.what());
      return 2;
    }
  }
  if (!json_path.empty()) {
    std::ofstream out(json_path);
    out << "[";
    for (std::size_t i = 0; i < reports.size(); ++i) out << (i ? ",\n" : "\n") << reports[i];
    out << "\n]\n";
    if (!out) {
      std::fprintf(stderr, "cannot write %s\n", json_path.c_str());
      return 2;
    }
  }
  return all_ok ? 0 : 1;
}
