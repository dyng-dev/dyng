// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// cbst_trace_tool verify <trace.cbt>...: rebuilds every tree of a CBST trace after every op with
// the host-only reader (cpp/tests/parity/escher/cbst_trace.hpp) and checks every hash the
// recorder wrote, and that the trace ends normally (ADR 0036). The fast twin of
// `parity/escher_trace.py verify`, used by parity/cbst_trace_goldens.py for the large traces.
// Built by parity/exporters/escher/build_crossreplay.sh (host code only).
#include "parity/escher/cbst_trace.hpp"

#include <cstdio>
#include <exception>
#include <string>

namespace tr = dyng::test::cbst_trace;

int main(int argc, char** argv) {
  if (argc < 3 || std::string(argv[1]) != "verify") {
    std::fprintf(stderr, "usage: %s verify <trace.cbt>...\n", argv[0]);
    return 2;
  }
  int bad = 0;
  for (int i = 2; i < argc; ++i) {
    try {
      auto r = tr::reader::open(argv[i]);
      tr::state_tracker tracker;
      tr::record rec;
      long long ops = 0;
      std::string exit_kind;
      std::string first;
      while (r->next(rec)) {
        if (rec.kind == tr::record_kind::op) ++ops;
        if (rec.kind == tr::record_kind::end) exit_kind = rec.text("exit");
        for (const auto& p : tracker.apply(rec, *r))
          if (first.empty()) first = "seq " + std::to_string(rec.integer("seq")) + ": " + p;
      }
      if (exit_kind != "normal" && first.empty()) first = "the trace does not end normally";
      if (first.empty()) {
        std::printf("%s: %lld ops verified\n", argv[i], ops);
      } else {
        std::printf("%s: FAILED: %s\n", argv[i], first.c_str());
        ++bad;
      }
    } catch (const std::exception& e) {
      std::printf("%s: FAILED: %s\n", argv[i], e.what());
      ++bad;
    }
  }
  return bad ? 1 : 0;
}
