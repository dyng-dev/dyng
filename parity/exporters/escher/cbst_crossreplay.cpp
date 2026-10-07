// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// The cross-check driver of the two ESCHER cores (PLAN 6.3 step 2; ADR 0036): replays the
// CBST operations of one original's trace through the OTHER original's unchanged store.
//
// parity/exporters/escher/build_crossreplay.sh compiles it against the PATCHED copy's unchanged
// headers (with DYNG_REPLAY_STORE_ESCHER_GPU or DYNG_REPLAY_STORE_MOSP_ESCHER) and links it like
// a twin: the copy's unchanged objects, the recorder and -Xlinker --wrap, so the replay writes
// its own trace (DYNG_CBST_TRACE_DIR) with the same state digests. Before each replayed op it
// writes a NOTE "replay source_seq=<seq> source_tree=<id> op=<op>", by which
// parity/escher_crosscheck.py pairs the two traces and finds the first divergence of every tree.
//
//   cbst_crossreplay <trace.cbt> [--trees h2v,V2H,...] [--capacity-floor N] [--counts scan|null]
//
// --trees         dataset names to replay (default: every tree)
// --capacity-floor the store's payload capacity is max(the trace's capacity, N) (a store that
//                 compacts needs room for the other's growth)
// --counts        MOSP_ESCHER's construct of an ESCHER-GPU trace (which has no row counts): pass
//                 the row counts of the payload (scan: the number of values before each row's
//                 terminator; the default) or nullptr (null: its legacy behaviour, occupancy 0).
//                 A MOSP_ESCHER trace's construct passes what its caller passed.
#include "cbst_recorder.hpp"
#include "parity/escher/cbst_trace.hpp"
#include "structure.hpp"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace rec = dyng_parity::cbst_rec;
namespace tr = dyng::test::cbst_trace;

#if defined(DYNG_REPLAY_STORE_ESCHER_GPU)
constexpr const char* store_name = "ESCHER-GPU@abcf9b2";
#elif defined(DYNG_REPLAY_STORE_MOSP_ESCHER)
constexpr const char* store_name = "MOSP_ESCHER@4b86159";
#else
#error "define DYNG_REPLAY_STORE_ESCHER_GPU or DYNG_REPLAY_STORE_MOSP_ESCHER"
#endif

namespace {

struct source_tree {
  std::string name;
  long long capacity = 0;
  long long alignment = 4;
};

std::set<std::string> split(const std::string& s) {
  std::set<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.insert(item);
  }
  return out;
}

// The number of values of every row of a flattened payload (positive values before the row's
// terminator, 0 or INT_MIN; construct payloads have no chains).
[[maybe_unused]] std::vector<int> row_counts(const std::vector<int>& starts,
                                             const std::vector<int>& flat) {
  std::vector<int> out(starts.size(), 0);
  for (std::size_t i = 0; i < starts.size(); ++i) {
    for (long long p = starts[i]; p >= 0 && p < static_cast<long long>(flat.size()); ++p) {
      const int v = flat[static_cast<std::size_t>(p)];
      if (v == 0 || v == INT_MIN || v < 0) break;
      ++out[i];
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <trace.cbt> [--trees a,b] [--capacity-floor N] "
                 "[--counts scan|null]\n",
                 argv[0]);
    return 2;
  }
  std::string path = argv[1];
  std::set<std::string> trees;
  long long floor = 0;
  bool counts_scan = true;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--trees")
      trees = split(argv[i + 1]);
    else if (a == "--capacity-floor")
      floor = std::atoll(argv[i + 1]);
    else if (a == "--counts")
      counts_scan = std::string(argv[i + 1]) != "null";
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  rec::recorder* r = rec::original_recorder();
  if (r == nullptr) {
    std::fprintf(stderr, "set DYNG_CBST_TRACE_DIR: the replay is recorded\n");
    return 2;
  }
  std::unique_ptr<tr::reader> in;
  try {
    in = tr::reader::open(path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  rec::note(r, std::string("replay of ") + path.substr(path.find_last_of('/') + 1) +
                   " through the store of " + store_name);
  std::map<std::uint32_t, source_tree> sources;
  std::map<std::uint32_t, std::unique_ptr<CBSTOperations>> stores;
  std::deque<std::string>
      names;  // dataset names outlive the stores (CBSTContext keeps the pointer)
  std::set<std::uint32_t> stopped;
  long long replayed = 0, skipped = 0, threw = 0;
  tr::record rec_;
  try {
    while (in->next(rec_)) {
      if (rec_.kind == tr::record_kind::tree) {
        source_tree& t = sources[static_cast<std::uint32_t>(rec_.integer("tree"))];
        t.name = rec_.text("dataset_name");
        t.capacity = rec_.integer("payload_capacity");
        t.alignment = rec_.integer("alignment", 4);
        continue;
      }
      if (rec_.kind != tr::record_kind::op) continue;
      const auto id = static_cast<std::uint32_t>(rec_.integer("tree"));
      const source_tree& src = sources[id];
      if (!trees.empty() && trees.count(src.name) == 0) continue;
      if (stopped.count(id)) {
        ++skipped;
        continue;
      }
      const std::string op = rec_.text("op");
      std::ostringstream label;
      label << "replay source_seq=" << rec_.integer("seq") << " source_tree=" << id << " op=" << op;
      auto ints = [&](const char* name) { return in->ints(rec_.array(std::string("in.") + name)); };
      try {
        if (op == "construct") {
          names.push_back(src.name);
          const long long cap = std::max<long long>(rec_.integer("in.capacity"), floor);
          rec::note(r, label.str() + " capacity=" + std::to_string(cap));
          auto store = std::make_unique<CBSTOperations>(
              names.back().c_str(), static_cast<int>(std::min<long long>(cap, INT_MAX)),
              static_cast<int>(rec_.integer("in.alignment", src.alignment)));
          std::vector<int> keys = ints("keys"), starts = ints("start_offsets"),
                           flat = ints("flat_payload");
          const int n = static_cast<int>(rec_.integer("in.num_records"));
          const int size = static_cast<int>(rec_.integer("in.flat_payload_size"));
          stores[id] = std::move(store);
#if defined(DYNG_REPLAY_STORE_MOSP_ESCHER)
          // A MOSP_ESCHER trace says what its caller passed (the row counts or nullptr); an
          // ESCHER-GPU trace has no counts: scan the payload (--counts scan) or pass nullptr.
          std::vector<int> counts;
          const int* occupancy = nullptr;
          if (rec_.has("in.row_occupancy_null")) {
            if (rec_.integer("in.row_occupancy_null") == 0) {
              counts = ints("row_occupancy");
              occupancy = counts.data();
            }
          } else if (counts_scan) {
            counts = row_counts(starts, flat);
            occupancy = counts.data();
          }
          stores[id]->construct(keys.data(), starts.data(), n, flat.data(), size, occupancy);
#else
          (void)counts_scan;
          stores[id]->construct(keys.data(), starts.data(), n, flat.data(), size);
#endif
        } else if (stores.count(id) == 0) {
          rec::note(r, label.str() + " skipped: the tree was not constructed in this replay");
          ++skipped;
          continue;
        } else if (op == "insert") {
          rec::note(r, label.str());
          (void)stores[id]->insert(ints("keys"), ints("payload"), ints("prefix"));
        } else if (op == "fill") {
          rec::note(r, label.str());
          stores[id]->fill(ints("keys"), ints("payload"), ints("prefix"));
        } else if (op == "erase") {
          rec::note(r, label.str());
          stores[id]->erase(ints("keys"));
        } else if (op == "unfill") {
          rec::note(r, label.str());
#if defined(DYNG_REPLAY_STORE_MOSP_ESCHER)
          unfillCBST(ints("keys"), ints("payload"), ints("prefix"),
                     const_cast<CBSTContext&>(stores[id]->context()));
#else
          stores[id]->unfill(ints("keys"), ints("payload"), ints("prefix"));
#endif
        } else if (op == "compact") {
#if defined(DYNG_REPLAY_STORE_MOSP_ESCHER)
          rec::note(r, label.str());
          (void)stores[id]->compact();
#else
          rec::note(r, label.str() + " skipped: ESCHER-GPU has no compact()");
          ++skipped;
          continue;
#endif
        } else if (op == "rebuild") {
#if defined(DYNG_REPLAY_STORE_ESCHER_GPU)
          rec::note(r, label.str());
          // The rebuild's inputs are device arrays the tree takes ownership of.
          std::vector<int> starts = ints("start_offsets"), flat = ints("flat_payload");
          int *d_starts = nullptr, *d_flat = nullptr;
          cudaMalloc(&d_starts, std::max<std::size_t>(1, starts.size()) * sizeof(int));
          cudaMalloc(&d_flat, std::max<std::size_t>(1, flat.size()) * sizeof(int));
          cudaMemcpy(d_starts, starts.data(), starts.size() * sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(d_flat, flat.data(), flat.size() * sizeof(int), cudaMemcpyHostToDevice);
          stores[id]->rebuildFromDevice(d_starts, static_cast<int>(rec_.integer("in.num_records")),
                                        d_flat,
                                        static_cast<int>(rec_.integer("in.flat_payload_size")),
                                        static_cast<int>(rec_.integer("in.capacity")));
#else
          rec::note(r, label.str() +
                           " stop: MOSP_ESCHER has no rebuildFromDevice; the tree "
                           "is not replayed further");
          stopped.insert(id);
          ++skipped;
          continue;
#endif
        } else {
          rec::note(r, label.str() + " skipped: unknown op");
          ++skipped;
          continue;
        }
        ++replayed;
      } catch (const tr::trace_error& e) {
        rec::note(r, label.str() + " stop: " + e.what());
        stopped.insert(id);
        ++skipped;
      } catch (const std::exception&) {
        ++threw;  // recorded in the replay's trace (status 1)
        ++replayed;
      }
    }
  } catch (const tr::trace_error& e) {
    std::fprintf(stderr, "%s: %s\n", path.c_str(), e.what());
    return 1;
  }
  std::printf("replayed %lld ops (%lld threw), skipped %lld, through %s\n", replayed, threw,
              skipped, store_name);
  return 0;
}
