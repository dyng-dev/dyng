// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// The H-SOSP dump adapter of MOSP_ESCHER@4b86159 (PLAN 6.4.8, 8.3; ADR 0081): the __wrap_
// functions of the six H-SOSP entry points of parity/export_patches/MOSP_ESCHER/wrap_hsosp.txt,
// and the dump side of the batch hooks (bulkLoad, beginBatch), which
// parity/exporters/escher/recorder_mosp_escher_hooks.cpp calls through the weak functions
// dyng_hsosp_* below when this object is linked.
//
//   escher_mosp::DynamicHypergraph::DynamicHypergraph(int, const Caps&)   the session: Caps
//   escher_mosp::hsosp::buildDeviceH2H                                     the initial headroom
//   escher_mosp::hsosp::applyBatch                                         the delta and counters
//   escher_mosp::hsosp::hsospUpdate, ::hsospRecompute                      distances and parents
//   escher_mosp::hsosp::compareDistances                                   the mismatch count
//
// Linked into a twin only when its link command has hsosp/src/hsospDelta.cu.o (relink_twins.sh
// --extra). --wrap sees calls between objects only: the update's own fallback recompute
// (hsospUpdate -> hsospRecompute inside hsospDevice.cu) and applyBatch's CSR rebuild
// (detail::buildCsr) are not wrapped; the fallback shows in the update's stats and its result,
// the rebuild in BatchTimes::csrRebuilt. Every __wrap_ calls __real_, the original, unchanged,
// and records only after the original returned (outside every timer of hsospBench except
// t_sosp_update_ms and t_static_ms, which enclose the update and recompute calls: the patched
// copy's timings are not measurements, PLAN 8.3).
//
// Compiled against the PATCHED scratch copy's unchanged headers (hypergraph/include,
// hsosp/include) with the host compiler; cuda_runtime.h comes from the CUDA toolkit.
#include "DynamicHypergraph.hpp"
#include "HostHypergraph.hpp"
#include "hsosp.cuh"
#include "hsosp_dump.hpp"

#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hd = dyng_parity::hsosp_dump;

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

using escher_mosp::DynamicHypergraph;
using escher_mosp::HgBatch;
using escher_mosp::HostHypergraph;
using escher_mosp::LineGraphCSR;
namespace hs = escher_mosp::hsosp;

namespace {

// A session is the life of one DynamicHypergraph (from its constructor); its host model, device
// graphs and states are found by address. A later constructor at the same address starts a new
// session.
struct session {
  long long id = 0;
  const HostHypergraph* host = nullptr;
  std::vector<int> touched;  // the touched hyperedges of the last beginBatch (1-based ids)
  long long batches = 0;     // applyBatch calls
  long long begin_batches = 0;
  std::unordered_map<const void*, long long> states;  // HsospState* -> ordinal (first use)
};

std::recursive_mutex g_mutex;
std::vector<session*> g_sessions;
std::unordered_map<const void*, session*> g_by_dh, g_by_host, g_by_dev;

session* by(const std::unordered_map<const void*, session*>& m, const void* p) {
  auto it = m.find(p);
  return it == m.end() ? nullptr : it->second;
}

long long sid(const session* s) {
  return s ? s->id : -1;
}

long long state_ordinal(session* s, const void* st) {
  if (!s) return -1;
  auto it = s->states.find(st);
  if (it != s->states.end()) return it->second;
  const long long n = static_cast<long long>(s->states.size());
  s->states.emplace(st, n);
  return n;
}

// The rows of `ids` (1-based) of the host model: sorted vertex lists (empty when dead), weights,
// liveness.
void put_rows(hd::fields& f, const HostHypergraph& hg, const std::vector<int>& ids) {
  std::vector<std::vector<int>> rows;
  std::vector<long long> w;
  std::vector<int> alive;
  rows.reserve(ids.size());
  for (int id : ids) {
    const bool ok = id >= 1 && id <= hg.maxId();
    rows.push_back(ok ? hg.heVerts[id - 1] : std::vector<int>{});
    w.push_back(ok ? hg.heW[id - 1] : 0);
    alive.push_back(ok ? static_cast<int>(hg.alive[id - 1]) : 0);
  }
  f.ints("rows.ids", ids);
  f.rows("rows", rows);
  f.longs("rows.weight", w);
  f.ints("rows.alive", alive);
}

void put_host(hd::fields& f, const HostHypergraph& hg) {
  f.i64("max_id", hg.maxId());
  f.i64("num_vertices", hg.numVertices);
  f.i64("h2h_pair_count", hg.h2hPairCount);
  f.i64("alive_count", hg.aliveCount);
  f.i64("source_he", hg.sourceHe);
  f.i64("target_he", hg.targetHe);
  f.i64("free_ids", static_cast<long long>(hg.freeIds.size()));
}

void put_device(hd::fields& f, const hs::DeviceH2H& dev) {
  f.i64("dev.max_nodes", dev.maxNodes);
  f.i64("dev.num_nodes", dev.numNodes);
  f.i64("dev.cap_entries", dev.capEntries);
  f.i64("dev.used_entries", dev.usedEntries);
  f.i64("dev.num_entries", dev.numEntries);
  f.i64("dev.inc.max_nodes", dev.inc.maxNodes);
  f.i64("dev.inc.num_vertices", dev.inc.numVertices);
  f.i64("dev.inc.he_capacity", dev.inc.heCapacity);
  f.i64("dev.inc.he_tail", dev.inc.heTail);
  f.i64("dev.inc.v_capacity", dev.inc.vCapacity);
  f.i64("dev.inc.v_tail", dev.inc.vTail);
  f.i64("dev.device_bytes", dev.deviceBytes());
}

}  // namespace

extern "C" {
void __real__ZN11escher_mosp17DynamicHypergraphC1EiRKNS0_4CapsE(DynamicHypergraph*, int,
                                                                const DynamicHypergraph::Caps&);
void __real__ZN11escher_mosp5hsosp14buildDeviceH2HERNS0_9DeviceH2HERKNS_14HostHypergraphERKNS_12LineGraphCSREid(
    hs::DeviceH2H&, const HostHypergraph&, const LineGraphCSR&, int, double);
hs::BatchTimes
__real__ZN11escher_mosp5hsosp10applyBatchERNS_17DynamicHypergraphERNS0_9DeviceH2HERKNS_7HgBatchEd(
    DynamicHypergraph&, hs::DeviceH2H&, const HgBatch&, double);
hs::UpdateStats
__real__ZN11escher_mosp5hsosp11hsospUpdateERKNS0_9DeviceH2HERNS0_10HsospStateEiRKNS0_12UpdateConfigE(
    const hs::DeviceH2H&, hs::HsospState&, int, const hs::UpdateConfig&);
hs::UpdateStats
__real__ZN11escher_mosp5hsosp14hsospRecomputeERKNS0_9DeviceH2HERNS0_10HsospStateEiRKNS0_12UpdateConfigE(
    const hs::DeviceH2H&, hs::HsospState&, int, const hs::UpdateConfig&);
long long __real__ZN11escher_mosp5hsosp16compareDistancesERKNS0_10HsospStateES3_i(
    const hs::HsospState&, const hs::HsospState&, int);
}

// --- The session: DynamicHypergraph(int numVertices, const Caps&) ------------------------------

extern "C" void __wrap__ZN11escher_mosp17DynamicHypergraphC1EiRKNS0_4CapsE(
    DynamicHypergraph* self, int num_vertices, const DynamicHypergraph::Caps& caps) {
  hd::dump* d = hd::get();
  if (d == nullptr) {
    __real__ZN11escher_mosp17DynamicHypergraphC1EiRKNS0_4CapsE(self, num_vertices, caps);
    return;
  }
  const DynamicHypergraph::Caps copy = caps;  // Caps is private to the object afterwards
  std::lock_guard<std::recursive_mutex> lock(g_mutex);
  auto* s = new session;
  s->id = static_cast<long long>(g_sessions.size()) + 1;
  g_sessions.push_back(s);
  g_by_dh[self] = s;
  hd::fields f;
  f.i64("session", s->id);
  f.i64("num_vertices", num_vertices);
  f.i64("caps.max_hyperedges", copy.maxHyperedges);
  f.real("caps.headroom_factor", copy.headroomFactor);
  f.i64("caps.extra_payload_ints", copy.extraPayloadInts);
  try {
    __real__ZN11escher_mosp17DynamicHypergraphC1EiRKNS0_4CapsE(self, num_vertices, caps);
  } catch (...) {
    f.str("exception", hd::current_exception_text());
    hd::write(d, hd::tag_session, f);
    g_by_dh.erase(self);
    throw;
  }
  f.str("exception", "");
  hd::write(d, hd::tag_session, f);
}

// --- The batch hooks' dump side (called by recorder_mosp_escher_hooks.cpp) ---------------------

extern "C" bool dyng_hsosp_dump_active() {
  return hd::get() != nullptr;
}

extern "C" void dyng_hsosp_on_bulk_load_enter(const void* self,
                                              const std::vector<std::vector<int>>* rows,
                                              const std::vector<long long>* weights, int source_he,
                                              int target_he) {
  hd::dump* d = hd::get();
  if (d == nullptr) return;
  std::lock_guard<std::recursive_mutex> lock(g_mutex);
  session* s = by(g_by_dh, self);
  hd::fields f;
  f.i64("session", sid(s));
  f.str("phase", "enter");
  // The rows as given (0-based vertices; the virtual source and target hyperedges included).
  f.rows("rows", *rows);
  f.longs("weights", *weights);
  f.i64("source_he", source_he);
  f.i64("target_he", target_he);
  hd::write(d, hd::tag_load, f);
}

extern "C" void dyng_hsosp_on_bulk_load_exit(const void* self, const HostHypergraph* host,
                                             const LineGraphCSR* lg, const char* error) {
  hd::dump* d = hd::get();
  if (d == nullptr) return;
  std::lock_guard<std::recursive_mutex> lock(g_mutex);
  session* s = by(g_by_dh, self);
  hd::fields f;
  f.i64("session", sid(s));
  f.str("phase", "exit");
  f.str("exception", error ? error : "");
  if (error == nullptr && host != nullptr) {
    if (s) {
      s->host = host;
      g_by_host[host] = s;
    }
    put_host(f, *host);
    f.i64("line_graph_entries", lg ? lg->numEntries() : -1);
  }
  hd::write(d, hd::tag_load, f);
}

extern "C" void dyng_hsosp_on_begin_batch_enter(const void* self, const HgBatch* batch) {
  hd::dump* d = hd::get();
  if (d == nullptr) return;
  std::lock_guard<std::recursive_mutex> lock(g_mutex);
  session* s = by(g_by_dh, self);
  hd::fields f;
  f.i64("session", sid(s));
  f.str("phase", "enter");
  f.i64("index", s ? s->begin_batches : -1);
  // The batch in B's op order (hyperedge deletions, incidence removals, incidence additions,
  // hyperedge insertions), 1-based hyperedge ids and 0-based vertices as given.
  f.ints("he_delete", batch->heDelete);
  std::vector<int> a, c;
  for (const auto& x : batch->vtxDelete) a.push_back(x.heId), c.push_back(x.vertex);
  f.ints("vtx_delete.he", a);
  f.ints("vtx_delete.vertex", c);
  a.clear();
  c.clear();
  for (const auto& x : batch->vtxInsert) a.push_back(x.heId), c.push_back(x.vertex);
  f.ints("vtx_insert.he", a);
  f.ints("vtx_insert.vertex", c);
  std::vector<std::vector<int>> verts;
  std::vector<long long> w;
  for (const auto& x : batch->heInsert) verts.push_back(x.vertices), w.push_back(x.weight);
  f.rows("he_insert", verts);
  f.longs("he_insert.weight", w);
  hd::write(d, hd::tag_begin_batch, f);
}

extern "C" void dyng_hsosp_on_begin_batch_exit(const void* self,
                                               const DynamicHypergraph::BatchResult* res,
                                               const HostHypergraph* host, const char* error) {
  hd::dump* d = hd::get();
  if (d == nullptr) return;
  std::lock_guard<std::recursive_mutex> lock(g_mutex);
  session* s = by(g_by_dh, self);
  hd::fields f;
  f.i64("session", sid(s));
  f.str("phase", "exit");
  f.i64("index", s ? s->begin_batches : -1);
  if (s) ++s->begin_batches;
  f.str("exception", error ? error : "");
  if (error == nullptr && res != nullptr) {
    const escher_mosp::IncidenceBatch& inc = res->inc;
    f.ints("final_ids", inc.newHe);
    f.longs("final_weights", inc.newW);
    f.ints("dead_he", inc.deadHe);
    f.i64("skipped_ops", inc.skippedOps);
    // The incidence changes the GPU derives the line-graph delta from: (vertex << 32 | id) and
    // their signs, the touched hyperedges and vertices (with the vertices' post lengths).
    std::vector<long long> keys(inc.incKey.begin(), inc.incKey.end());
    f.longs("inc.key", keys);
    f.ints("inc.sign", inc.incSign);
    f.ints("inc.touched", inc.touched);
    f.ints("inc.touched_vertices", inc.touchedVertices);
    f.ints("inc.touched_vertex_len", inc.touchedVertexLen);
    if (s) s->touched = inc.touched;
    if (host) f.i64("max_id", host->maxId());
  }
  hd::write(d, hd::tag_begin_batch, f);
}

// --- hsosp::buildDeviceH2H ----------------------------------------------------------------------

extern "C" void
__wrap__ZN11escher_mosp5hsosp14buildDeviceH2HERNS0_9DeviceH2HERKNS_14HostHypergraphERKNS_12LineGraphCSREid(
    hs::DeviceH2H& dev, const HostHypergraph& hg, const LineGraphCSR& lg, int max_nodes,
    double entry_headroom) {
  hd::dump* d = hd::get();
  if (d == nullptr) {
    __real__ZN11escher_mosp5hsosp14buildDeviceH2HERNS0_9DeviceH2HERKNS_14HostHypergraphERKNS_12LineGraphCSREid(
        dev, hg, lg, max_nodes, entry_headroom);
    return;
  }
  std::lock_guard<std::recursive_mutex> lock(g_mutex);
  session* s = by(g_by_host, &hg);
  hd::fields f;
  f.i64("session", sid(s));
  f.i64("max_nodes", max_nodes);
  f.real("entry_headroom", entry_headroom);
  f.i64("line_graph_entries", lg.numEntries());
  try {
    __real__ZN11escher_mosp5hsosp14buildDeviceH2HERNS0_9DeviceH2HERKNS_14HostHypergraphERKNS_12LineGraphCSREid(
        dev, hg, lg, max_nodes, entry_headroom);
  } catch (...) {
    f.str("exception", hd::current_exception_text());
    hd::write(d, hd::tag_device_build, f);
    throw;
  }
  f.str("exception", "");
  if (s) g_by_dev[&dev] = s;
  put_device(f, dev);
  hd::write(d, hd::tag_device_build, f);
}

// --- hsosp::applyBatch --------------------------------------------------------------------------

extern "C" hs::BatchTimes
__wrap__ZN11escher_mosp5hsosp10applyBatchERNS_17DynamicHypergraphERNS0_9DeviceH2HERKNS_7HgBatchEd(
    DynamicHypergraph& dh, hs::DeviceH2H& dev, const HgBatch& batch, double entry_headroom) {
  hd::dump* d = hd::get();
  if (d == nullptr)
    return __real__ZN11escher_mosp5hsosp10applyBatchERNS_17DynamicHypergraphERNS0_9DeviceH2HERKNS_7HgBatchEd(
        dh, dev, batch, entry_headroom);
  hs::BatchTimes t;
  std::string error;
  try {
    t = __real__ZN11escher_mosp5hsosp10applyBatchERNS_17DynamicHypergraphERNS0_9DeviceH2HERKNS_7HgBatchEd(
        dh, dev, batch, entry_headroom);
  } catch (...) {
    error = hd::current_exception_text();
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    session* s = by(g_by_dh, &dh);
    hd::fields f;
    f.i64("session", sid(s));
    f.i64("index", s ? s->batches : -1);
    if (s) ++s->batches;
    f.real("entry_headroom", entry_headroom);
    f.str("exception", error);
    hd::write(d, hd::tag_apply_batch, f);
    throw;
  }
  // After applyBatch returned (its stages are synchronized; the sorted keys are on the host).
  std::lock_guard<std::recursive_mutex> lock(g_mutex);
  session* s = by(g_by_dh, &dh);
  const HostHypergraph& hg = dh.host();
  hd::fields f;
  f.i64("session", sid(s));
  f.i64("index", s ? s->batches : -1);
  if (s) ++s->batches;
  f.real("entry_headroom", entry_headroom);
  f.str("exception", "");
  f.i64("bt.candidates", t.candidates);
  f.i64("bt.ins_pairs", t.insPairs);
  f.i64("bt.del_pairs", t.delPairs);
  f.i64("bt.new_he", t.newHe);
  f.i64("bt.dead_he", t.deadHe);
  f.i64("bt.skipped_ops", t.skippedOps);
  f.i64("bt.csr_rebuilt", t.csrRebuilt ? 1 : 0);
  f.i64("bt.mirror_reuploads", t.mirrorReuploads);
  const hs::DeviceDelta& dd = dev.delta;
  f.i64("delta.num_del", dd.numDel);
  f.i64("delta.num_ins", dd.numIns);
  f.i64("delta.num_new", dd.numNew);
  f.i64("delta.num_dead", dd.numDead);
  f.i64("delta.num_sorted_keys", dd.numSortedKeys);
  // The pairs, from the pinned sorted directed keys (row << 33 | isInsert << 32 | col, both
  // directions of every pair): row < col, 0-based node indices (= B id - 1), sorted.
  std::vector<int> del_a, del_b, ins_a, ins_b;
  const auto* keys = reinterpret_cast<const std::uint64_t*>(dd.h_sortedKeys);
  for (long long k = 0; keys != nullptr && k < dd.numSortedKeys; ++k) {
    const std::uint64_t key = keys[k];
    const int row = static_cast<int>(key >> 33);
    const bool ins = ((key >> 32) & 1U) != 0;
    const int col = static_cast<int>(key & 0xffffffffULL);
    if (row >= col) continue;
    (ins ? ins_a : del_a).push_back(row);
    (ins ? ins_b : del_b).push_back(col);
  }
  f.ints("pairs.deleted.a", del_a);
  f.ints("pairs.deleted.b", del_b);
  f.ints("pairs.inserted.a", ins_a);
  f.ints("pairs.inserted.b", ins_b);
  put_host(f, hg);
  put_device(f, dev);
  // The post-batch rows: every hyperedge (DYNG_HSOSP_DUMP_ROWS=all) or the batch's touched ones.
  std::vector<int> ids;
  if (hd::all_rows(d)) {
    ids.resize(static_cast<std::size_t>(hg.maxId()));
    for (int i = 0; i < hg.maxId(); ++i) ids[static_cast<std::size_t>(i)] = i + 1;
    f.str("rows.kind", "all");
  } else {
    ids = s ? s->touched : std::vector<int>{};
    f.str("rows.kind", "touched");
  }
  put_rows(f, hg, ids);
  hd::write(d, hd::tag_apply_batch, f);
  return t;
}

// --- hsosp::hsospUpdate / hsospRecompute --------------------------------------------------------

namespace {

void record_sssp(hd::dump* d, const char* kind, const hs::DeviceH2H& dev, hs::HsospState& st,
                 int source_id, const hs::UpdateConfig& cfg, const hs::UpdateStats* us,
                 const std::string& error) {
  std::lock_guard<std::recursive_mutex> lock(g_mutex);
  session* s = by(g_by_dev, &dev);
  hd::fields f;
  f.i64("session", sid(s));
  f.str("kind", kind);
  f.i64("state", state_ordinal(s, &st));
  f.i64("batches", s ? s->batches : -1);
  f.i64("source", source_id);
  f.i64("cfg.max_iterations", cfg.maxIterations);
  f.real("cfg.work_budget", cfg.workBudget);
  f.i64("cfg.block_size", cfg.blockSize);
  f.str("exception", error);
  if (us != nullptr) {
    // Logged only (not parity targets; HD7): iterations, work, frontier sizes.
    f.i64("stats.iterations", us->iterations);
    f.i64("stats.fallback_recompute", us->fallbackRecompute ? 1 : 0);
    f.i64("stats.fallback_iterations", us->fallbackIterations);
    f.i64("stats.max_frontier", us->maxFrontier);
    f.i64("stats.seed_count", us->seedCount);
    f.i64("stats.jump_rounds", us->jumpRounds);
    f.i64("stats.work", us->work);
    f.i64("stats.num_entries", dev.numEntries);
    const int n = dev.numNodes;
    std::vector<long long> dist;
    std::vector<int> parent;
    st.downloadDistances(dist, n);
    st.downloadParents(parent, n);  // 0-based node indices, -1 = none
    f.i64("n", n);
    f.longs("distances", dist);
    f.ints("parents", parent);
  }
  hd::write(d, hd::tag_sssp, f);
}

}  // namespace

extern "C" hs::UpdateStats
__wrap__ZN11escher_mosp5hsosp11hsospUpdateERKNS0_9DeviceH2HERNS0_10HsospStateEiRKNS0_12UpdateConfigE(
    const hs::DeviceH2H& dev, hs::HsospState& st, int source_id, const hs::UpdateConfig& cfg) {
  hd::dump* d = hd::get();
  if (d == nullptr)
    return __real__ZN11escher_mosp5hsosp11hsospUpdateERKNS0_9DeviceH2HERNS0_10HsospStateEiRKNS0_12UpdateConfigE(
        dev, st, source_id, cfg);
  hs::UpdateStats us;
  try {
    us =
        __real__ZN11escher_mosp5hsosp11hsospUpdateERKNS0_9DeviceH2HERNS0_10HsospStateEiRKNS0_12UpdateConfigE(
            dev, st, source_id, cfg);
  } catch (...) {
    record_sssp(d, "update", dev, st, source_id, cfg, nullptr, hd::current_exception_text());
    throw;
  }
  record_sssp(d, "update", dev, st, source_id, cfg, &us, "");
  return us;
}

extern "C" hs::UpdateStats
__wrap__ZN11escher_mosp5hsosp14hsospRecomputeERKNS0_9DeviceH2HERNS0_10HsospStateEiRKNS0_12UpdateConfigE(
    const hs::DeviceH2H& dev, hs::HsospState& st, int source_id, const hs::UpdateConfig& cfg) {
  hd::dump* d = hd::get();
  if (d == nullptr)
    return __real__ZN11escher_mosp5hsosp14hsospRecomputeERKNS0_9DeviceH2HERNS0_10HsospStateEiRKNS0_12UpdateConfigE(
        dev, st, source_id, cfg);
  hs::UpdateStats us;
  try {
    us =
        __real__ZN11escher_mosp5hsosp14hsospRecomputeERKNS0_9DeviceH2HERNS0_10HsospStateEiRKNS0_12UpdateConfigE(
            dev, st, source_id, cfg);
  } catch (...) {
    record_sssp(d, "recompute", dev, st, source_id, cfg, nullptr, hd::current_exception_text());
    throw;
  }
  record_sssp(d, "recompute", dev, st, source_id, cfg, &us, "");
  return us;
}

// --- hsosp::compareDistances --------------------------------------------------------------------

extern "C" long long __wrap__ZN11escher_mosp5hsosp16compareDistancesERKNS0_10HsospStateES3_i(
    const hs::HsospState& a, const hs::HsospState& b, int n) {
  hd::dump* d = hd::get();
  const long long r =
      __real__ZN11escher_mosp5hsosp16compareDistancesERKNS0_10HsospStateES3_i(a, b, n);
  if (d == nullptr) return r;
  std::lock_guard<std::recursive_mutex> lock(g_mutex);
  // The states' session: the one that used them last (by their ordinals).
  session* s = nullptr;
  for (auto it = g_sessions.rbegin(); it != g_sessions.rend() && s == nullptr; ++it)
    if ((*it)->states.count(&a) != 0 || (*it)->states.count(&b) != 0) s = *it;
  hd::fields f;
  f.i64("session", sid(s));
  f.i64("state_a", state_ordinal(s, &a));
  f.i64("state_b", state_ordinal(s, &b));
  f.i64("n", n);
  f.i64("mismatches", r);
  hd::write(d, hd::tag_compare, f);
  return r;
}
