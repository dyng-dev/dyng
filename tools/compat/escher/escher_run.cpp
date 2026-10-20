// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:src/driver.cu (run: the stages, their timers and the stdout
// lines of runDynamicCounting)
/**
 * @file escher_run.cpp
 * @brief `dyng-compat-escher run`: ESCHER-GPU@abcf9b2's triad-counting driver (build/main,
 *        build/type1..3) on dynG, the port side of the triad_count gates (M9a; PLAN 6.4.6, 8.6;
 *        parity/timed_regions/triad_count.toml; ADR 0051).
 *
 *     dyng-compat-escher run <A's arguments> [--pattern h_motif|type1|type2|type3]
 *                            [--backend sequential|openmp|cuda] [--threads N]
 *                            [--sync-stages] [--nvtx] [--no-count] [--dump-batches FILE]
 *
 * A's arguments are taken verbatim (utils/options.cpp): `N maxCard minCard V [capacity]
 * [alignment]` or `--input F [capacity] [alignment]`, --batches, --batch-size, --deletions,
 * --insertions, --seed, --insert-mode, --check and --timing FILE (A's JSON {meta, stages[{stage,
 * batch, ms}]}). What it does, in A's order (src/driver.cu run()):
 *
 *  1. the clock starts at the top of main(); resources::warm_up() (the CUDA context and, under
 *     CUDA_MODULE_LOADING=EAGER, the modules; A's first CUDA call does the same inside its total);
 *  2. load (`load_ms`): io::read_hypergraph(F, hypergraph_read_options::escher_gpu()) or
 *     generators::legacy::escher_hypergraph(std::mt19937_64(seed), {N, maxCard, minCard - 1,
 *     V - 1}) after A's lines "Generating hypergraph with: ..." (printHypergraphParams); A's line
 *     "Hypergraph: E hyperedges, max vertex id V, P pins";
 *  3. the hypergraph under {id_reuse::erase_first, capacity_policy::escher_gpu(capacity),
 *     line_graph_policy::sorted_merge}: the line graph and the row views exist after the build
 *     (hypergraph.build, hypergraph.upload), and generators::legacy::escher_host_model, the host
 *     rows the batches are drawn from (A's DynamicHypergraph keeps the same rows);
 *  4. the full count, triad_count::compute() of the pattern (`full_count_ms`, between device
 *     synchronizations); A's count line ("Motif counts (30 bins): ..." for h_motif, "[GPU] TypeN
 *     baseline motif count...: N" for the types), byte for byte;
 *  5. per batch: the batch drawn by escher_batch_generator::next(model) from the one engine
 *     (`generate_ms`, outside the update region), then the update (`update_ms` around
 *     triad_count::update()) or, with --no-count, hypergraph::apply() alone (`apply_ms`), each
 *     started after and ended with a device synchronization; the model follows the summary; A's
 *     lines "Batch b (D deletions, I insertions)" (when --batches > 1) and the delta ("Motif delta
 *     counts (30 bins): ..." or "[GPU] TypeN motif delta count: N"); with --check,
 *     check_integrity() and "baseline + deltas == full recount" (a fresh compute()), outside the
 *     timers;
 *  6. `total_ms` after the last batch; the --timing file holds the driver timers (load_ms,
 *     full_count_ms, generate_ms, update_ms or apply_ms, total_ms; batch 0 = before the batches)
 *     and every profiler stage per batch.
 *
 * --dump-batches FILE (never in a timed run) writes every batch drawn as A's dumps are converted
 * (`to-dgb`: flags deleted_members and expected_ids; the deleted rows from the host model, the
 * ids from hyper_apply_summary::inserted_ids), so the batches can be compared with the goldens.
 *
 * The gated process keeps no public view of the hypergraph: the generator reads the host model,
 * never hypergraph::h2v() (whose device CSR would stay live across the next apply and add to the
 * Nsight Systems peak, M9a criterion 13). Exit status: 0 done, 1 a --check failure, 2 a usage or
 * input error.
 */
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/hypergraph_io.hpp>
#include <dyng/triad_count.hpp>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dyng::compat::escher {

namespace {

using clock_type = std::chrono::steady_clock;
using hgraph = hypergraph<std::int32_t, unweighted>;
namespace legacy = generators::legacy;

struct usage_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// A's options (utils/options.cpp parseOptions) and the driver's own.
struct run_options {
  // A's
  std::string input;
  std::int64_t num_hyperedges = 0;
  std::int32_t max_card = 1, min_vertex = 1, max_vertex = 1;
  std::int64_t payload_capacity = 0;
  std::int64_t batches = 1;
  bool composition_set = false;
  std::int64_t deletions = 0, insertions = 0;
  std::uint64_t seed = 1;
  bool insert_mode_set = false, sample = false;
  bool check = false;
  std::string timing;
  std::vector<std::string> argv;  // A's arguments as given (the --timing file's meta)
  // the driver's
  std::string pattern = "h_motif";
  std::string backend = "cuda";
  int threads = 0;
  bool sync_stages = false, nvtx = false, count = true;
  std::string dump_batches;  // --dump-batches FILE: every batch drawn, as A's dumps (.dgb)
};

long long parse_int(const std::string& s, long long lo, long long hi, const std::string& what) {
  char* end = nullptr;
  errno = 0;
  const long long v = std::strtoll(s.c_str(), &end, 10);
  if (s.empty() || end == s.c_str() || *end != '\0' || errno == ERANGE || v < lo || v > hi)
    throw usage_error(what + " expects an integer in " + std::to_string(lo) + ".." +
                      std::to_string(hi) + " (got " + s + ")");
  return v;
}

run_options parse(int argc, char** argv) {
  run_options o;
  std::vector<std::string> positional;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw usage_error(a + " expects a value");
      return argv[++i];
    };
    if (a.rfind("--", 0) != 0) {
      positional.push_back(a);
      o.argv.push_back(a);
      continue;
    }
    if (a == "--pattern") {
      o.pattern = value();
      if (o.pattern == "main") o.pattern = "h_motif";
      if (o.pattern != "h_motif" && o.pattern != "type1" && o.pattern != "type2" &&
          o.pattern != "type3")
        throw usage_error("--pattern expects h_motif, type1, type2 or type3");
      continue;
    }
    if (a == "--backend") {
      o.backend = value();
      continue;
    }
    if (a == "--threads") {
      o.threads = static_cast<int>(parse_int(value(), 0, 4096, a));
      continue;
    }
    if (a == "--sync-stages") {
      o.sync_stages = true;
      continue;
    }
    if (a == "--nvtx") {
      o.nvtx = true;
      continue;
    }
    if (a == "--no-count") {
      o.count = false;
      continue;
    }
    if (a == "--dump-batches") {
      o.dump_batches = value();
      continue;
    }
    // A's options.
    o.argv.push_back(a);
    if (a == "--input") {
      o.input = value();
      o.argv.push_back(o.input);
    } else if (a == "--batches") {
      o.batches = parse_int(value(), 0, INT32_MAX, a);
      o.argv.push_back(std::to_string(o.batches));
    } else if (a == "--batch-size") {
      const long long v = parse_int(value(), 0, INT32_MAX, a);
      o.deletions = v / 2;
      o.insertions = v - v / 2;
      o.composition_set = true;
      o.argv.push_back(std::to_string(v));
    } else if (a == "--deletions" || a == "--insertions") {
      const long long v = parse_int(value(), 0, INT32_MAX, a);
      (a == "--deletions" ? o.deletions : o.insertions) = v;
      o.composition_set = true;
      o.argv.push_back(std::to_string(v));
    } else if (a == "--seed") {
      const std::string s = value();
      char* end = nullptr;
      errno = 0;
      o.seed = std::strtoull(s.c_str(), &end, 10);
      if (s.empty() || s[0] == '-' || *end != '\0' || errno == ERANGE)
        throw usage_error("--seed expects a non-negative 64-bit integer");
      o.argv.push_back(s);
    } else if (a == "--insert-mode") {
      const std::string m = value();
      if (m != "random" && m != "sample")
        throw usage_error("--insert-mode expects random or sample");
      o.sample = m == "sample";
      o.insert_mode_set = true;
      o.argv.push_back(m);
    } else if (a == "--check") {
      o.check = true;
    } else if (a == "--timing") {
      o.timing = value();
      o.argv.push_back(o.timing);
    } else if (a == "--reference") {
      throw usage_error(
          "--reference: the CPU reference is not part of the timed driver (the oracle replays "
          "are parity/compare.py triad_count)");
    } else {
      throw usage_error("unknown option " + a);
    }
  }
  const bool from_file = !o.input.empty();
  if ((!from_file && (positional.size() < 4 || positional.size() > 6)) ||
      (from_file && positional.size() > 2))
    throw usage_error(
        "expects N maxCard minCard V [capacity] [alignment] or --input F "
        "[capacity] [alignment]");
  std::size_t k = 0;
  if (!from_file) {
    o.num_hyperedges = parse_int(positional[k++], 1, INT32_MAX, "num_hyperedges");
    o.max_card =
        static_cast<std::int32_t>(parse_int(positional[k++], 1, INT32_MAX, "max_vertices"));
    o.min_vertex =
        static_cast<std::int32_t>(parse_int(positional[k++], 1, INT32_MAX - 1, "min_vertex_id"));
    o.max_vertex =
        static_cast<std::int32_t>(parse_int(positional[k++], 1, INT32_MAX - 1, "max_vertex_id"));
    if (o.min_vertex >= o.max_vertex)
      throw usage_error("min_vertex_id must be less than max_vertex_id");
  }
  if (k < positional.size())
    o.payload_capacity = parse_int(positional[k++], 0, INT32_MAX, "payload_capacity");
  if (k < positional.size() && parse_int(positional[k], 1, INT32_MAX, "alignment") != 4)
    throw usage_error(
        "alignment: only A's default 4 is supported (capacity_policy::row_alignment)");
  if (!o.insert_mode_set) o.sample = from_file;
  if (o.backend != "sequential" && o.backend != "openmp" && o.backend != "cuda")
    throw usage_error("--backend expects sequential, openmp or cuda");
  if (!o.timing.empty()) {  // checked at start-up, as A does (a bad path fails before the run)
    std::ofstream probe(o.timing, std::ios::app);
    if (!probe) throw usage_error("cannot write the --timing file " + o.timing);
  }
  return o;
}

/// The driver's timers and the profiler's samples, per batch (0: before the batches).
class timing_log {
 public:
  explicit timing_log(const profiler& prof) : prof_(prof) {}
  void timer(const std::string& name, std::int64_t batch, double ms) {
    records_.push_back({name, batch, ms});
  }
  /// The profiler samples completed since the last call belong to `batch`.
  void stages(std::int64_t batch) {
    const auto& s = prof_.samples();
    for (; seen_ < s.size(); ++seen_) records_.push_back({s[seen_].name, batch, s[seen_].host_ms});
  }
  [[nodiscard]] double sum(const std::string& name, std::int64_t batch) const {
    double t = 0;
    for (const auto& r : records_)
      if (r.name == name && r.batch == batch) t += r.ms;
    return t;
  }
  void write(const std::string& path, const run_options& o) const {
    std::ofstream out(path);
    out << std::fixed << std::setprecision(3);
    out << "{\n  \"meta\": {\"driver\": \"dyng-compat-escher run\", \"command\": \"";
    for (std::size_t i = 0; i < o.argv.size(); ++i) out << (i ? " " : "") << o.argv[i];
    out << "\", \"seed\": \"" << o.seed << "\", \"batches\": \"" << o.batches
        << "\", \"pattern\": \"" << (o.count ? o.pattern : "none") << "\", \"backend\": \""
        << o.backend << "\", \"sync_stages\": \"" << (o.sync_stages ? 1 : 0) << "\"},\n";
    out << "  \"stages\": [\n";
    for (std::size_t i = 0; i < records_.size(); ++i) {
      out << "    {\"stage\": \"" << records_[i].name << "\", \"batch\": " << records_[i].batch
          << ", \"ms\": " << records_[i].ms << "}" << (i + 1 < records_.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    if (!out) throw std::runtime_error("cannot write the --timing file " + path);
  }

 private:
  struct record {
    std::string name;
    std::int64_t batch;
    double ms;
  };
  const profiler& prof_;
  std::size_t seen_ = 0;
  std::vector<record> records_;
};

double ms_since(clock_type::time_point t0) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

triad_count::pattern kind_of(const std::string& pattern) {
  if (pattern == "type1") return triad_count::pattern::stathyper_type1;
  if (pattern == "type2") return triad_count::pattern::stathyper_type2;
  if (pattern == "type3") return triad_count::pattern::stathyper_type3;
  return triad_count::pattern::h_motif;
}

/// A's count lines (src/main.cu and src/type1.cu .. type3.cu, print()).
template <typename value_t>
void print_values(std::ostream& out, triad_count::pattern kind, const value_t* values,
                  bool is_delta) {
  switch (kind) {
    case triad_count::pattern::h_motif:
      out << (is_delta ? "Motif delta counts (30 bins): " : "Motif counts (30 bins): ");
      for (int i = 0; i < 30; ++i) out << values[i] << (i + 1 < 30 ? ' ' : '\n');
      return;
    case triad_count::pattern::stathyper_type1:
      if (is_delta)
        out << "[GPU] Type1 motif delta count: " << values[0] << std::endl;
      else
        out << "[GPU] Type1 baseline motif count (sum_e C(|e|, 3)): " << values[0] << std::endl;
      return;
    case triad_count::pattern::stathyper_type2:
    case triad_count::pattern::stathyper_type3: {
      const char* name = kind == triad_count::pattern::stathyper_type2 ? "Type2" : "Type3";
      if (is_delta)
        out << "[GPU] " << name << " motif delta count: " << values[0] << std::endl;
      else
        out << "[GPU] " << name << " baseline motif count: " << values[0] << std::endl;
      return;
    }
  }
}

int run(const run_options& o, clock_type::time_point start) {
  resources res = o.backend == "cuda"     ? resources::cuda(0)
                  : o.backend == "openmp" ? resources::openmp(o.threads)
                                          : resources::sequential();
  profiler_options popt;
  popt.sync_stages = o.sync_stages;
  popt.nvtx = o.nvtx;
  profiler prof(popt);
  res.attach_profiler(&prof);
  timing_log log(prof);
  res.warm_up();

  // 2. Load (A's init.load: no synchronization).
  std::mt19937_64 rng(o.seed);
  hyperedge_list<std::int32_t> list;
  legacy::escher_hypergraph_options graph;
  {
    const auto t0 = clock_type::now();
    if (!o.input.empty()) {
      scoped_stage stage(res, "io.read_hypergraph");
      list = io::read_hypergraph<std::int32_t, unweighted>(
          o.input, io::hypergraph_read_options::escher_gpu());
    } else {
      // A's printHypergraphParams (utils/utils.cpp), inside its load stage.
      std::cout << "Generating hypergraph with:" << std::endl;
      std::cout << "  - " << o.num_hyperedges << " hyperedges" << std::endl;
      std::cout << "  - Up to " << o.max_card << " vertices per hyperedge" << std::endl;
      std::cout << "  - Vertex IDs in range [" << o.min_vertex << ", " << o.max_vertex << "]"
                << std::endl;
      std::cout << std::endl;
      graph.num_hyperedges = o.num_hyperedges;
      graph.max_cardinality = o.max_card;
      graph.min_vertex = o.min_vertex - 1;
      graph.max_vertex = o.max_vertex - 1;
      list = legacy::escher_hypergraph<std::int32_t, unweighted>(rng, graph);
    }
    log.timer("load_ms", 0, ms_since(t0));
  }
  std::int64_t pins = static_cast<std::int64_t>(list.members.size());
  if (!o.input.empty()) {
    // A's random insertions with --input: its file's largest cardinality, vertices 1..max id.
    std::int64_t max_card = 1;
    for (std::int64_t i = 0; i < list.num_hyperedges(); ++i)
      max_card = std::max(max_card, list.offsets[static_cast<std::size_t>(i) + 1] -
                                        list.offsets[static_cast<std::size_t>(i)]);
    graph.max_cardinality = static_cast<std::int32_t>(max_card);
    graph.min_vertex = 0;
    graph.max_vertex = list.num_vertices - 1;
  }
  std::cout << "Hypergraph: " << list.num_hyperedges() << " hyperedges, max vertex id "
            << (o.input.empty() ? static_cast<std::int64_t>(o.max_vertex)
                                : static_cast<std::int64_t>(list.num_vertices))
            << ", " << pins << " pins" << std::endl;

  // 3. The hypergraph (line graph and row views built here) and the host model.
  hypergraph_properties props;
  props.reuse = id_reuse::erase_first;
  props.capacity = capacity_policy::escher_gpu(o.payload_capacity);
  props.line_graph = line_graph_policy::sorted_merge;
  hgraph h = hgraph::from_hyperedges(res, list.view(), props);
  legacy::escher_host_model<std::int32_t> model(list.view());
  list = {};  // A moves its loaded rows into its host model
  res.synchronize();
  log.stages(0);
  // 4. The full count (A's init.full_count: synchronized at both ends).
  const triad_count::pattern kind = kind_of(o.pattern);
  std::unique_ptr<triad_count::result> counts;
  if (o.count) {
    triad_count::options topt;
    topt.kind = kind;
    res.synchronize();
    const auto t0 = clock_type::now();
    counts = std::make_unique<triad_count::result>(triad_count::compute(res, h, topt));
    res.synchronize();
    log.timer("full_count_ms", 0, ms_since(t0));
    log.stages(0);
    print_values(std::cout, kind, counts->counts().data(), false);
  }
  const std::int64_t initial_bytes = h.store_stats().device_bytes;
  int failures = 0;
  if (o.check) {
    h.check_integrity(res);
    std::cerr << "[check] batch 0: data-structure integrity OK" << std::endl;
  }

  // 5. The batches.
  legacy::escher_batch_options bopt;
  if (o.composition_set) {
    bopt.num_deletions = o.deletions;
    bopt.num_insertions = o.insertions;
  }
  bopt.insert_mode =
      o.sample ? legacy::escher_insert_mode::sample : legacy::escher_insert_mode::random;
  bopt.graph = graph;
  legacy::escher_batch_generator generator(rng, bopt);
  // --dump-batches (the proof that the host model draws A's batches; never in a timed run): every
  // batch with the member sets of its deletions and the ids its insertions got, as
  // `dyng-compat-escher to-dgb` writes A's dumps.
  std::unique_ptr<io::batch_writer> dump;
  if (!o.dump_batches.empty()) {
    dump = std::make_unique<io::batch_writer>(
        o.dump_batches,
        io::hyperedge_batch_file_header<std::int32_t, unweighted>(
            io::batch_file_flags::deleted_members | io::batch_file_flags::expected_ids));
  }
  for (std::int64_t b = 1; b <= o.batches; ++b) {
    hyperedge_batch<std::int32_t, unweighted> batch;
    {
      const auto t0 = clock_type::now();
      scoped_stage stage(res, "driver.batch.generate");
      batch = generator.next(model);
      stage.stop();
      log.timer("generate_ms", b, ms_since(t0));
    }
    std::vector<std::int64_t> deleted_offsets{0};
    std::vector<std::int32_t> deleted_members;
    if (dump) {
      for (const std::int64_t id : batch.delete_hyperedges()) {
        const auto& r = model.row(id);
        deleted_members.insert(deleted_members.end(), r.begin(), r.end());
        deleted_offsets.push_back(static_cast<std::int64_t>(deleted_members.size()));
      }
    }
    res.synchronize();
    const auto t0 = clock_type::now();
    hyper_apply_summary s;
    triad_count::stats st;
    if (counts) {
      st = triad_count::update(res, h, batch.view(), *counts);
      s = st.batch;
    } else {
      s = h.apply(res, batch.view());
    }
    res.synchronize();
    log.timer(counts ? "update_ms" : "apply_ms", b, ms_since(t0));
    {
      // The host model's maintenance (the rows A's DynamicHypergraph::update keeps in h2v_
      // inside its timed update.maintain): timed apart, outside update_ms, and reported next to
      // the gated update (parity/timed_regions/triad_count.toml, scope rules).
      const auto t1 = clock_type::now();
      model.apply(batch.view(), s);
      log.timer("host_model_ms", b, ms_since(t1));
    }
    if (dump) {
      io::hyperedge_batch_annotations<std::int32_t> a;
      a.deleted_offsets = host_view(deleted_offsets);
      a.deleted_members = host_view(deleted_members);
      a.expected_ids = host_view(s.inserted_ids);
      dump->write(batch.view(), a);
    }
    log.stages(b);
    if (o.batches > 1)
      std::cout << "Batch " << b << " (" << batch.num_deletions() << " deletions, "
                << batch.num_insertions() << " insertions)" << std::endl;
    if (counts) print_values(std::cout, kind, st.delta.data(), true);
    std::cerr << "[memory] batch " << b << ": store device bytes " << h.store_stats().device_bytes
              << std::endl;
    if (o.check) {
      try {
        h.check_integrity(res);
        std::cerr << "[check] batch " << b << ": data-structure integrity OK" << std::endl;
      } catch (const std::exception& e) {
        ++failures;
        std::cerr << "[check] batch " << b << ": data-structure integrity FAILED: " << e.what()
                  << std::endl;
      }
      if (counts) {
        triad_count::options topt;
        topt.kind = kind;
        const triad_count::result fresh = triad_count::compute(res, h, topt);
        const auto a = counts->counts();
        const auto c = fresh.counts();
        bool same = a.size() == c.size();
        for (std::size_t i = 0; same && i < a.size(); ++i) same = a[i] == c[i];
        failures += same ? 0 : 1;
        std::cerr << "[check] batch " << b
                  << ": baseline + deltas == full recount: " << (same ? "OK" : "FAILED")
                  << std::endl;
      }
    }
  }
  if (dump) dump->close();
  res.synchronize();
  const double total = ms_since(start);
  log.timer("total_ms", 0, total);
  log.timer("driver.total", 0, total);
  std::cerr << std::fixed << std::setprecision(1) << "[timing] init: load " << log.sum("load_ms", 0)
            << " ms, build " << log.sum("hypergraph.build", 0) << " ms, upload "
            << log.sum("hypergraph.upload", 0) << " ms" << std::endl;
  if (counts)
    std::cerr << "[timing] full count " << log.sum("full_count_ms", 0) << " ms" << std::endl;
  for (std::int64_t b = 1; b <= o.batches; ++b)
    std::cerr << "[timing] batch " << b << ": " << (counts ? "update " : "apply ")
              << log.sum(counts ? "update_ms" : "apply_ms", b) << " ms" << std::endl;
  std::cerr << "[timing] total " << total << " ms" << std::endl;
  std::cerr << "[memory] dynG store device bytes: " << initial_bytes << " after init, "
            << h.store_stats().device_bytes << " after the last batch" << std::endl;
  if (!o.timing.empty()) log.write(o.timing, o);
  if (failures) {
    std::cerr << "[check] " << failures << " check(s) FAILED" << std::endl;
    return 1;
  }
  return 0;
}

}  // namespace

int run_main(int argc, char** argv) {
  const auto start = clock_type::now();  // A's total starts at the top of run()
  run_options o;
  try {
    o = parse(argc, argv);
  } catch (const usage_error& e) {
    std::fprintf(stderr, "dyng-compat-escher run: %s\n", e.what());
    return 2;
  }
  try {
    return run(o, start);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dyng-compat-escher run: %s\n", e.what());
    return 2;
  }
}

}  // namespace dyng::compat::escher
