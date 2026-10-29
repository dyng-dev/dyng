// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file lp_replay_main.cpp
 * @brief dyng_lp_replay: replays of the legacy generator and of the label_propagation oracles
 *        against LabelPropagation-CUDA@a276a3a's recorded traces (M10a step 2; ADRs 0067, 0068).
 *        parity/label_propagation_goldens.py (compare.py label_propagation generator|model|oracle)
 *        drives it over the golden corpus.
 *
 *   dyng_lp_replay generator --nodes N --batches B --degree D --seed S [--threads T]
 *                            [--golden <generator.dlt>] [--write-graph <prefix>]
 *       Generates the stream (generators' internal dynlp_legacy_generate) and compares every array
 *       with the golden's generator / batch_data records (raw: element by element; digest mode:
 *       SHA-256); --write-graph writes O's four CSR text files with the library's writers.
 *   dyng_lp_replay model --trace <trace.dlt> [--threads T]
 *       Replays O's own batches (run_begin / batch_data; in the digest mode the generator's, after
 *       checking their digests) through lp_host_model and compares every predicted field with
 *       the trace's records; also checks the generator's tau against the trace's.
 *   dyng_lp_replay container --trace <trace.dlt> [--threads T] [--backend sequential|cuda]
 *       M10a step 3 (ADR 0069): O's own batches (as for model) applied to
 *       graph::with_capacity() with slotted rows and batch_semantics::net_effect() (the library's
 *       host apply), and after every batch the graph's row ends, slot columns and value bits and
 *       its states (mapped to O's codes by name with the seed labels of the batches) against the
 *       trace's records, check_integrity(), and the same fields against lp_host_model.
 *   dyng_lp_replay oracle --nodes N --batches B --degree D --seed S [--threads T]
 *                         (--labels-trace <dlt> | --labels-bits <file> | --final-labels <txt>)
 *       Per batch: the harmonic solution of the graph after the batch (the generator's batches
 *       through lp_host_model), and the comparison of the given labels with it.
 * Each prints one JSON object; the exit status is 0 when every comparison is equal (generator,
 * model) or the run completed (oracle), 1 on a mismatch, 2 on a usage or input error.
 */
#include "io/dynlp_text.hpp"
#include "parity/label_propagation/dlt_reader.hpp"
#include "parity/label_propagation/lp_replay.hpp"
#include "support/sha256.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/graph/graph_view.hpp>
#include <dyng/testing/harmonic_solution.hpp>
#include <dyng/testing/lp_host_model.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace lp = dyng::test::lp;
namespace dlt = dyng::test::dlt;

struct args {
  std::map<std::string, std::string> kv;
  [[nodiscard]] bool has(const std::string& k) const {
    return kv.count(k) != 0;
  }
  [[nodiscard]] std::string get(const std::string& k) const {
    const auto it = kv.find(k);
    if (it == kv.end()) {
      throw std::invalid_argument("missing --" + k);
    }
    return it->second;
  }
  [[nodiscard]] std::int64_t num(const std::string& k, std::int64_t fallback) const {
    return has(k) ? std::stoll(get(k)) : fallback;
  }
};

std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  return out + "\"";
}

std::string json_double(double x) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", x);
  return buf;
}

/// Compare an expected array with a trace field: raw element bytes, or the digest.
struct field_result {
  bool equal = false;
  std::string detail;
};

field_result compare(const lp::named_array& ours, const dlt::field& theirs) {
  field_result r;
  if (ours.count != theirs.count) {
    r.detail = "count " + std::to_string(ours.count) + " vs " + std::to_string(theirs.count);
    return r;
  }
  if (theirs.digest) {
    r.equal = lp::digest(ours) == theirs.sha256;
    if (!r.equal) {
      r.detail = "digest differs";
    }
    return r;
  }
  if (ours.size() != theirs.raw.size()) {
    r.detail = "element size differs";
    return r;
  }
  r.equal = ours.size() == 0 || std::memcmp(ours.data(), theirs.raw.data(), ours.size()) == 0;
  if (!r.equal) {
    const std::size_t elem = ours.size() / std::max<std::size_t>(ours.count, 1);
    for (std::size_t i = 0; i < ours.size(); ++i) {
      if (ours.data()[i] != theirs.raw[i]) {
        r.detail =
            "first difference at element " + std::to_string(i / std::max<std::size_t>(elem, 1));
        break;
      }
    }
  }
  return r;
}

dyng::generators::vertex_arrival_options options_of(const args& a) {
  dyng::generators::vertex_arrival_options o;
  o.total_vertices = a.num("nodes", 0);
  o.num_batches = static_cast<std::int32_t>(a.num("batches", 10));
  o.average_degree = static_cast<std::int32_t>(a.num("degree", 5));
  o.seed = static_cast<std::uint32_t>(a.num("seed", 42));
  return o;
}

dyng::generators::vertex_arrival_options options_of(const dlt::record& run) {
  dyng::generators::vertex_arrival_options o;
  o.total_vertices = run["total_nodes"].integer();
  o.num_batches = static_cast<std::int32_t>(run["num_batches"].integer());
  o.average_degree = static_cast<std::int32_t>(run["avg_degree"].integer());
  o.seed = static_cast<std::uint32_t>(run["seed"].integer());
  std::uint32_t gt = run["ground_truth_fraction"].f32_bits();
  std::uint32_t del = run["deleted_fraction"].f32_bits();
  std::memcpy(&o.ground_truth_fraction, &gt, 4);
  std::memcpy(&o.deleted_fraction, &del, 4);
  return o;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/// generator: the arrays against the golden's generator / batch_data records.
int run_generator(const args& a) {
  const auto opt = options_of(a);
  const int threads = static_cast<int>(a.num("threads", 1));
  const auto t0 = std::chrono::steady_clock::now();
  const dyng::detail::dynlp_legacy_output out = dyng::detail::dynlp_legacy_generate(opt, threads);
  const double gen_seconds = seconds_since(t0);
  const auto arrays = lp::generator_arrays(out);
  std::ostringstream js;
  js << "{\"mode\":\"generator\",\"threads\":" << threads
     << ",\"generate_seconds\":" << json_double(gen_seconds)
     << ",\"undirected_edges\":" << out.undirected_edges;
  std::uint32_t tau_bits = 0;
  std::memcpy(&tau_bits, &out.tau, 4);
  js << ",\"tau_bits\":" << tau_bits << ",\"digests\":{";
  bool first = true;
  for (const auto& [name, arr] : arrays) {
    js << (first ? "" : ",") << json_string(name) << ":" << json_string(lp::digest(arr));
    first = false;
  }
  js << "}";
  bool all_equal = true;
  if (a.has("golden")) {
    const dlt::trace t = dlt::read(a.get("golden"));
    const dlt::record& g = t.one("generator");
    std::vector<std::pair<std::string, const dlt::field*>> expected = {
        {"row_ptr", &g["row_ptr"]},
        {"col_ind", &g["col_ind"]},
        {"values", &g["values"]},
        {"true_labels", &g["true_labels"]}};
    const auto batch_records = t.all("batch_data");
    if (static_cast<std::int64_t>(batch_records.size()) != opt.num_batches) {
      throw std::runtime_error("the golden has " + std::to_string(batch_records.size()) +
                               " batches");
    }
    for (const dlt::record* r : batch_records) {
      const std::string p = "b" + std::to_string(r->batch) + ".";
      for (const char* f :
           {"inserted", "gt_ids", "gt_labels", "deleted", "edge_src", "edge_dst", "edge_weight"}) {
        expected.emplace_back(p + f, &(*r)[f]);
      }
    }
    js << ",\"compared\":" << expected.size() << ",\"mismatches\":[";
    bool first_mismatch = true;
    for (const auto& [name, field] : expected) {
      const auto it = arrays.find(name);
      field_result res;
      if (it == arrays.end()) {
        res.detail = "not generated";
      } else {
        res = compare(it->second, *field);
      }
      if (!res.equal) {
        all_equal = false;
        js << (first_mismatch ? "" : ",") << "{\"field\":" << json_string(name)
           << ",\"detail\":" << json_string(res.detail) << "}";
        first_mismatch = false;
      }
    }
    js << "]";
  }
  if (a.has("write-graph")) {
    std::vector<std::int8_t> labels(out.true_labels.size());
    for (std::size_t i = 0; i < labels.size(); ++i) {
      labels[i] = out.true_labels[i] >= 0.5F ? 1 : 0;
    }
    dyng::detail::write_dynlp_graph_files<std::int32_t, std::int32_t>(
        a.get("write-graph"), out.row_ptr, out.col_ind, out.values, labels);
  }
  js << ",\"equal\":" << (all_equal ? "true" : "false") << "}";
  std::cout << js.str() << "\n";
  return all_equal ? 0 : 1;
}

/// O's inputs of a trace: raw run_begin / batch_data, or (digest mode) the generator's after
/// checking that their digests equal the trace's.
dyng::detail::dynlp_legacy_output inputs_of(const dlt::trace& t, int threads, bool& generated,
                                            std::string& problem) {
  const dlt::record& run = t.one("run_begin");
  dyng::detail::dynlp_legacy_output in;
  generated = run["row_ptr"].digest;
  if (!generated) {
    in.num_nodes = static_cast<std::int32_t>(run["num_nodes"].integer());
    in.row_ptr = run["row_ptr"].as<std::int32_t>();
    in.col_ind = run["col_ind"].as<std::int32_t>();
    in.values = run["values"].as<float>();
    in.true_labels = run["true_labels"].as<float>();
    for (const dlt::record* r : t.all("batch_data")) {
      dyng::detail::dynlp_legacy_batch b;
      b.inserted = (*r)["inserted"].as<std::int32_t>();
      b.gt_ids = (*r)["gt_ids"].as<std::int32_t>();
      b.gt_labels = (*r)["gt_labels"].as<float>();
      b.deleted = (*r)["deleted"].as<std::int32_t>();
      b.edge_src = (*r)["edge_src"].as<std::int32_t>();
      b.edge_dst = (*r)["edge_dst"].as<std::int32_t>();
      b.edge_weight = (*r)["edge_weight"].as<float>();
      in.batches.push_back(std::move(b));
    }
    return in;
  }
  in = dyng::detail::dynlp_legacy_generate(options_of(run), threads);
  const auto arrays = lp::generator_arrays(in);
  std::vector<std::pair<std::string, const dlt::field*>> expected = {
      {"row_ptr", &run["row_ptr"]},
      {"col_ind", &run["col_ind"]},
      {"values", &run["values"]},
      {"true_labels", &run["true_labels"]}};
  for (const dlt::record* r : t.all("batch_data")) {
    const std::string p = "b" + std::to_string(r->batch) + ".";
    for (const char* f :
         {"inserted", "gt_ids", "gt_labels", "deleted", "edge_src", "edge_dst", "edge_weight"}) {
      expected.emplace_back(p + f, &(*r)[f]);
    }
  }
  for (const auto& [name, field] : expected) {
    const auto it = arrays.find(name);
    if (it == arrays.end() || !compare(it->second, *field).equal) {
      problem += (problem.empty() ? "" : "; ") + std::string("generated input differs: ") + name;
    }
  }
  return in;
}

/// model: lp_host_model against every predicted field of the trace.
int run_model(const args& a) {
  const int threads = static_cast<int>(a.num("threads", 1));
  const dlt::trace t = dlt::read(a.get("trace"));
  bool generated = false;
  std::string problem;
  const auto t0 = std::chrono::steady_clock::now();
  const dyng::detail::dynlp_legacy_output in = inputs_of(t, threads, generated, problem);
  const float tau = dyng::testing::lp_host_model::average_weight(in.col_ind, in.values);
  // The generator's tau (computeAvgEdgeWeight of its own graph) against the trace's.
  const dyng::detail::dynlp_legacy_output gen =
      generated ? dyng::detail::dynlp_legacy_output{}
                : dyng::detail::dynlp_legacy_generate(options_of(t.one("run_begin")), threads);
  const float generator_tau = generated ? in.tau : gen.tau;
  dyng::testing::lp_host_model model(lp::row_capacities(in.row_ptr));
  std::ostringstream js;
  js << "{\"mode\":\"model\",\"digest_mode\":" << (t.digest_mode ? "true" : "false")
     << ",\"inputs\":" << json_string(generated ? "generated" : "recorded");
  std::int64_t compared = 0;
  std::int64_t mismatched = 0;
  std::ostringstream mm;
  std::ostringstream per_batch;
  bool tau_equal = true;
  for (std::size_t b = 0; b < in.batches.size(); ++b) {
    const auto bi = static_cast<std::int32_t>(b);
    const auto report = model.apply(lp::model_batch(in.batches[b], in.num_nodes), tau);
    const auto fields = lp::model_fields(model, report, tau, bi);
    std::int64_t batch_compared = 0;
    for (const auto& [key, ours] : fields) {
      const std::size_t dot = key.find('.');
      const std::string tag = key.substr(0, dot);
      const std::string name = key.substr(dot + 1);
      const dlt::record* rec = t.in_batch(tag, bi);
      field_result res;
      if (rec == nullptr || !rec->has(name)) {
        res.detail = "not in the trace";
      } else {
        res = compare(ours, (*rec)[name]);
      }
      ++compared;
      ++batch_compared;
      if (!res.equal) {
        ++mismatched;
        mm << (mismatched == 1 ? "" : ",") << "{\"batch\":" << b
           << ",\"field\":" << json_string(key) << ",\"detail\":" << json_string(res.detail) << "}";
      }
    }
    // A stage the model says does not run must be absent from the trace too.
    if (report.inserted.empty()) {
      for (const char* tag : {"sparsify", "connected_components", "label_init"}) {
        if (t.in_batch(tag, bi) != nullptr) {
          ++mismatched;
          mm << (mismatched == 1 ? "" : ",") << "{\"batch\":" << b
             << ",\"field\":" << json_string(tag)
             << ",\"detail\":\"recorded but Ins_eff is empty\"}";
        }
      }
    }
    if (const dlt::record* sp = t.in_batch("sparsify", bi)) {
      std::uint32_t gb = 0;
      std::memcpy(&gb, &generator_tau, 4);
      tau_equal = tau_equal && (*sp)["tau"].f32_bits() == gb;
    }
    per_batch << (b == 0 ? "" : ",") << "{\"batch\":" << b << ",\"fields\":" << batch_compared
              << ",\"inserted\":" << report.inserted.size()
              << ",\"deleted\":" << report.deleted.size()
              << ",\"affected\":" << report.affected.size()
              << ",\"components\":" << report.num_components << ",\"appended\":" << report.appended
              << "}";
  }
  if (t.all("batch_data").size() != in.batches.size()) {
    problem += (problem.empty() ? "" : "; ") + std::string("batch count differs");
  }
  if (!tau_equal) {
    problem += (problem.empty() ? "" : "; ") + std::string("the generator's tau differs");
  }
  const bool ok = mismatched == 0 && problem.empty();
  js << ",\"seconds\":" << json_double(seconds_since(t0)) << ",\"compared\":" << compared
     << ",\"mismatched\":" << mismatched
     << ",\"generator_tau_equal\":" << (tau_equal ? "true" : "false")
     << ",\"problem\":" << json_string(problem) << ",\"batches\":[" << per_batch.str()
     << "],\"mismatches\":[" << mm.str() << "],\"equal\":" << (ok ? "true" : "false") << "}";
  std::cout << js.str() << "\n";
  return ok ? 0 : 1;
}

/// An owned named array of a vector (the original's layout of a field).
template <typename value_t>
lp::named_array owned(const std::vector<value_t>& v) {
  lp::named_array a;
  a.count = v.size();
  a.bytes.resize(v.size() * sizeof(value_t));
  if (!v.empty()) {
    std::memcpy(a.bytes.data(), v.data(), a.bytes.size());
  }
  return a;
}

/// container: O's batches on graph::with_capacity() against the trace and lp_host_model.
int run_container(const args& a) {
  const int threads = static_cast<int>(a.num("threads", 1));
  const dlt::trace t = dlt::read(a.get("trace"));
  bool generated = false;
  std::string problem;
  const auto t0 = std::chrono::steady_clock::now();
  const dyng::detail::dynlp_legacy_output in = inputs_of(t, threads, generated, problem);
  const auto n = static_cast<std::size_t>(in.num_nodes);
  std::vector<std::int32_t> capacity(n);
  for (std::size_t v = 0; v < n; ++v) {
    capacity[v] = in.row_ptr[v + 1] - in.row_ptr[v];
  }
  dyng::graph_properties props;
  props.directed = false;
  props.layout = dyng::row_layout::slotted;
  props.order = dyng::row_order::append;
  props.parallel_edges = dyng::multi_edges::allow;
  props.semantics = dyng::batch_semantics::net_effect();
  // --backend cuda: the device apply of ADR 0074 (the rows are downloaded for the comparison).
  const std::string backend = a.kv.count("backend") != 0 ? a.get("backend") : "sequential";
  if (backend != "sequential" && backend != "cuda") {
    throw std::runtime_error("--backend: sequential or cuda");
  }
  const dyng::resources res =
      backend == "cuda" ? dyng::resources::cuda(0) : dyng::resources::sequential();
  const std::vector<std::int32_t>& cap = capacity;
  auto g = dyng::graph<std::int32_t, std::int32_t, float>::with_capacity(
      res, static_cast<std::int32_t>(n), dyng::host_view(cap), props);
  dyng::testing::lp_host_model model(lp::row_capacities(in.row_ptr));
  std::vector<std::int8_t> seeds(n, -1);
  std::int64_t compared = 0;
  std::int64_t mismatched = 0;
  std::ostringstream mm;
  const auto mismatch = [&](std::size_t b, const std::string& field, const std::string& detail) {
    ++mismatched;
    mm << (mismatched == 1 ? "" : ",") << "{\"batch\":" << b << ",\"field\":" << json_string(field)
       << ",\"detail\":" << json_string(detail) << "}";
  };
  for (std::size_t b = 0; b < in.batches.size(); ++b) {
    const auto bi = static_cast<std::int32_t>(b);
    const dyng::testing::lp_model_batch mb = lp::model_batch(in.batches[b], in.num_nodes);
    dyng::edge_batch_view<std::int32_t, float> view;
    view.insert_vertices = dyng::host_view(mb.insert_vertices);
    view.insert_vertex_labels = dyng::host_view(mb.insert_labels);
    view.delete_vertices = dyng::host_view(mb.delete_vertices);
    view.insert_src = dyng::host_view(mb.edge_sources);
    view.insert_dst = dyng::host_view(mb.edge_destinations);
    view.insert_weights = dyng::host_view(mb.edge_weights);
    view.num_weights = 1;
    try {
      (void)g.apply(res, view);
    } catch (const std::exception& e) {
      mismatch(b, "apply", e.what());
      break;
    }
    for (std::size_t i = 0; i < mb.insert_vertices.size(); ++i) {
      if (mb.insert_labels[i] >= 0) {
        seeds[static_cast<std::size_t>(mb.insert_vertices[i])] = mb.insert_labels[i];
      }
    }
    (void)model.apply(mb, in.tau);
    try {
      g.check_integrity(res);
    } catch (const std::exception& e) {
      mismatch(b, "check_integrity", e.what());
    }
    const auto gv = g.view();
    std::vector<std::int32_t> status(n);
    for (std::size_t v = 0; v < n; ++v) {
      switch (gv.status[v]) {
        case dyng::vertex_status::active:
          status[v] = seeds[v] >= 0 ? lp::original_ground_truth : lp::original_active;
          break;
        case dyng::vertex_status::deleted:
          status[v] = lp::original_deleted;
          break;
        case dyng::vertex_status::inactive:
          status[v] = lp::original_inactive;
          break;
      }
    }
    std::vector<std::int32_t> row_end(gv.row_end.begin(), gv.row_end.end());
    std::vector<std::int32_t> cols(gv.out.col_ind.begin(), gv.out.col_ind.end());
    std::vector<float> values(gv.out.weights.begin(), gv.out.weights.end());
    std::map<std::string, lp::named_array> fields;
    fields["apply_vertex_changes.status"] = owned(status);
    fields["append_edges.row_end"] = owned(row_end);
    fields["append_edges.slot_col_ind"] = owned(cols);
    fields["append_edges.slot_values"] = owned(values);
    if (b == 0) {
      std::vector<std::int32_t> row_ptr(gv.out.row_ptr.begin(), gv.out.row_ptr.end());
      fields["append_edges.row_ptr"] = owned(row_ptr);
    }
    for (const auto& [key, ours] : fields) {
      const std::size_t dot = key.find('.');
      const dlt::record* rec = t.in_batch(key.substr(0, dot), bi);
      ++compared;
      if (rec == nullptr || !rec->has(key.substr(dot + 1))) {
        mismatch(b, key, "not in the trace");
        continue;
      }
      const field_result res_field = compare(ours, (*rec)[key.substr(dot + 1)]);
      if (!res_field.equal) {
        mismatch(b, key, res_field.detail);
      }
    }
    // check_integrity agrees with lp_host_model: the same rows, slots and states.
    ++compared;
    bool same = std::equal(model.row_end().begin(), model.row_end().end(), row_end.begin(),
                           row_end.end(), [](std::int64_t x, std::int32_t y) { return x == y; }) &&
                model.slot_columns() == cols &&
                std::memcmp(model.slot_weights().data(), values.data(),
                            values.size() * sizeof(float)) == 0 &&
                lp::original_statuses(model) == status;
    if (!same) {
      mismatch(b, "lp_host_model", "the graph differs from the host model");
    }
  }
  const bool ok = mismatched == 0 && problem.empty();
  std::cout << "{\"mode\":\"container\",\"digest_mode\":" << (t.digest_mode ? "true" : "false")
            << ",\"inputs\":" << json_string(generated ? "generated" : "recorded")
            << ",\"seconds\":" << json_double(seconds_since(t0)) << ",\"compared\":" << compared
            << ",\"mismatched\":" << mismatched << ",\"problem\":" << json_string(problem)
            << ",\"mismatches\":[" << mm.str() << "],\"equal\":" << (ok ? "true" : "false")
            << "}\n";
  return ok ? 0 : 1;
}

/// The labels of every batch from one of the sources (missing batches: empty).
std::vector<std::vector<float>> labels_of(const args& a, std::size_t n, std::size_t batches) {
  std::vector<std::vector<float>> out(batches);
  if (a.has("labels-trace")) {
    const dlt::trace t = dlt::read(a.get("labels-trace"));
    for (const dlt::record* r : t.all("propagation_end")) {
      if (r->batch >= 0 && static_cast<std::size_t>(r->batch) < batches) {
        out[static_cast<std::size_t>(r->batch)] = (*r)["labels"].as<float>();
      }
    }
  } else if (a.has("labels-bits")) {
    std::ifstream f(a.get("labels-bits"), std::ios::binary);
    for (std::size_t b = 0; b < batches; ++b) {
      out[b].resize(n);
      f.read(reinterpret_cast<char*>(out[b].data()), static_cast<std::streamsize>(n * 4));
      if (!f) {
        throw std::runtime_error("labels-bits: short file");
      }
    }
  } else if (a.has("final-labels")) {
    std::ifstream f(a.get("final-labels"));
    std::vector<float> last(n, 0.0F);
    std::string line;
    std::size_t lines = 0;
    while (std::getline(f, line)) {
      std::istringstream s(line);
      std::size_t i = 0;
      double x = 0.0;
      if (!(s >> i >> x) || i >= n) {
        throw std::runtime_error("final-labels: bad line " + std::to_string(lines + 1));
      }
      last[i] = static_cast<float>(x);
      ++lines;
    }
    if (lines != n) {
      throw std::runtime_error("final-labels: " + std::to_string(lines) + " lines");
    }
    out.back() = std::move(last);
  } else {
    throw std::invalid_argument("oracle needs --labels-trace, --labels-bits or --final-labels");
  }
  return out;
}

/// oracle: the harmonic solution per batch and the labels' error against it.
int run_oracle(const args& a) {
  const auto opt = options_of(a);
  const int threads = static_cast<int>(a.num("threads", 1));
  const dyng::detail::dynlp_legacy_output in = dyng::detail::dynlp_legacy_generate(opt, threads);
  const auto n = static_cast<std::size_t>(in.num_nodes);
  const auto labels = labels_of(a, n, in.batches.size());
  dyng::testing::lp_host_model model(lp::row_capacities(in.row_ptr));
  dyng::testing::harmonic_options hopt;
  hopt.num_threads = threads;
  std::ostringstream js;
  js << "{\"mode\":\"oracle\",\"threads\":" << threads << ",\"batches\":[";
  for (std::size_t b = 0; b < in.batches.size(); ++b) {
    (void)model.apply(lp::model_batch(in.batches[b], in.num_nodes), in.tau);
    js << (b == 0 ? "" : ",") << "{\"batch\":" << b;
    if (labels[b].empty()) {
      js << ",\"labels\":false}";
      continue;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto f = dyng::testing::harmonic_solution(in.row_ptr, in.col_ind, in.values,
                                                    model.present(), model.seed_labels(), hopt);
    const double seconds = seconds_since(t0);
    const auto err =
        dyng::testing::label_error_against(labels[b], f, model.present(), model.seed_labels());
    const auto agree = dyng::testing::class_agreement_with(labels[b], f, model.present(),
                                                           model.seed_labels(), 0.0);
    js << ",\"labels\":true,\"pcg_iterations\":" << f.iterations
       << ",\"relative_residual\":" << json_double(f.relative_residual)
       << ",\"converged\":" << (f.converged ? "true" : "false")
       << ",\"seconds\":" << json_double(seconds) << ",\"compared\":" << err.compared
       << ",\"excluded\":" << err.excluded << ",\"mean\":" << json_double(err.mean)
       << ",\"max\":" << json_double(err.max) << ",\"flips\":" << err.flips
       << ",\"largest_flip_distance\":" << json_double(agree.largest_flip_distance) << "}";
  }
  js << "]}";
  std::cout << js.str() << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::invalid_argument(
          "usage: dyng_lp_replay generator|model|container|oracle [--key value ...]");
    }
    const std::string mode = argv[1];
    args a;
    for (int i = 2; i < argc; ++i) {
      std::string k = argv[i];
      if (k.rfind("--", 0) != 0 || i + 1 >= argc) {
        throw std::invalid_argument("bad argument " + k);
      }
      a.kv[k.substr(2)] = argv[++i];
    }
    if (mode == "generator") {
      return run_generator(a);
    }
    if (mode == "model") {
      return run_model(a);
    }
    if (mode == "container") {
      return run_container(a);
    }
    if (mode == "oracle") {
      return run_oracle(a);
    }
    throw std::invalid_argument("unknown mode " + mode);
  } catch (const std::exception& e) {
    std::cerr << "dyng_lp_replay: " << e.what() << "\n";
    return 2;
  }
}
