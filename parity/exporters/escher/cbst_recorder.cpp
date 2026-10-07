// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// The CBST trace recorder core (see cbst_recorder.hpp and ADR 0036 for the format). Host-only
// C++; it reads device memory with the CUDA runtime and writes the trace with stdio.
#include "cbst_recorder.hpp"

#include <cuda_runtime.h>

#include <cxxabi.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <typeinfo>
#include <unordered_map>

// Marks a recording twin (parity/exporters/escher/relink_twins.sh tells a twin from the original's
// executable by this string).
extern "C" __attribute__((used)) const char dyng_cbst_recorder_marker[] =
    "dyng-cbst-recorder-twin format 1";

namespace dyng_parity::cbst_rec {
namespace {

// --- Encoding --------------------------------------------------------------------------------

enum : std::uint32_t {
  tag_header = 1,
  tag_tree = 2,
  tag_op = 3,
  tag_event = 4,
  tag_batch = 5,
  tag_note = 6,
  tag_end = 7,
};
enum : std::uint8_t {
  type_i64 = 1,
  type_str = 2,
  type_array = 3,
  type_strings = 4,
};
enum : std::uint8_t {
  array_inline = 0,
  array_blob = 1,
  array_hash = 2,
  array_null = 3,
};

constexpr std::uint64_t fnv_offset = 0xcbf29ce484222325ULL;
constexpr std::uint64_t fnv_prime = 0x100000001b3ULL;

std::uint64_t fnv_bytes(std::uint64_t h, const void* data, std::size_t n) {
  const auto* p = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= fnv_prime;
  }
  return h;
}
std::uint64_t fnv_u64(std::uint64_t h, std::uint64_t v) {
  unsigned char b[8];
  for (int i = 0; i < 8; ++i) b[i] = static_cast<unsigned char>(v >> (8 * i));
  return fnv_bytes(h, b, 8);
}
std::uint64_t fnv_i32s(std::uint64_t h, const int* data, std::size_t n) {
  // Little-endian int32 bytes (the host is little-endian; checked in the recorder).
  return fnv_bytes(h, data, n * sizeof(int));
}

/// A field list (the payload of every record).
class fields {
 public:
  fields() {
    put_u32(0);
  }
  void i64(const char* name, long long v) {
    head(type_i64, name);
    put_u64(static_cast<std::uint64_t>(v));
  }
  void str(const char* name, const std::string& s) {
    head(type_str, name);
    put_u32(static_cast<std::uint32_t>(s.size()));
    buf_.insert(buf_.end(), s.begin(), s.end());
  }
  void strings(const char* name, const std::vector<std::string>& v) {
    head(type_strings, name);
    put_u32(static_cast<std::uint32_t>(v.size()));
    for (const auto& s : v) {
      put_u32(static_cast<std::uint32_t>(s.size()));
      buf_.insert(buf_.end(), s.begin(), s.end());
    }
  }
  void array_inline_bytes(const char* name, std::uint8_t elem, const void* data,
                          std::uint64_t count) {
    head(type_array, name);
    buf_.push_back(array_inline);
    buf_.push_back(elem);
    put_u64(count);
    const auto* p = static_cast<const unsigned char*>(data);
    buf_.insert(buf_.end(), p, p + count * elem);
  }
  void array_ref(const char* name, std::uint8_t kind, std::uint8_t elem, std::uint64_t count,
                 std::uint64_t offset, std::uint64_t hash) {
    head(type_array, name);
    buf_.push_back(kind);
    buf_.push_back(elem);
    put_u64(count);
    if (kind == array_blob) put_u64(offset);
    if (kind == array_blob || kind == array_hash) put_u64(hash);
  }
  const std::vector<unsigned char>& bytes() {
    std::memcpy(buf_.data(), &count_, 4);
    return buf_;
  }

 private:
  void head(std::uint8_t type, const char* name) {
    ++count_;
    buf_.push_back(type);
    const auto len = static_cast<std::uint16_t>(std::strlen(name));
    buf_.push_back(static_cast<unsigned char>(len & 0xff));
    buf_.push_back(static_cast<unsigned char>(len >> 8));
    buf_.insert(buf_.end(), name, name + len);
  }
  void put_u32(std::uint32_t v) {
    for (int i = 0; i < 4; ++i) buf_.push_back(static_cast<unsigned char>(v >> (8 * i)));
  }
  void put_u64(std::uint64_t v) {
    for (int i = 0; i < 8; ++i) buf_.push_back(static_cast<unsigned char>(v >> (8 * i)));
  }
  std::vector<unsigned char> buf_;
  std::uint32_t count_ = 0;
};

std::string demangle(const char* name) {
  int status = 0;
  char* out = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  std::string s = status == 0 && out ? out : name;
  std::free(out);
  return s;
}

std::string env_or(const char* name, const char* fallback) {
  const char* v = std::getenv(name);
  return v ? v : fallback;
}

// --- The state of one tree ------------------------------------------------------------------

struct key_record {
  int key, avail, length, size, occupancy, tail_capacity, value, tail_base;
  int segments, row_len, terminated;
  std::uint64_t row_hash, hash;
  long long row_begin;  // offset of the row in tree_state::rows
};

struct tree_state {
  bool read_ok = true;
  std::string read_error;
  long long num_records = 0, fixed_size = 0, initial_payload_size = 0, alignment = 0;
  std::vector<int> keys;           // in-order (BST order) keys
  std::vector<int> subtree_avail;  // by heap position
  std::vector<key_record> recs;    // in-order
  std::vector<int> rows;           // rows of recs, concatenated
  std::uint64_t keys_hash = 0, subtree_hash = 0, state_hash = 0;
  long long bad_subtree = 0, unterminated = 0, key_order = 0, unreachable = 0;
  long long duplicate_keys = 0, max_segments = 0;
};

std::uint64_t record_hash(const key_record& r) {
  const int f[11] = {r.key,   r.avail,     r.length,   r.size,    r.occupancy, r.tail_capacity,
                     r.value, r.tail_base, r.segments, r.row_len, r.terminated};
  return fnv_u64(fnv_i32s(fnv_offset, f, 11), r.row_hash);
}

template <class T>
bool copy_down(std::vector<T>& out, const void* device, long long count, std::string& error) {
  out.assign(static_cast<std::size_t>(std::max(0LL, count)), T{});
  if (count <= 0) return true;
  if (device == nullptr) {
    error = "null device pointer";
    return false;
  }
  cudaError_t e = cudaMemcpy(out.data(), device, sizeof(T) * static_cast<std::size_t>(count),
                             cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    error = cudaGetErrorString(e);
    return false;
  }
  return true;
}

tree_state read_state(const ctx_view& v) {
  tree_state s;
  s.num_records = v.num_records;
  s.fixed_size = v.fixed_size;
  s.initial_payload_size = v.initial_payload_size;
  s.alignment = v.alignment;
  const long long n = v.d_nodes ? std::max(0LL, v.num_records) : 0;
  std::vector<node_mirror> nodes;
  std::vector<int> avail, flat;
  cudaDeviceSynchronize();
  const long long used = std::max(0LL, std::min(v.initial_payload_size, v.fixed_size));
  if (!copy_down(nodes, v.d_nodes, n, s.read_error) ||
      !copy_down(avail, v.d_avail, v.d_avail ? n : 0, s.read_error) ||
      !copy_down(s.subtree_avail, v.d_subtree_avail, v.d_subtree_avail ? n : 0, s.read_error) ||
      !copy_down(flat, v.d_payload, v.d_payload ? used : 0, s.read_error)) {
    s.read_ok = false;
    cudaGetLastError();
    return s;
  }
  if (avail.empty()) avail.assign(static_cast<std::size_t>(n), 0);
  if (s.subtree_avail.empty()) s.subtree_avail.assign(static_cast<std::size_t>(n), 0);
  const auto base = reinterpret_cast<std::uintptr_t>(v.d_nodes);
  auto child = [&](const node_mirror* p) -> long long {
    if (p == nullptr) return -1;
    const auto a = reinterpret_cast<std::uintptr_t>(p);
    if (a < base || (a - base) % sizeof(node_mirror) != 0) return -2;
    const long long t = static_cast<long long>((a - base) / sizeof(node_mirror));
    return t < n ? t : -2;
  };
  // In-order traversal from the root (heap position 0); nodes it cannot reach follow in key
  // order and are counted.
  std::vector<long long> order;
  order.reserve(static_cast<std::size_t>(n));
  std::vector<char> seen(static_cast<std::size_t>(n), 0);
  if (n > 0) {
    std::vector<long long> stack;
    long long t = 0;
    while (t >= 0 || !stack.empty()) {
      while (t >= 0 && !seen[static_cast<std::size_t>(t)]) {
        seen[static_cast<std::size_t>(t)] = 1;
        stack.push_back(t);
        t = child(nodes[static_cast<std::size_t>(t)].left);
        if (t == -2) ++s.unreachable, t = -1;
      }
      if (t >= 0) {  // a cycle: the node was seen already
        ++s.unreachable;
        t = -1;
      }
      if (stack.empty()) break;
      t = stack.back();
      stack.pop_back();
      order.push_back(t);
      t = child(nodes[static_cast<std::size_t>(t)].right);
      if (t == -2) ++s.unreachable, t = -1;
    }
    std::vector<long long> rest;
    for (long long u = 0; u < n; ++u)
      if (!seen[static_cast<std::size_t>(u)]) rest.push_back(u);
    s.unreachable += static_cast<long long>(rest.size());
    std::stable_sort(rest.begin(), rest.end(), [&](long long a, long long b) {
      return nodes[static_cast<std::size_t>(a)].index < nodes[static_cast<std::size_t>(b)].index;
    });
    order.insert(order.end(), rest.begin(), rest.end());
  }
  const long long limit = static_cast<long long>(flat.size());
  s.recs.reserve(order.size());
  s.keys.reserve(order.size());
  for (long long t : order) {
    const node_mirror& nd = nodes[static_cast<std::size_t>(t)];
    key_record r{};
    r.key = nd.index;
    r.avail = avail[static_cast<std::size_t>(t)];
    r.length = nd.length;
    r.size = nd.size;
    r.occupancy = nd.occupancy;
    r.tail_capacity = nd.tail_capacity;
    r.value = nd.value;
    r.tail_base = nd.tail_base;
    r.row_begin = static_cast<long long>(s.rows.size());
    // The row: positive values from `value`, a negative value other than INT_MIN continues at
    // its negation, 0 or INT_MIN ends it. A walk longer than the payload does not terminate.
    long long loc = nd.value;
    int segments = 1;
    bool terminated = false;
    for (long long steps = 0; steps <= limit; ++steps) {
      if (loc < 0 || loc >= limit) break;
      const int x = flat[static_cast<std::size_t>(loc)];
      if (x < 0 && x != INT_MIN) {
        loc = -static_cast<long long>(x);
        ++segments;
        continue;
      }
      if (x == 0 || x == INT_MIN) {
        terminated = true;
        break;
      }
      s.rows.push_back(x);
      ++loc;
    }
    r.segments = segments;
    r.terminated = terminated ? 1 : 0;
    r.row_len = static_cast<int>(static_cast<long long>(s.rows.size()) - r.row_begin);
    r.row_hash =
        fnv_i32s(fnv_offset, s.rows.data() + r.row_begin, static_cast<std::size_t>(r.row_len));
    r.hash = record_hash(r);
    s.unterminated += terminated ? 0 : 1;
    s.max_segments = std::max<long long>(s.max_segments, segments);
    s.keys.push_back(r.key);
    s.recs.push_back(r);
  }
  for (std::size_t i = 1; i < s.keys.size(); ++i) s.key_order += s.keys[i - 1] >= s.keys[i];
  {
    std::vector<int> sorted = s.keys;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 1; i < sorted.size(); ++i) s.duplicate_keys += sorted[i - 1] == sorted[i];
  }
  for (long long t = 0; t < n; ++t) {
    long long want = avail[static_cast<std::size_t>(t)];
    if (2 * t + 1 < n) want += s.subtree_avail[static_cast<std::size_t>(2 * t + 1)];
    if (2 * t + 2 < n) want += s.subtree_avail[static_cast<std::size_t>(2 * t + 2)];
    s.bad_subtree += want != s.subtree_avail[static_cast<std::size_t>(t)];
  }
  s.keys_hash = fnv_i32s(fnv_offset, s.keys.data(), s.keys.size());
  s.subtree_hash = fnv_i32s(fnv_offset, s.subtree_avail.data(), s.subtree_avail.size());
  std::uint64_t h = fnv_offset;
  for (long long x : {s.num_records, s.fixed_size, s.initial_payload_size, s.alignment})
    h = fnv_u64(h, static_cast<std::uint64_t>(x));
  h = fnv_u64(h, s.keys_hash);
  h = fnv_u64(h, s.subtree_hash);
  for (const auto& r : s.recs) h = fnv_u64(h, r.hash);
  s.state_hash = h;
  return s;
}

/// What the recorder keeps of a tree's last recorded state (to write deltas).
struct tree_memory {
  bool known = false;
  std::uint64_t state_hash = 0, keys_hash = 0, subtree_hash = 0;
  std::unordered_map<int, std::uint64_t> key_hash;
};

struct tree_info {
  std::uint32_t id;
  std::string name;
  long long capacity, alignment;
  std::string origin;
  tree_memory memory;
};

std::string normalize_arg(const std::string& a) {
  if (!a.empty() && a[0] == '/') {
    const auto slash = a.find_last_of('/');
    return "<abs>/" + a.substr(slash + 1);
  }
  return a;
}

}  // namespace

// --- The recorder --------------------------------------------------------------------------

class recorder {
 public:
  explicit recorder(const adapter& a, std::string dir) : adapter_(a), dir_(std::move(dir)) {}

  bool open();
  void write(std::uint32_t tag, fields& f);
  std::uint64_t next_seq() {
    return seq_++;
  }
  std::uint64_t blob(const void* data, std::size_t bytes);
  void finalize(const char* how);

  tree_info& tree_for_object(const void* object, const void* context);
  tree_info& tree_for_context(const void* context);
  tree_info& new_tree(const std::string& name, long long capacity, long long alignment,
                      const std::string& origin);

  void note(const std::string& text) {
    fields f;
    f.i64("seq", static_cast<long long>(next_seq()));
    f.str("text", text);
    write(tag_note, f);
  }

  const adapter& adapter_;
  std::string dir_;
  std::string mode_ = "auto";  // full | digest | auto | batches
  long long full_max_ = 4096;
  std::set<std::string> hash_only_;  // "<dataset>:<what>"
  std::recursive_mutex mutex_;
  FILE* main_ = nullptr;
  FILE* blob_ = nullptr;
  std::string path_;
  std::uint64_t seq_ = 0;
  std::uint64_t blob_bytes_ = 16;
  std::uint32_t next_tree_ = 1;
  long long ops_ = 0;
  bool in_op_ = false;
  std::string op_desc_;
  bool finalized_ = false;
  std::vector<std::unique_ptr<tree_info>> trees_;
  std::unordered_map<const void*, tree_info*> by_object_, by_context_;
};

namespace {

recorder* g_recorder = nullptr;
std::once_flag g_once;

void at_exit_handler() {
  if (g_recorder) {
    std::lock_guard<std::recursive_mutex> lock(g_recorder->mutex_);
    if (g_recorder->in_op_)
      g_recorder->note("the process exited during op " + g_recorder->op_desc_);
    g_recorder->finalize(g_recorder->in_op_ ? "exit-during-op" : "normal");
  }
}

std::vector<std::string> read_argv() {
  std::vector<std::string> args;
  std::ifstream in("/proc/self/cmdline", std::ios::binary);
  std::string a;
  while (std::getline(in, a, '\0')) args.push_back(a);
  return args;
}

}  // namespace

bool recorder::open() {
  {
    const unsigned one = 1;
    unsigned char b;
    std::memcpy(&b, &one, 1);
    if (b != 1) return false;  // the format is little-endian
  }
  mode_ = env_or("DYNG_CBST_TRACE_MODE", "auto");
  if (mode_ != "full" && mode_ != "digest" && mode_ != "auto" && mode_ != "batches") mode_ = "auto";
  full_max_ = std::atoll(env_or("DYNG_CBST_TRACE_FULL_MAX", "4096").c_str());
  {
    std::stringstream ss(env_or("DYNG_CBST_TRACE_HASH_ONLY", ""));
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (!item.empty()) hash_only_.insert(item);
    }
  }
  ::mkdir(dir_.c_str(), 0777);
  int fd = -1;
  for (int i = 1; i < 1000000 && fd < 0; ++i) {
    char name[32];
    std::snprintf(name, sizeof name, "%06d.cbt", i);
    path_ = dir_ + "/" + name;
    fd = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0 && errno != EEXIST) return false;
  }
  if (fd < 0) return false;
  main_ = ::fdopen(fd, "wb");
  if (!main_) return false;
  const char magic[8] = {'D', 'Y', 'N', 'G', 'C', 'B', 'T', '1'};
  std::fwrite(magic, 1, 8, main_);
  const std::uint32_t head[2] = {format_version, 0};
  std::fwrite(head, 4, 2, main_);

  fields f;
  f.i64("seq", static_cast<long long>(next_seq()));
  f.i64("format_version", format_version);
  f.str("recorder_version", recorder_version);
  f.str("repo", adapter_.repo);
  f.str("commit", adapter_.commit);
  f.str("mode", mode_);
  f.i64("full_max", full_max_);
  f.str("hash", "fnv1a64");
  {
    std::vector<std::string> args;
    std::vector<std::string> raw = read_argv();
    for (const auto& a : raw) args.push_back(normalize_arg(a));
    f.strings("argv", args);
    // Input files named on the command line: content hash and size; files up to 16 MiB are
    // copied into <dir>/inputs/<name>.<hash> (the awk-generated test inputs).
    std::vector<std::string> inputs;
    for (std::size_t i = 1; i < raw.size(); ++i) {
      struct stat st{};
      if (::stat(raw[i].c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
      std::ifstream in(raw[i], std::ios::binary);
      std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      const std::uint64_t h = fnv_bytes(fnv_offset, data.data(), data.size());
      char hex[17];
      std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(h));
      const std::string base = normalize_arg(raw[i]).substr(raw[i][0] == '/' ? 6 : 0);
      std::string stored = "-";
      if (data.size() <= (16u << 20)) {
        const std::string sub = dir_ + "/inputs";
        ::mkdir(sub.c_str(), 0777);
        std::string leaf = base;
        std::replace(leaf.begin(), leaf.end(), '/', '_');
        stored = "inputs/" + leaf + "." + hex;
        const std::string dst = dir_ + "/" + stored;
        struct stat ds{};
        if (::stat(dst.c_str(), &ds) != 0) {
          std::ofstream out(dst, std::ios::binary);
          out.write(data.data(), static_cast<std::streamsize>(data.size()));
        }
      }
      inputs.push_back(std::to_string(i) + " " + base + " " + hex + " " +
                       std::to_string(data.size()) + " " + stored);
    }
    f.strings("inputs", inputs);
  }
  {
    std::vector<std::string> env;
    for (const char* k : {"DYNG_CBST_TRACE_MODE", "DYNG_CBST_TRACE_FULL_MAX",
                          "DYNG_CBST_TRACE_HASH_ONLY", "DYNG_CBST_TRACE_LABEL",
                          "CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "OMP_NUM_THREADS"}) {
      const char* v = std::getenv(k);
      if (v) env.push_back(std::string(k) + "=" + v);
    }
    f.strings("env", env);
  }
  {
    std::vector<std::string> ho(hash_only_.begin(), hash_only_.end());
    f.strings("hash_only", ho);
  }
  write(tag_header, f);
  std::atexit(at_exit_handler);
  return true;
}

void recorder::write(std::uint32_t tag, fields& f) {
  if (!main_ || finalized_) return;
  const auto& b = f.bytes();
  const std::uint64_t len = b.size();
  std::fwrite(&tag, 4, 1, main_);
  std::fwrite(&len, 8, 1, main_);
  std::fwrite(b.data(), 1, b.size(), main_);
  std::fflush(main_);
}

std::uint64_t recorder::blob(const void* data, std::size_t bytes) {
  if (!blob_) {
    const std::string p = path_.substr(0, path_.size() - 4) + ".blob";
    blob_ = std::fopen(p.c_str(), "wb");
    if (!blob_) return 0;
    const char magic[8] = {'D', 'Y', 'N', 'G', 'C', 'B', 'B', '1'};
    std::fwrite(magic, 1, 8, blob_);
    const std::uint32_t head[2] = {format_version, 0};
    std::fwrite(head, 4, 2, blob_);
  }
  const std::uint64_t at = blob_bytes_;
  std::fwrite(data, 1, bytes, blob_);
  std::fflush(blob_);
  blob_bytes_ += bytes;
  return at;
}

void recorder::finalize(const char* how) {
  if (finalized_ || !main_) return;
  fields f;
  f.i64("seq", static_cast<long long>(next_seq()));
  f.str("exit", how);
  f.i64("num_ops", ops_);
  f.i64("blob_bytes", blob_ ? static_cast<long long>(blob_bytes_) : 0);
  std::vector<long long> ids, caps, aligns;
  std::vector<std::string> names, origins;
  for (const auto& t : trees_) {
    ids.push_back(t->id);
    caps.push_back(t->capacity);
    aligns.push_back(t->alignment);
    names.push_back(t->name);
    origins.push_back(t->origin);
  }
  f.array_inline_bytes("trees.id", 8, ids.data(), ids.size());
  f.strings("trees.name", names);
  f.array_inline_bytes("trees.capacity", 8, caps.data(), caps.size());
  f.array_inline_bytes("trees.alignment", 8, aligns.data(), aligns.size());
  f.strings("trees.origin", origins);
  write(tag_end, f);
  finalized_ = true;
  std::fclose(main_);
  main_ = nullptr;
  if (blob_) std::fclose(blob_);
  blob_ = nullptr;
}

tree_info& recorder::new_tree(const std::string& name, long long capacity, long long alignment,
                              const std::string& origin) {
  trees_.push_back(std::make_unique<tree_info>());
  tree_info& t = *trees_.back();
  t.id = next_tree_++;
  t.name = name;
  t.capacity = capacity;
  t.alignment = alignment;
  t.origin = origin;
  fields f;
  f.i64("seq", static_cast<long long>(next_seq()));
  f.i64("tree", t.id);
  f.str("dataset_name", name);
  f.i64("payload_capacity", capacity);
  f.i64("alignment", alignment);
  f.str("origin", origin);
  write(tag_tree, f);
  return t;
}

tree_info& recorder::tree_for_context(const void* context) {
  auto it = by_context_.find(context);
  if (it != by_context_.end()) return *it->second;
  const ctx_view v = adapter_.view_of_context(context);
  tree_info& t =
      new_tree(v.dataset_name ? v.dataset_name : "", v.fixed_size, v.alignment, "context");
  by_context_[context] = &t;
  return t;
}

tree_info& recorder::tree_for_object(const void* object, const void* context) {
  if (object == nullptr) return tree_for_context(context);
  auto it = by_object_.find(object);
  if (it != by_object_.end()) return *it->second;
  const ctx_view v = adapter_.view_of_object(object);
  tree_info& t =
      new_tree(v.dataset_name ? v.dataset_name : "", v.fixed_size, v.alignment, "object");
  by_object_[object] = &t;
  by_context_[adapter_.context_of(object)] = &t;
  return t;
}

recorder* get(const adapter& a) {
  std::call_once(g_once, [&] {
    const char* dir = std::getenv("DYNG_CBST_TRACE_DIR");
    if (dir == nullptr || *dir == '\0') return;
    auto* r = new recorder(a, dir);
    if (r->open()) {
      g_recorder = r;
    } else {
      delete r;
    }
  });
  return g_recorder && !g_recorder->finalized_ ? g_recorder : nullptr;
}

// --- The invocation log (DYNG_CBST_INVOCATION_LOG) ------------------------------------------
//
// With DYNG_CBST_INVOCATION_LOG=<dir> every twin writes <dir>/NNNNNN.inv at start-up: its working
// directory and arguments (one "arg=" line each), the input files among them (copied to
// <dir>/inputs/), and at exit the number of CBST operations it called ("calls=") and "exit=normal"
// (missing when the process was killed). parity/exporters/escher/check_twins.py replays every
// logged invocation in the unpatched and the patched copy; the trace export checks that every
// invocation with calls > 0 wrote a trace. Without the variable nothing is written or counted.

namespace {

std::atomic<long long> g_calls{0};
std::string g_inv_path;

void invocation_at_exit() {
  FILE* f = std::fopen(g_inv_path.c_str(), "a");
  if (!f) return;
  std::fprintf(f, "calls=%lld\nexit=normal\n", g_calls.load());
  std::fclose(f);
}

std::string copy_input(const std::string& dir, const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (data.size() > (16u << 20)) return "-";
  char hex[17];
  std::snprintf(hex, sizeof hex, "%016llx",
                static_cast<unsigned long long>(fnv_bytes(fnv_offset, data.data(), data.size())));
  std::string leaf = path.substr(path.find_last_of('/') + 1);
  const std::string stored = "inputs/" + leaf + "." + hex;
  const std::string sub = dir + "/inputs";
  ::mkdir(sub.c_str(), 0777);
  struct stat st{};
  if (::stat((dir + "/" + stored).c_str(), &st) != 0) {
    std::ofstream out(dir + "/" + stored, std::ios::binary);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
  }
  return stored;
}

bool start_invocation_log() {
  const char* dir = std::getenv("DYNG_CBST_INVOCATION_LOG");
  if (dir == nullptr || *dir == '\0') return false;
  ::mkdir(dir, 0777);
  int fd = -1;
  for (int i = 1; i < 1000000 && fd < 0; ++i) {
    char name[32];
    std::snprintf(name, sizeof name, "%06d.inv", i);
    g_inv_path = std::string(dir) + "/" + name;
    fd = ::open(g_inv_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0 && errno != EEXIST) return false;
  }
  if (fd < 0) return false;
  FILE* f = ::fdopen(fd, "w");
  if (!f) return false;
  char cwd[4096];
  std::fprintf(f, "cwd=%s\n", ::getcwd(cwd, sizeof cwd) ? cwd : "?");
  const std::vector<std::string> args = read_argv();
  for (std::size_t i = 0; i < args.size(); ++i) {
    std::fprintf(f, "arg=%s\n", args[i].c_str());
    struct stat st{};
    if (i > 0 && ::stat(args[i].c_str(), &st) == 0 && S_ISREG(st.st_mode))
      std::fprintf(f, "input=%zu %s\n", i, copy_input(dir, args[i]).c_str());
  }
  std::fclose(f);
  std::atexit(invocation_at_exit);
  return true;
}

const bool g_invocation_logged = start_invocation_log();

}  // namespace

void note_call() {
  if (g_invocation_logged) ++g_calls;
}

void note(recorder* r, const std::string& text) {
  if (!r) return;
  std::lock_guard<std::recursive_mutex> lock(r->mutex_);
  r->note(text);
}

// --- Lifecycle -------------------------------------------------------------------------------

namespace {
void event(recorder* r, const char* kind, long long a, long long b) {
  fields f;
  f.i64("seq", static_cast<long long>(r->next_seq()));
  f.str("event", kind);
  f.i64("tree", a);
  f.i64("other", b);
  r->write(tag_event, f);
}
}  // namespace

void on_construct_object(recorder* r, const void* object, const char* name, int capacity,
                         int alignment) {
  if (!r) return;
  std::lock_guard<std::recursive_mutex> lock(r->mutex_);
  r->by_object_.erase(object);
  tree_info& t = r->new_tree(name ? name : "", capacity, alignment, "object");
  r->by_object_[object] = &t;
  r->by_context_[r->adapter_.context_of(object)] = &t;
  event(r, "construct_object", t.id, 0);
}

void on_destroy_object(recorder* r, const void* object) {
  if (!r) return;
  std::lock_guard<std::recursive_mutex> lock(r->mutex_);
  auto it = r->by_object_.find(object);
  if (it == r->by_object_.end()) return;
  event(r, "destroy_object", it->second->id, 0);
  r->by_context_.erase(r->adapter_.context_of(object));
  r->by_object_.erase(it);
}

void on_move_construct(recorder* r, const void* dst, const void* src) {
  if (!r) return;
  std::lock_guard<std::recursive_mutex> lock(r->mutex_);
  auto it = r->by_object_.find(src);
  if (it == r->by_object_.end()) return;
  tree_info* t = it->second;
  r->by_object_.erase(it);
  r->by_context_.erase(r->adapter_.context_of(src));
  r->by_object_[dst] = t;
  r->by_context_[r->adapter_.context_of(dst)] = t;
  event(r, "move_construct", t->id, 0);
}

void on_move_assign(recorder* r, const void* dst, const void* src) {
  if (!r) return;
  std::lock_guard<std::recursive_mutex> lock(r->mutex_);
  long long ended = 0;
  auto old = r->by_object_.find(dst);
  if (old != r->by_object_.end()) {
    ended = old->second->id;
    r->by_object_.erase(old);
  }
  auto it = r->by_object_.find(src);
  if (it == r->by_object_.end()) {
    r->by_context_.erase(r->adapter_.context_of(dst));
    event(r, "move_assign", 0, ended);
    return;
  }
  tree_info* t = it->second;
  r->by_object_.erase(it);
  r->by_context_.erase(r->adapter_.context_of(src));
  r->by_object_[dst] = t;
  r->by_context_[r->adapter_.context_of(dst)] = t;
  event(r, "move_assign", t->id, ended);
}

array_policy input_policy(recorder* r, const char* dataset_name, const char* what) {
  if (r && dataset_name && r->hash_only_.count(std::string(dataset_name) + ":" + what) != 0)
    return array_policy::hash_only;
  return array_policy::input;
}

// --- Operations ------------------------------------------------------------------------------

namespace {

const char* op_name(op_kind k) {
  switch (k) {
    case op_kind::construct:
      return "construct";
    case op_kind::rebuild:
      return "rebuild";
    case op_kind::insert:
      return "insert";
    case op_kind::fill:
      return "fill";
    case op_kind::erase:
      return "erase";
    case op_kind::unfill:
      return "unfill";
    case op_kind::compact:
      return "compact";
  }
  return "?";
}

/// Writes an int array under a policy: inline, or (digest mode, input arrays over 4096 values)
/// into the side file, or as a hash only.
void put_ints(recorder* r, fields& f, const char* name, const int* data, long long count,
              array_policy policy, bool digest) {
  if (count < 0 || (count > 0 && data == nullptr)) {
    f.array_ref(name, array_null, 4, 0, 0, 0);
    return;
  }
  const auto n = static_cast<std::uint64_t>(count);
  if (policy == array_policy::hash_only) {
    f.array_ref(name, array_hash, 4, n, 0, fnv_i32s(fnv_offset, data, n));
    return;
  }
  if (policy == array_policy::input && digest && count > 4096) {
    const std::uint64_t at = r->blob(data, n * sizeof(int));
    f.array_ref(name, array_blob, 4, n, at, fnv_i32s(fnv_offset, data, n));
    return;
  }
  f.array_inline_bytes(name, 4, data, n);
}

void put_scalars(fields& f, const char* prefix, const ctx_view& v, long long device_bytes,
                 long long d) {
  const std::string p = prefix;
  f.i64((p + ".fixed_size").c_str(), v.fixed_size);
  f.i64((p + ".num_records").c_str(), v.num_records);
  f.i64((p + ".initial_payload_size").c_str(), v.initial_payload_size);
  f.i64((p + ".alignment").c_str(), v.alignment);
  f.i64((p + ".scratch_items").c_str(), v.scratch_items);
  f.i64((p + ".scratch_payload").c_str(), v.scratch_payload);
  f.i64((p + ".device_bytes").c_str(), device_bytes);
  f.i64((p + ".d").c_str(), d);
}

long long root_avail(const ctx_view& v) {
  if (v.num_records <= 0 || v.d_subtree_avail == nullptr) return -1;
  int d = -1;
  cudaDeviceSynchronize();
  if (cudaMemcpy(&d, v.d_subtree_avail, sizeof d, cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaGetLastError();
    return -2;
  }
  return d;
}

void describe_exception(std::exception_ptr e, std::string& type, std::string& what) {
  try {
    std::rethrow_exception(e);
  } catch (const std::exception& x) {
    type = demangle(typeid(x).name());
    what = x.what();
  } catch (...) {
    const std::type_info* t = abi::__cxa_current_exception_type();
    type = t ? demangle(t->name()) : "unknown";
  }
}

}  // namespace

struct op_record::impl {
  impl(recorder* r_, op_kind k, const void* o, const void* c)
      : r(r_), kind(k), object(o), context(c) {}
  recorder* r;
  op_kind kind;
  const void* object;
  const void* context;
  tree_info* tree = nullptr;
  fields f;
  bool digest_mode = false;  // large inputs go to the side file
  bool batches_only = false;
  std::vector<int> named;
  long long bins_before[9] = {0};
  bool have_bins = false;
  bool threw = false;
  std::string exc_type, exc_what;
  std::vector<int> mapping;
  bool has_mapping = false;
  ctx_view view() const {
    return object ? r->adapter_.view_of_object(object) : r->adapter_.view_of_context(context);
  }
  long long device_bytes() const {
    return object && r->adapter_.device_bytes ? r->adapter_.device_bytes(object) : -1;
  }
};

op_record::op_record(recorder* r, op_kind kind, const void* object, const void* context)
    : impl_(nullptr) {
  if (!r) return;
  r->mutex_.lock();
  impl_ = new impl(r, kind, object, context);
  try {
    impl_->tree = &r->tree_for_object(object, context);
    const ctx_view v = impl_->view();
    impl_->batches_only = r->mode_ == "batches";
    const long long n = std::max(v.num_records, 0LL);
    impl_->digest_mode = r->mode_ == "digest" || (r->mode_ == "auto" && n > r->full_max_);
    auto& f = impl_->f;
    f.i64("seq", static_cast<long long>(r->next_seq()));
    f.i64("tree", impl_->tree->id);
    f.str("op", op_name(kind));
    // The state at entry must be the state recorded after the tree's last op.
    if (!impl_->batches_only) {
      if (!impl_->tree->memory.known) {
        f.i64("pre.matches", 2);
      } else {
        const tree_state s = read_state(v);
        f.i64("pre.matches", s.state_hash == impl_->tree->memory.state_hash ? 1 : 0);
        f.i64("pre.hash", static_cast<long long>(s.state_hash));
      }
    }
    put_scalars(f, "before", v, impl_->device_bytes(), root_avail(v));
    if (r->adapter_.bin_launches) {
      r->adapter_.bin_launches(impl_->bins_before);
      impl_->have_bins = true;
    }
    r->in_op_ = true;
    r->op_desc_ = std::string(op_name(kind)) + " on tree " + std::to_string(impl_->tree->id);
  } catch (const std::exception& e) {
    r->note(std::string("recorder error at op entry: ") + e.what());
  }
}

op_record::~op_record() {
  if (!impl_) return;
  recorder* r = impl_->r;
  delete impl_;
  r->mutex_.unlock();
}

void op_record::scalar(const char* name, long long value) {
  if (impl_) impl_->f.i64((std::string("in.") + name).c_str(), value);
}
void op_record::output(const char* name, long long value) {
  if (impl_) impl_->f.i64((std::string("out.") + name).c_str(), value);
}
void op_record::text(const char* name, const std::string& value) {
  if (impl_) impl_->f.str((std::string("in.") + name).c_str(), value);
}
void op_record::ints(const char* name, const int* data, long long count, array_policy policy) {
  if (!impl_) return;
  put_ints(impl_->r, impl_->f, (std::string("in.") + name).c_str(), data, count, policy,
           impl_->digest_mode || impl_->batches_only);
}
void op_record::device_ints(const char* name, const int* device, long long count,
                            array_policy policy) {
  if (!impl_) return;
  std::vector<int> host;
  std::string error;
  cudaDeviceSynchronize();
  if (count < 0 || !copy_down(host, device, count, error)) {
    impl_->f.array_ref((std::string("in.") + name).c_str(), array_null, 4, 0, 0, 0);
    return;
  }
  ints(name, host.data(), count, policy);
}
void op_record::named_keys(const std::vector<int>& keys) {
  if (impl_) impl_->named.insert(impl_->named.end(), keys.begin(), keys.end());
}
void op_record::named_keys(const int* keys, long long count) {
  if (impl_ && keys && count > 0) impl_->named.insert(impl_->named.end(), keys, keys + count);
}
void op_record::exception(std::exception_ptr e) {
  if (!impl_) return;
  impl_->threw = true;
  describe_exception(e, impl_->exc_type, impl_->exc_what);
}
void op_record::mapping(const std::vector<int>& item_to_key) {
  if (!impl_) return;
  impl_->mapping = item_to_key;
  impl_->has_mapping = true;
  named_keys(item_to_key);
}

void op_record::finish() {
  if (!impl_) return;
  recorder* r = impl_->r;
  auto& f = impl_->f;
  try {
    f.i64("status", impl_->threw ? 1 : 0);
    if (impl_->threw) {
      f.str("exception.type", impl_->exc_type);
      f.str("exception.what", impl_->exc_what);
    }
    if (impl_->has_mapping)
      f.array_inline_bytes("out.mapping", 4, impl_->mapping.data(), impl_->mapping.size());
    const ctx_view v = impl_->view();
    put_scalars(f, "after", v, impl_->device_bytes(), root_avail(v));
    if (impl_->have_bins) {
      long long now[9];
      r->adapter_.bin_launches(now);
      long long delta[9];
      for (int i = 0; i < 9; ++i) delta[i] = now[i] - impl_->bins_before[i];
      f.array_inline_bytes("bins.fill", 8, delta, 3);
      f.array_inline_bytes("bins.reuse", 8, delta + 3, 3);
      f.array_inline_bytes("bins.unfill", 8, delta + 6, 3);
    }
    if (!impl_->batches_only) {
      const tree_state s = read_state(v);
      tree_memory& mem = impl_->tree->memory;
      // Delta digests in digest mode, and in auto mode for trees with more than full_max keys
      // or row values; full digests otherwise, and always for a tree's first digest.
      const bool delta =
          r->mode_ == "digest" ||
          (r->mode_ == "auto" && (static_cast<long long>(s.recs.size()) > r->full_max_ ||
                                  static_cast<long long>(s.rows.size()) > r->full_max_));
      const bool full = !delta || !mem.known || s.duplicate_keys > 0;
      f.i64("digest.mode", full ? 0 : 1);
      if (!s.read_ok) f.str("digest.read_error", s.read_error);
      f.i64("state.num_records", static_cast<long long>(s.recs.size()));
      if (full || s.keys_hash != mem.keys_hash)
        f.array_inline_bytes("state.keys", 4, s.keys.data(), s.keys.size());
      else
        f.i64("state.keys_same", 1);
      f.i64("state.keys_hash", static_cast<long long>(s.keys_hash));
      if (full || s.subtree_hash != mem.subtree_hash)
        f.array_inline_bytes("state.subtree_avail", 4, s.subtree_avail.data(),
                             s.subtree_avail.size());
      else
        f.i64("state.subtree_avail_same", 1);
      f.i64("state.subtree_avail_hash", static_cast<long long>(s.subtree_hash));
      // The keys written in full: all (full digest), else those that changed or the op names.
      std::vector<std::size_t> touched;
      if (full) {
        touched.resize(s.recs.size());
        for (std::size_t i = 0; i < touched.size(); ++i) touched[i] = i;
      } else {
        std::unordered_map<int, char> named;
        for (int k : impl_->named) named[k] = 1;
        for (std::size_t i = 0; i < s.recs.size(); ++i) {
          auto it = mem.key_hash.find(s.recs[i].key);
          if (it == mem.key_hash.end() || it->second != s.recs[i].hash ||
              named.count(s.recs[i].key))
            touched.push_back(i);
        }
      }
      std::vector<int> col(touched.size());
      std::vector<long long> hashes(touched.size());
      auto column = [&](const char* name, int key_record::* field) {
        for (std::size_t j = 0; j < touched.size(); ++j) col[j] = s.recs[touched[j]].*field;
        f.array_inline_bytes(name, 4, col.data(), col.size());
      };
      column("key.key", &key_record::key);
      column("key.avail", &key_record::avail);
      column("key.length", &key_record::length);
      column("key.size", &key_record::size);
      column("key.occupancy", &key_record::occupancy);
      column("key.tail_capacity", &key_record::tail_capacity);
      column("key.value", &key_record::value);
      column("key.tail_base", &key_record::tail_base);
      column("key.segments", &key_record::segments);
      column("key.row_len", &key_record::row_len);
      column("key.terminated", &key_record::terminated);
      for (std::size_t j = 0; j < touched.size(); ++j)
        hashes[j] = static_cast<long long>(s.recs[touched[j]].row_hash);
      f.array_inline_bytes("key.row_hash", 8, hashes.data(), hashes.size());
      {
        std::vector<int> rows;
        for (std::size_t j : touched) {
          const key_record& rec = s.recs[j];
          rows.insert(rows.end(), s.rows.begin() + rec.row_begin,
                      s.rows.begin() + rec.row_begin + rec.row_len);
        }
        const bool hash_rows = r->hash_only_.count(impl_->tree->name + ":rows") != 0 && delta;
        put_ints(r, f, "key.rows", rows.data(), static_cast<long long>(rows.size()),
                 hash_rows ? array_policy::hash_only : array_policy::normal, false);
      }
      if (!full) {
        std::vector<long long> all(s.recs.size());
        for (std::size_t i = 0; i < all.size(); ++i)
          all[i] = static_cast<long long>(s.recs[i].hash);
        f.array_inline_bytes("state.key_hashes", 8, all.data(), all.size());
      }
      f.i64("state.hash", static_cast<long long>(s.state_hash));
      f.i64("rec.bad_subtree_avail", s.bad_subtree);
      f.i64("rec.unterminated", s.unterminated);
      f.i64("rec.key_order", s.key_order);
      f.i64("rec.unreachable", s.unreachable);
      f.i64("rec.duplicate_keys", s.duplicate_keys);
      f.i64("rec.max_segments", s.max_segments);
      // The original's own integrity verdict on the rows as read (order-sensitive).
      const void* ctx = impl_->object ? r->adapter_.context_of(impl_->object) : impl_->context;
      long long max_key = 0;
      for (const auto& rec : s.recs)
        if (rec.avail == 0 && rec.key >= 1) max_key = std::max<long long>(max_key, rec.key);
      if (s.read_ok && max_key <= (1LL << 28) && r->adapter_.check_rows) {
        std::vector<std::vector<int>> expected(static_cast<std::size_t>(max_key));
        for (const auto& rec : s.recs)
          if (rec.avail == 0 && rec.key >= 1)
            expected[static_cast<std::size_t>(rec.key - 1)].assign(
                s.rows.begin() + rec.row_begin, s.rows.begin() + rec.row_begin + rec.row_len);
        std::string log;
        f.i64("check.rows", r->adapter_.check_rows(ctx, expected, impl_->tree->name.c_str(), log));
        if (r->adapter_.check_subtree)
          f.i64("check.subtree_avail",
                r->adapter_.check_subtree(ctx, impl_->tree->name.c_str(), log));
        if (log.size() > 4096) log.resize(4096);
        f.str("check.log", log);
      } else {
        f.i64("check.rows", -1);
      }
      mem.known = true;
      mem.state_hash = s.state_hash;
      mem.keys_hash = s.keys_hash;
      mem.subtree_hash = s.subtree_hash;
      mem.key_hash.clear();
      mem.key_hash.reserve(s.recs.size());
      for (const auto& rec : s.recs) mem.key_hash[rec.key] = rec.hash;
    }
    r->write(tag_op, f);
    ++r->ops_;
  } catch (const std::exception& e) {
    r->note(std::string("recorder error at op exit: ") + e.what());
  }
  r->in_op_ = false;
}

// --- Batch hooks ---------------------------------------------------------------------------

struct batch_record::impl {
  explicit impl(recorder* r_) : r(r_) {}
  recorder* r;
  fields f;
};

batch_record::batch_record(recorder* r, const char* hook, const char* phase) : impl_(nullptr) {
  if (!r) return;
  r->mutex_.lock();
  impl_ = new impl(r);
  impl_->f.i64("seq", static_cast<long long>(r->next_seq()));
  impl_->f.str("hook", hook);
  impl_->f.str("phase", phase);
}
batch_record::~batch_record() {
  if (!impl_) return;
  recorder* r = impl_->r;
  delete impl_;
  r->mutex_.unlock();
}
void batch_record::scalar(const char* name, long long value) {
  if (impl_) impl_->f.i64(name, value);
}
void batch_record::text(const char* name, const std::string& value) {
  if (impl_) impl_->f.str(name, value);
}
void batch_record::ints(const char* name, const std::vector<int>& v) {
  if (impl_)
    put_ints(impl_->r, impl_->f, name, v.data(), static_cast<long long>(v.size()),
             array_policy::input, impl_->r->mode_ != "full");
}
void batch_record::longs(const char* name, const std::vector<long long>& v) {
  if (!impl_) return;
  if (impl_->r->mode_ != "full" && v.size() > 4096) {
    const std::uint64_t at = impl_->r->blob(v.data(), v.size() * 8);
    impl_->f.array_ref(name, array_blob, 8, v.size(), at,
                       fnv_bytes(fnv_offset, v.data(), v.size() * 8));
  } else {
    impl_->f.array_inline_bytes(name, 8, v.data(), v.size());
  }
}
void batch_record::rows(const char* name, const std::vector<std::vector<int>>& rows) {
  if (!impl_) return;
  std::vector<int> values;
  std::vector<long long> prefix;
  prefix.reserve(rows.size());
  for (const auto& row : rows) {
    values.insert(values.end(), row.begin(), row.end());
    prefix.push_back(static_cast<long long>(values.size()));
  }
  ints((std::string(name) + ".values").c_str(), values);
  longs((std::string(name) + ".prefix").c_str(), prefix);
}
void batch_record::exception(std::exception_ptr e) {
  if (!impl_) return;
  std::string type, what;
  describe_exception(e, type, what);
  impl_->f.str("exception.type", type);
  impl_->f.str("exception.what", what);
}
void batch_record::write() {
  if (impl_) impl_->r->write(tag_batch, impl_->f);
}

}  // namespace dyng_parity::cbst_rec
