// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_trace.cpp
 * @brief The CBST trace reader and state tracker (trace format 1, ADR 0036).
 */
#include "parity/escher/cbst_trace.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <istream>
#include <set>
#include <sstream>
#include <streambuf>

namespace dyng::test::cbst_trace {
namespace {

constexpr std::uint64_t fnv_offset = 0xcbf29ce484222325ULL;
constexpr char magic[8] = {'D', 'Y', 'N', 'G', 'C', 'B', 'T', '1'};
constexpr char blob_magic[8] = {'D', 'Y', 'N', 'G', 'C', 'B', 'B', '1'};

bool little_endian() {
  const std::uint32_t one = 1;
  unsigned char b = 0;
  std::memcpy(&b, &one, 1);
  return b == 1;
}

/// A bounds-checked little-endian cursor over one record's payload.
class cursor {
 public:
  cursor(const std::vector<unsigned char>& b, std::size_t pos) : b_(b), pos_(pos) {}
  template <class T>
  T get() {
    need(sizeof(T));
    T v{};
    std::memcpy(&v, b_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return v;
  }
  std::string bytes(std::size_t n) {
    need(n);
    std::string s(reinterpret_cast<const char*>(b_.data() + pos_), n);
    pos_ += n;
    return s;
  }
  template <class T>
  std::vector<T> values(std::uint64_t n) {
    if (n > (b_.size() - pos_) / sizeof(T))
      throw trace_error("CBST trace: array overruns its record");
    std::vector<T> out(static_cast<std::size_t>(n));
    if (n > 0) std::memcpy(out.data(), b_.data() + pos_, static_cast<std::size_t>(n) * sizeof(T));
    pos_ += static_cast<std::size_t>(n) * sizeof(T);
    return out;
  }

 private:
  void need(std::size_t n) const {
    if (n > b_.size() - pos_) throw trace_error("CBST trace: field overruns its record");
  }
  const std::vector<unsigned char>& b_;
  std::size_t pos_;
};

std::uint64_t fnv_u64(std::uint64_t h, std::uint64_t v) {
  unsigned char b[8];
  for (int i = 0; i < 8; ++i) b[i] = static_cast<unsigned char>(v >> (8 * i));
  return fnv1a64(b, 8, h);
}

std::uint64_t ints_hash(const std::vector<std::int32_t>& v) {
  return fnv1a64(v.data(), v.size() * sizeof(std::int32_t));
}

std::uint64_t record_hash(const key_state& k) {
  const std::int32_t f[11] = {k.key,       k.avail,         k.length,    k.size,
                              k.occupancy, k.tail_capacity, k.value,     k.tail_base,
                              k.segments,  k.row_len,       k.terminated};
  return fnv_u64(fnv1a64(f, sizeof f), k.row_hash);
}

}  // namespace

const char* to_string(record_kind kind) {
  switch (kind) {
    case record_kind::header:
      return "header";
    case record_kind::tree:
      return "tree";
    case record_kind::op:
      return "op";
    case record_kind::event:
      return "event";
    case record_kind::batch:
      return "batch";
    case record_kind::note:
      return "note";
    case record_kind::end:
      return "end";
  }
  return "?";
}

std::uint64_t fnv1a64(const void* data, std::size_t n, std::uint64_t h) {
  const auto* p = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 0x100000001b3ULL;
  }
  return h;
}

const field* record::find(const std::string& name) const {
  for (const auto& f : fields)
    if (f.first == name) return &f.second;
  return nullptr;
}

std::int64_t record::integer(const std::string& name, std::int64_t fallback) const {
  const field* f = find(name);
  return f && f->kind == field::type::integer ? f->integer : fallback;
}

std::string record::text(const std::string& name) const {
  const field* f = find(name);
  return f && f->kind == field::type::text ? f->text : std::string();
}

const array_field& record::array(const std::string& name) const {
  const field* f = find(name);
  if (f == nullptr || f->kind != field::type::array)
    throw trace_error("CBST trace: no array field " + name);
  return f->array;
}

struct reader::impl {
  std::unique_ptr<std::ifstream> side;
  std::vector<unsigned char> buf;
};

reader::reader(std::istream& in) : in_(&in), impl_(std::make_unique<impl>()) {
  if (!little_endian()) throw trace_error("CBST trace: the reader needs a little-endian host");
  char head[16];
  if (!in_->read(head, 16) || std::memcmp(head, magic, 8) != 0)
    throw trace_error("CBST trace: not a trace (magic)");
  std::uint32_t version = 0;
  std::memcpy(&version, head + 8, 4);
  if (version != format_version)
    throw trace_error("CBST trace: format " + std::to_string(version) + ", the reader reads " +
                      std::to_string(format_version));
}

reader::~reader() = default;

std::unique_ptr<reader> reader::open(const std::string& path) {
  auto file = std::make_unique<std::ifstream>(path, std::ios::binary);
  if (!*file) throw trace_error("CBST trace: cannot open " + path);
  auto r = std::make_unique<reader>(*file);
  r->owned_ = std::move(file);
  std::string stem = path;
  if (stem.size() > 4 && stem.compare(stem.size() - 4, 4, ".cbt") == 0) {
    stem.resize(stem.size() - 4);
    std::ifstream probe(stem + ".blob", std::ios::binary);
    if (probe) r->set_side_file(stem + ".blob");
  }
  return r;
}

void reader::set_side_file(const std::string& path) {
  auto f = std::make_unique<std::ifstream>(path, std::ios::binary);
  char head[16];
  if (!*f || !f->read(head, 16) || std::memcmp(head, blob_magic, 8) != 0)
    throw trace_error("CBST trace: bad side file " + path);
  impl_->side = std::move(f);
}

bool reader::next(record& out) {
  unsigned char head[12];
  in_->read(reinterpret_cast<char*>(head), 12);
  if (in_->gcount() == 0) return false;
  if (in_->gcount() != 12) throw trace_error("CBST trace: truncated record header");
  std::uint32_t tag = 0;
  std::uint64_t len = 0;
  std::memcpy(&tag, head, 4);
  std::memcpy(&len, head + 4, 8);
  if (tag < 1 || tag > 7)
    throw trace_error("CBST trace: unknown record tag " + std::to_string(tag));
  if (len > (std::uint64_t{1} << 40)) throw trace_error("CBST trace: record too long");
  auto& b = impl_->buf;
  b.resize(static_cast<std::size_t>(len));
  if (!in_->read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(len)))
    throw trace_error("CBST trace: truncated record");
  out.kind = static_cast<record_kind>(tag);
  out.fields.clear();
  cursor c(b, 0);
  const auto count = c.get<std::uint32_t>();
  for (std::uint32_t i = 0; i < count; ++i) {
    const auto type = c.get<std::uint8_t>();
    const auto name_len = c.get<std::uint16_t>();
    std::string name = c.bytes(name_len);
    field f;
    switch (type) {
      case 1:
        f.kind = field::type::integer;
        f.integer = c.get<std::int64_t>();
        break;
      case 2:
        f.kind = field::type::text;
        f.text = c.bytes(c.get<std::uint32_t>());
        break;
      case 3: {
        f.kind = field::type::array;
        const auto kind = c.get<std::uint8_t>();
        const auto elem = c.get<std::uint8_t>();
        if (elem != 4 && elem != 8) throw trace_error("CBST trace: bad array element size");
        array_field& a = f.array;
        a.elem_bytes = elem;
        a.count = c.get<std::uint64_t>();
        if (kind == 0) {
          a.kind = array_field::storage::inline_data;
          if (elem == 4)
            a.i32 = c.values<std::int32_t>(a.count);
          else
            a.i64 = c.values<std::int64_t>(a.count);
        } else if (kind == 1) {
          a.kind = array_field::storage::side_file;
          a.offset = c.get<std::uint64_t>();
          a.hash = c.get<std::uint64_t>();
        } else if (kind == 2) {
          a.kind = array_field::storage::hash_only;
          a.hash = c.get<std::uint64_t>();
        } else if (kind == 3) {
          a.kind = array_field::storage::null;
        } else {
          throw trace_error("CBST trace: bad array storage");
        }
        break;
      }
      case 4: {
        f.kind = field::type::texts;
        const auto n = c.get<std::uint32_t>();
        for (std::uint32_t j = 0; j < n; ++j) f.texts.push_back(c.bytes(c.get<std::uint32_t>()));
        break;
      }
      default:
        throw trace_error("CBST trace: unknown field type of " + name);
    }
    out.fields.emplace_back(std::move(name), std::move(f));
  }
  return true;
}

std::vector<std::int32_t> reader::ints(const array_field& a) {
  if (a.elem_bytes != 4) throw trace_error("CBST trace: not an int32 array");
  switch (a.kind) {
    case array_field::storage::inline_data:
      return a.i32;
    case array_field::storage::null:
      return {};
    case array_field::storage::hash_only:
      throw trace_error("CBST trace: a hash-only array has no values");
    case array_field::storage::side_file:
      break;
  }
  if (!impl_->side) throw trace_error("CBST trace: the side file is missing");
  std::vector<std::int32_t> out(static_cast<std::size_t>(a.count));
  impl_->side->seekg(static_cast<std::streamoff>(a.offset));
  if (!impl_->side->read(reinterpret_cast<char*>(out.data()),
                         static_cast<std::streamsize>(a.count * 4)))
    throw trace_error("CBST trace: side file too short");
  if (ints_hash(out) != a.hash) throw trace_error("CBST trace: side-file array hash differs");
  return out;
}

std::vector<std::int64_t> reader::longs(const array_field& a) {
  if (a.elem_bytes != 8) throw trace_error("CBST trace: not an int64 array");
  switch (a.kind) {
    case array_field::storage::inline_data:
      return a.i64;
    case array_field::storage::null:
      return {};
    case array_field::storage::hash_only:
      throw trace_error("CBST trace: a hash-only array has no values");
    case array_field::storage::side_file:
      break;
  }
  if (!impl_->side) throw trace_error("CBST trace: the side file is missing");
  std::vector<std::int64_t> out(static_cast<std::size_t>(a.count));
  impl_->side->seekg(static_cast<std::streamoff>(a.offset));
  if (!impl_->side->read(reinterpret_cast<char*>(out.data()),
                         static_cast<std::streamsize>(a.count * 8)))
    throw trace_error("CBST trace: side file too short");
  if (fnv1a64(out.data(), out.size() * 8) != a.hash)
    throw trace_error("CBST trace: side-file array hash differs");
  return out;
}

const tree_state* state_tracker::tree(std::uint32_t id) const {
  const auto it = trees_.find(id);
  return it == trees_.end() ? nullptr : &it->second;
}

std::vector<std::string> state_tracker::apply(const record& r, reader& source) {
  std::vector<std::string> problems;
  if (r.kind == record_kind::tree) {
    tree_state& t = trees_[static_cast<std::uint32_t>(r.integer("tree"))];
    t.id = static_cast<std::uint32_t>(r.integer("tree"));
    t.name = r.text("dataset_name");
    t.payload_capacity = r.integer("payload_capacity");
    t.alignment = r.integer("alignment");
    return problems;
  }
  if (r.kind != record_kind::op || !r.has("digest.mode")) return problems;
  tree_state& t = trees_[static_cast<std::uint32_t>(r.integer("tree"))];
  t.id = static_cast<std::uint32_t>(r.integer("tree"));
  if (r.integer("rec.duplicate_keys") > 0) {
    // Two nodes with the same key cannot be tracked by key (the recorder writes a full digest).
    problems.emplace_back("duplicate keys in the tree");
  }
  const bool full = r.integer("digest.mode") == 0;
  if (r.has("state.keys")) t.keys = source.ints(r.array("state.keys"));
  if (r.has("state.subtree_avail")) t.subtree_avail = source.ints(r.array("state.subtree_avail"));
  static const char* const columns[11] = {"key.key",     "key.avail",     "key.length",
                                          "key.size",    "key.occupancy", "key.tail_capacity",
                                          "key.value",   "key.tail_base", "key.segments",
                                          "key.row_len", "key.terminated"};
  std::vector<std::int32_t> col[11];
  for (int i = 0; i < 11; ++i) col[i] = source.ints(r.array(columns[i]));
  const std::vector<std::int64_t> row_hash = source.longs(r.array("key.row_hash"));
  const array_field& rows_field = r.array("key.rows");
  const bool rows_known = rows_field.kind != array_field::storage::hash_only;
  const std::vector<std::int32_t> rows =
      rows_known ? source.ints(rows_field) : std::vector<std::int32_t>();
  if (full) t.by_key.clear();
  std::size_t at = 0;
  for (std::size_t i = 0; i < col[0].size(); ++i) {
    key_state k;
    k.key = col[0][i];
    k.avail = col[1][i];
    k.length = col[2][i];
    k.size = col[3][i];
    k.occupancy = col[4][i];
    k.tail_capacity = col[5][i];
    k.value = col[6][i];
    k.tail_base = col[7][i];
    k.segments = col[8][i];
    k.row_len = col[9][i];
    k.terminated = col[10][i];
    k.row_hash = static_cast<std::uint64_t>(row_hash[i]);
    k.row_known = rows_known;
    const auto n = static_cast<std::size_t>(std::max(0, k.row_len));
    if (rows_known) {
      if (at + n > rows.size()) throw trace_error("CBST trace: digest rows too short");
      k.row.assign(rows.begin() + static_cast<std::ptrdiff_t>(at),
                   rows.begin() + static_cast<std::ptrdiff_t>(at + n));
      if (ints_hash(k.row) != k.row_hash)
        problems.push_back("key " + std::to_string(k.key) + ": row hash");
    }
    at += n;
    t.by_key[k.key] = std::move(k);
  }
  {
    const std::set<std::int32_t> live(t.keys.begin(), t.keys.end());
    for (auto it = t.by_key.begin(); it != t.by_key.end();)
      it = live.count(it->first) ? std::next(it) : t.by_key.erase(it);
  }
  t.fixed_size = r.integer("after.fixed_size");
  t.num_records = r.integer("after.num_records");
  t.initial_payload_size = r.integer("after.initial_payload_size");
  t.alignment = r.integer("after.alignment");
  // Every recorded hash must match the rebuilt state.
  const auto keys_hash = ints_hash(t.keys);
  const auto subtree_hash = ints_hash(t.subtree_avail);
  if (keys_hash != static_cast<std::uint64_t>(r.integer("state.keys_hash")))
    problems.emplace_back("keys hash");
  if (subtree_hash != static_cast<std::uint64_t>(r.integer("state.subtree_avail_hash")))
    problems.emplace_back("subtree_avail hash");
  std::uint64_t h = fnv_offset;
  for (std::int64_t x : {t.num_records, t.fixed_size, t.initial_payload_size, t.alignment})
    h = fnv_u64(h, static_cast<std::uint64_t>(x));
  h = fnv_u64(h, keys_hash);
  h = fnv_u64(h, subtree_hash);
  std::vector<std::int64_t> key_hashes;
  if (r.has("state.key_hashes")) key_hashes = source.longs(r.array("state.key_hashes"));
  for (std::size_t i = 0; i < t.keys.size(); ++i) {
    const auto it = t.by_key.find(t.keys[i]);
    if (it == t.by_key.end()) {
      problems.push_back("key " + std::to_string(t.keys[i]) + ": no record");
      return problems;
    }
    const std::uint64_t kh = record_hash(it->second);
    if (!key_hashes.empty() && static_cast<std::uint64_t>(key_hashes[i]) != kh)
      problems.push_back("key " + std::to_string(t.keys[i]) + ": record hash");
    h = fnv_u64(h, kh);
  }
  if (h != static_cast<std::uint64_t>(r.integer("state.hash"))) problems.emplace_back("state hash");
  return problems;
}

namespace {

std::string quote_for_shell(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  return out + "'";
}

}  // namespace

/// An input stream buffer over a pipe (popen).
struct trace_source::pipe : std::streambuf {
  explicit pipe(FILE* f) : file(f) {}
  ~pipe() override {
    if (file) pclose(file);
  }
  pipe(const pipe&) = delete;
  pipe& operator=(const pipe&) = delete;
  int_type underflow() override {
    if (gptr() < egptr()) return traits_type::to_int_type(*gptr());
    if (file == nullptr) return traits_type::eof();
    const std::size_t n = std::fread(buf, 1, sizeof buf, file);
    if (n == 0) return traits_type::eof();
    setg(buf, buf, buf + n);
    return traits_type::to_int_type(*gptr());
  }
  FILE* file;
  char buf[1 << 16];
};

trace_source::trace_source(const std::string& path, const std::string& tmp_dir) {
  namespace fs = std::filesystem;
  if (path.size() < 4 || path.compare(path.size() - 4, 4, ".zst") != 0) {
    reader_ = reader::open(path);
    return;
  }
  if (!fs::is_regular_file(path)) throw trace_error("CBST trace: cannot open " + path);
  FILE* f = popen(("zstd -q -dc --long=31 " + quote_for_shell(path)).c_str(), "r");
  if (f == nullptr) throw trace_error("CBST trace: cannot run zstd for " + path);
  pipe_ = std::make_unique<pipe>(f);
  stream_ = std::make_unique<std::istream>(pipe_.get());
  reader_ = std::make_unique<cbst_trace::reader>(*stream_);
  const fs::path blob = fs::path(path).parent_path() / "trace.blob.zst";
  if (fs::exists(blob)) {
    static int counter = 0;
    blob_tmp_ = (fs::path(tmp_dir) / ("cbst_replay_" + std::to_string(getpid()) + "_" +
                                      std::to_string(++counter) + ".blob"))
                    .string();
    const std::string cmd = "zstd -q -d --long=31 -f " + quote_for_shell(blob.string()) + " -o " +
                            quote_for_shell(blob_tmp_);
    if (std::system(cmd.c_str()) != 0)
      throw trace_error("CBST trace: cannot decompress " + blob.string());
    reader_->set_side_file(blob_tmp_);
  }
}

trace_source::~trace_source() {
  reader_.reset();
  stream_.reset();
  pipe_.reset();
  if (!blob_tmp_.empty()) {
    std::error_code ec;
    std::filesystem::remove(blob_tmp_, ec);
  }
}

}  // namespace dyng::test::cbst_trace
