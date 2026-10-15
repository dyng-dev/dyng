// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file batch_binary.cpp
 * @brief The dynG binary batch format `.dgb` (PLAN Section 5.7; ADR 0045;
 *        docs/api/file_formats.md): batch_writer, batch_reader and the whole-file helpers.
 *
 * Layout (little-endian): a 64-byte header (magic `DYNGBAT1`, format version, header size,
 * container kind, vertex / weight / id dtypes, K, flags, batch count, reserved zeros, CRC32 of the
 * first 60 bytes), then one record per batch: its payload length (u64), the CRC32 of the payload
 * (u32), 4 zero bytes, and the payload: u64 counts followed by the arrays, each starting at a
 * multiple of 8 bytes (zero padding). Every size field is checked against the rest of the file
 * before anything is allocated, so a file of N bytes makes the reader allocate O(N) bytes at most.
 */
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "hypergraph/instantiate.hpp"
#include "io/batch_checks.hpp"
#include "io/io_instantiate.hpp"
#include "util/allocation.hpp"
#include "util/crc32.hpp"
#include "util/parser.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/io/batch_io.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the .dgb reader and writer assume a little-endian host"
#endif

namespace dyng::io {

std::string_view to_string(batch_dtype dtype) noexcept {
  switch (dtype) {
    case batch_dtype::none:
      return "none";
    case batch_dtype::int32:
      return "int32";
    case batch_dtype::int64:
      return "int64";
  }
  return "unknown";
}

namespace {

constexpr std::array<char, 8> magic = {'D', 'Y', 'N', 'G', 'B', 'A', 'T', '1'};
constexpr std::uint32_t current_version = 1;
constexpr std::size_t header_bytes = 64;
constexpr std::size_t record_header_bytes = 16;
constexpr std::size_t header_crc_offset = 60;

/// The dtype of a vertex or weight type.
template <typename value_t>
constexpr batch_dtype dtype_of() noexcept {
  if constexpr (is_unweighted_v<value_t>) {
    return batch_dtype::none;
  } else {
    static_assert(std::is_integral_v<value_t> && std::is_signed_v<value_t> &&
                      (sizeof(value_t) == 4 || sizeof(value_t) == 8),
                  ".dgb stores 32- or 64-bit signed integers");
    return sizeof(value_t) == 4 ? batch_dtype::int32 : batch_dtype::int64;
  }
}

constexpr std::size_t dtype_bytes(batch_dtype t) noexcept {
  return t == batch_dtype::int32 ? 4 : t == batch_dtype::int64 ? 8 : 0;
}

constexpr std::size_t pad8(std::size_t n) noexcept {
  return (n + 7) / 8 * 8;
}

void put_u32(unsigned char* p, std::uint32_t v) noexcept {
  std::memcpy(p, &v, 4);
}
void put_u64(unsigned char* p, std::uint64_t v) noexcept {
  std::memcpy(p, &v, 8);
}
std::uint32_t get_u32(const unsigned char* p) noexcept {
  std::uint32_t v = 0;
  std::memcpy(&v, p, 4);
  return v;
}
std::uint64_t get_u64(const unsigned char* p) noexcept {
  std::uint64_t v = 0;
  std::memcpy(&v, p, 8);
  return v;
}

/// The header's 64 bytes.
std::array<unsigned char, header_bytes> encode_header(const batch_file_header& h) noexcept {
  std::array<unsigned char, header_bytes> b{};
  std::memcpy(b.data(), magic.data(), magic.size());
  put_u32(b.data() + 8, h.format_version);
  put_u32(b.data() + 12, static_cast<std::uint32_t>(header_bytes));
  b[16] = static_cast<unsigned char>(h.kind);
  b[17] = static_cast<unsigned char>(h.vertex_type);
  b[18] = static_cast<unsigned char>(h.weight_type);
  b[19] = static_cast<unsigned char>(h.kind == container_kind::hypergraph ? batch_dtype::int64
                                                                          : batch_dtype::none);
  put_u32(b.data() + 20, static_cast<std::uint32_t>(h.num_weights));
  put_u64(b.data() + 24, h.flags);
  put_u64(b.data() + 32, static_cast<std::uint64_t>(h.num_batches));
  put_u32(b.data() + header_crc_offset, detail::crc32(b.data(), header_crc_offset));
  return b;
}

/// Whether a header describes a file this library can write and read (empty: yes).
std::string header_problem(const batch_file_header& h) {
  if (h.format_version != current_version) {
    return "format version " + std::to_string(h.format_version) + " (this library reads and " +
           "writes version " + std::to_string(current_version) + ")";
  }
  if (h.kind != container_kind::graph && h.kind != container_kind::hypergraph) {
    return "unknown container kind " + std::to_string(static_cast<int>(h.kind));
  }
  if (h.vertex_type != batch_dtype::int32 && h.vertex_type != batch_dtype::int64) {
    return "vertex type " + std::to_string(static_cast<int>(h.vertex_type)) +
           " (int32 = 1 or int64 = 2)";
  }
  if (h.weight_type != batch_dtype::none && h.weight_type != batch_dtype::int32 &&
      h.weight_type != batch_dtype::int64) {
    return "weight type " + std::to_string(static_cast<int>(h.weight_type)) +
           " (none = 0, int32 = 1 or int64 = 2)";
  }
  if (h.num_weights < 0 || h.num_weights > 64 ||
      (h.weight_type == batch_dtype::none && h.num_weights != 0) ||
      (h.weight_type != batch_dtype::none && h.num_weights == 0)) {
    return "K = " + std::to_string(h.num_weights) + " weights per insertion with weight type " +
           std::string(to_string(h.weight_type)) +
           " (a .dgb file holds 1 to 64 weights per insertion with a weight type, 0 without)";
  }
  if (h.kind == container_kind::hypergraph &&
      h.num_weights != (h.weight_type == batch_dtype::none ? 0 : 1)) {
    return "K = " + std::to_string(h.num_weights) +
           " for hypergraph batches (one weight per weighted hyperedge)";
  }
  if ((h.flags & ~batch_file_flags::known) != 0) {
    return "unknown flags " + std::to_string(h.flags & ~batch_file_flags::known) +
           " (written by a newer dynG?)";
  }
  if (h.kind == container_kind::graph && h.flags != 0) {
    return "flags " + std::to_string(h.flags) + " in a file of graph batches";
  }
  return {};
}

/// A typed read of element i of an array of dtype `t` at `p` (host memory).
std::int64_t element(const unsigned char* p, batch_dtype t, std::size_t i) noexcept {
  if (t == batch_dtype::int32) {
    std::int32_t v = 0;
    std::memcpy(&v, p + 4 * i, 4);
    return v;
  }
  std::int64_t v = 0;
  std::memcpy(&v, p + 8 * i, 8);
  return v;
}

/// One array of a record: its byte offset in the payload, count and dtype.
struct array_slot {
  std::size_t offset = 0;
  std::size_t count = 0;
  batch_dtype type = batch_dtype::none;
};

/// The arrays of a record, in payload order (graph: delete_src, delete_dst, insert_src,
/// insert_dst, weights; hypergraph: delete ids, insert offsets, members, weights, deleted
/// offsets, deleted members, expected ids).
struct record_layout {
  std::int64_t batch_id = -1;
  std::array<array_slot, 7> slots{};
};

enum graph_slot : std::size_t { g_del_src, g_del_dst, g_ins_src, g_ins_dst, g_weights };
enum hyper_slot : std::size_t {
  h_del_ids,
  h_offsets,
  h_members,
  h_weights,
  h_del_offsets,
  h_del_members,
  h_expected
};

}  // namespace

// ---------------------------------------------------------------------------- headers -------

template <typename vertex_t, typename weight_t>
batch_file_header edge_batch_file_header(int num_weights) {
  DYNG_EXPECTS(num_weights >= 0, "edge_batch_file_header: K must be >= 0, got ", num_weights);
  DYNG_EXPECTS(!is_unweighted_v<weight_t> || num_weights == 0,
               "edge_batch_file_header: an unweighted batch has K = 0, got ", num_weights);
  batch_file_header h;
  h.kind = container_kind::graph;
  h.vertex_type = dtype_of<vertex_t>();
  h.weight_type = num_weights == 0 ? batch_dtype::none : dtype_of<weight_t>();
  h.num_weights = num_weights;
  return h;
}

template <typename vertex_t, typename weight_t>
batch_file_header hyperedge_batch_file_header(std::uint64_t flags) {
  DYNG_EXPECTS((flags & ~batch_file_flags::known) == 0,
               "hyperedge_batch_file_header: unknown flags ", flags & ~batch_file_flags::known);
  batch_file_header h;
  h.kind = container_kind::hypergraph;
  h.vertex_type = dtype_of<vertex_t>();
  h.weight_type = dtype_of<weight_t>();
  h.num_weights = is_unweighted_v<weight_t> ? 0 : 1;
  h.flags = flags;
  return h;
}

// ---------------------------------------------------------------------------- writer --------

struct batch_writer::impl {
  std::string path;
  std::FILE* file = nullptr;
  batch_file_header header;
  std::int64_t count = 0;
  bool failed = false;
  std::vector<unsigned char> payload;

  ~impl() {
    if (file != nullptr) std::fclose(file);
  }

  void write_bytes(const void* data, std::size_t n) {
    if (n > 0 && std::fwrite(data, 1, n, file) != n) failed = true;
  }

  void check_open() const {
    if (file == nullptr) {
      throw invalid_argument_error("dyng: batch_writer: the file " + path + " is closed");
    }
  }

  // Append one array of `count` values of the batch's type as values of dtype `t`.
  template <typename value_t>
  void put_array(std::size_t& off, const value_t* values, std::size_t count, batch_dtype t) {
    const std::size_t width = dtype_bytes(t);
    for (std::size_t i = 0; i < count; ++i) {
      if (width == 4) {
        const auto v = static_cast<std::int32_t>(values[i]);
        std::memcpy(payload.data() + off + 4 * i, &v, 4);
      } else {
        const auto v = static_cast<std::int64_t>(values[i]);
        std::memcpy(payload.data() + off + 8 * i, &v, 8);
      }
    }
    off += pad8(count * width);
  }

  void emit() {
    const std::array<unsigned char, record_header_bytes> rec = [&] {
      std::array<unsigned char, record_header_bytes> r{};
      put_u64(r.data(), payload.size());
      put_u32(r.data() + 8, detail::crc32(payload.data(), payload.size()));
      return r;
    }();
    write_bytes(rec.data(), rec.size());
    write_bytes(payload.data(), payload.size());
    if (failed) detail::throw_io_error(path, 0, 0, "write error");
    ++count;
  }
};

batch_writer::batch_writer(const std::string& path, const batch_file_header& header) try
    : impl_(std::make_unique<impl>()) {
  const std::string problem = header_problem(header);
  DYNG_EXPECTS(problem.empty(), "batch_writer: ", problem);
  impl_->path = path;
  impl_->header = header;
  impl_->header.num_batches = 0;
  const std::filesystem::path parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);  // on failure fopen fails below
  }
  impl_->file = std::fopen(path.c_str(), "wb");
  if (impl_->file == nullptr) detail::throw_io_error(path, 0, 0, "cannot open file for writing");
  const auto bytes = encode_header(impl_->header);
  impl_->write_bytes(bytes.data(), bytes.size());
  if (impl_->failed) detail::throw_io_error(path, 0, 0, "write error");
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::batch_writer (", path, ")")

batch_writer::~batch_writer() = default;
batch_writer::batch_writer(batch_writer&& other) noexcept = default;
batch_writer& batch_writer::operator=(batch_writer&& other) noexcept = default;

std::int64_t batch_writer::num_batches() const noexcept {
  return impl_ ? impl_->count : 0;
}

void batch_writer::close() {
  DYNG_EXPECTS(impl_ != nullptr, "batch_writer::close: the writer was moved from");
  if (impl_->file == nullptr) return;
  impl_->header.num_batches = impl_->count;
  const auto bytes = encode_header(impl_->header);
  if (std::fseek(impl_->file, 0, SEEK_SET) != 0) impl_->failed = true;
  impl_->write_bytes(bytes.data(), bytes.size());
  if (std::fclose(impl_->file) != 0) impl_->failed = true;
  impl_->file = nullptr;
  if (impl_->failed) detail::throw_io_error(impl_->path, 0, 0, "write error");
}

template <typename vertex_t, typename weight_t>
void batch_writer::write(const edge_batch_view<vertex_t, weight_t>& batch) try {
  DYNG_EXPECTS(impl_ != nullptr, "batch_writer::write: the writer was moved from");
  impl_->check_open();
  const batch_file_header& h = impl_->header;
  DYNG_EXPECTS(h.kind == container_kind::graph && h.vertex_type == dtype_of<vertex_t>() &&
                   (h.weight_type == batch_dtype::none || h.weight_type == dtype_of<weight_t>()),
               "batch_writer::write: the file ", impl_->path, " holds ", to_string(h.kind),
               " batches with ", to_string(h.vertex_type), " vertices and ",
               to_string(h.weight_type), " weights, not this batch's types");
  const auto index = static_cast<std::size_t>(impl_->count);
  detail::check_edge_batch_arrays(batch, "batch_writer::write", index);
  const std::size_t ni = batch.insert_src.size();
  const std::size_t nd = batch.delete_src.size();
  DYNG_EXPECTS(ni == 0 || batch.num_weights == h.num_weights, "batch_writer::write: batch ", index,
               " has K = ", batch.num_weights, " weights per insertion, the file ", h.num_weights);
  const std::size_t vb = dtype_bytes(h.vertex_type);
  const std::size_t wb = dtype_bytes(h.weight_type);
  const auto k = static_cast<std::size_t>(h.num_weights);
  impl_->payload.assign(24 + 2 * pad8(nd * vb) + 2 * pad8(ni * vb) + pad8(ni * k * wb), 0);
  put_u64(impl_->payload.data(), static_cast<std::uint64_t>(impl_->count));
  put_u64(impl_->payload.data() + 8, nd);
  put_u64(impl_->payload.data() + 16, ni);
  std::size_t off = 24;
  impl_->put_array(off, batch.delete_src.data(), nd, h.vertex_type);
  impl_->put_array(off, batch.delete_dst.data(), nd, h.vertex_type);
  impl_->put_array(off, batch.insert_src.data(), ni, h.vertex_type);
  impl_->put_array(off, batch.insert_dst.data(), ni, h.vertex_type);
  if constexpr (!is_unweighted_v<weight_t>) {
    if (k > 0) impl_->put_array(off, batch.insert_weights.data(), ni * k, h.weight_type);
  }
  impl_->emit();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::batch_writer::write")

template <typename vertex_t, typename weight_t>
void batch_writer::write(const hyperedge_batch_view<vertex_t, weight_t>& batch,
                         const hyperedge_batch_annotations<vertex_t>& annotations) try {
  DYNG_EXPECTS(impl_ != nullptr, "batch_writer::write: the writer was moved from");
  impl_->check_open();
  const batch_file_header& h = impl_->header;
  DYNG_EXPECTS(h.kind == container_kind::hypergraph && h.vertex_type == dtype_of<vertex_t>() &&
                   h.weight_type == dtype_of<weight_t>(),
               "batch_writer::write: the file ", impl_->path, " holds ", to_string(h.kind),
               " batches with ", to_string(h.vertex_type), " vertices and ",
               to_string(h.weight_type), " weights, not this batch's types");
  const auto index = static_cast<std::size_t>(impl_->count);
  if (!batch.delete_incidence_hyperedge.empty() || !batch.delete_incidence_vertex.empty() ||
      !batch.insert_incidence_hyperedge.empty() || !batch.insert_incidence_vertex.empty()) {
    throw not_supported_error(detail::concat_message(
        "dyng: batch_writer::write: batch ", index,
        " has incidence edits, which the .dgb files of this release do not hold (planned for "
        "0.4)"));
  }
  detail::check_hyperedge_batch_arrays(batch, "batch_writer::write", index);
  const std::size_t nd = batch.num_deletions();
  const std::size_t ni = batch.num_insertions();
  const std::size_t nm = batch.insert_members.size();
  const bool with_deleted = (h.flags & batch_file_flags::deleted_members) != 0;
  const bool with_expected = (h.flags & batch_file_flags::expected_ids) != 0;
  const auto host = [](const auto& v) { return detail::host_readable(v); };
  DYNG_EXPECTS(host(annotations.deleted_offsets) && host(annotations.deleted_members) &&
                   host(annotations.expected_ids),
               "batch_writer::write: the annotations of batch ", index,
               " must be in host-accessible memory");
  std::size_t ndm = 0;
  if (with_deleted) {
    const auto& off = annotations.deleted_offsets;
    DYNG_EXPECTS(off.size() == nd + 1 && off[0] == 0 &&
                     off[nd] == static_cast<std::int64_t>(annotations.deleted_members.size()),
                 "batch_writer::write: batch ", index, " needs num_deletions() + 1 = ", nd + 1,
                 " deleted-member offsets from 0 to the member count (flag deleted_members)");
    for (std::size_t j = 0; j < nd; ++j) {
      DYNG_EXPECTS(off[j] < off[j + 1], "batch_writer::write: deleted hyperedge ", j, " of batch ",
                   index, " has no members (or decreasing offsets)");
    }
    for (std::size_t j = 0; j < annotations.deleted_members.size(); ++j) {
      DYNG_EXPECTS(annotations.deleted_members[j] >= 0, "batch_writer::write: batch ", index,
                   " has a negative deleted member");
    }
    ndm = annotations.deleted_members.size();
  } else {
    DYNG_EXPECTS(annotations.deleted_offsets.empty() && annotations.deleted_members.empty(),
                 "batch_writer::write: deleted member sets given, but the file has no flag "
                 "deleted_members");
  }
  if (with_expected) {
    DYNG_EXPECTS(annotations.expected_ids.size() == ni, "batch_writer::write: batch ", index,
                 " needs one expected id per insertion (flag expected_ids)");
    for (std::size_t i = 0; i < ni; ++i) {
      DYNG_EXPECTS(annotations.expected_ids[i] >= 0, "batch_writer::write: batch ", index,
                   " has a negative expected id");
    }
  } else {
    DYNG_EXPECTS(annotations.expected_ids.empty(),
                 "batch_writer::write: expected ids given, but the file has no flag expected_ids");
  }
  const std::size_t vb = dtype_bytes(h.vertex_type);
  const std::size_t wb = dtype_bytes(h.weight_type);
  std::size_t total = 40 + 8 * nd + 8 * (ni + 1) + pad8(nm * vb) + pad8(ni * wb);
  if (with_deleted) total += 8 * (nd + 1) + pad8(ndm * vb);
  if (with_expected) total += 8 * ni;
  impl_->payload.assign(total, 0);
  unsigned char* p = impl_->payload.data();
  put_u64(p, static_cast<std::uint64_t>(impl_->count));
  put_u64(p + 8, nd);
  put_u64(p + 16, ni);
  put_u64(p + 24, nm);
  put_u64(p + 32, ndm);
  std::size_t off = 40;
  impl_->put_array(off, batch.delete_hyperedges.data(), nd, batch_dtype::int64);
  if (ni > 0) {
    impl_->put_array(off, batch.insert_offsets.data(), ni + 1, batch_dtype::int64);
  } else {
    off += 8;  // the single offset 0
  }
  impl_->put_array(off, batch.insert_members.data(), nm, h.vertex_type);
  if constexpr (!is_unweighted_v<weight_t>) {
    impl_->put_array(off, batch.insert_weights.data(), ni, h.weight_type);
  }
  if (with_deleted) {
    impl_->put_array(off, annotations.deleted_offsets.data(), nd + 1, batch_dtype::int64);
    impl_->put_array(off, annotations.deleted_members.data(), ndm, h.vertex_type);
  }
  if (with_expected) {
    impl_->put_array(off, annotations.expected_ids.data(), ni, batch_dtype::int64);
  }
  impl_->emit();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::batch_writer::write")

// ---------------------------------------------------------------------------- reader --------

struct batch_reader::impl {
  std::string path;
  batch_reader_options options;
  std::FILE* file = nullptr;
  std::uint64_t file_size = 0;
  std::uint64_t pos = 0;  // byte offset of the next record
  batch_file_header header;
  std::int64_t id_max = 0;  // the largest vertex id accepted
  std::int64_t records = 0;
  record_layout current;
  bool has_current = false;
  // Host staging: the payload of the current record (8-byte aligned).
  std::vector<std::uint64_t> host_payload;
  // CUDA staging: two pinned buffers used in turn, the device copy of the current record and a
  // fence per pinned buffer (its copy to the device).
  bool device = false;
  std::optional<resources> res;
  std::optional<memory_resource_ref> pinned_mr;
  std::array<void*, 2> pinned{nullptr, nullptr};
  std::array<std::size_t, 2> pinned_bytes{0, 0};
  std::array<detail::cuda_stream_fence, 2> fences;
  int turn = 0;
  buffer<std::uint64_t> device_payload;
  const unsigned char* base = nullptr;  // the current payload (host or device)
  // Fail-stop (M8b review): after an error of next() the stream is past the damaged record while
  // pos and records still describe it, so every later call throws.
  bool failed = false;

  impl() = default;
  impl(const impl&) = delete;
  impl& operator=(const impl&) = delete;
  impl(impl&&) = delete;
  impl& operator=(impl&&) = delete;
  ~impl() {
    if (file != nullptr) std::fclose(file);
    for (int i = 0; i < 2; ++i) {
      if (pinned[static_cast<std::size_t>(i)] != nullptr) {
        fences[static_cast<std::size_t>(i)].synchronize();
        pinned_mr->deallocate_sync(pinned[static_cast<std::size_t>(i)],
                                   pinned_bytes[static_cast<std::size_t>(i)], 8);
      }
    }
  }

  [[noreturn]] void fail(const std::string& message) const {
    detail::throw_io_error(path, 0, 0, message);
  }
  [[noreturn]] void fail_record(std::uint64_t at, const std::string& message) const {
    fail("batch record " + std::to_string(records) + " at byte " + std::to_string(at) + ": " +
         message);
  }

  void read_exact(void* out, std::size_t n, std::uint64_t at) {
    if (n > 0 && std::fread(out, 1, n, file) != n) {
      fail_record(at, "read error or unexpected end of file");
    }
  }

  void open(const std::string& p, const batch_reader_options& o) {
    path = p;
    options = o;
    file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) fail("cannot open file for reading");
    if (std::fseek(file, 0, SEEK_END) != 0) fail("cannot determine the file size");
    const long size = std::ftell(file);
    if (size < 0) fail("cannot determine the file size");
    file_size = static_cast<std::uint64_t>(size);
    std::rewind(file);
    if (file_size < header_bytes) {
      fail("not a .dgb file: " + std::to_string(file_size) + " bytes, shorter than the " +
           std::to_string(header_bytes) + "-byte header");
    }
    std::array<unsigned char, header_bytes> b{};
    if (std::fread(b.data(), 1, b.size(), file) != b.size()) fail("cannot read the header");
    if (std::memcmp(b.data(), magic.data(), magic.size()) != 0) {
      fail("not a .dgb file (the magic is not DYNGBAT1)");
    }
    header.format_version = get_u32(b.data() + 8);
    if (header.format_version != current_version) {
      fail("format version " + std::to_string(header.format_version) +
           " is not supported (this library reads version " + std::to_string(current_version) +
           ")");
    }
    if (get_u32(b.data() + 12) != header_bytes) {
      fail("header size " + std::to_string(get_u32(b.data() + 12)) + " (version 1 has " +
           std::to_string(header_bytes) + ")");
    }
    if (get_u32(b.data() + header_crc_offset) != detail::crc32(b.data(), header_crc_offset)) {
      fail("the header's CRC32 does not match (a damaged file)");
    }
    for (std::size_t i = 40; i < header_crc_offset; ++i) {
      if (b[i] != 0) fail("the header's reserved bytes are not zero");
    }
    header.kind = static_cast<container_kind>(b[16]);
    header.vertex_type = static_cast<batch_dtype>(b[17]);
    header.weight_type = static_cast<batch_dtype>(b[18]);
    const auto id_type = static_cast<batch_dtype>(b[19]);
    const std::uint32_t k = get_u32(b.data() + 20);
    if (k > 64) {
      fail("invalid header: K = " + std::to_string(k) +
           " weights per insertion (a .dgb file holds at most 64)");
    }
    header.num_weights = static_cast<int>(k);
    header.flags = get_u64(b.data() + 24);
    const std::uint64_t count = get_u64(b.data() + 32);
    const std::string problem = header_problem(header);
    if (!problem.empty()) fail("invalid header: " + problem);
    if (id_type !=
        (header.kind == container_kind::hypergraph ? batch_dtype::int64 : batch_dtype::none)) {
      fail("invalid header: id type " + std::to_string(static_cast<int>(id_type)));
    }
    // Every record takes at least its 16-byte header and a 24-byte payload.
    if (count > (file_size - header_bytes) / (record_header_bytes + 24)) {
      fail("the header announces " + std::to_string(count) + " batches, more than the " +
           std::to_string(file_size) + "-byte file can hold");
    }
    header.num_batches = static_cast<std::int64_t>(count);
    const std::int64_t type_max = header.vertex_type == batch_dtype::int32
                                      ? std::numeric_limits<std::int32_t>::max()
                                      : std::numeric_limits<std::int64_t>::max();
    DYNG_EXPECTS(options.num_vertices >= -1 && options.num_vertices <= type_max,
                 "batch_reader_options::num_vertices must be -1 or in [0, ", type_max, "]");
    id_max = options.num_vertices >= 0 ? options.num_vertices - 1 : type_max;
    pos = header_bytes;
  }

  // The staging buffer of the next record (host vector, or the next pinned buffer).
  unsigned char* stage(std::size_t bytes) {
    if (!device) {
      host_payload.resize(bytes / 8);
      return reinterpret_cast<unsigned char*>(host_payload.data());
    }
    turn ^= 1;
    const auto t = static_cast<std::size_t>(turn);
    fences[t].synchronize();  // its previous copy to the device has completed
    if (pinned_bytes[t] < bytes) {
      if (pinned[t] != nullptr) pinned_mr->deallocate_sync(pinned[t], pinned_bytes[t], 8);
      pinned[t] = nullptr;
      pinned_bytes[t] = 0;
      pinned[t] = pinned_mr->allocate_sync(bytes, 8);
      pinned_bytes[t] = bytes;
    }
    return static_cast<unsigned char*>(pinned[t]);
  }

  bool next() {
    has_current = false;
    if (records == header.num_batches) {
      if (pos != file_size) {
        fail("the header announces " + std::to_string(header.num_batches) + " batches, but " +
             std::to_string(file_size - pos) + " bytes follow the last one");
      }
      return false;
    }
    const std::uint64_t at = pos;
    if (file_size - pos < record_header_bytes) {
      fail_record(at, "the file ends after " + std::to_string(records) + " of " +
                          std::to_string(header.num_batches) + " batches (a truncated file)");
    }
    std::array<unsigned char, record_header_bytes> rh{};
    read_exact(rh.data(), rh.size(), at);
    const std::uint64_t length = get_u64(rh.data());
    const std::uint32_t crc = get_u32(rh.data() + 8);
    if (get_u32(rh.data() + 12) != 0)
      fail_record(at, "the record header's reserved bytes are not zero");
    const std::uint64_t rest = file_size - pos - record_header_bytes;
    if (length > rest) {
      fail_record(at, "the record's length " + std::to_string(length) +
                          " exceeds the rest of the file (" + std::to_string(rest) + " bytes)");
    }
    const std::uint64_t fixed = header.kind == container_kind::graph ? 24 : 40;
    if (length < fixed || length % 8 != 0) {
      fail_record(at, "the record's length " + std::to_string(length) +
                          " is not a multiple of 8 of at least " + std::to_string(fixed));
    }
    const auto bytes = static_cast<std::size_t>(length);
    unsigned char* p = stage(bytes);
    read_exact(p, bytes, at);
    if (detail::crc32(p, bytes) != crc) fail_record(at, "the CRC32 of the record does not match");
    pos += record_header_bytes + length;
    current = parse(p, bytes, at);
    if (records > 0 && current.batch_id <= last_id) {
      fail_record(at, "batch id " + std::to_string(current.batch_id) + " after " +
                          std::to_string(last_id) + " (ids must be strictly increasing)");
    }
    last_id = current.batch_id;
    if (device) {
      if (device_payload.size() < bytes / 8) {
        device_payload = buffer<std::uint64_t>(*res, bytes / 8);
      }
      if (bytes > 0) {
        detail::copy_bytes(device_payload.data(), memory_space::device, p,
                           memory_space::pinned_host, bytes, res->stream(), res->device());
      }
      fences[static_cast<std::size_t>(turn)].record(res->device(), res->stream());
      base = reinterpret_cast<const unsigned char*>(device_payload.data());
    } else {
      base = p;
    }
    ++records;
    has_current = true;
    return true;
  }

  // The layout of a payload, every count and value checked.
  record_layout parse(const unsigned char* p, std::size_t length, std::uint64_t at) const {
    record_layout l;
    const std::uint64_t id = get_u64(p);
    if (id > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      fail_record(at, "batch id " + std::to_string(id) + " out of range");
    }
    l.batch_id = static_cast<std::int64_t>(id);
    std::size_t off = header.kind == container_kind::graph ? 24 : 40;
    // Place an array of `count` values of width `width`: count is checked against the bytes
    // left before anything is multiplied.
    const auto place = [&](std::size_t slot, std::uint64_t count, batch_dtype t, const char* what) {
      const std::size_t width = dtype_bytes(t);
      const std::size_t left = length - off;
      if (count > left / width) {
        fail_record(at, std::string(what) + ": " + std::to_string(count) +
                            " values do not fit the record's " + std::to_string(left) +
                            " remaining bytes");
      }
      const auto n = static_cast<std::size_t>(count);
      l.slots[slot] = {off, n, t};
      off += pad8(n * width);
      if (off > length) fail_record(at, std::string(what) + ": the padding exceeds the record");
    };
    const batch_dtype vt = header.vertex_type;
    if (header.kind == container_kind::graph) {
      const std::uint64_t nd = get_u64(p + 8);
      const std::uint64_t ni = get_u64(p + 16);
      place(g_del_src, nd, vt, "deletions");
      place(g_del_dst, nd, vt, "deletions");
      place(g_ins_src, ni, vt, "insertions");
      place(g_ins_dst, ni, vt, "insertions");
      const auto k = static_cast<std::uint64_t>(header.num_weights);
      if (k > 0) {
        if (ni > (length - off) / 4 / k) fail_record(at, "the weights do not fit the record");
        place(g_weights, ni * k, header.weight_type, "weights");
      }
      if (off != length) {
        fail_record(at,
                    "the record has " + std::to_string(length - off) + " bytes beyond its arrays");
      }
      for (std::size_t s = g_del_src; s <= g_ins_dst; ++s) {
        const array_slot& a = l.slots[s];
        for (std::size_t i = 0; i < a.count; ++i) {
          const std::int64_t v = element(p + a.offset, a.type, i);
          if (v < 0 || v > id_max) {
            fail_record(at, "vertex id " + std::to_string(v) + " out of range [0, " +
                                std::to_string(id_max) + "]");
          }
        }
      }
      return l;
    }
    const std::uint64_t nd = get_u64(p + 8);
    const std::uint64_t ni = get_u64(p + 16);
    const std::uint64_t nm = get_u64(p + 24);
    const std::uint64_t ndm = get_u64(p + 32);
    const bool with_deleted = (header.flags & batch_file_flags::deleted_members) != 0;
    const bool with_expected = (header.flags & batch_file_flags::expected_ids) != 0;
    if (!with_deleted && ndm != 0) fail_record(at, "deleted members without the flag");
    if (ni >= length / 8) fail_record(at, "the insertion count does not fit the record");
    place(h_del_ids, nd, batch_dtype::int64, "deletions");
    place(h_offsets, ni + 1, batch_dtype::int64, "insertion offsets");
    place(h_members, nm, vt, "members");
    if (header.weight_type != batch_dtype::none) {
      place(h_weights, ni, header.weight_type, "weights");
    }
    if (with_deleted) {
      if (nd >= length / 8) fail_record(at, "the deletion count does not fit the record");
      place(h_del_offsets, nd + 1, batch_dtype::int64, "deleted-member offsets");
      place(h_del_members, ndm, vt, "deleted members");
    }
    if (with_expected) place(h_expected, ni, batch_dtype::int64, "expected ids");
    if (off != length) {
      fail_record(at,
                  "the record has " + std::to_string(length - off) + " bytes beyond its arrays");
    }
    const auto check_offsets = [&](std::size_t slot, std::uint64_t total, const char* what) {
      const array_slot& a = l.slots[slot];
      if (element(p + a.offset, a.type, 0) != 0) {
        fail_record(at, std::string(what) + " do not start at 0");
      }
      for (std::size_t i = 0; i + 1 < a.count; ++i) {
        if (element(p + a.offset, a.type, i) >= element(p + a.offset, a.type, i + 1)) {
          fail_record(at, std::string(what) + " are not increasing at " + std::to_string(i) +
                              " (an empty hyperedge or a decreasing offset)");
        }
      }
      if (static_cast<std::uint64_t>(element(p + a.offset, a.type, a.count - 1)) != total) {
        fail_record(at,
                    std::string(what) + " do not end at the member count " + std::to_string(total));
      }
    };
    const auto check_vertices = [&](std::size_t slot) {
      const array_slot& a = l.slots[slot];
      for (std::size_t i = 0; i < a.count; ++i) {
        const std::int64_t v = element(p + a.offset, a.type, i);
        if (v < 0 || v > id_max) {
          fail_record(at, "vertex id " + std::to_string(v) + " out of range [0, " +
                              std::to_string(id_max) + "]");
        }
      }
    };
    const auto check_non_negative = [&](std::size_t slot, const char* what) {
      const array_slot& a = l.slots[slot];
      for (std::size_t i = 0; i < a.count; ++i) {
        if (element(p + a.offset, a.type, i) < 0) fail_record(at, std::string("negative ") + what);
      }
    };
    check_offsets(h_offsets, nm, "the insertion offsets");
    check_vertices(h_members);
    check_non_negative(h_del_ids, "hyperedge id of a deletion");
    if (header.weight_type != batch_dtype::none) {
      const array_slot& a = l.slots[h_weights];
      const std::int64_t wmax = header.weight_type == batch_dtype::int32
                                    ? std::numeric_limits<std::int32_t>::max()
                                    : std::numeric_limits<std::int64_t>::max();
      for (std::size_t i = 0; i < a.count; ++i) {
        const std::int64_t w = element(p + a.offset, a.type, i);
        if (w < 1 || w > wmax) {
          fail_record(at, "hyperedge weight " + std::to_string(w) + " out of range [1, " +
                              std::to_string(wmax) + "]");
        }
      }
    }
    if (with_deleted) {
      check_offsets(h_del_offsets, ndm, "the deleted-member offsets");
      check_vertices(h_del_members);
    }
    if (with_expected) check_non_negative(h_expected, "expected id");
    return l;
  }

  [[noreturn]] void fail_stopped() const {
    fail("an earlier call of next() failed at batch record " + std::to_string(records) +
         "; the reader stops there (reopen the file to read it again)");
  }

  void require_current(container_kind kind, batch_dtype vt, batch_dtype wt, bool check_weight,
                       const char* what) const {
    if (failed) fail_stopped();
    DYNG_EXPECTS(has_current, "batch_reader::", what, ": no current batch (call next() first)");
    DYNG_EXPECTS(header.kind == kind, "batch_reader::", what, ": the file ", path, " holds ",
                 to_string(header.kind), " batches");
    DYNG_EXPECTS(header.vertex_type == vt && (!check_weight || header.weight_type == wt),
                 "batch_reader::", what, ": the file ", path, " holds ",
                 to_string(header.vertex_type), " vertices and ", to_string(header.weight_type),
                 " weights, not the requested types (", to_string(vt), ", ", to_string(wt), ")");
  }

  template <typename value_t>
  array_view<const value_t> view(std::size_t slot) const {
    const array_slot& a = current.slots[slot];
    const auto* data = reinterpret_cast<const value_t*>(base + a.offset);
    return array_view<const value_t>(a.count == 0 ? nullptr : data, a.count,
                                     device ? memory_space::device : memory_space::host,
                                     device ? res->device() : -1);
  }

  std::int64_t last_id = -1;
};

batch_reader::batch_reader(const std::string& path, const batch_reader_options& options) try
    : impl_(std::make_unique<impl>()) {
  impl_->open(path, options);
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::batch_reader (", path, ")")

batch_reader::batch_reader(const resources& res, const std::string& path,
                           const batch_reader_options& options) try
    : impl_(std::make_unique<impl>()) {
  impl_->res = res;
  impl_->device = res.get_backend() == backend::cuda;
  if (impl_->device) impl_->pinned_mr = detail::resources_access::staging_memory(res);
  impl_->open(path, options);
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::batch_reader (", path, ")")

batch_reader::~batch_reader() = default;
batch_reader::batch_reader(batch_reader&& other) noexcept = default;
batch_reader& batch_reader::operator=(batch_reader&& other) noexcept = default;

const batch_file_header& batch_reader::header() const noexcept {
  return impl_->header;
}

bool batch_reader::next() try {
  DYNG_EXPECTS(impl_ != nullptr, "batch_reader::next: the reader was moved from");
  if (impl_->failed) impl_->fail_stopped();
  try {
    return impl_->next();
  } catch (...) {
    impl_->failed = true;
    impl_->has_current = false;
    throw;
  }
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::batch_reader::next (", impl_->path, ")")

std::int64_t batch_reader::batch_id() const noexcept {
  return impl_ != nullptr && impl_->has_current ? impl_->current.batch_id : -1;
}

template <typename vertex_t, typename weight_t>
edge_batch_view<vertex_t, weight_t> batch_reader::edge_batch() const {
  DYNG_EXPECTS(impl_ != nullptr, "batch_reader::edge_batch: the reader was moved from");
  const bool weighted = impl_->header.num_weights > 0;
  impl_->require_current(container_kind::graph, dtype_of<vertex_t>(), dtype_of<weight_t>(),
                         weighted, "edge_batch");
  edge_batch_view<vertex_t, weight_t> v;
  v.delete_src = impl_->view<vertex_t>(g_del_src);
  v.delete_dst = impl_->view<vertex_t>(g_del_dst);
  v.insert_src = impl_->view<vertex_t>(g_ins_src);
  v.insert_dst = impl_->view<vertex_t>(g_ins_dst);
  if constexpr (!is_unweighted_v<weight_t>) {
    if (weighted) v.insert_weights = impl_->view<weight_t>(g_weights);
  }
  v.num_weights = impl_->header.num_weights;
  return v;
}

template <typename vertex_t, typename weight_t>
hyperedge_batch_view<vertex_t, weight_t> batch_reader::hyperedge_batch() const {
  DYNG_EXPECTS(impl_ != nullptr, "batch_reader::hyperedge_batch: the reader was moved from");
  impl_->require_current(container_kind::hypergraph, dtype_of<vertex_t>(), dtype_of<weight_t>(),
                         true, "hyperedge_batch");
  hyperedge_batch_view<vertex_t, weight_t> v;
  v.delete_hyperedges = impl_->view<std::int64_t>(h_del_ids);
  if (impl_->current.slots[h_members].count > 0) {
    v.insert_offsets = impl_->view<std::int64_t>(h_offsets);
  }
  v.insert_members = impl_->view<vertex_t>(h_members);
  if constexpr (!is_unweighted_v<weight_t>) {
    v.insert_weights = impl_->view<weight_t>(h_weights);
  }
  return v;
}

template <typename vertex_t>
hyperedge_batch_annotations<vertex_t> batch_reader::annotations() const {
  DYNG_EXPECTS(impl_ != nullptr, "batch_reader::annotations: the reader was moved from");
  impl_->require_current(container_kind::hypergraph, dtype_of<vertex_t>(), batch_dtype::none, false,
                         "annotations");
  hyperedge_batch_annotations<vertex_t> a;
  if ((impl_->header.flags & batch_file_flags::deleted_members) != 0) {
    a.deleted_offsets = impl_->view<std::int64_t>(h_del_offsets);
    a.deleted_members = impl_->view<vertex_t>(h_del_members);
  }
  if ((impl_->header.flags & batch_file_flags::expected_ids) != 0) {
    a.expected_ids = impl_->view<std::int64_t>(h_expected);
  }
  return a;
}

// ---------------------------------------------------------------------------- whole files ---

template <typename vertex_t, typename weight_t>
void write_binary_batches(const std::string& path,
                          const std::vector<edge_batch_view<vertex_t, weight_t>>& batches) try {
  int k = 0;
  for (std::size_t b = 0; b < batches.size(); ++b) {
    detail::check_edge_batch_arrays(batches[b], "write_binary_batches", b);
    if (!batches[b].insert_src.empty()) {
      DYNG_EXPECTS(k == 0 || k == batches[b].num_weights, "write_binary_batches: batch ", b,
                   " has K = ", batches[b].num_weights, " weights per insertion, an earlier ",
                   "batch ", k, " (a .dgb file has one K)");
      k = batches[b].num_weights;
    }
  }
  batch_writer out(path, edge_batch_file_header<vertex_t, weight_t>(k));
  for (const auto& b : batches) out.write(b);
  out.close();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::write_binary_batches (", path, ")")

template <typename vertex_t, typename weight_t>
std::vector<edge_batch<vertex_t, weight_t>> read_binary_batches(
    const std::string& path, const batch_reader_options& options) try {
  batch_reader in(path, options);
  const batch_file_header& h = in.header();
  // The header decides before any record is read (M8b review): another kind is an io_error, as
  // in .dgt; other types are the caller's error.
  if (h.kind != container_kind::graph) {
    detail::throw_io_error(path, 0, 0,
                           "this file holds " + std::string(to_string(h.kind)) +
                               " batches (its header says so): read it with "
                               "read_binary_hyperedge_batches()");
  }
  DYNG_EXPECTS(h.vertex_type == dtype_of<vertex_t>() &&
                   (h.num_weights == 0 || h.weight_type == dtype_of<weight_t>()),
               "read_binary_batches: the file ", path, " holds ", to_string(h.vertex_type),
               " vertices and ", to_string(h.weight_type), " weights, not the requested types (",
               to_string(dtype_of<vertex_t>()), ", ", to_string(dtype_of<weight_t>()), ")");
  std::vector<edge_batch<vertex_t, weight_t>> out;
  while (in.next()) {
    const auto v = in.edge_batch<vertex_t, weight_t>();
    edge_batch<vertex_t, weight_t> b(h.num_weights);
    b.reserve(v.num_insertions(), v.num_deletions());
    for (std::size_t j = 0; j < v.num_deletions(); ++j)
      b.delete_edge(v.delete_src[j], v.delete_dst[j]);
    const auto k = static_cast<std::size_t>(h.num_weights);
    for (std::size_t i = 0; i < v.num_insertions(); ++i) {
      if (k == 0) {
        b.insert_edge(v.insert_src[i], v.insert_dst[i]);
      } else {
        b.insert_edge(v.insert_src[i], v.insert_dst[i], v.insert_weights.subview(i * k, k));
      }
    }
    out.push_back(std::move(b));
  }
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::read_binary_batches (", path, ")")

template <typename vertex_t, typename weight_t>
void write_binary_hyperedge_batches(
    const std::string& path,
    const std::vector<hyperedge_batch_view<vertex_t, weight_t>>& batches) try {
  batch_writer out(path, hyperedge_batch_file_header<vertex_t, weight_t>());
  for (const auto& b : batches) out.write(b);
  out.close();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::write_binary_hyperedge_batches (", path, ")")

template <typename vertex_t, typename weight_t>
std::vector<hyperedge_batch<vertex_t, weight_t>> read_binary_hyperedge_batches(
    const std::string& path, const batch_reader_options& options) try {
  batch_reader in(path, options);
  const batch_file_header& h = in.header();
  if (h.kind != container_kind::hypergraph) {
    detail::throw_io_error(path, 0, 0,
                           "this file holds " + std::string(to_string(h.kind)) +
                               " batches (its header says so): read it with read_binary_batches()");
  }
  DYNG_EXPECTS(h.vertex_type == dtype_of<vertex_t>() && h.weight_type == dtype_of<weight_t>(),
               "read_binary_hyperedge_batches: the file ", path, " holds ",
               to_string(h.vertex_type), " vertices and ", to_string(h.weight_type),
               " weights, not the requested types (", to_string(dtype_of<vertex_t>()), ", ",
               to_string(dtype_of<weight_t>()), ")");
  std::vector<hyperedge_batch<vertex_t, weight_t>> out;
  while (in.next()) {
    const auto v = in.hyperedge_batch<vertex_t, weight_t>();
    hyperedge_batch<vertex_t, weight_t> b;
    b.reserve(v.num_insertions(), v.insert_members.size(), v.num_deletions());
    for (const std::int64_t id : v.delete_hyperedges) b.delete_hyperedge(id);
    for (std::size_t i = 0; i < v.num_insertions(); ++i) {
      const auto first = static_cast<std::size_t>(v.insert_offsets[i]);
      const auto last = static_cast<std::size_t>(v.insert_offsets[i + 1]);
      weight_t w{};
      if constexpr (!is_unweighted_v<weight_t>) w = v.insert_weights[i];
      b.insert_hyperedge(v.insert_members.subview(first, last - first), w);
    }
    out.push_back(std::move(b));
  }
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::read_binary_hyperedge_batches (", path, ")")

#define DYNG_INSTANTIATE_BATCH_BINARY_EDGES(V, W)                                      \
  template batch_file_header edge_batch_file_header<V, W>(int);                        \
  template void batch_writer::write<V, W>(const edge_batch_view<V, W>&);               \
  template edge_batch_view<V, W> batch_reader::edge_batch<V, W>() const;               \
  template void write_binary_batches<V, W>(const std::string&,                         \
                                           const std::vector<edge_batch_view<V, W>>&); \
  template std::vector<edge_batch<V, W>> read_binary_batches<V, W>(const std::string&, \
                                                                   const batch_reader_options&);
DYNG_FOR_EACH_VERTEX_WEIGHT_TYPE(DYNG_INSTANTIATE_BATCH_BINARY_EDGES)
DYNG_FOR_EACH_VERTEX_TYPE_UNWEIGHTED(DYNG_INSTANTIATE_BATCH_BINARY_EDGES)
#undef DYNG_INSTANTIATE_BATCH_BINARY_EDGES

#define DYNG_INSTANTIATE_BATCH_BINARY_HYPEREDGES(V, W)                             \
  template batch_file_header hyperedge_batch_file_header<V, W>(std::uint64_t);     \
  template void batch_writer::write<V, W>(const hyperedge_batch_view<V, W>&,       \
                                          const hyperedge_batch_annotations<V>&);  \
  template hyperedge_batch_view<V, W> batch_reader::hyperedge_batch<V, W>() const; \
  template void write_binary_hyperedge_batches<V, W>(                              \
      const std::string&, const std::vector<hyperedge_batch_view<V, W>>&);         \
  template std::vector<hyperedge_batch<V, W>> read_binary_hyperedge_batches<V, W>( \
      const std::string&, const batch_reader_options&);
DYNG_FOR_EACH_HYPERGRAPH_TYPE(DYNG_INSTANTIATE_BATCH_BINARY_HYPEREDGES)
#undef DYNG_INSTANTIATE_BATCH_BINARY_HYPEREDGES

template hyperedge_batch_annotations<std::int32_t> batch_reader::annotations<std::int32_t>() const;

}  // namespace dyng::io
