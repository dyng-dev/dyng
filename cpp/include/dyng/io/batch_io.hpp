// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file batch_io.hpp
 * @brief Batch files: the dynG batch text format `.dgt` (read_batches(), write_batches() for
 *        graphs; read_hyperedge_batches(), write_hyperedge_batches() for hypergraphs), the
 *        binary batch format `.dgb` (batch_reader, batch_writer and the whole-file helpers) and
 *        the legacy MOSP files `insert.txt` ("u v w1 .. wK") and `delete.txt` ("u v").
 * @ingroup io
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/registry.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace dyng::io {

/**
 * @brief Options of read_batches().
 * @ingroup io
 */
struct batch_file_options {
  /**
   * @brief Weights per edge insertion (K), or -1: taken from the first insertion of the file
   *        (then every insertion must carry as many). Must be 0 or -1 for unweighted batches.
   */
  int num_weights = -1;
  std::int64_t num_vertices = -1;  ///< ids must be < num_vertices; -1: only ids >= 0 are checked
};

/**
 * @brief Read a `.dgt` batch text file: a sequence of batches (PLAN Section 5.7).
 *
 * One operation per line; blank lines and lines whose first token starts with `#` are skipped.
 * An optional first line `%dgt 1` names the format version (only version 1 exists). A line
 * `%batch <id>` starts a batch (the ids are non-negative and strictly increasing); a file
 * without `%batch` lines holds one batch (or none if it holds no operation). The graph
 * operations of 0.1 are `+e u v [w1 .. wK]` (an edge insertion with exactly K weights) and
 * `-e u v` (an edge deletion), ids 0-based. Within a batch the insertions and the deletions each
 * keep file order (which of the two applies first is the graph's batch semantics, not the
 * file's). A file holds the batches of one container kind: the first line may name it
 * (`%dgt 1 graph`); a hypergraph file (`%dgt 1 hypergraph`, or a hyperedge operation `+h` /
 * `-h`) is an io_error here (read it with read_hyperedge_batches()). The vertex operations
 * `+v u [label]` and `-v u` (label_propagation) and the incidence edits `+i`, `-i` are reserved
 * for 0.4 and rejected with an io_error. Every token must be a whole decimal integer; weights
 * must fit `weight_t`.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type (integral, or dyng::unweighted).
 * @param[in] path    The file.
 * @param[in] options K and the vertex count used for validation.
 * @return The batches in file order.
 * @throws io_error               if the file is missing or malformed (with path, line and column).
 * @throws invalid_argument_error if the options are invalid.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] std::vector<edge_batch<vertex_t, weight_t>> read_batches(
    const std::string& path, const batch_file_options& options = {});

/**
 * @brief Write batches as a `.dgt` batch text file (read back by read_batches()).
 *
 * Writes `%dgt 1`, then for batch i the line `%batch i`, its deletions (`-e u v`) and its
 * insertions (`+e u v w1 .. wK`), each in batch order.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type (integral, or dyng::unweighted).
 * @param[in] path    The file (parent directories are created).
 * @param[in] batches The batches (host memory; every batch with insertions has the same K).
 * @throws io_error               if the file cannot be written.
 * @throws invalid_argument_error if a batch has vertex operations, inconsistent sizes, a negative
 *                                vertex id (it would not read back), or a K that differs from
 *                                another batch's.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
void write_batches(const std::string& path,
                   const std::vector<edge_batch_view<vertex_t, weight_t>>& batches);

/**
 * @brief Options of read_hyperedge_batches().
 * @ingroup io
 */
struct hyperedge_batch_file_options {
  std::int64_t num_vertices =
      -1;  ///< members must be < num_vertices; -1: only ids >= 0 are checked
};

/**
 * @brief Read the hypergraph batches of a `.dgt` batch text file (PLAN Section 5.7).
 *
 * The format of read_batches() with the hyperedge operations: `+h v1 v2 .. [; w]` inserts a
 * hyperedge with the listed members (at least one; repeats are kept here and dropped by
 * hypergraph::apply()) and, for a weighted type, its weight in [1, max(weight_t)] after a `;`
 * (a token of its own: `+h 0 1 ; 5`, or attached: `+h 0 1;5`); `-h id` deletes the hyperedge
 * with the id `id` (>= 0; dead or out-of-range ids are kept and skipped by apply()). A file holds
 * the batches of one container kind: graph operations (`+e`, `-e`) or a `%dgt 1 graph` line are
 * an io_error here. The incidence edits `+i h v` and `-i h v` are reserved for 0.4 and rejected
 * with an io_error.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type (dyng::unweighted, or integral).
 * @param[in] path    The file.
 * @param[in] options The vertex count used for validation.
 * @return The batches in file order.
 * @throws io_error               if the file is missing or malformed (with path, line and column).
 * @throws invalid_argument_error if the options are invalid.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] std::vector<hyperedge_batch<vertex_t, weight_t>> read_hyperedge_batches(
    const std::string& path, const hyperedge_batch_file_options& options = {});

/**
 * @brief Write hypergraph batches as a `.dgt` batch text file (read back by
 *        read_hyperedge_batches()).
 *
 * Writes `%dgt 1 hypergraph`, then for batch i the line `%batch i`, its deletions (`-h id`) and
 * its insertions (`+h v1 .. vk`, with ` ; w` for a weighted type), each in batch order.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type (dyng::unweighted, or integral).
 * @param[in] path    The file (parent directories are created).
 * @param[in] batches The batches (host memory).
 * @throws io_error               if the file cannot be written.
 * @throws invalid_argument_error if a batch is malformed or would not read back: offsets that do
 *                                not start at 0 or end at the member count, an empty insertion, a
 *                                negative member or deletion id, a weight below 1 or a missing
 *                                one, an array outside host-accessible memory.
 * @throws not_supported_error    if a batch has incidence edits (planned for 0.4).
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
void write_hyperedge_batches(const std::string& path,
                             const std::vector<hyperedge_batch_view<vertex_t, weight_t>>& batches);

/**
 * @brief The element types a `.dgb` file can store (its header names one per array kind).
 * @ingroup io
 */
enum class batch_dtype : std::uint8_t {
  none = 0,   ///< no such array (an unweighted batch's weights, a graph's hyperedge ids)
  int32 = 1,  ///< 32-bit signed integers
  int64 = 2,  ///< 64-bit signed integers
};

/**
 * @brief The lower-case name of a batch_dtype ("none", "int32", "int64").
 * @param[in] dtype The value.
 * @return Its name.
 * @ingroup io
 */
[[nodiscard]] std::string_view to_string(batch_dtype dtype) noexcept;

/**
 * @brief The flags of a `.dgb` file (batch_file_header::flags; docs/api/file_formats.md). A
 *        reader rejects a file with a flag it does not know.
 * @ingroup io
 */
struct batch_file_flags {
  /// Hypergraph files: every deletion also carries the member set of the hyperedge it deleted
  /// when the batch was generated (replay by content, PLAN 6.4.6).
  static constexpr std::uint64_t deleted_members = 1U;
  /// Hypergraph files: every insertion also carries the id the generating run gave it (replay
  /// by key, checked against hyper_apply_summary::inserted_ids).
  static constexpr std::uint64_t expected_ids = 2U;
  /// Hypergraph files: the batches were generated against a hypergraph with
  /// id_reuse::insert_first (the expected ids hold under that order; without the flag,
  /// id_reuse::erase_first).
  static constexpr std::uint64_t insert_first = 4U;
  /// Every flag this library knows.
  static constexpr std::uint64_t known = deleted_members | expected_ids | insert_first;
};

/**
 * @brief The header of a `.dgb` file: what its batch records hold.
 * @ingroup io
 */
struct batch_file_header {
  std::uint32_t format_version = 1;              ///< the `.dgb` format version (1)
  container_kind kind = container_kind::graph;   ///< graph or hypergraph batches
  batch_dtype vertex_type = batch_dtype::int32;  ///< vertex ids
  batch_dtype weight_type = batch_dtype::none;   ///< weights (none: unweighted)
  int num_weights = 0;           ///< weights per insertion: K for graphs, 0 or 1 for hypergraphs
  std::uint64_t flags = 0;       ///< batch_file_flags (0 for graph files)
  std::int64_t num_batches = 0;  ///< batch records in the file (written by batch_writer::close())
};

/**
 * @brief The header of a `.dgb` file of graph batches of the types (vertex_t, weight_t).
 * @tparam vertex_t Vertex id type (int32 or int64).
 * @tparam weight_t Weight type (dyng::unweighted, int32 or int64).
 * @param[in] num_weights Weights per insertion (K; 0 for unweighted).
 * @return The header (num_batches 0).
 * @throws invalid_argument_error if num_weights is negative, or positive for unweighted.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] batch_file_header edge_batch_file_header(int num_weights);

/**
 * @brief The header of a `.dgb` file of hypergraph batches of the types (vertex_t, weight_t).
 * @tparam vertex_t Vertex id type (int32 or int64).
 * @tparam weight_t Hyperedge weight type (dyng::unweighted, int32 or int64).
 * @param[in] flags batch_file_flags.
 * @return The header (one weight per insertion for a weighted type; num_batches 0).
 * @throws invalid_argument_error if a flag is unknown.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] batch_file_header hyperedge_batch_file_header(std::uint64_t flags = 0);

/**
 * @brief What a hyperedge batch record of a `.dgb` file carries besides the batch, for the
 *        flags batch_file_flags::deleted_members and batch_file_flags::expected_ids.
 * @tparam vertex_t Vertex id type.
 * @ingroup io
 */
template <typename vertex_t>
struct hyperedge_batch_annotations {
  /// deleted_members: num_deletions() + 1 offsets from 0 into deleted_members.
  array_view<const std::int64_t> deleted_offsets;
  /// deleted_members: the member set of every deleted hyperedge, back to back.
  array_view<const vertex_t> deleted_members;
  /// expected_ids: the id of every insertion in the generating run.
  array_view<const std::int64_t> expected_ids;
};

/**
 * @brief Writes a `.dgb` binary batch file record by record (PLAN Section 5.7; the format is in
 *        docs/api/file_formats.md).
 *
 * The constructor writes the header with a batch count of 0; every write() appends one
 * length-prefixed record with its CRC32; close() writes the count. A file whose writer was not
 * closed is rejected by batch_reader (its count does not match).
 * @ingroup io
 */
class batch_writer {
 public:
  /**
   * @brief Create (truncate) `path` and write the header.
   * @param[in] path   The file (parent directories are created).
   * @param[in] header The kind, types, K and flags (num_batches is ignored).
   * @throws io_error               if the file cannot be opened or written.
   * @throws invalid_argument_error if the header is inconsistent (a graph file with flags or no
   *                                vertex type, a hypergraph file with K > 1, an unknown flag or
   *                                format version).
   */
  batch_writer(const std::string& path, const batch_file_header& header);
  /// @brief Close the file if close() was not called (errors are ignored).
  ~batch_writer();
  batch_writer(const batch_writer&) = delete;             ///< not copyable
  batch_writer& operator=(const batch_writer&) = delete;  ///< not copyable
  /**
   * @brief Move constructor.
   * @param[in,out] other The writer to move from.
   */
  batch_writer(batch_writer&& other) noexcept;
  /**
   * @brief Move assignment.
   * @param[in,out] other The writer to move from.
   * @return *this.
   */
  batch_writer& operator=(batch_writer&& other) noexcept;

  /**
   * @brief Append a graph batch.
   * @tparam vertex_t Vertex id type (the header's).
   * @tparam weight_t Weight type (the header's).
   * @param[in] batch The batch (host memory; K = the header's num_weights).
   * @throws invalid_argument_error if the file is not a graph file of these types, or the batch is
   *                                malformed (edge_batch rules of write_batches()).
   * @throws io_error               if the file cannot be written.
   * @throws out_of_memory_error    if host memory cannot be allocated.
   */
  template <typename vertex_t, typename weight_t>
  void write(const edge_batch_view<vertex_t, weight_t>& batch);

  /**
   * @brief Append a hypergraph batch.
   * @tparam vertex_t Vertex id type (the header's).
   * @tparam weight_t Hyperedge weight type (the header's).
   * @param[in] batch       The batch (host memory; the rules of write_hyperedge_batches()).
   * @param[in] annotations What the header's flags require (deleted member sets, expected ids).
   * @throws invalid_argument_error if the file is not a hypergraph file of these types, the batch
   *                                is malformed, or the annotations do not match the flags or the
   *                                batch.
   * @throws not_supported_error    if the batch has incidence edits (planned for 0.4).
   * @throws io_error               if the file cannot be written.
   * @throws out_of_memory_error    if host memory cannot be allocated.
   */
  template <typename vertex_t, typename weight_t>
  void write(const hyperedge_batch_view<vertex_t, weight_t>& batch,
             const hyperedge_batch_annotations<vertex_t>& annotations = {});

  /**
   * @brief Write the batch count and close the file.
   * @throws io_error if a write or the close failed.
   */
  void close();

  /**
   * @brief The batches written so far.
   * @return The count.
   */
  [[nodiscard]] std::int64_t num_batches() const noexcept;

 private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

/**
 * @brief Options of batch_reader.
 * @ingroup io
 */
struct batch_reader_options {
  /// Vertex ids must be < num_vertices; -1: only ids >= 0 (and the type's range) are checked.
  std::int64_t num_vertices = -1;
};

/**
 * @brief Reads a `.dgb` binary batch file one batch at a time, in bounded memory (PLAN Section
 *        5.7; the format is in docs/api/file_formats.md).
 *
 * The constructor reads and checks the header; next() reads the next record, checks its
 * length against the rest of the file before allocating anything, its CRC32 and its contents,
 * and makes it the current batch, whose views stay valid until the next call of next(). The
 * memory held is that of the largest record (twice, plus its device copy, with CUDA resources).
 *
 * With the resources of the cuda backend the views are in device memory: each record is read
 * into one of two pinned host buffers from the resources' staging memory resource and copied to
 * the device asynchronously on the resources' stream (the views are ready in stream order; the
 * reader waits for a pinned buffer's earlier copy only before it reuses that buffer). With other
 * resources, or none, the views are in host memory.
 *
 * Every malformed input raises io_error with the path and the byte offset of the record (in the
 * column of the error; the line is 0): a wrong magic, a newer format version, an unknown flag,
 * a size field larger than the rest of the file, a record that is shorter or longer than its
 * counts, a CRC32 that does not match, offsets that are not monotonic, ids out of range, a file
 * that ends before (or after) the header's batch count.
 * @ingroup io
 */
class batch_reader {
 public:
  /**
   * @brief Open `path` and read its header (views in host memory).
   * @param[in] path    The file.
   * @param[in] options The vertex count used for validation.
   * @throws io_error               if the file is missing or its header is malformed.
   * @throws invalid_argument_error if the options are invalid.
   * @throws out_of_memory_error    if host memory cannot be allocated.
   */
  explicit batch_reader(const std::string& path, const batch_reader_options& options = {});

  /**
   * @brief Open `path` and read its header; the views are in the default memory space of `res`
   *        (device memory for the cuda backend, staged through pinned buffers).
   * @param[in] res     Execution resources.
   * @param[in] path    The file.
   * @param[in] options The vertex count used for validation.
   * @throws io_error               if the file is missing or its header is malformed.
   * @throws invalid_argument_error if the options are invalid.
   * @throws out_of_memory_error    if host memory cannot be allocated.
   * @sync Enqueues no work.
   */
  batch_reader(const resources& res, const std::string& path,
               const batch_reader_options& options = {});

  /// @brief Release the buffers (device and pinned memory in stream order).
  ~batch_reader();
  batch_reader(const batch_reader&) = delete;             ///< not copyable
  batch_reader& operator=(const batch_reader&) = delete;  ///< not copyable
  /**
   * @brief Move constructor.
   * @param[in,out] other The reader to move from.
   */
  batch_reader(batch_reader&& other) noexcept;
  /**
   * @brief Move assignment.
   * @param[in,out] other The reader to move from.
   * @return *this.
   */
  batch_reader& operator=(batch_reader&& other) noexcept;

  /**
   * @brief The file's header.
   * @return The header (num_batches as the file states it).
   */
  [[nodiscard]] const batch_file_header& header() const noexcept;

  /**
   * @brief Read the next batch.
   * @return False at the end of the file (after the header's batch count).
   * @throws io_error            if the record is malformed or the file ends early.
   * @throws out_of_memory_error if host, pinned or device memory cannot be allocated.
   * @throws cuda_error          if the CUDA runtime reports an error.
   * @async With CUDA resources the copy to the device is enqueued on the resources' stream; the
   *        call may wait for the copy of the batch before the previous one.
   */
  bool next();

  /**
   * @brief The record's batch id (strictly increasing within a file; the writers number 0, 1,
   *        ...).
   * @return The id of the current batch, -1 before the first next().
   */
  [[nodiscard]] std::int64_t batch_id() const noexcept;

  /**
   * @brief The current batch of a graph file.
   * @tparam vertex_t Vertex id type (the header's vertex type).
   * @tparam weight_t Weight type (the header's weight type).
   * @return The view, valid until the next call of next().
   * @throws invalid_argument_error if the file holds hypergraph batches, the types differ from the
   *                                header's, or there is no current batch.
   */
  template <typename vertex_t, typename weight_t>
  [[nodiscard]] edge_batch_view<vertex_t, weight_t> edge_batch() const;

  /**
   * @brief The current batch of a hypergraph file.
   * @tparam vertex_t Vertex id type (the header's vertex type).
   * @tparam weight_t Hyperedge weight type (the header's weight type).
   * @return The view, valid until the next call of next().
   * @throws invalid_argument_error if the file holds graph batches, the types differ from the
   *                                header's, or there is no current batch.
   */
  template <typename vertex_t, typename weight_t>
  [[nodiscard]] hyperedge_batch_view<vertex_t, weight_t> hyperedge_batch() const;

  /**
   * @brief The annotations of the current hypergraph batch (empty arrays for the flags the file
   *        does not have).
   * @tparam vertex_t Vertex id type (the header's vertex type).
   * @return The views, valid until the next call of next().
   * @throws invalid_argument_error if the file holds graph batches, the vertex type differs, or
   *                                there is no current batch.
   */
  template <typename vertex_t>
  [[nodiscard]] hyperedge_batch_annotations<vertex_t> annotations() const;

 private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

/**
 * @brief Write graph batches as a `.dgb` file (batch_writer over every batch).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type (dyng::unweighted, or integral).
 * @param[in] path    The file (parent directories are created).
 * @param[in] batches The batches (host memory, one K).
 * @throws io_error               if the file cannot be written.
 * @throws invalid_argument_error if a batch is malformed (the rules of write_batches()).
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
void write_binary_batches(const std::string& path,
                          const std::vector<edge_batch_view<vertex_t, weight_t>>& batches);

/**
 * @brief Read every batch of a `.dgb` file of graph batches into host memory.
 * @tparam vertex_t Vertex id type (the file's).
 * @tparam weight_t Weight type (the file's).
 * @param[in] path    The file.
 * @param[in] options The vertex count used for validation.
 * @return The batches in file order.
 * @throws io_error               if the file is missing or malformed.
 * @throws invalid_argument_error if the file holds hypergraph batches or other types.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] std::vector<edge_batch<vertex_t, weight_t>> read_binary_batches(
    const std::string& path, const batch_reader_options& options = {});

/**
 * @brief Write hypergraph batches as a `.dgb` file (batch_writer over every batch, no flags).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type (dyng::unweighted, or integral).
 * @param[in] path    The file (parent directories are created).
 * @param[in] batches The batches (host memory).
 * @throws io_error               if the file cannot be written.
 * @throws invalid_argument_error if a batch is malformed (the rules of write_hyperedge_batches()).
 * @throws not_supported_error    if a batch has incidence edits (planned for 0.4).
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
void write_binary_hyperedge_batches(
    const std::string& path, const std::vector<hyperedge_batch_view<vertex_t, weight_t>>& batches);

/**
 * @brief Read every batch of a `.dgb` file of hypergraph batches into host memory (the
 *        annotations are not kept: use batch_reader for them).
 * @tparam vertex_t Vertex id type (the file's).
 * @tparam weight_t Hyperedge weight type (the file's).
 * @param[in] path    The file.
 * @param[in] options The vertex count used for validation.
 * @return The batches in file order.
 * @throws io_error               if the file is missing or malformed.
 * @throws invalid_argument_error if the file holds graph batches or other types.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] std::vector<hyperedge_batch<vertex_t, weight_t>> read_binary_hyperedge_batches(
    const std::string& path, const batch_reader_options& options = {});

/**
 * @brief Options of read_legacy_batch().
 * @ingroup io
 */
struct legacy_batch_options {
  int num_weights = 1;             ///< weights per insertion line (K of the graph), >= 0
  std::int64_t num_vertices = -1;  ///< ids must be < num_vertices; -1: only ids >= 0 are checked
  /**
   * @brief MOSP's accept/reject rules instead of dynG's stricter ones.
   *
   * As readChangeBatch() of MOSP-OpenMP\@c352151 / MOSP-CUDA\@e220ee2: integers after the first
   * 2 + K of an insertion line and after the first 2 of a deletion line are read and ignored,
   * and a deletion line with a single integer is skipped. The compatibility driver
   * dyng-compat-mosp sets it (the parity harness needs the original's input-rejection
   * semantics); ADR 0010 records why the default is strict.
   */
  bool mosp_lenient = false;
};

/**
 * @brief Read a MOSP batch: insertions from `insert_path`, deletions from `delete_path`.
 *
 * Each non-blank line of the insert file holds exactly 2 + K integers "u v w1 .. wK", and each
 * non-blank line of the delete file exactly 2 integers "u v"; ids are 0-based and must lie in
 * [0, num_vertices), weights in [1, max(weight_t)] (MOSP readChangeBatch()). By default dynG is
 * stricter than MOSP in three places: extra integers on an insertion or a deletion line, and a
 * deletion line with a single integer, are errors (MOSP ignores them);
 * `options.mosp_lenient` restores MOSP's decisions. In both modes every token must be a whole
 * decimal integer ("12abc" is an error, as in MOSP). Operations keep file order.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type (integral).
 * @param[in] insert_path The insertion file.
 * @param[in] delete_path The deletion file.
 * @param[in] options     K and the vertex count used for validation.
 * @return The batch.
 * @throws io_error               if a file is missing or malformed (with path, line and column).
 * @throws invalid_argument_error if the options are invalid.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] edge_batch<vertex_t, weight_t> read_legacy_batch(
    const std::string& insert_path, const std::string& delete_path,
    const legacy_batch_options& options = {});

/**
 * @brief Write a batch as MOSP `insert.txt` / `delete.txt` (byte-identical to MOSP's
 *        writeChangeBatch()).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type (integral).
 * @param[in] insert_path The insertion file (parent directories are created).
 * @param[in] delete_path The deletion file (parent directories are created).
 * @param[in] batch       The batch (host memory; vertex operations are not representable).
 * @throws io_error               if a file cannot be written.
 * @throws invalid_argument_error if the batch has vertex operations or inconsistent sizes.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
void write_legacy_batch(const std::string& insert_path, const std::string& delete_path,
                        const edge_batch_view<vertex_t, weight_t>& batch);

}  // namespace dyng::io
