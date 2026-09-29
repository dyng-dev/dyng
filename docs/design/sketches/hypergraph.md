# Sketch: `hypergraph` and `hyperedge_batch` (0.2)

**Status:** reviewed sketch (M3, against the frozen 0.1 contract, ADR 0023); frozen in M8 with the
ESCHER store. **Origin:** ESCHER-GPU's CBST store (`structure/*`, `kernel/*`) and MOSP_ESCHER's
copy of it, merged in stages (PLAN Section 6.4.5); MOSP_ESCHER's `HostHypergraph`, `HgBatch` and
`IncidenceBatch` for the batch model; H-SOSP's `DeviceIncidence` / `DeviceH2H` for the slack
storage of 0.3.

The hypergraph is the second container. It mirrors `graph` wherever the concepts match (pimpl,
move-only, a version counter and a state identity for stale-result detection, `clone(res)` and
`to_backend(res)`, `apply()` for the structure only, `check_integrity()`), so that the algorithm
contract, `dyng::update()` and the conformance kit carry over unchanged.

## The headers

```cpp
// cpp/include/dyng/hypergraph/hypergraph_properties.hpp
namespace dyng {
enum class hypergraph_storage : std::uint8_t { escher, slack_csr };           ///< slack_csr: 0.3
enum class line_graph_policy : std::uint8_t { none, automatic, sorted_merge, slack_csr };
enum class id_reuse : std::uint8_t { erase_first, insert_first };            ///< O16: erase_first
struct capacity_policy {                    ///< how the store grows (CBST payload, slack rows)
  double initial_headroom = 0.125;
  double grow_factor = 1.5;
  bool compact_before_grow = true;          ///< MOSP_ESCHER's compaction first, then grow
  std::int64_t max_payload = 0;             ///< 0 = up to the int32 offset limit (capacity_error)
  [[nodiscard]] static constexpr capacity_policy escher_gpu() noexcept;     ///< ESCHER-GPU exactly
  [[nodiscard]] static constexpr capacity_policy mosp_escher() noexcept;    ///< MOSP_ESCHER exactly
};
struct hypergraph_properties {
  hypergraph_storage storage = hypergraph_storage::escher;
  line_graph_policy line_graph = line_graph_policy::automatic;  ///< h2h built when an algorithm needs it
  id_reuse reuse = id_reuse::erase_first;
  capacity_policy capacity{};
  bool store_v2h = true;                    ///< vertex -> hyperedges (needed by triad_count)
};
}  // namespace dyng

// cpp/include/dyng/hypergraph/hypergraph_view.hpp
namespace dyng {
template <typename id_t> struct incidence_view;   ///< rows of sorted members, any memory space;
                                                   ///< row_ptr/cols as csr_view, dead rows empty
template <typename weight_t> struct line_graph_view;  ///< h2h: hyperedge -> overlapping hyperedges
                                                       ///< (+ overlap sizes), sorted rows
struct storage_stats { std::int64_t device_bytes = 0, payload_capacity = 0, compactions = 0,
                       thread_path_rows = 0, warp_path_rows = 0, block_path_rows = 0; };
}

// cpp/include/dyng/hypergraph/hyperedge_batch.hpp
namespace dyng {
template <typename vertex_t, typename weight_t>
struct hyperedge_batch_view {              ///< non-owning; empty array = none of this kind
  array_view<const std::int64_t> delete_hyperedges;     ///< ids of live hyperedges
  array_view<const std::int64_t> insert_offsets;        ///< CSR of the new hyperedges:
  array_view<const vertex_t> insert_members;            ///<   members of hyperedge i are
  array_view<const weight_t> insert_weights;            ///<   [offsets[i], offsets[i+1]); one weight each
  array_view<const std::int64_t> delete_incidence_hyperedge;  ///< (h, v) incidence edits (0.3)
  array_view<const vertex_t> delete_incidence_vertex;
  array_view<const std::int64_t> insert_incidence_hyperedge;
  array_view<const vertex_t> insert_incidence_vertex;
  [[nodiscard]] std::size_t num_insertions() const noexcept;
  [[nodiscard]] std::size_t num_deletions() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
};
template <typename vertex_t, typename weight_t>
class hyperedge_batch {                    ///< owning host builder, operations in order
 public:
  explicit hyperedge_batch(bool weighted = !is_unweighted_v<weight_t>);
  void delete_hyperedge(std::int64_t id);
  void insert_hyperedge(array_view<const vertex_t> members, weight_t w = {});
  void insert_hyperedge(std::initializer_list<vertex_t> members, weight_t w = {});
  void delete_incidence(std::int64_t h, vertex_t v);     ///< 0.3
  void insert_incidence(std::int64_t h, vertex_t v);     ///< 0.3
  void reserve(std::size_t insertions, std::size_t members, std::size_t deletions);
  void clear() noexcept;
  [[nodiscard]] hyperedge_batch_view<vertex_t, weight_t> view() const noexcept;
};
struct hyper_apply_summary {               ///< returned by apply(), inside every hypergraph stats
  std::int64_t inserted_hyperedges = 0, deleted_hyperedges = 0;
  std::int64_t inserted_incidences = 0, deleted_incidences = 0;
  std::int64_t skipped_ops = 0;            ///< deletions of dead ids, duplicate members (counted, not errors)
  std::int64_t num_hyperedges_after = 0, id_bound_after = 0;
  std::vector<std::int64_t> inserted_ids;  ///< the id given to each inserted hyperedge, in batch
                                           ///< order (best-fit reuse: authoritative)
};
}

// cpp/include/dyng/hypergraph/hypergraph.hpp
namespace dyng {
template <typename vertex_t = std::int32_t, typename weight_t = unweighted>
class hypergraph {                          ///< pimpl, move-only, resident
 public:
  using vertex_type = vertex_t; using weight_type = weight_t;
  [[nodiscard]] static hypergraph from_hyperedges(const resources& res,
                                                  hyperedge_list_view<vertex_t, weight_t> hyperedges,
                                                  const hypergraph_properties& props = {});
  [[nodiscard]] hypergraph clone(const resources& res) const;
  [[nodiscard]] hypergraph to_backend(const resources& res) const;
  [[nodiscard]] vertex_t num_vertices() const noexcept;
  [[nodiscard]] std::int64_t num_hyperedges() const noexcept;       ///< live hyperedges
  [[nodiscard]] std::int64_t hyperedge_id_bound() const noexcept;   ///< ids in [0, bound)
  [[nodiscard]] const hypergraph_properties& properties() const noexcept;
  [[nodiscard]] memory_space space() const noexcept;
  [[nodiscard]] std::uint64_t version() const noexcept;
  [[nodiscard]] incidence_view<vertex_t> h2v() const;                ///< hyperedge -> sorted members
  [[nodiscard]] incidence_view<std::int64_t> v2h() const;            ///< vertex -> hyperedges
  [[nodiscard]] line_graph_view<weight_t> h2h() const;               ///< throws if line_graph == none
  hyper_apply_summary apply(const resources& res, const hyperedge_batch_view<vertex_t, weight_t>& batch);
  [[nodiscard]] hyperedge_list<vertex_t, weight_t> to_hyperedge_list(const resources& res) const;
  [[nodiscard]] storage_stats store_stats() const;
  void check_integrity(const resources& res) const;    ///< ESCHER checkIntegrity + checkEscher
};
}
```

- **Ids.** Public hyperedge ids are 0-based `std::int64_t`; the CBST's `+1` shift, its sentinels
  and its int32 payload offsets never leave `hypergraph/escher/` (PLAN 4.4.2). Deleted ids are
  reused by the store's best-fit rule; `hyper_apply_summary::inserted_ids` is the mapping the
  algorithms and the user read (the triad frontier of the insert phase is exactly this list).
- **Batch semantics.** One fixed order per `id_reuse`: `erase_first` (ESCHER's paper order,
  deletions then insertions, so a deleted id may be reused by the same batch) or `insert_first`
  (H-SOSP's order). A deletion of a dead id and a duplicate member are counted in `skipped_ops`;
  an out-of-range vertex is an `invalid_argument_error` before anything changes (strong
  guarantee, as `graph::apply()`).
- **Views on the host and the device.** The views describe the resident storage (device memory
  on CUDA), with the same `array_view` members as `csr_view`, so the framework's `old_view` /
  `new_view` wrap them unchanged.
- **Stages:** `hypergraph.apply` (sub-stages `hypergraph.apply.delete`, `.insert`, `.line_graph`),
  `hypergraph.build`, `hypergraph.upload`.

## What 0.1 has to provide (and does)

| Need | 0.1 |
|---|---|
| `array_view`, `resources`, errors, profiler, `unweighted`, `invalid_id` | frozen in `core/*` |
| `dyng::update()` on a hypergraph | `update.hpp` dispatches on `detail::participant_of<container_t>`; M8 adds the hypergraph specialization and a `hypergraph` `run_update()` (one commit, the same poison rules). The frozen `update()` template needs no change |
| the registry's container kind | `container_kind::hypergraph` exists in `core/registry.hpp` |
| stale-result detection | the same `version()` + state identity scheme as `graph` |

## Review against the 0.1 contract

| Check | Result |
|---|---|
| Naming | snake_case; counts `num_`; accessors nouns (`h2v`, `v2h`, `h2h` are the papers' names for the three incidence relations, kept as nouns) |
| Ownership and ABI | pimpl, move-only; `hyper_apply_summary` is an aggregate with a vector (the id mapping must be owned by the caller) |
| Host-compilability | no CUDA types; the CBST kernels live in `cpp/src/hypergraph/escher/` |
| Differences from PLAN 5.2 | `v2h()` loses its `sorted` flag (always sorted: every reader so far wants it, and a flag would make two view types of one relation); `store_v2h` and `capacity_policy`'s two presets are added so each original's behaviour is one line; `hyper_apply_summary`'s counts are spelled out |
| Python | `dyng.Hypergraph.from_hyperedges(offsets, members, ...)`, `from_lists([[0, 1, 2], ...])`; `dyng.HyperedgeBatch(delete_hyperedges=..., insert_hyperedges=[[...], ...])`; `inserted_ids` as an array |

## Open questions (decided in M8)

1. `incidence_view`'s dead rows: empty rows (simple for readers) or a separate liveness bitmap
   (cheaper for the CBST, which does not compact on delete). The sketch says empty rows.
2. Whether `hyper_apply_summary::inserted_ids` is a `std::vector` (host) or a `buffer` in the
   backend's space (the triad insert phase reads it on the device). A device copy may be kept
   internally either way.
3. The weighted hypergraph for `hyper_sssp` (0.3): one weight per hyperedge (`weight_t`), stored
   next to the members; whether `escher` storage supports weights or only `slack_csr` does.
