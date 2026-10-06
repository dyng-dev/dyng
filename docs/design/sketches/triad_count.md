# Sketch: `triad_count` (0.3)

**Status:** reviewed sketch (M3, against the frozen 0.1 contract, ADR 0023); frozen in M9.
**Computes:** the 30-bin closed h-motif counts of a hypergraph (the triads of three hyperedges and
how they overlap), or the StatHyper type 1 / 2 / 3 counts, kept exact under hyperedge batches.
**Paper:** ESCHER (IPDPS 2026), ESCHER+ (TKDE 2026). **Origin:** ESCHER-GPU `HMotifCount`,
`HMotifCountUpdate`, `type1/2/3`, the wedge engine `kernel/wedge.cuh` (PLAN Section 6.4.6).

## The header

```cpp
// cpp/include/dyng/triad_count.hpp
namespace dyng::triad_count {

enum class pattern : std::uint8_t {
  h_motif,           ///< the 30 closed h-motifs (bins 21-26, the open triads, are always 0)
  stathyper_type1,   ///< StatHyper's three triad types, one bin each
  stathyper_type2,
  stathyper_type3,
};

struct options {                            ///< an aggregate; fields are only appended
  pattern kind = pattern::h_motif;          ///< fixed at compute()
  engine cuda_engine = engine::automatic;   ///< fused: the wedge engine (Tier B). A tunable
};

struct stats : update_stats {               ///< update_stats: affected = bins whose count changed
  hyper_apply_summary batch;                ///<   (deterministic); iterations = 0; fallback_used false
  std::int64_t deleted_frontier = 0;        ///< deterministic: hyperedges of the subtract phase
  std::int64_t inserted_frontier = 0;       ///< deterministic: hyperedges of the add phase
  std::uint64_t triads_removed = 0;         ///< deterministic: summed over the bins
  std::uint64_t triads_added = 0;
};

class result {                              ///< opaque, move-only (pimpl); a host array of counts
 public:
  [[nodiscard]] pattern kind() const noexcept;
  [[nodiscard]] int num_bins() const noexcept;                  ///< 30 or 1
  [[nodiscard]] array_view<const std::uint64_t> counts() const; ///< host memory on every backend
  [[nodiscard]] std::uint64_t count(int bin) const;
  [[nodiscard]] std::uint64_t total() const;
  [[nodiscard]] const options& get_options() const;
  void set_options(const options& opt);                         ///< tunables only
  [[nodiscard]] std::uint64_t graph_version() const noexcept;
  [[nodiscard]] memory_space space() const noexcept;            ///< memory_space::host
  [[nodiscard]] result clone(const resources& res) const;
};

template <typename vertex_t, typename weight_t>
[[nodiscard]] result compute(const resources& res, const hypergraph<vertex_t, weight_t>& hg,
                             const options& opt = {});

template <typename vertex_t, typename weight_t>
stats update(const resources& res, hypergraph<vertex_t, weight_t>& hg,
             const hyperedge_batch_view<vertex_t, weight_t>& batch, result& r);
}  // namespace dyng::triad_count
```

- **The update is cycle_count's template on a hypergraph** (aggregate delta, PLAN 4.5.1):
  `count(-)` on the old h2h / h2v views with the deleted ids as the frontier, the commit, then
  `count(+)` on the new views with `hyper_apply_summary::inserted_ids` as the frontier; each
  affected triad is owned by its smallest frontier member (`ownership::min_member`, invariant
  I2). The bins are signed 64-bit on the device and folded into the unsigned histogram in
  `finalize` (a bin that would become negative is an `internal_error`, as in cycle_count).
- **Stages:** `triad_count.count_minus`, `triad_count.commit`, `triad_count.identify_affected`,
  `triad_count.count_plus`, `triad_count.finalize`; `triad_count.count` for `compute()` (the
  paper's full count); `hypergraph.apply` inside the commit.
- **Requirements:** a line graph (`line_graph_policy::automatic` materializes `sorted_merge`)
  and the vertex-to-hyperedge rows (`store_v2h`). `compute()` checks them and names the property.
- **Rule of two:** the `count` operator with `ownership::min_member` is the second user of the
  aggregate-delta machinery after cycle_count; that is when `operators::count_delta` enters
  `cpp/src/operators/` (PLAN 4.5.3).

## Contract and registry

| Item | Value |
|---|---|
| Family | aggregate delta |
| Oracle | `compute`: `update()` equals `compute()` on the new hypergraph exactly ("baseline + deltas == recount") |
| Determinism | `exact_value` |
| Container requirements | `line_graph != none`, `store_v2h = true`; weights ignored |
| Backends | sequential and OpenMP (the CPU reference `utils/reference.cpp`), cuda (the wedge engine, Tier B) |
| `@guarantee` | strong before the commit; basic after it (result poisoned) |

## Review against the 0.1 contract

| Check | Result |
|---|---|
| Uniform skeleton | yes; `result` is not a template (a histogram of counts, as `cycle_count::result`) |
| Frozen headers needed | `stats : update_stats`, `engine`, `array_view`; `dyng::update()` works once the hypergraph provides `participant_of` (see {doc}`hypergraph`) |
| Differences from PLAN 5.3 / 5.5 | the plan's `stats::delta` (a signed per-bin vector) is replaced by `triads_removed` / `triads_added` plus the result's counts before and after, like cycle_count's `cycles_removed` / `cycles_added`: the stats of an update are returned by value from every call, and a per-bin `std::vector` would add one more allocation to each (the `batch` summary's `inserted_ids` already allocates, and mosp's per-objective stats do; a fixed-size `std::array<std::int64_t, 30>` would not, open question 1); the per-bin delta is `counts()` after minus before, which the Python layer can offer as `st.delta` |
| Python | `dyng.triad_count.compute(hg)`, `kind="h_motif"`; `hist.counts` (30 values) |

## Open questions (decided in M9)

1. Whether the per-bin signed delta is worth a `stats` field after all (the paper's plots show
   per-batch deltas; a fixed-size `std::array<std::int64_t, 30>` would not allocate).
2. The `experimental::count_local_patterns<op_t>` extension point (0.5) and how its Op concept
   relates to `pattern` (a new pattern as a new Op, not a new enumerator).
