# Sketch: `hyper_sssp` (0.3)

**Status:** reviewed sketch (M3, against the frozen 0.1 contract, ADR 0023); frozen in M11.
**Computes:** single-source shortest hyperpaths on a hypergraph with hyperedge weights: the
distance D(h) of every hyperedge from a source vertex (through the line graph, where two
hyperedges are adjacent when they share a vertex and entering h costs w(h)), its canonical parent
hyperedge, and the distance of every vertex, kept up to date under hyperedge and incidence
batches. **Paper:** H-SOSP (IA3 2026). **Origin:** MOSP_ESCHER `hsosp/` (`hsosp.cuh`,
`hsospDelta.cu`, `hsospDevice.cu`), `hypergraph/` (`HostHypergraph`, `HgBatch`,
`emulateSospUpdate`) (PLAN Section 6.4.8).

## The header

```cpp
// cpp/include/dyng/hyper_sssp.hpp
namespace dyng::hyper_sssp {

enum class budget_action : std::uint8_t {
  recompute,   ///< over budget: recompute from scratch (stats::fallback_used = true)
  error,       ///< over budget: throw convergence_error (nothing of the update is kept)
};

struct options {                                   ///< an aggregate; fields are only appended
  std::int64_t work_budget = 0;                    ///< frontier visits allowed per update before
                                                   ///< the budget action; 0 = automatic (H-SOSP's
                                                   ///< rule). A tunable
  budget_action on_budget = budget_action::recompute;   ///< a tunable
  std::int64_t delta = 0;                          ///< near-far width of the sssp engine; 0 = automatic
  engine cuda_engine = engine::automatic;          ///< the ported H-SOSP kernels first, then the
                                                   ///< shared sssp engine (a node-weight functor)
  bool validate_inputs = true;
};

struct stats : update_stats {             ///< update_stats: affected, iterations, frontier_visits,
  hyper_apply_summary batch;              ///<   fallback_used (the budget), converged, engine_used
  std::int64_t line_graph_insertions = 0; ///< deterministic: h2h pairs the batch added
  std::int64_t line_graph_deletions = 0;  ///< deterministic: h2h pairs the batch removed
  std::int64_t invalidated = 0;           ///< deterministic: hyperedges in invalidated subtrees
};

template <typename vertex_t, typename distance_t = std::int64_t>
class result {                            ///< opaque, move-only (pimpl)
 public:
  [[nodiscard]] vertex_t source() const noexcept;                          ///< a vertex
  [[nodiscard]] array_view<const distance_t> hyperedge_distances() const;  ///< D(h), by hyperedge id;
                                                                           ///< infinite_distance() for dead or unreachable ids
  [[nodiscard]] array_view<const std::int64_t> hyperedge_parents() const;  ///< canonical; -1 = none
  [[nodiscard]] distance_t distance_to(vertex_t target) const;             ///< min over h containing
                                                                           ///< target of D(h); @sync
  [[nodiscard]] const options& get_options() const;
  void set_options(const options& opt);
  [[nodiscard]] std::uint64_t graph_version() const noexcept;
  [[nodiscard]] memory_space space() const noexcept;
  [[nodiscard]] result clone(const resources& res) const;
};

template <typename vertex_t, typename weight_t>
[[nodiscard]] result<vertex_t> compute(const resources& res, const hypergraph<vertex_t, weight_t>& hg,
                                       vertex_t source, const options& opt = {});

template <typename vertex_t, typename weight_t>
stats update(const resources& res, hypergraph<vertex_t, weight_t>& hg,
             const hyperedge_batch_view<vertex_t, weight_t>& batch, result<vertex_t>& r);
}  // namespace dyng::hyper_sssp
```

- **The source is a vertex.** The original's virtual source and target hyperedges become
  internal: the hyperedges containing the source start at their own weight, and a vertex's
  distance is the minimum over its hyperedges (`distance_to`). Parent ids are 0-based hyperedge ids
  everywhere (the original's host / device mismatch behind its defect S4 cannot recur).
- **The update is sssp's template on the line graph.** `translate` turns the hyperedge and
  incidence changes into line-graph edge changes (`line_graph/incidence_delta.cu`, order-free;
  ESCHER's whole-hyperedge path is a special case), then Step 1 and Step 2 of `sssp` run on the
  h2h view with the hyperedge weight as a node weight. The `translate` hook is the one the
  framework left out in 0.1 (no user then; `docs/developer/framework.md`, "Differences from
  PLAN 4.5"): `hyper_sssp` is its first user and adds it, and `mosp` its second.
- **The work budget** is the framework's budget policy made public: over `work_budget` frontier
  visits the update recomputes (`fallback_used`), or throws under `budget_action::error`.
- **Stages:** `hyper_sssp.normalize`, `.commit` (`hypergraph.apply` inside), `.translate`,
  `.identify_affected`, `.seed`, `.loop`, `.finalize`; the paper's per-stage times
  (`t_escher_ms`, `t_delta_ms`, `t_csr_apply_ms`, `t_sosp_update_ms`, `t_static_ms`) map onto
  them in `parity/timed_regions/hyper_sssp.toml`.

## Contract and registry

| Item | Value |
|---|---|
| Family | fixed point (on the line graph) |
| Oracle | `compute`: `update()` equals `compute()` exactly (distances and canonical parents) |
| Determinism | `bitwise` |
| Container requirements | weighted hyperedges (`weight_t = int32_t`, weights in [1, 2^28], a stricter limit than sssp's, documented), a line graph (`line_graph_policy::slack_csr` for incidence edits) |
| Backends | sequential (the host emulation `emulateSospUpdate`), cuda |
| `@guarantee` | strong before the commit (and for `budget_action::error`: the check runs before anything is kept); basic after it (result poisoned) |

## Review against the 0.1 contract

| Check | Result |
|---|---|
| Uniform skeleton | yes; the same shape as `sssp` (source fixed at `compute()`, tunables through `set_options`) |
| Frozen headers needed | `update_stats::fallback_used` (the budget) and `converged`; `infinite_distance()`; `engine`; `hyper_apply_summary` from {doc}`hypergraph` |
| Differences from PLAN 5.3 | `stats::fell_back` is `update_stats::fallback_used` (the frozen common field); `iterations` is inherited; `delta` and `cuda_engine` are added as in `sssp::options`; `distance_to` returns `distance_t` |
| Python | `dyng.hyper_sssp.compute(hg, source=v)`; `sp.distance_to(t)`; `st.fallback_used` |

## Open questions (decided in M11)

1. Whether `result` also exposes per-vertex distances as an array (n values) or only
   `distance_to()`; an array costs one extra pass per update and is what most users will plot.
2. Parity by id needs `id_reuse::insert_first`; with the default `erase_first` the results are
   compared keyed by (sorted vertex set, weight). The documented default stays `erase_first` (O16).
