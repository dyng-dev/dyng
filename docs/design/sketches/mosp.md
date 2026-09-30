# Sketch: `mosp` (0.2)

**Status:** reviewed sketch (M3, against the frozen 0.1 contract, ADR 0023); frozen in M7.
**Computes:** multi-objective shortest paths from one source on a graph with K weight columns:
the K single-objective shortest-path trees, the preference-weighted combined graph, its
shortest-path tree (the MOSP tree) and the K path costs of every vertex along it.
**Paper:** DynaMOSP (IPDPS 2025, IEEE TPDS 2025). **Origin:** MOSP-CUDA `src/mospUpdate.cu`,
`combinedGraphGpu.cu`; MOSP-OpenMP `combinedGraphCpu.cpp` (PLAN Section 6.4.4).

## The header

```cpp
// cpp/include/dyng/mosp.hpp
namespace dyng::mosp {

struct options {                          ///< an aggregate; fields are only appended
  std::vector<std::int32_t> preferences;  ///< one per objective, >= 1; empty = all ones. Fixed at
                                          ///< compute(); lcm(preferences) <= 2^20
  std::int64_t delta = 0;                 ///< near-far width of the K updates and of the combined
                                          ///< solve; 0 = automatic (sssp's rule). A tunable
  engine cuda_engine = engine::automatic; ///< the engine of the K sssp updates (sssp::options). A tunable
  bool compute_path_costs = true;         ///< Step 3's last line (the K costs along the MOSP tree). A tunable
  bool validate_inputs = true;            ///< O(K n) checks of imported trees (from_arrays)
};

struct stats : update_stats {             ///< update_stats: affected (vertices whose MOSP parent
  apply_summary batch;                    ///<   or a cost changed), fallback_used, engine_used, ...
  std::vector<sssp::stats> objectives;    ///< one per objective, deterministic fields as in sssp
  std::int64_t combined_edges = 0;        ///< deterministic: edges of the combined graph
  std::int64_t preference_scale = 0;      ///< deterministic: L = lcm(preferences)
};

template <typename vertex_t, typename distance_t = std::int64_t>
class result {                            ///< opaque, move-only (pimpl)
 public:
  [[nodiscard]] vertex_t source() const noexcept;
  [[nodiscard]] int num_objectives() const noexcept;
  [[nodiscard]] array_view<const distance_t> distances(int objective) const;   ///< the K sssp trees
  [[nodiscard]] array_view<const vertex_t> parents(int objective) const;
  [[nodiscard]] array_view<const distance_t> combined_distances() const;       ///< in units of 1/L
  [[nodiscard]] array_view<const vertex_t> combined_parents() const;           ///< the MOSP tree
  [[nodiscard]] array_view<const distance_t> path_costs() const;               ///< n * K, vertex-major
  [[nodiscard]] std::int64_t preference_scale() const noexcept;
  [[nodiscard]] const options& get_options() const;
  void set_options(const options& opt);                                        ///< tunables only
  [[nodiscard]] std::uint64_t graph_version() const noexcept;
  [[nodiscard]] memory_space space() const noexcept;
  [[nodiscard]] result clone(const resources& res) const;
  template <typename edge_t, typename weight_t>            ///< adopt K trees (MOSP's obj<k>/ files)
  [[nodiscard]] static result from_arrays(const resources& res, const graph<vertex_t, edge_t, weight_t>& g,
                                          vertex_t source,
                                          array_view<const array_view<const distance_t>> distances,
                                          array_view<const array_view<const vertex_t>> parents,
                                          bool canonicalize = true, const options& opt = {});
};

template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] result<vertex_t> compute(const resources& res, const graph<vertex_t, edge_t, weight_t>& g,
                                       vertex_t source, const options& opt = {});

template <typename vertex_t, typename edge_t, typename weight_t>
stats update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
             const edge_batch_view<vertex_t, weight_t>& batch, result<vertex_t>& r);
}  // namespace dyng::mosp
```

- **Objectives are the weight columns of the graph** (`g.num_weights()`, 1 to
  `max_objectives = 64`); `graph_properties::num_weights` sets K and the batch carries K weights
  per insertion (`edge_batch_view::num_weights`). No new container or batch type is needed.
- **Composition, not a new engine.** `update()` runs K `sssp` problems over objective views of
  one graph through one `run_update()` (one commit, one Step 0), sharing the handle's workspace
  (ADR 0015), then `finalize` builds the combined graph (weights `L(K+1) - sum_i L / pref_i` per
  edge: count, scan, fill), solves it statically with the `sssp` engine, and computes the path
  costs. This is the framework's composition of PLAN 4.5.1; `mosp_problem` holds the K
  `sssp_problem`s.
- **Stages:** `mosp.normalize`, `mosp.commit`, `mosp.objective` around each sssp update (one
  stage for all K objectives, called K times; whose own `sssp.*` stages nest inside; the frozen
  profiler rule wants every dotted part to start with a letter, so the objective's index is not a
  name part: `mosp.objective.0` would be rejected, and the per-objective times are the stage's
  samples in order), `mosp.combine`, `mosp.combined_sssp`, `mosp.path_costs`, `mosp.finalize`. The paper's regions: "(a) compute" = the K updates + combine + combined solve,
  "(b) end to end".

## Contract and registry

| Item | Value |
|---|---|
| Family | fixed point (K times `sssp`, then a finalize step) |
| Oracle | `compute`: `update()` equals `compute()` on the new graph exactly |
| Determinism | `bitwise` (canonical lowest-id ties in every tree, the combined tree included) |
| Graph requirements | `store_transposed = true`, `num_weights >= 1`; any row order and multigraph switch (`mosp_compatible()` for byte parity with MOSP's CSR) |
| Backends | sequential, openmp, cuda (fused sssp engine; operators engine from M7) |
| `@guarantee` | strong before the commit; basic after it (result poisoned) |

## Review against the 0.1 contract

| Check | Result |
|---|---|
| Uniform skeleton | yes; `result` is a template (its arrays hold vertex ids and distances), as `sssp::result` |
| Frozen headers needed | none changed: K weight columns, `edge_batch_view::num_weights`, `sssp::stats`, `run_update()` and the workspace lease exist in 0.1 |
| Differences from PLAN 5.1-5.3 | `stats` derives from `update_stats` and names its combine counters directly (the plan's `combine_stats` predates the frozen `stats` shape); `result::from_arrays` takes K views (MOSP's `obj<k>/` trees) |
| Python | `dyng.mosp.compute(g, source=0, preferences=[4, 1, 4])`; `result.path_costs` as an `(n, K)` array; `from_arrays` takes lists of arrays |

## Open questions (decided in M7)

1. Whether `path_costs()` is computed on the device in 0.2 (the original computes it on the host)
   or stays a host step behind `compute_path_costs`.
2. `preferences` as `std::vector` makes `options` non-trivially-copyable; an inline array of
   `max_objectives` is the alternative if the Python binding or the ABI prefers it.
3. The combined graph's storage: a second `graph` owned by the result (simplest, reuses
   `graph::from_csr`) or a workspace-leased CSR (no per-update allocation, invariant I9). The
   latter is the target; the former is the straight port.
