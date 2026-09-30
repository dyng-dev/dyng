# Sketch: `label_propagation` (0.3)

**Status:** reviewed sketch (M3, against the frozen 0.1 contract, ADR 0023); frozen in M10.
**Computes:** binary harmonic label propagation (labels F in [0, 1] from a set of seed vertices
with class 0 or 1) on an undirected weighted graph, kept up to date under **vertex batches**
(vertices arrive with their similarity edges, or leave). **Paper:** DynLP (ICS 2026).
**Origin:** LabelPropagation-CUDA `src/{dynLP,iterativePropagation,sparsifySubgraph,
connectedComponents,labelInitialization}.cu` (PLAN Section 6.4.7).

## The header

```cpp
// cpp/include/dyng/label_propagation.hpp
namespace dyng::label_propagation {

enum class schedule : std::uint8_t {
  in_place,   ///< the published rule (two hashed half-sweeps); fast, not bitwise reproducible
  damped,     ///< 1/2-damped Jacobi: bitwise reproducible per backend and build
};

/// What happens at max_iterations: the framework's three policies (framework/policies.hpp),
/// public here because label_propagation is their first public user.
enum class on_limit : std::uint8_t { error, report, fallback_recompute };

struct options {                                     ///< an aggregate; fields are only appended
  std::optional<float> similarity_threshold;         ///< tau (sparsification: keep w > tau);
                                                     ///< nullopt = the average edge weight of the
                                                     ///< current graph. A tunable
  float tolerance = 1e-4f;                           ///< delta, the stopping rule. A tunable
  std::int64_t max_iterations = 100000;              ///< per update (the original's default)
  on_limit on_iteration_limit = on_limit::report;    ///< report (stats::converged = false),
                                                     ///< error (convergence_error) or
                                                     ///< fallback_recompute
  schedule update_schedule = schedule::in_place;     ///< a tunable
  engine cuda_engine = engine::automatic;            ///< fused propagation engine first (Tier B)
};

struct stats : update_stats {             ///< update_stats: affected = vertices in the affected
  apply_summary batch;                    ///<   set V_aff (deterministic), iterations and
  std::int64_t components = 0;            ///<   frontier_visits (schedule-dependent), converged
  std::int64_t sparsified_edges = 0;      ///< deterministic: edges kept by the tau filter
};

template <typename vertex_t>
class result {                            ///< opaque, move-only (pimpl)
 public:
  [[nodiscard]] array_view<const float> labels() const;          ///< F per vertex, in [0, 1];
                                                                 ///< seeds exactly 0 or 1
  [[nodiscard]] array_view<const std::int8_t> seed_labels() const;  ///< -1 none, 0 / 1 class
  [[nodiscard]] const options& get_options() const;
  void set_options(const options& opt);
  [[nodiscard]] std::uint64_t graph_version() const noexcept;
  [[nodiscard]] memory_space space() const noexcept;
  [[nodiscard]] result clone(const resources& res) const;
};

/// The static harmonic propagation from the seeds (IrLP, the published baseline).
template <typename vertex_t, typename edge_t>
[[nodiscard]] result<vertex_t> compute(const resources& res, const graph<vertex_t, edge_t, float>& g,
                                       array_view<const vertex_t> seed_vertices,
                                       array_view<const std::int8_t> seed_labels,
                                       const options& opt = {});

/// A vertex batch: batch.insert_vertices (+ insert_vertex_labels: ground truth of new seeds),
/// insert_src / insert_dst / insert_weights for their similarity edges, delete_vertices.
template <typename vertex_t, typename edge_t>
stats update(const resources& res, graph<vertex_t, edge_t, float>& g,
             const edge_batch_view<vertex_t, float>& batch, result<vertex_t>& r);
}  // namespace dyng::label_propagation
```

- **The graph is the frozen `graph`**, with what 0.3 adds to it: `row_layout::slotted` (DynLP's
  fixed per-vertex slots and `row_end`; `graph::with_capacity`) and `slack`, vertex insertions
  and deletions in `graph::apply()` (the `insert_vertices`, `insert_vertex_labels` and
  `delete_vertices` arrays of `edge_batch_view` exist since 0.1 and throw `not_supported_error`
  until then), soft vertex status, and `batch_semantics::net_effect()`. All of these are
  additions, allowed by PLAN 5.9's additive rules; nothing frozen changes.
- **Hooks:** `before_apply` marks the neighbours of deleted vertices on the rows before the batch;
  `identify_affected` = the new vertices and their neighbours, sparsified by tau, grouped into
  components (Shiloach-Vishkin; `operators::connected_components`); `seed` = the supernode label
  initialization (64-bit fixed-point sums); Step 2 = the propagation engine until the change is
  below `tolerance` or `max_iterations` (the framework's `on_limit` policy).
- **Stages:** `label_propagation.normalize`, `.commit`, `.identify_affected` (with
  `.identify_affected.sparsify` and `.identify_affected.components`), `.seed`, `.loop`,
  `.finalize`.

## Contract and registry

| Item | Value |
|---|---|
| Family | fixed point |
| Oracle | `reference`: neither `update()` nor `compute()` is compared with the other; both are compared with the converged harmonic solution (PCG) of the current graph within a tolerance tied to delta (mean \|F - F*\| <= 2000 delta, CI at delta = 1e-6), plus class agreement (PLAN 5.1) |
| Determinism | `tolerance` (in-place); the `damped` schedule is bitwise per backend and build |
| Graph requirements | `directed = false`, `weight_t = float`; `row_layout::slotted` or `slack` for vertex batches |
| Backends | sequential (from the checker's host model, Gauss-Seidel), cuda; OpenMP if contributed (0.4) |
| `@guarantee` | strong before the commit; basic after it (result poisoned) |

## Review against the 0.1 contract

| Check | Result |
|---|---|
| Uniform skeleton | yes; the seeds are inputs fixed at `compute()` (arguments), tau / tolerance / schedule tunables |
| Frozen headers needed | `update_stats` (`iterations`, `frontier_visits`, `converged` fit the propagation loop), `edge_batch_view`'s vertex arrays (reserved in 0.1), `determinism::tolerance`, `oracle_kind::reference`, `on_limit` (in the framework; public as `label_propagation::options::on_iteration_limit`) |
| Differences from PLAN 5.3 and the API-first design | `max_iterations` is `int64` (the other counters are); `on_limit` reuses the framework's three policies (report, error, fallback_recompute) instead of a new `limit_action`; `classes()` (thresholded labels) is left to the caller / Python (it is one comparison) |
| Python | `dyng.label_propagation.compute(g, seeds, seed_labels, tolerance=1e-4, update_schedule="damped")`; `labels.labels` zero-copy through DLPack |

## Open questions (decided in M10)

1. Where `on_limit` lives publicly: in `core/types.hpp` (the second public user would be
   `hyper_sssp`'s budget) or per algorithm. The sketch nests it under the algorithm until a second
   algorithm needs the same enumeration (rule of two).
2. `std::optional<float>` in `options` is bindable (nanobind supports it) but not a trivial
   aggregate member for C; acceptable, since the C++ API is the contract.
3. The published setup (full graph known up front, `with_capacity`) as the documented default
   (O9), streaming (`slack`, tau from the data seen so far) as the advanced mode.
