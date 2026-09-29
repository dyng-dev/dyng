# Algorithms

Every algorithm has the same public shape: a header `<dyng/<algo>.hpp>` with `options`,
`result`, `stats`, `compute()` and `update()` (ADR 0006), a sequential reference backend, and a
page here with the same nine sections (problem, template mapping, API, backends and
determinism, performance, limitations, differences from the paper, mapping from the original
code, how to cite).

| Algorithm | Computes | Container | Family | Backends | Maturity | Release |
|---|---|---|---|---|---|---|
| {doc}`sssp` | dynamic single-source shortest paths | graph | fixed point | sequential, OpenMP, CUDA | experimental | 0.1 |
| {doc}`cycle_count` | k-bounded directed simple-cycle histograms | graph | aggregate delta | sequential, OpenMP, CUDA (M2b) | experimental | 0.1 |
| `mosp` | multi-objective shortest paths | graph | fixed point (composition of K `sssp`) | sequential, OpenMP, CUDA | planned | 0.2 |
| `triad_count` | hypergraph h-motif triad counts | hypergraph | aggregate delta | sequential, OpenMP, CUDA | planned | 0.2 |
| `label_propagation` | binary harmonic label propagation under vertex batches | graph | fixed point | sequential, CUDA | planned | 0.3 |
| `hyper_sssp` | shortest hyperpaths | hypergraph | fixed point | sequential, CUDA | planned | 0.3 |

Columns:

- **Family**: *fixed point* (a value per element, iterated until nothing changes) or *aggregate
  delta* (a global count, updated by one signed recount); see {doc}`../concepts/update_model`.
- **Maturity**: `experimental` (may change in any release), `stable` (SemVer applies from
  0.1.0), `planned` (not written yet). A planned algorithm gets its page when its port starts.
- **Release**: the first release that contains it ({doc}`../roadmap`).

The table will be generated from each algorithm's `manifest.toml` by `scripts/regen.py`
(milestone M3), like the README's.

```{toctree}
:maxdepth: 1
:glob:

*
```
