# Hypergraphs

The dynamic hypergraph container `dyng::hypergraph` (ADR 0041): hyperedges with 0-based `int64`
ids on the ESCHER store, changed by batches of hyperedge deletions and insertions whose new ids
come from the store's best-fit reuse (`hyper_apply_summary::inserted_ids`); its properties
(`hypergraph_storage`, `line_graph_policy`, `id_reuse`, `capacity_policy` with the presets
`escher_gpu()` and `mosp_escher(extra)`), the hyperedge list, the batch, the incidence and
line-graph views and the storage counters. These headers are tracked, not frozen, until the end
of M8b (0.3.0).

**The line graph** (ADR 0042). `hypergraph::h2h()` is the line graph: one sorted row per id, the
ids of the hyperedges that share a vertex with it, and `out[r]`, the offset of the first neighbour
above `r` (the oriented wedges of `triad_count`). Under `line_graph_policy::sorted_merge` it is
built with the hypergraph and kept by every `apply()` with ESCHER-GPU's sorted delta merge (on the
CPU backends a host form of the same merge, with the same layout); under `automatic` (the default)
the first `h2h()` builds it, and `apply()` keeps it from then on; under `none` `h2h()` throws. The
views (`h2v()`, `v2h()`, `h2h()`) are built on first use per version, in the hypergraph's memory
space, and never by `apply()`; `store_stats().device_bytes` counts the stores and the row views the
container keeps per commit (ESCHER-GPU's `DynamicHypergraph::deviceBytes()` scope).
`dyng::testing::rebuild_line_graph` (`dyng/testing/line_graph_oracle.hpp`) recomputes a line
graph from the hyperedges for tests.

```{doxygengroup} hypergraph
:content-only:
:members:
```
