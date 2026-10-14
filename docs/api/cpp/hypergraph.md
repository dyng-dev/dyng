# Hypergraphs

The dynamic hypergraph container `dyng::hypergraph` (ADR 0041): hyperedges with 0-based `int64`
ids on the ESCHER store, changed by batches of hyperedge deletions and insertions whose new ids
come from the store's best-fit reuse (`hyper_apply_summary::inserted_ids`); its properties
(`hypergraph_storage`, `line_graph_policy`, `id_reuse`, `capacity_policy` with the presets
`escher_gpu()` and `mosp_escher(extra)`), the hyperedge list, the batch, the incidence and
line-graph views and the storage counters. These headers are tracked, not frozen, until the end
of M8b (0.3.0).

```{doxygengroup} hypergraph
:content-only:
:members:
```
