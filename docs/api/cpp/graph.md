# Graphs

The dynamic graph container `dyng::graph`, its properties (row order, parallel edges, batch semantics), views, CSR and edge-list inputs, and the summary of an applied batch.

These headers are frozen since 0.1 (ADR 0023) and **stable** (SemVer, PLAN 5.9). The additions of
0.4 for `label_propagation` (M10; reviewed and frozen at the end of M10b, ADR 0109) are part of
the stable tier **from 0.4.0** (the author's decision of 2026-11-18 on AD15, GOVERNANCE.md):
`graph::with_capacity`, the slot and slack rows (`row_layout::slotted` and `slack`,
{doc}`../../concepts/row_layouts`), `vertex_status`, `graph_view::status` and `row_end`,
`graph::num_active_vertices`, `batch_semantics::net_effect()` (and `as_net_effect`), the vertex
operations of `edge_batch` (`insert_vertex`, `delete_vertex` and their read-back
`insert_vertices()`, `insert_vertex_labels()`, `delete_vertices()`), the `apply_summary` fields
`dropped_insertions`, `relocated_rows` and `compactions`, and the float instantiations of the graph
(`weight_t = float`). Their Python counterparts are stable from 0.4.0 too (`Graph.with_capacity`,
`Graph.vertex_status()`, `dyng.VertexStatus`, the vertex arrays of `EdgeBatch`, the float32 graph
classes and the `weight_dtype` keywords; {doc}`../python/index`).

```{doxygengroup} graph
:content-only:
:members:
```
