# Use hypergraphs from Python

Build a hypergraph, change it with batches, read its incidence and line-graph rows, and read and
write hypergraph files, from the `dyng` package. The concepts (ids, `id_reuse`, capacity
presets, the line graph) are in {doc}`../concepts/hypergraphs`; the C++ API is
{doc}`../api/cpp/hypergraph`.

## Build a hypergraph

<!-- snippet: hypergraph-python-build -->
```python
import dyng

h = dyng.Hypergraph.from_lists([[0, 1, 2], [1, 3], [2, 3, 4]], line_graph="sorted_merge")
print(h.num_vertices, h.num_hyperedges, h.hyperedge_id_bound)
b = dyng.HyperedgeBatch(delete_hyperedges=[1], insert_hyperedges=[[0, 4], [4, 4, 1]])
s = h.apply(b)
print(s.inserted_ids, s.deleted_hyperedges, s.dropped_members, h.version)
lg = h.h2h()
print(lg.row(0).tolist(), lg.col_ind[lg.out[0] : lg.row_ptr[1]].tolist())
```
<!-- snippet-output: hypergraph-python-build -->
```text
5 3 3
[1 3] 1 1 1
[1, 2, 3] [1, 2, 3]
```

Hyperedge `i` of the list gets id `i`; members are sorted and repeated members dropped. A CSR
works too: `dyng.Hypergraph.from_hyperedges(offsets, members)` with `offsets` the
`num_hyperedges + 1` offsets of the `members` array. Vertex ids are int32: pass
`vertex_dtype="int32"` for an int64 NumPy array (every value is checked; nothing is narrowed
silently). Weights (one int32 per hyperedge, in [1, 2^31 - 1]) are the second argument:
`from_lists(lists, [5, 7, 2])`.

The properties are keywords (or a `dyng.HypergraphProperties`): `line_graph="none" |
"automatic" | "sorted_merge"`, `id_reuse="erase_first" | "insert_first"`, `capacity=
"default" | "escher_gpu" | "mosp_escher"` (or a `dyng.CapacityPolicy`, e.g.
`dyng.CapacityPolicy.mosp_escher(extra=1 << 20)`), `store_v2h=True`. A hypergraph belongs to the
backend of its resources: `resources=dyng.Resources("openmp")`, or `dyng.Resources.cuda()` with a
CUDA plugin (`pip install "dyng[cu13]"`); `h.to_backend(res)` copies it to another backend.

## Apply batches and read the ids

```python
b = dyng.HyperedgeBatch(delete_hyperedges=[1], insert_hyperedges=[[0, 4], [4, 4, 1]])
s = h.apply(b)
print(s.inserted_ids)          # int64 ids chosen by the store, e.g. [1 3]
print(s.deleted_hyperedges, s.dropped_members, s.skipped_ops)   # 1 1 1
print(h.version)               # 1
```

**Read the ids from `inserted_ids`; never predict them**: the store reuses the slots of deleted
hyperedges by best fit, and the batch order (`id_reuse`) changes which slots are free
({doc}`../concepts/hypergraphs`). Deletions of dead or unknown ids are skipped and counted
(`ignored_deletions`). A batch with an invalid member or an empty hyperedge raises
`dyng.InvalidArgumentError` and changes nothing; incidence edits (`delete_incidences=`,
`insert_incidences=`) raise `dyng.NotSupportedError` until 0.4.

## Read the rows

```python
v = h.h2v()                    # hyperedge -> members, one row per id (empty for a dead id)
print([v.row(i).tolist() for i in range(v.num_rows)])
print(h.v2h().row(4))          # the ids of vertex 4's hyperedges, increasing (int64)
lg = h.h2h()                   # the line graph: the hyperedges each one overlaps
r = 0
print(lg.row(r), lg.col_ind[lg.out[r]:lg.row_ptr[r + 1]])   # all neighbours; those above r
```

The views are host copies of the state they describe (`v.version`); `h.weights` is the weight
of every id. `dyng.testing.rebuild_line_graph(h)` recomputes the line graph from the rows, and
`h.check_integrity()` checks the store's invariants.

## Files

```python
h = dyng.io.read_hypergraph("coauth-DBLP.hg", options="escher_gpu", line_graph="sorted_merge")
w = dyng.io.read_hypergraph("coauth-DBLP.hg", options="hsosp", id_reuse="insert_first")
dyng.io.write_hypergraph("out.hg", h)                    # its live hyperedges, in id order

dyng.io.write_hyperedge_batches("batches.dgt", [b])    # text: +h v1 v2 .. ; -h id
batches = dyng.io.read_hyperedge_batches("batches.dgt")

dyng.io.write_binary_batches("batches.dgb", [b], expected_ids=[s.inserted_ids])
for rec in dyng.io.BatchReader("batches.dgb"):         # streaming, CRC-checked
    print(rec.batch_id, rec.batch.num_insertions, rec.expected_ids)
```

`options="escher_gpu"` reads a `.hg` file as ESCHER-GPU does (ids 1..n kept, every line a
hyperedge); `options="hsosp"` as H-SOSP does (at most 25 members, vertices by first appearance,
weights U[1, 100] from a fixed seed). The keywords `dedupe_hyperedges`, `max_cardinality`,
`relabel` and `random_weights=(lo, hi, seed)` override them. Malformed files raise
`dyng.FileFormatError` with `.path`, `.line` and `.column`. The formats are specified in
{doc}`../api/file_formats`.

## Generate hypergraphs and batches

```python
h = dyng.generators.uniform_hypergraph(num_vertices=1000, num_hyperedges=5000, seed=1)
for k in range(3):
    b = dyng.generators.random_hyperedge_batch(h, mix="reuse", size=100, seed=k)
    h.apply(b)
```

The streams are bit-exact per seed and major version on every platform. The mixes include the
degenerate cases (`"delete_all"`, `"dead_ids"`, `"limits"`, `"skewed"` with a hyperedge of 1,100
members, a batch `"larger"` than the hypergraph). ESCHER-GPU's own generators are in
`dyng.generators.legacy` (`EscherEngine(seed)`, `escher_hypergraph()`, `EscherBatchGenerator`),
which redraw its published runs exactly.

## Test with Hypothesis

```python
from hypothesis import given, strategies as st
from dyng.testing import strategies as dst

@given(dst.hypergraphs(), st.data())
def test_ids_are_valid(hg, data):
    h, model = hg.to_hypergraph(), hg.model()
    for _ in range(3):
        b = data.draw(dst.hyperedge_batches(model))
        s = h.apply(b.to_batch())
        model.apply(b.delete_hyperedges, b.insert_hyperedges, None, s.inserted_ids)
        v = h.h2v()
        assert [v.row(i).tolist() for i in range(v.num_rows)] == model.rows_by_id(
            h.hyperedge_id_bound)
```

Batches are drawn from the model's current state, because the ids a batch may delete depend on
the ids the earlier batches got; failing cases shrink to small hypergraphs and short batches.

## Not yet

`dyng.update()` over hypergraphs (several results, one batch) arrives with the first hypergraph
algorithm, `triad_count` (0.3); until then `Hypergraph.apply()` changes the structure. The list
of C++ symbols without a Python counterpart is {doc}`../developer/python_gaps`.
