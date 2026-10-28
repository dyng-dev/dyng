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
CUDA plugin of the same version as `dyng` (`pip install "dyng[cu13]"`,
{doc}`../getting_started/install`, "CUDA from Python"); `h.to_backend(res)` copies it to another
backend.

## Apply batches and read the ids

<!-- snippet: hypergraph-python-apply -->
```python
import dyng

h = dyng.Hypergraph.from_lists([[0, 1, 2], [1, 3], [2, 3, 4]])
b = dyng.HyperedgeBatch(delete_hyperedges=[1], insert_hyperedges=[[0, 4], [4, 4, 1]])
s = h.apply(b)                 # one commit: version 0 -> 1
print(s.inserted_ids)          # int64 ids chosen by the store
print(s.deleted_hyperedges, s.dropped_members, s.skipped_ops, s.ignored_deletions)
print(h.version)
```
<!-- snippet-output: hypergraph-python-apply -->
```text
[1 3]
1 1 1 0
1
```

Hyperedge 1 is deleted and the two insertions get the ids 1 and 3 (here the first takes the
freed slot 1); the repeated member of `[4, 4, 1]` is dropped (`dropped_members`), and
`skipped_ops` is `ignored_deletions + dropped_members`. Every `apply()` is a new commit: the same
batch applied again would be version 2, with ids chosen again.

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

<!-- snippet: hypergraph-python-hypothesis -->
```python
from hypothesis import given, settings, strategies as st
from dyng.testing import strategies as dst

@settings(deadline=None)  # the first CUDA use initialises the device (some 0.3 s)
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

test_ids_are_valid()  # pytest finds it by its name; called here, it runs the examples once
```

Batches are drawn from the model's current state, because the ids a batch may delete depend on
the ids the earlier batches got; failing cases shrink to small hypergraphs and short batches.
`deadline=None` matters with a CUDA plugin: the hypergraphs are built on the default resources,
cuda there, and the first CUDA call of a process initialises the device, which takes longer
than Hypothesis's default deadline of 200 ms for one example (a `FlakyFailure` /
`DeadlineExceeded` on the first example otherwise). `resources=dyng.Resources("sequential")` in
the test avoids the device altogether.

## Several results, one batch: `dyng.update()`

`h.apply(batch)` changes the structure alone, and every result computed on `h` is then stale. To
keep results up to date, apply the batch through them: `dyng.update(h, batch, *results)` applies
it to the hypergraph **once** and updates every result (from 0.3 the results of
`dyng.triad_count`), returning their stats as a tuple in the order of the results:

<!-- snippet: hypergraph-python-update -->
```python
import dyng

h = dyng.Hypergraph.from_lists([[0, 1, 2], [1, 3], [2, 3, 4]], line_graph="sorted_merge")
motifs = dyng.triad_count.compute(h)                     # h_motif, 30 bins (reads the line graph)
type1 = dyng.triad_count.compute(h, kind="stathyper_type1")
batch = dyng.HyperedgeBatch(delete_hyperedges=[1], insert_hyperedges=[[0, 4], [4, 4, 1]])
st, st1 = dyng.update(h, batch, motifs, type1)           # one commit, both counts exact
print(st.batch.inserted_ids, motifs.total, type1.total)
```
<!-- snippet-output: hypergraph-python-update -->
```text
[1 3] 4 2
```

The algorithm is {doc}`../algorithms/triad_count`; the {doc}`../tutorials/hypergraph_triads`
tutorial does it step by step.

## Not yet

Incidence edits (`delete_incidences=`, `insert_incidences=`) and the `slack_csr` storage are
planned for 0.4. The list of C++ symbols without a Python counterpart is
{doc}`../developer/python_gaps`.
