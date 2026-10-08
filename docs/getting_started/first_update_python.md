# A first update in Python

This program builds a four-vertex graph, computes shortest paths from vertex 0, and then applies
a batch with `dyng.sssp.update()`, which changes the graph **and** brings the result up to date.
It is `examples/python/first_update.py`, the Python twin of {doc}`first_update_cpp`; the test
suite runs it on both CPU backends and checks that it prints exactly this:

```text
invalidated 2, affected 2
distances 0 4 1 2
parents -1 0 0 2
```

```{literalinclude} ../../examples/python/first_update.py
:language: python
:start-at: "import sys"
```

What happened:

1. `dyng.Resources("openmp")` chose the backend (the CPU wheel has `"sequential"` and
   `"openmp"`; {doc}`../concepts/backends_and_resources`). The graph keeps its resources, so the
   calls below need no `resources=` argument.
2. `Graph.from_edges(src, dst, weights)` built the graph. The id arrays are Python lists, so the
   vertex ids are int32 (an int64 NumPy array would give int64 ids: dtypes are never narrowed
   silently; {doc}`../api/python/index`).
3. `dyng.sssp.compute()` produced the static result, the canonical shortest-path tree: distances
   (0, 3, 1, 4), vertex 1 reached through vertex 2.
4. `dyng.sssp.update()` applied the batch to `g` (its version went up by one) and repaired the
   tree: deleting the tree edge (2, 1) invalidated the subtree below vertex 1 ({1, 3}), and the
   insertion (2, 3) gave vertex 3 a shorter path. `st` holds the counters of this update.
5. `tree.distances` and `tree.parents` are {py:class}`dyng.Array` views of the result's memory
   (no copy); `tolist()` and `to_numpy()` copy them. After the next update of `tree` these views
   raise `StaleResultError`: read the properties again
   ({doc}`../concepts/results_and_versions`).

Keeping several results on one graph current takes one call, which applies the batch once:

<!-- snippet: first-update-multi -->
```python
import dyng

g = dyng.Graph.from_edges([0, 0, 2, 1, 3], [1, 2, 1, 3, 0], [4, 1, 2, 1, 1])
tree = dyng.sssp.compute(g, source=0)
hist = dyng.cycle_count.compute(g, max_length=4)       # cycles of length 2..4 (weights ignored)
batch = dyng.EdgeBatch(insert=([2], [3], [1]), delete=([2], [1]))
st_tree, st_hist = dyng.update(g, batch, tree, hist)
print(tree.distances.tolist(), hist.counts.tolist(), st_hist.cycles_removed, st_hist.cycles_added)
```

<!-- snippet-output: first-update-multi -->
```text
[0, 4, 1, 2] [0, 0, 0, 2, 0] 1 1
```

`dyng.sssp.update(g, batch, tree)` alone would have left `hist` stale: its next use raises
`dyng.StaleResultError`.

The same steps from the command line, on files in MOSP's formats, are in {doc}`../api/cli`
(`dyng sssp compute` and `dyng sssp update`) and in the tutorial
{doc}`../tutorials/sssp_mosp_files`. The next steps are the algorithm pages, {doc}`../algorithms/sssp`,
{doc}`../algorithms/mosp` (several weights per edge: multi-objective shortest paths) and
{doc}`../algorithms/cycle_count`, which have longer Python and CLI examples; the tutorial
algorithms {doc}`../algorithms/dynamic_bfs` and {doc}`../algorithms/triangle_delta` show how an
algorithm is written.
