# dynG

**dynG** keeps the results of graph algorithms up to date while the graph changes in batches of
edge insertions and deletions, without recomputing from scratch. It unifies the research codes of
DynaMOSP (dynamic shortest paths) and TruCy / DynTruCy (cycle counting), with more algorithms
(multi-objective shortest paths, hypergraph motifs, label propagation) to follow.

<!-- snippet: pypi-readme -->
```python
import dyng

g = dyng.Graph.from_edges([0, 0, 1, 2], [1, 2, 2, 3], [4, 1, 1, 5])   # src, dst, weights
tree = dyng.sssp.compute(g, source=0)                                # shortest-path tree
batch = dyng.EdgeBatch(insert=([1], [3], [1]), delete=([0], [2]))
stats = dyng.sssp.update(g, batch, tree)                             # incremental update
print(tree.distances.to_numpy(), stats.invalidated)

hist = dyng.cycle_count.compute(g, max_length=4)                     # simple cycles by length
print(hist.counts.tolist(), hist.total)
```

The package also installs the `dyng` command line, which reads and writes the formats of the
original tools:

```console
$ dyng cycle_count compute --graph DD_A.txt --max-length 6       # histogram CSV
$ dyng sssp compute --graph roadNet-CA_ --out init                # MOSP's distance and tree files
$ dyng sssp update --graph roadNet-CA_ --changes batch --init init --out updated
```

This wheel contains the sequential and OpenMP backends (`dyng.Resources("openmp")`); the CUDA
backends follow as plugin wheels in the 0.1.x releases. The C++ library, the documentation and
the parity reports with the original codes are at <https://github.com/dyng-dev/dyng>.

Cite dynG and the papers behind the algorithms you use: `dyng.citation("sssp")`.

License: Apache-2.0.
