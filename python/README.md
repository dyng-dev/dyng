# dynG

**dynG** keeps the results of graph and hypergraph algorithms up to date while the structure
changes in batches of insertions and deletions, without recomputing from scratch. It unifies the
research codes of DynaMOSP (dynamic single- and multi-objective shortest paths), TruCy /
DynTruCy (cycle counting) and ESCHER / ESCHER+ (hypergraph triads): `dyng.sssp`, `dyng.mosp`,
`dyng.cycle_count`, the hypergraph `dyng.Hypergraph` and `dyng.triad_count` are stable, with more
algorithms (label propagation, shortest hyperpaths) to follow; `dyng.dynamic_bfs` and
`dyng.triangle_delta` are small tutorial algorithms.

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

w2 = [[4, 1], [1, 5], [1, 1], [5, 2]]                                # two objectives per edge
g2 = dyng.Graph.from_edges([0, 0, 1, 2], [1, 2, 2, 3], w2)
paths = dyng.mosp.compute(g2, source=0)                              # multi-objective (MOSP)
dyng.mosp.update(g2, dyng.EdgeBatch(delete=([0], [2])), paths)
print(paths.path_costs.to_numpy().tolist())                          # the K costs per vertex

h = dyng.Hypergraph.from_lists([[0, 1, 2], [1, 2, 3], [0, 2, 3]])    # three hyperedges
triads = dyng.triad_count.compute(h)                                 # the 30 h-motif bins
dyng.triad_count.update(h, dyng.HyperedgeBatch(insert_hyperedges=[[0, 1, 3]]), triads)
print(triads.total, triads.counts.nonzero()[0].tolist())             # triads, non-zero bins
```

The package also installs the `dyng` command line, which reads and writes the formats of the
original tools:

```console
$ dyng cycle_count compute --graph DD_A.txt --max-length 6       # histogram CSV
$ dyng sssp compute --graph roadNet-CA_ --out init                # MOSP's distance and tree files
$ dyng sssp update --graph roadNet-CA_ --changes batch --init init --out updated
$ dyng mosp update --graph roadNet-CA_ --changes batch --init init --preferences 4,1,4 --out out
$ dyng triad_count update --input coauth.hg --batches batches.dgb --format json   # triad bins JSON
```

This wheel contains the sequential and OpenMP backends (`dyng.Resources("openmp")`). The CUDA
backend comes as a plugin wheel of the same version, from 0.2: `pip install "dyng[cu13]"` (an
NVIDIA driver of CUDA 13) or `pip install "dyng[cu12]"` (CUDA 12); `dyng.Resources.cuda()` then
runs every algorithm on the GPU, and `dyng.show_config()` reports which module is used.
Upgrading from 0.2.0 or 0.1.0: `pip install -U "dyng[cu13]"` (or `"dyng[cu12]"`, or `-U dyng`
without the plugin); without `-U` pip keeps the installed release. The C++ library, the
documentation and the parity reports with the original codes are at
<https://github.com/dyng-dev/dyng>.

Cite dynG and the papers behind the algorithms you use: `dyng.citation("sssp")`.

License: dynG is Apache-2.0. The wheel also contains nanobind (BSD-3-Clause), robin-map (MIT)
and the GCC runtime (GPL-3.0-or-later WITH GCC-exception-3.1), which the distributions' licence
expression names; their licences are in `THIRD_PARTY_LICENSES.txt`.
