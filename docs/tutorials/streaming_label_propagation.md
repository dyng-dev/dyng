# Streaming label propagation

In this tutorial you run DynLP, dynG's `label_propagation`, twice. **Part 1** reproduces the
paper's setup on LabelPropagation-CUDA's own synthetic stream and gets the original's labels
byte for byte. **Part 2** runs your own data: feature vectors turned into a kNN similarity graph by
the recipe `datasets/preprocess/knn_graph.py`, arriving in batches, in the streaming mode (rows
that grow, the threshold taken from the current graph), with the labels handed to PyTorch without
a copy. It takes about ten minutes on a laptop CPU; every step also runs on a GPU with
`dyng.Resources.cuda()` (a CUDA plugin wheel: `pip install "dyng[cu13]"`).

What you need: `pip install dyng` (and PyTorch for the last step, optional), and a clone of the
repository for the recipe (`datasets/` is not in the wheel). The algorithm itself is described
on {doc}`../algorithms/label_propagation`; the row layouts on {doc}`../concepts/row_layouts`.

## The problem in two sentences

Some vertices of a similarity graph are labelled with a class, 0 or 1 (the seeds); every other
vertex should get a label F in [0, 1] that is the weighted average of its neighbours' labels
(the harmonic solution), and its class is F >= 0.5. Vertices arrive in batches with their edges
and leave, and `dyng.label_propagation.update()` brings the labels up to date by propagating only
from the vertices a batch affects.

## Part 1: the paper's setup, the original's bytes

LabelPropagation-CUDA generates a synthetic Erdős-Rényi stream: the full graph, the true class of
every vertex, and batches of arriving and departing vertices with their edges. dynG generates the
same stream byte for byte (`dyng.generators.legacy.dynlp_vertex_arrival_stream`) and runs the
**published setup**: the graph gets the stream's full degrees as its row capacities (slot rows that
never move), `compute()` runs on the empty graph (every label 0.5), and every batch is one
`update()` with the published threshold (the full graph's average edge weight). This is
`examples/python/label_propagation_torch.py`:

% quote: examples/python/label_propagation_torch.py#L20-L35 6d7e34fc1546
```{literalinclude} ../../examples/python/label_propagation_torch.py
:language: python
:lines: 20-35
```

The resources are explicit: `sequential` or `cuda` (label_propagation has no OpenMP backend in
0.4, and `dyng.Resources("openmp")` raises `NotSupportedError` naming the fix). The `damped`
schedule makes the labels bitwise reproducible: on the sequential backend they equal
`dyng-compat-dynlp`'s sequential output, and on cuda they equal the original's own
`dynlp_labels.txt` byte for byte (the example's CI test compares both against committed digests).
Run it:

```console
$ python examples/python/label_propagation_torch.py out sequential
batch 0: ...
...
$ dyng label_propagation update --generate legacy --total-vertices 5000 --num-batches 5 \
      --update-schedule damped --backend sequential --out run
```

Both write the labels in the original's format (`i label` with six decimals,
`dyng.io.write_labels`); `run/dynlp_labels.txt` and `out/dynlp_labels.txt` are the same bytes.

The accuracy the example prints is about 51 %: the synthetic weights carry little signal with 1 %
seeds, as the original's own README says ("it is not a quality measure of the update"). Quality is
measured against the harmonic solution (`dyng.testing.harmonic_solution`) and IrLP; real data,
part 2, is where the labels mean something.

## Part 2: your own data, streaming

### Feature vectors

Any n x d matrix of feature vectors works (image embeddings, TF-IDF rows); here two Gaussian blobs
stand in for them:

% quote: python/tests/test_label_propagation_tutorial.py#features 9c70697ecf4d
```{literalinclude} ../../python/tests/test_label_propagation_tutorial.py
:language: python
:start-after: "# [tutorial: features]"
:end-before: "# [tutorial: features end]"
:dedent: 4
```

### The kNN graph and the arrival batches

The DynLP paper built its non-graph datasets as cosine-similarity kNN graphs with k = 5; the
recipe does the same (scikit-learn when it is installed, a NumPy brute force otherwise: both write
the same files) and, with `--arrival-batches`, cuts the vertices into batches in a seeded random
order, a share of each batch's arrivals with their class as seeds, every edge in the batch of its
later endpoint, and departures of earlier vertices:

% quote: python/tests/test_label_propagation_tutorial.py#knn 2a9651e95a7d
```{literalinclude} ../../python/tests/test_label_propagation_tutorial.py
:language: python
:start-after: "# [tutorial: knn]"
:end-before: "# [tutorial: knn end]"
:dedent: 4
```

It writes `g.txt` (the graph, `u v w` with float weights in (0, 1]; `dyng.io.read_edge_list(...,
weight_dtype="float32")` and `dyng label_propagation compute --graph g.txt` read it), `g.dgb` (the
five batches, with vertex records: arrivals with their seed labels, the edges they bring,
departures), `g.row_capacity.txt` (each vertex's final degree), `g.true_labels.txt` and
`g.stream.json` (the published threshold). The paper's own datasets are not redistributed (the
recipe's header names where to get them).

### The streaming mode

You could run these batches in the published setup, since the recipe wrote the final degrees and
the threshold. A real stream does not know them, so run the **streaming mode**: slack rows
without capacity hints (a row that outgrows its slots moves to the tail of the slot arena, and the
arena is compacted when the dead slots outnumber the live ones) and tau of the current graph
(`similarity_threshold` left unset):

% quote: python/tests/test_label_propagation_tutorial.py#streaming 463c4c600ecf
```{literalinclude} ../../python/tests/test_label_propagation_tutorial.py
:language: python
:start-after: "# [tutorial: streaming]"
:end-before: "# [tutorial: streaming end]"
:dedent: 4
```

Each `update()` reports what the batch did to the rows (`st.batch.relocated_rows`,
`st.batch.compactions`) next to the propagation's counts. The two lines that differ from the
published setup are the layout and the threshold; the streaming mode has no performance gate, and
at 500k vertices it takes about 1.4x the published setup's update time.

### The labels

`lp.labels` is a read-only float32 `dyng.Array` over the result's memory: NumPy and PyTorch read
it without a copy.

% quote: python/tests/test_label_propagation_tutorial.py#labels 8be06ac0d3d6
```{literalinclude} ../../python/tests/test_label_propagation_tutorial.py
:language: python
:start-after: "# [tutorial: labels]"
:end-before: "# [tutorial: labels end]"
:dedent: 4
```

On the two blobs the accuracy is above 95 % (the test checks it, and that the labels are within
2000 delta of the harmonic solution of the final graph). With PyTorch, `torch.from_dlpack` views
the same memory (on cuda, device memory on `cuda:0`):

% quote: python/tests/test_label_propagation_tutorial.py#torch f5db32097bd7
```{literalinclude} ../../python/tests/test_label_propagation_tutorial.py
:language: python
:start-after: "# [tutorial: torch]"
:end-before: "# [tutorial: torch end]"
:dedent: 4
```

A view keeps showing the labels it was made from: if a view is still alive when the next `update()`
runs, the update copies the result's per-vertex state first and changes the copy (copy-on-write),
so drop views you no longer need.

## Where to go next

- {doc}`../algorithms/label_propagation`: the options (tolerance, the iteration cap and what happens
  at it, the engines), the determinism of each schedule, the gates against the original, and the
  differences from the paper.
- {doc}`../concepts/row_layouts`: slot rows, slack rows, vertex status and net-effect batches.
- `dyng label_propagation --help` and {doc}`../api/cli`: the same runs from the shell, with the
  original's output files.
