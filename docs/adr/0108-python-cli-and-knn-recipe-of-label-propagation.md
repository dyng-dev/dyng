# ADR 0108: label_propagation in Python and on the command line: the dtype rule, the surfaces and the kNN recipe

- **Status:** Accepted (M10b, step python-cli-knn-examples-wheels). The rules are the author's,
  **decided by the author on 2026-10-28**: AD14 (sheet E6; the Python dtype rule: the row "API
  semantics of 0.4 (sheet E2-E10)" of GOVERNANCE.md's approvals log), AD18 (sheet G1; the kNN
  recipe's licensing: the row "Licensing, data provenance, budgets and scope of 0.4 (sheet G1-G3,
  H1-H3)"), AD12 with E1's changes (the names, the CLI commands: the row "The new public names of
  0.4 (AD12, HD8, HD12; sheet E1)"), AD19 (explicit resources, the OpenMP refusal: the row
  "Licensing, data provenance, budgets and scope of 0.4") and C8 (the compat tool's argv: the row
  "The parity rules of 0.4 (sheet C1-C8)"). Under delegation (GOVERNANCE.md, "Delegation of
  technical ADRs"), technical scope only: everything else below. The names beyond AD12's list
  (section 5) are listed in GOVERNANCE.md's open-decision sub-block `M10 (AD)` for the API ADR of
  M10b's freeze (the row "Public names beyond AD12's list" of 2026-10-29). Provisional until that
  freeze (AD15).
- **Date:** 2026-11-12
- **Deciders:** the AI assistant, on the author's behalf
- **Builds on:** ADR 0011 (item 3: dtype dispatch), ADR 0025 (the Python CLI), ADR 0031 (device
  arrays and DLPack), ADR 0046 (the hypergraph Python layer), ADR 0059 (triad_count's Python and
  CLI precedent), ADR 0069 / 0103 (slotted and slack rows), ADR 0071 (the label_propagation API),
  ADRs 0105-0107 (the formats, the label files, the library generator).
- **Number:** M10's ADR range 0063-0079, 0100-0109 has two numbers left for M10b's steps 3 and 4
  (0108, 0109), so this one ADR records the three subjects the step names (the dtype rule, the
  Python and CLI surfaces, the kNN recipe), and 0109 stays for the API review of the freeze.

## Context

M10a and M10b steps 1-2 built `label_propagation` in C++ (CUDA and sequential), the float graphs,
the slot rows, the vertex batches, the generators and the formats. The Python package of 0.3 had
five graph types, all with int32 or no weights; its readers refused the vertex operations and
float32 weights of AD13 (step 2, deviation 11). PLAN 5.4-5.6 ask for the same names in Python as
in C++, the dtype dispatch of ADR 0011, zero-copy result arrays, a CLI whose flags are the option
fields and whose outputs are the originals' formats, and O10 for a kNN recipe.

## Decision

### 1. The dtype rule (AD14)

- `GraphType` gains a weight axis: int32, float32 or none. The package binds two float graphs,
  `(int32, int32, float32)` and `(int32, int64, float32)` (the library's instantiations; native
  classes `GraphI32I32F32`, `GraphI32I64F32` and the batch `EdgeBatchI32F32`).
- **float32 weights select the float graph.** float64 weights are accepted only with an explicit
  `weight_dtype="float32"` (`Graph.from_edges`, `Graph.from_csr`, `EdgeBatch`), and are then
  **checked**: every value finite and within float32's range, then rounded to the nearest float32
  (the conversion the caller asked for). Without it a float64 weight array raises
  `InvalidArgumentError` naming both fixes; nothing is narrowed silently. Integer weights keep
  their rule (int32, checked) and become a float graph only on `weight_dtype="float32"`. NaN and
  infinities are refused in every case.
- **Where the graph is already chosen** (a batch applied to a float graph, a writer): float32
  weights are used as they are; other weights only when every value is exactly a float32 (no
  rounding without the request). An int32 graph refuses floating-point batch weights.
- `Graph.weighted` keeps its 0.1 meaning (int32 weights); **`Graph.weight_dtype`** (int32,
  float32 or None) is added. `EdgeBatch` keeps its arrays as given (floating-point weights are
  accepted at construction and converted at use, the rule of 0.1 for ids).
- Readers: `dyng.io.read_edge_list(..., weight_dtype="float32")` (the C++ float reader, int32
  ids), `read_batches(..., weight_dtype="float32")` for a `.dgt` file of a float graph, and the
  `.dgb` reader by the header's weight code 3. The float32 fields of `Options`
  (`similarity_threshold`, `tolerance`) and the stream's threshold read back as the shortest
  decimal of the float32 (`1e-4` reads back as `1e-4`, and passing it back gives the same float).

### 2. The graph surface and the batches

- `Graph.with_capacity(row_capacity, *, vertex_capacity=None, directed=False, layout="slotted" |
  "slack", semantics="net_effect", headroom=0.125, edge_dtype=None, weight_dtype="float32",
  resources=None)`: C++ `graph::with_capacity` with the properties it requires (append order,
  parallel edges allowed, no transposed rows). The default edge offset type is int32 when the
  capacities fit it (slack rows: twice the capacities with the headroom, room for moved rows),
  else int64.
- `BatchSemantics.net_effect()`, its field `as_net_effect` and the semantics name `"net_effect"`;
  `ApplySummary.dropped_insertions`, `relocated_rows` and `compactions`;
  `Graph.num_active_vertices`; `Graph.vertex_status()` (a host copy, one `dyng.VertexStatus` code
  per vertex; the codes are mapped from the library's **by name**, C8's rule).
- `EdgeBatch(insert_vertices=..., vertex_labels=..., delete_vertices=...)` are applied by slot
  rows; the readers and generators fill them (the native batch tuple gains the three arrays);
  `EdgeBatch.insert_vertices`, `vertex_labels` and `delete_vertices` read them back. The `.dgb`
  writer sets the flag `vertex_records` only when a batch has vertex operations (AD13: flags
  fixed at open; an edge-only file is 0.3's, byte for byte).

### 3. `dyng.label_propagation`

- `Options` (the six C++ fields, keyword for keyword), `Stats` (the inherited fields and
  `components`, `sparsified_edges`, `initial_frontier`), `Result` and `compute(graph,
  seed_vertices=None, seed_labels=None, *, options=None, resources=None, **fields)`,
  `update(graph, batch, result, *, resources=None)`; `dyng.update` takes its results.
- `Result.labels` is a read-only float32 `dyng.Array` that views the result's state (host or
  device; ADR 0031's stream ordering for DLPack and `__cuda_array_interface__`);
  `torch.from_dlpack(result.labels)` and `numpy.from_dlpack` are zero-copy. **Copy-on-write** is
  the 0.1 holder's rule (ADR 0011 item 6): an export keeps the labels it was read from, the next
  update copies the result's per-vertex state once and writes the copy; with no export alive the
  update writes in place. The state is more than the n labels the step's text names: the seed
  labels and the propagation tags (the original's stamps) are copied with them, 13 bytes per
  vertex on the host (float label, int8 seed, int64 tag) and 17 on cuda (float label, int32 class,
  two int32 stamps, the host's int8 seeds); a clone of the labels alone would leave a state the
  next update cannot continue from. Also `seed_labels`, `options`, `set_options`,
  `graph_version`, `space`, `clone`, `reserve`.
- **AD19:** OpenMP resources raise `NotSupportedError` naming `'sequential'`, `'cuda'` and the fix
  `dyng.Resources("sequential")` (checked in Python before the call, since the default resources
  of the CPU wheel are OpenMP's, ADR 0070's C++ message having no Python spelling).
- `dyng.generators.vertex_arrival_stream(...)` and `dyng.generators.legacy.
  dynlp_vertex_arrival_stream(...)` return a `VertexArrivalStream` (the full graph's CSR, the
  true labels, the batches as `EdgeBatch` objects, `row_capacity`, `similarity_threshold`,
  `max_batch_vertices`, `max_batch_edges`, `write_graph_files(dir)` in the original's bytes
  through the library's writer). `dyng.testing.harmonic_solution(graph, seed_labels)` returns a
  `HarmonicResult` with `error(labels)`. `dyng.io.write_labels` / `read_labels`.

### 4. The command line (ADR 0025's style; AD12 with E1)

- `dyng label_propagation compute` (an edge list with float weights and `v class` seeds) and
  `update` (`--generate legacy|library` with the generator's keyword flags, or `--batches FILE..`
  with `--capacity auto|FILE` and `--true-labels`), one kebab-case flag per `Options` field (a
  test checks both directions; `--similarity-threshold` also takes `published` and `current`),
  `--layout`, `--backend sequential|cuda` (no OpenMP choice), `--per-batch-labels`, `--quiet`;
  outputs `dynlp_labels.txt` (the original's bytes) and `stats.csv` (the original's
  `results.csv` columns, then dynG's). M10's CLI is its own module
  (`python/dyng/cli/_label_propagation.py`), registered after mosp's.
- `dyng generate vertex_arrival_stream` (library) and `dyng generate dynlp_vertex_arrival_stream`
  (legacy), named as their Python functions (E1); each writes the batches (`.dgb` or `.dgt`), the
  capacities, the true labels, `stream.json` (tau and the sizes) and optionally the original's
  graph files. The group's help no longer says legacy-only.
- `dyng-compat-dynlp` already takes the original's complete flag set (M10a, ADR 0073); its tests
  now pass every flag at once and a negative `--maxIter` among C8's refusals.
- With `--batches`, `published` is the average weight of the files' edges as the original's
  `computeAvgEdgeWeight` sums them (each undirected edge twice, a double sum, rounded to float);
  the accumulation order differs from the original's CSR order, so the value may differ from a
  generator's tau in the last float bit: the exact tau is the number in `stream.json`
  (`--similarity-threshold <it>`); on the 5k5 stream both give the same labels, byte for byte.

### 5. The kNN recipe (O10; AD18)

`datasets/preprocess/knn_graph.py`, a repository script (not in the wheel):

- the symmetric kNN graph (k = 5) of cosine similarities, the union of the k-nearest pairs, each
  edge once with u < v, weight = s(u, v) in double precision, clipped to 1, written as float32
  by `dyng.io.write_edge_list` (the shortest round-trip decimal, read back exactly by the float
  reader and `dyng label_propagation compute`);
- **negative similarities:** `--negative drop` (the default) keeps only pairs with s > 0, so an
  anti-similar neighbour adds no edge (the paper's features, ReLU activations and TF-IDF, have
  s >= 0); `--negative shift` maps every weight to (1 + s) / 2 (dropping s = -1). Rows of norm 0
  are refused (their cosine is undefined);
- **scikit-learn** (BSD-3-Clause) `NearestNeighbors(metric="cosine", algorithm="brute")` is an
  optional import inside one function, with a NumPy brute force over row blocks as the fallback;
  both feed one ranking rule (similarity, then the smaller index) and one weighting, so they
  write identical files on inputs without ties at the k-th neighbour. No code is vendored or
  copied (the prototypes in the author's study repository were read, not copied);
- `--arrival-batches B` writes a vertex-arrival stream in `dyng generate`'s files: a seeded
  permutation split into B parts, a `--seed-fraction` of each part's arrivals with their class,
  every edge in batch max(batch(u), batch(v)) (the original's placement), from batch 1 on
  `--deleted-fraction` of a batch's arrivals in departures of earlier present vertices; the
  capacities (degrees) and the published tau (the original's average);
- **no dataset** is shipped or downloaded: the script's documentation links the paper's datasets
  (IMDB, ImageNet, Yelp, Amazon) at their publishers and describes the features the paper used.

### 6. Examples and wheels

`examples/cpp/label_propagation_stream.cpp` and `examples/python/label_propagation_torch.py` (at
most 60 lines, explicit resources) write `dynlp_labels.txt`; `examples/data/
label_propagation_5k5_damped.sha256` holds its digest per backend (cuda: the original's own file;
sequential: dyng-compat-dynlp's). The CTests `example.label_propagation_stream.*`,
`ci/cmake_consumer.sh` (against the installed package) and `test_examples.py` compare them. The
plugin smoke test runs the 5k5 damped stream on cuda under the oracle (AD6: a wheel is judged by
the oracles only), and `parity/wheel_vs_parity.py --task label_propagation` is a reported row.

## Consequences

- The Python batches carry the vertex operations and float32 weights of AD13; step 2's
  refusal is gone (its test now reads both).
- A float64 weight array that worked nowhere before still fails, with a message that names the
  two fixes; a floating-point batch weight is accepted at construction and refused by an int32
  graph at use (it was refused at construction in 0.3).
- `GraphType.weighted` keeps its meaning for every existing caller; the float axis is
  `GraphType.weight`.

## Alternatives considered

- **Narrowing float64 silently (as NumPy's `astype`):** rejected by AD14.
- **Rounding float64 batch weights for a float graph without a request:** rejected; only exact
  values pass, so a batch never changes a weight the caller wrote.
- **`Graph.weighted` true for float graphs:** rejected by AD14 (it keeps its meaning).
- **A `published` mode object in Python:** rejected as in C++ (ADR 0071 section 2): the setup is
  data (`stream.row_capacity`, `stream.similarity_threshold`).
- **scikit-learn as a required dependency, or its distances as the weights:** rejected (AD18: an
  optional dependency; one weighting keeps both paths' files identical).
- **A Python writer of the original's graph files:** rejected for the library's writer (one
  implementation, already checked against the original's digests).
