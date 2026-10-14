# ADR 0046: The hypergraph in Python

- **Status:** Accepted under delegation (2026-10-14; GOVERNANCE.md, "Delegation of technical
  ADRs"). It carries out the Python part of M8b step 5 (PLAN 5.4, 5.5, 5.7, 5.8; ADR 0011; the
  M8b acceptance criterion 17) and records where it differs from the plan's sketch. It changes no
  rule the author approved: the names PLAN 5.4 and 5.5 fix are kept (`dyng.Hypergraph`,
  `from_hyperedges`, `from_lists`, `dyng.HyperedgeBatch`, `dyng.io.read_hypergraph` with
  `dedupe_hyperedges`, `max_cardinality`, `random_weights=(lo, hi, seed)`, `storage=`,
  `line_graph=`, `id_reuse=`), and the rules of PLAN 5.4 hold as written (rule 2: no silent
  narrowing; rule 3: the GIL is released; rule 5: enumerations as strings; rule 8: every stable
  C++ symbol bound or listed in `python_gaps.md`).
- **Date:** 2026-10-14
- **Deciders:** the AI assistant, on the author's behalf (M8b, step python-freeze-records)
- **Builds on:** ADR 0011 (the Python layer), ADR 0030 / 0031 (the CUDA plugins, device
  arrays), ADR 0041 (the container), ADR 0045 (`.hg`, `.dgb`, the generators).

## Context

The hypergraph container, its files and its generators exist in C++ (ADRs 0041-0045). PLAN 5.4
lists `dyng.Hypergraph.from_hyperedges(offsets, members, ...)`, `dyng.Hypergraph.from_lists(...)`
and `dyng.HyperedgeBatch(...)`; PLAN 5.5 shows `dyng.io.read_hypergraph("coauth-DBLP.hg",
dedupe_hyperedges=True, line_graph="sorted_merge")` and `dyng.HyperedgeBatch(delete_hyperedges=
[17, 42], insert_hyperedges=[[3, 9, 12], [9, 12, 40, 41]])` and reads `st.batch.inserted_ids`.
`dyng.update()` over hypergraphs needs a hypergraph algorithm, which M9 brings.

## Decision

1. **Native classes** follow the `type_code` naming of ADR 0011: `HypergraphI32U` and
   `HypergraphI32I32` (holders of `hypergraph<int32, unweighted>` and `<int32, int32>`, the two
   instantiations), `HyperedgeBatchI32U` / `HyperedgeBatchI32I32` (the batch over Python arrays,
   kept alive, never copied: `hyper_batch_arrays`), and the native functions with `_i32_u` /
   `_i32_i32` suffixes (`read_hypergraph_*`, `write_hypergraph_*`, `read_hyperedge_batches_*`,
   `write_hyperedge_batches_*`, `write_binary_hyperedge_batches_*`, `uniform_hypergraph_*`,
   `escher_hypergraph_*`). Files: `python/bindings/hypergraph.cpp` (the container, its
   properties, the summary and the storage counters), `batch.cpp` (both batch kinds), `io.cpp`
   (`.hg`, the `.dgt` hyperedge batches, `.dgb` with `BatchReader`), `generators.cpp`,
   `testing.cpp` (`rebuild_line_graph`). Every native call that runs a kernel or I/O releases the
   GIL; the container holder's reader/writer lock serializes `apply()` against the other calls
   (ADR 0011, item 5).
2. **The typed layer** `python/dyng/hypergraph.py`: `Hypergraph` (`from_hyperedges(offsets,
   members, weights=None, *, num_vertices, properties, storage, line_graph, id_reuse, capacity,
   store_v2h, vertex_dtype, resources)`, `from_lists(hyperedges, weights=None, *, ...)`,
   `num_vertices`, `num_hyperedges`, `hyperedge_id_bound`, `version`, `space`, `properties`,
   `weighted`, `weights`, `h2v()`, `v2h()`, `h2h()`, `apply()`, `to_hyperedge_list()`, `clone()`,
   `to_backend()`, `store_stats()`, `check_integrity()`); the dataclasses
   `HypergraphProperties` and `CapacityPolicy` (the C++ field names and defaults, tested equal to
   the native defaults; `CapacityPolicy.escher_gpu(min_initial_payload)`,
   `.mosp_escher(extra)`, `.preset(name)`); `HyperApplySummary` (frozen; `inserted_ids` a
   read-only **int64** NumPy array); `StorageStats`; `HyperedgeBatch(delete_hyperedges=None,
   insert_hyperedges=None, insert_weights=None, *, insert_offsets, insert_members,
   delete_incidences, insert_incidences)`; `HyperedgeList`, `Incidence`, `LineGraph` (named
   tuples of NumPy arrays).
3. **Keywords.** The properties' field is `reuse` (the C++ name, PLAN 5.4 rule 5); the keyword of
   the constructors and readers is `id_reuse=` (PLAN 5.5's spelling, the name of the enumeration).
   `capacity=` takes a `CapacityPolicy` or a preset name (`"escher_gpu"`, `"mosp_escher"`,
   `"default"`). Every enumeration is a string (`"sorted_merge"`, `"insert_first"`); an unknown
   one raises `InvalidArgumentError` listing the valid names.
4. **Dtype dispatch (PLAN 5.4 rule 2).** The library compiles int32 vertex ids only for
   hypergraphs (PLAN 4.4.3). Member arrays that declare int64 (NumPy's default integer type)
   raise `NotSupportedError` naming the supported combinations and the escape
   `vertex_dtype="int32"`, under which every value is range-checked; Python lists take int32 when
   they fit. Hyperedge ids are int64 everywhere and never narrowed: deletions, offsets,
   `inserted_ids`, `v2h` and `h2h` columns. Batches convert their members to the hypergraph's
   type with a range check at every use (the graph batches' rule).
5. **Views are host copies.** `h2v()`, `v2h()` and `h2h()` return NumPy copies with the version
   they describe: the C++ views are invalidated by the next `apply()` and live in device memory
   on CUDA, and a zero-copy device view of a container needs the staleness rule `dyng.Array` has
   for results (the rule of `graph::view()`, `python_gaps.md`).
6. **Files.** `dyng.io.read_hypergraph(path, *, options, index_base, dedupe_hyperedges,
   max_cardinality, relabel, random_weights, num_vertices, weighted, <the property keywords>,
   resources)` with `options="escher_gpu" | "hsosp" | HypergraphReadOptions` (the C++ option
   sets), `read_hypergraph_arrays()` (the arrays and `HypergraphFileInfo`), `write_hypergraph()`;
   `read_hyperedge_batches()` / `write_hyperedge_batches()` (`.dgt`); `.dgb` through
   `write_binary_batches(path, batches, *, deleted_members, expected_ids, insert_first,
   vertex_dtype)` (graph or hypergraph batches, one kind per file), `read_binary_batches()` and
   the streaming `BatchReader` (iterating gives `BinaryBatchRecord`: the batch id, the batch, the
   deleted member sets and the expected ids). Python batches are host objects, so `BatchReader`
   reads host views; the C++ reader's pinned staging serves CUDA callers (`python_gaps.md`).
7. **Generators.** `dyng.generators.uniform_hypergraph()` / `uniform_hypergraph_arrays()` and
   `random_hyperedge_batch(hypergraph, *, mix, size, ...)`, with `HyperedgeBatchMixName` the 13
   mixes as strings; `dyng.generators.legacy.EscherEngine(seed)` (the single
   `std::mt19937_64` of an ESCHER-GPU run, shared like the C++ engine),
   `escher_hypergraph[_arrays](engine, ...)`, `EscherBatchGenerator(engine, ..., batch_size)` and
   `standard_library_matches_legacy_streams()`.
8. **Testing.** `dyng.testing.rebuild_line_graph(hypergraph)` (the C++ oracle);
   `dyng.testing.oracles.HypergraphModel` (pure Python: the live member sets by id, the checks
   any valid id choice must pass, the counts of a summary, a brute-force line graph);
   `dyng.testing.strategies.hypergraphs()` and `hyperedge_batches(model)` (Hypothesis, with
   shrinking; a batch is drawn from the model's current state because ids depend on history).
9. **Python replays of the dumps.** `python/tests/test_hypergraph_parity.py` replays the
   committed batch extracts of one A case (016-main) and one B case (014-test_h2h_delta, 9
   sessions) through `dyng.Hypergraph` with the decoding rules of the C++ replay, read with the
   repository's trace reader `parity/escher_trace.py` (no new test data: the repository's test
   data is at its 1 MB budget); on CUDA too, in `test_hypergraph_cuda.py`.
10. **`dyng.update()` over hypergraphs** arrives with M9's `triad_count`, the first result type it
    can update: listed in `python_gaps.md`.

## Consequences

- PLAN 5.5's hypergraph lines run as written except `dyng.triad_count` (M9) and the
  `slack_csr` options, which raise `NotSupportedError` naming 0.4.
- The stubs `_core.pyi` gain the hypergraph classes and functions (`regen.py --stubs`), and
  `griffe` reports the new names as additions.
