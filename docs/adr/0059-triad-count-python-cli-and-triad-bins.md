# ADR 0059: triad_count in Python and on the command line, and the triad-bins JSON

- **Status:** Technical parts accepted under delegation (2026-10-21; GOVERNANCE.md, "Delegation
  of technical ADRs"); **the new public names await the author** (naming is not delegated: D9,
  the Open decisions table; taken on the plan's defaults until the 0.3.0 release). It carries out
  M9b step 2 (python-cli-wheels; the M9b acceptance criteria 5-7) under PLAN 5.4-5.7 and 7.7. It
  changes no rule the author approved: PLAN 5.4's rules hold as written (enumerations accept
  strings, no silent narrowing, the GIL released), every CLI option flag is still the kebab-case
  name of a field, the outputs are the original's bytes, and no frozen header is edited. The
  freeze of `dyng.triad_count` and `dyng triad_count` is M9b step 3's API review (D4).
- **Decided:** the author accepted the new public names (D9) on 2026-10-22 (GOVERNANCE.md,
  approvals log, 2026-10-22); `dyng.triad_count` and `dyng triad_count` are `stable` from 0.3.0 and
  `"dyng.triad_bins"` is a stable-tier format at version 1 (the same day's stability decision).
- **Date:** 2026-10-21
- **Deciders:** the AI assistant, on the author's behalf for the technical parts (M9b, step
  python-cli-wheels); the names: S M Shovan (2026-10-22)
- **Builds on:** ADR 0011 (the Python layer), ADR 0023 (the uniform contract), ADR 0025 (the
  command line), ADR 0041 section 1 and ADR 0046 (the hypergraph in Python; its item 10, the
  `dyng.update()` gap), ADR 0049 (the triad oracle), ADR 0052 (the triad_count API).

## Context

`dyng::triad_count` (ADR 0052) and `dyng::update()` over hypergraphs exist in C++ since M9a.
PLAN 5.5 shows the Python form (`dyng.triad_count.compute(hg)`, `kind="h_motif"`,
`dyng.triad_count.update(hg, hb, tri)`, `st.batch.inserted_ids`, `st.delta`,
`dyng.citation("triad_count")`); PLAN 5.6 asks for `dyng <algo> compute|update` with flags from
the option fields and the originals' output formats; PLAN 5.7 lists "triad bins (JSON)" among the
result writers. ADR 0046 left `dyng.update()` over hypergraphs to the first hypergraph algorithm.
Four points needed a decision: the name of the container parameter (the uniform contract of the
Python modules says `graph`), the form of an enumeration that the API names (`Pattern` in PLAN
5.5's `kind=str|enum`), whether `counts` is a `dyng.Array`, and what "A's count lines" means for
a command line that reads files where ESCHER-GPU generates its inputs.

## Decision

### 1. `dyng.triad_count`

| Name | Form |
|---|---|
| `Pattern` | `enum.StrEnum` with the four C++ enumerators (`h_motif`, `stathyper_type1`, `stathyper_type2`, `stathyper_type3`); a member is also its name (`Pattern.h_motif == "h_motif"`) |
| `PatternName` | `Literal[...]` of the four names (the type of `kind=` strings) |
| `MAX_BINS` | 30 (`triad_count::max_bins`) |
| `Options(kind="h_motif", cuda_engine="automatic")` | the C++ fields and defaults (tested equal to the native defaults); `kind` takes a name or a `Pattern` |
| `Stats` (frozen) | `update_stats`' fields, `batch` (`HyperApplySummary`, `inserted_ids` int64), `deleted_frontier`, `inserted_frontier`, `triads_removed`, `triads_added`, `delta` |
| `Result` | `counts`, `count(bin)`, `total`, `num_bins`, `kind` (a `Pattern`), `options`, `set_options()`, `graph_version`, `space`, `clone()`, `copy`/`deepcopy`; not picklable |
| `compute(hypergraph, *, kind=None, options=None, resources=None, **kwargs)` | `kind=` is the field (explicit for the signature and the help); the other fields as keywords |
| `update(hypergraph, batch, result, *, resources=None)` | `-> Stats` |

1. **The container parameter is `hypergraph`.** The uniform contract (`test_api_surface.py`)
   now reads the first parameter's name from the registry's `container` (`graph` for the graph
   algorithms): a `dyng.Graph` named `graph` and a `dyng.Hypergraph` named `hypergraph` say what
   they take. `dyng.update(graph, batch, *results)` keeps its first parameter's name (a frozen
   signature) and takes either container.
2. **`Pattern` is the first public enumeration class of the typed layer.** PLAN 5.4 rule 5
   (enumerations accept strings) holds: every function takes the names; the `StrEnum` adds a
   typed, discoverable form (PLAN 5.5's `kind=str|enum`) without a second spelling, since a
   member is its string. `Result.kind` returns a member, which compares equal to the name.
3. **`Result.counts` is a read-only uint64 NumPy copy**, not a `dyng.Array`: the counts are at
   most 30 host values on every backend (`triad_count::result::counts()` is host memory), so a
   copy costs nothing, and it never goes stale. **`Stats.delta`** is a read-only int64 array of
   the result's `num_bins` values (C++ keeps a fixed `std::array` of 30). Both are listed in
   `python_gaps.md`.
4. **Errors** are the library's (ADR 0011): an unknown name is `InvalidArgumentError` listing
   the names; a pattern that reads the line graph on a hypergraph without it is
   `InvalidArgumentError` whose message names the property (`hypergraph_properties::line_graph`,
   `store_v2h`); `cuda_engine="operators"` on cuda is `NotSupportedError`.

### 2. `dyng.update()` over hypergraphs (closes ADR 0046 item 10)

`python/bindings/update.cpp` gains, for both hypergraph instantiations, the branch of a
hypergraph: every result must be a `triad_count` result, its participant is
`detail::make_triad_count_participant<V, W>` on the holder's `for_update()` state, and the
participants run through `participant_of<hypergraph<V, W>>::run()`, the path of the C++
`dyng::update()` (every subtract phase on the old state, one commit, every add phase on the new
one). The hypergraph's lock and every result's are taken in address order. Two or more results
on one hypergraph (one per pattern) work; a result passed twice, no result, a graph result, or an
`EdgeBatch` are refused before anything changes.

### 3. Triad bins as JSON (`<dyng/io/result_io.hpp>`, tracked)

`io::triad_bins` (`kind`, `graph_version`, `counts`, optional `deltas`), `io::triad_bins_size()`,
`io::write_triad_bins_json()` (a stream and a path overload) and `io::read_triad_bins_json()`.
The document is RFC 8259 JSON, format `"dyng.triad_bins"`, version 1 (`docs/api/file_formats.md`,
"Triad bins"): keys `format`, `version`, `kind`, `num_bins`, `graph_version`, `counts`, `deltas`;
integers only where integers are meant; unknown keys ignored (forward compatible within version
1); the errors at line and column. The io module keeps its own list of the four names (it does
not depend on the algorithm; a test checks it against `triad_count::to_string()`). The reader
is a small strict recursive-descent parser (nesting at most 64 levels) with its libFuzzer target
`fuzz_triad_bins` (PLAN 5.7: every reader has one; the corpus, the replay and the writer
self-test in every test build). Python: `dyng.io.TriadBins` (`TriadBins.of(result, deltas)`),
`write_triad_bins()`, `read_triad_bins()`, `triad_bins_json()`. The oracle is bound too:
`dyng.testing.triad_reference(hypergraph, kind, threads=)` (ADR 0049's
`testing::triad_reference`).

### 4. `dyng triad_count compute|update`

1. **Flags from the fields** (ADR 0025 item 2): `--kind`, `--cuda-engine` from
   `dyng.triad_count.Options`; the read flags from `dyng.io.read_hypergraph`'s keywords
   (`--index-base`, `--dedupe-hyperedges`, `--max-cardinality`, `--relabel`, `--num-vertices`,
   `--weighted`, `--storage`, `--line-graph`, `--id-reuse`, `--capacity`, `--store-v2h`), with
   the help of `HypergraphReadOptions`' fields. The generator of the flags now unwraps
   `X | None` before it looks at `X` (a `bool | None` keyword is a `--x / --no-x` pair, an
   `int | None` an integer) and gives a union with one `Literal` that `Literal`'s choices
   (`kind: PatternName | Pattern`); no existing flag changes (none had such a hint).
2. **Not fields:** `--read-options escher_gpu|hsosp` is `read_hypergraph`'s `options=` (the
   name `--options` would read as the algorithm's options), `--min-initial-payload N` with
   `--capacity escher_gpu` is `CapacityPolicy.escher_gpu(N)` (ESCHER-GPU's `--payload-capacity`),
   `--input` (also `--hypergraph`), `--batches FILE...` (`.dgt` or `.dgb` by the file's magic, in
   order), `--format json|escher`, `--output`, `--compare-recompute` and the backend flags.
3. **`--format json`** (the default) writes the triad-bins JSON: the counts (after the last batch)
   and, for `update`, one delta row per batch. **`--format escher`** prints ESCHER-GPU@abcf9b2's
   lines byte for byte: `Hypergraph: N hyperedges, max vertex id V, P pins` (V is the vertex
   count, ESCHER-GPU's largest 1-based id), the baseline (`Motif counts (30 bins): ...` or
   `[GPU] TypeN baseline motif count...`), and per batch `Batch b (D deletions, I insertions)`
   (only when there is more than one batch, as the original) and the delta line. The `[GPU]`
   prefix is the original's text on every backend. ESCHER-GPU generates its hypergraph and
   batches; the parity harness records them as `.hg` and `.dgb` files, and on those files the
   output is A's standard output without its generator lines: the test
   `python/tests/test_triad_count_cli.py` diffs every make-test case (15 groups, 4 executables) against A's
   unpatched stdout of the golden set, and in hosted CI the regenerated groups against the
   committed fixture's counts.
4. **The expected ids of a `.dgb` recording are checked**: a batch whose inserted ids differ from
   the file's is an error (exit status 1), since later deletions would then name other
   hyperedges (the hypergraph was not built as the recording's run built it).

### 5. The new public names (for the author; D9)

Python: `dyng.triad_count` (`Pattern` and its four members, `PatternName`, `MAX_BINS`, `Options`
(`kind`, `cuda_engine`), `Stats` (`affected`, `iterations`, `frontier_visits`, `fallback_used`,
`converged`, `engine_used`, `batch`, `deleted_frontier`, `inserted_frontier`, `triads_removed`,
`triads_added`, `delta`), `Result` (`counts`, `count`, `total`, `num_bins`, `kind`, `options`,
`set_options`, `graph_version`, `space`, `clone`), `compute` (`hypergraph`, `kind`, `options`,
`resources`), `update` (`hypergraph`, `batch`, `result`, `resources`)); `dyng.update()` over a
`dyng.Hypergraph`; `dyng.io.TriadBins` (`kind`, `counts`, `graph_version`, `deltas`, `of`),
`dyng.io.write_triad_bins`, `read_triad_bins`, `triad_bins_json`;
`dyng.testing.triad_reference`. C++ (tracked, `<dyng/io/result_io.hpp>`): `io::triad_bins`
(`kind`, `graph_version`, `counts`, `deltas`), `io::triad_bins_size`, `io::write_triad_bins_json`,
`io::read_triad_bins_json`. The file format `"dyng.triad_bins"` version 1. The command line:
`dyng triad_count compute|update` with `--kind`, `--cuda-engine`, the read flags above,
`--read-options`, `--min-initial-payload`, `--input` / `--hypergraph`, `--batches`,
`--format json|escher`, `--output`, `--compare-recompute`.

## Consequences

- PLAN 5.5's triad_count lines run as ADR 0041 section 1 and ADR 0046 item 2 adapt them
  (`st.batch.inserted_ids` is already an int64 NumPy array, so the plan's `.to_numpy()` is
  dropped; `test_plan_examples.py`).
- `python_gaps.md` loses the `dyng.update()` gap and gains a triad_count table (the copies, the
  compile-time messages, the oracle's other entry points); `count_local_patterns` stays
  C++/nvcc-only (ADR 0058).
- Nine reader fuzzers: the fuzz workflow's dispatch limit is 660 s per target (was 750 s for
  eight) so that they and the build fit its 120 minutes.
- The stubs gain the native names (`TriadPattern`, `TriadCountOptions`, `TriadCountStats`,
  `TriadCountResult`, `triad_count_compute`, `triad_count_update`, `triad_bins_json`,
  `write_triad_bins`, `read_triad_bins`, `testing_triad_reference`, the hypergraph overloads of
  `update`); griffe reports additions only against `origin/main`.
