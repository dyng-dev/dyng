# Roadmap

What each release of dynG will contain, what is deliberately not planned, and where
contributors can help. Dates are not promised; the order is. The maintainers' detailed plan is
{doc}`developer/plan`.

## Where we are

**0.3.0 is released** (`pip install dyng`; with the CUDA backend `pip install "dyng[cu13]"` or
`"dyng[cu12]"`, upgrading from an earlier release with `pip install -U`,
{doc}`getting_started/install`; the release checklist is {doc}`developer/release`). Everything
listed for 0.3.0 below is done: the dynamic hypergraph, `triad_count` (C++, Python and the
command line) and the file formats `.hg`, `.dgb` and the triad bins at version 1 are stable from
0.3.0 (the author's decision of 2026-10-22), `triad_count` is exact against the original
ESCHER-GPU code and within its release gates against it, and `count_local_patterns` is
experimental. The parity certificate of the release is `benchmarks/results/0.3.0/`, carried over
from the release candidate 0.3.0rc1, which was published on TestPyPI only.

`label_propagation` and `hyper_sssp` make **0.4.0**, whose release candidate **0.4.0rc1** is
prepared on the branch `release-0.4.0` (TestPyPI only: `pip install -i
https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ dyng==0.4.0rc1`; with
the CUDA plugin, `pip install "numpy>=1.26"`, then `pip install -i https://test.pypi.org/simple/
--no-deps "dyng==0.4.0rc1" "dyng-cu13==0.4.0rc1"`, {doc}`getting_started/install`). Everything
listed for 0.4.0 below is in it: `label_propagation` and `hyper_sssp` (C++, Python and the
command line), the graph additions of 0.4, the vertex-arrival generators, the label files, the
`slack_csr` hypergraph storage with its incidence edits and the format extensions of 0.4 are
stable from 0.4.0 (the author's decision of 2026-11-18), with the release gates against
LabelPropagation-CUDA and MOSP_ESCHER in the certificate `benchmarks/results/0.4.0rc1/`. As for
0.3.0, the AI assistant pushes the tags and creates the GitHub Release on the author's behalf, and
the author merges the release pull requests and approves the three PyPI uploads (`dyng-cu12` and
`dyng-cu13`, then `dyng`; {doc}`developer/release`). The conda-forge packages and the aarch64
wheels are deferred beyond 0.4.0 (below).

Before it, **0.2.0** (2026-10-14) made `mosp` stable (the author's decision of 2026-10-07),
byte-identical to the original MOSP-OpenMP and MOSP-CUDA codes and within its own release gates
against them, and brought the CUDA backend to PyPI as the plugin wheels `dyng-cu12` and
`dyng-cu13` (certificate `benchmarks/results/0.2.0/`). **0.1.0** (2026-10-02) made `sssp` and
`cycle_count` work on all three backends, byte-identical to the original codes on their golden
corpora and within the performance gates against them (certificate `benchmarks/results/0.1.0/`),
with the Python package, the `dyng` command line and the documentation. After 0.1.0 the releases come in smaller steps (the
author's decision of 2026-10-02): the hardening planned as 0.1.x and `mosp` made 0.2.0.

## 0.1.0: the first release

- `sssp` (dynamic single-source shortest paths, DynaMOSP) and `cycle_count` (dynamic k-bounded
  directed cycle counts, TruCy / DynTruCy), each with sequential, OpenMP and CUDA backends and
  parity-tested against the original code.
- Several results kept on one graph with one call: `dyng::update(res, g, batch, r1, r2)`.
- The C++ API of these headers frozen for 0.1 (SemVer from here on).
- `pip install dyng`: a CPU wheel (sequential and OpenMP backends) and a command-line
  interface.
- Documentation: getting started (Python and C++), the update model, the algorithm pages, the
  C++, Python and CLI references, and the history of the ported research codes.

## 0.2.0: `mosp`, CUDA wheels and hardening

- `mosp`: multi-objective shortest paths (DynaMOSP), **stable** (SemVer applies from 0.2.0), and
  a second `sssp` engine for GPUs without cooperative launch.
- CUDA wheels as plugins: `pip install "dyng[cu12]"` or `"dyng[cu13]"`.
- A tutorial on writing your own dynamic algorithm ("Your first dynamic algorithm") and the two
  tutorial algorithms `dynamic_bfs` and `triangle_delta` (teaching material).
- Hosted documentation on GitHub Pages: <https://dyng-dev.github.io/dyng/>.
- Fuzzers for the file readers, mutation checks of the golden suites, sanitizer jobs (ASan,
  UBSan, TSan) on every pull request.

Planned after 0.2.0 for 0.2.x releases, of which none was made, and not yet assigned to a later
release: the Python tutorials (dynamic SSSP in ten minutes, cycle counting on a changing graph,
MOSP with preferences). A citable DOI per release needs Zenodo, which the author may connect
later (checkpoint A4); it is not part of 0.2.0 or 0.3.0, and 0.1.0 has none.

## 0.3.0: the dynamic hypergraph

Built in three milestones: M8a (the ESCHER store, internal: one store that reproduces both
ESCHER-GPU's and MOSP_ESCHER's CBST, proven by trace replay against both), M8b (the hypergraph
container on it, its I/O and Python, the header freeze) and M9 (`triad_count`, run as M9a,
the counting exact against ESCHER-GPU on every backend with its gates, and M9b, the
experimental `count_local_patterns`, Python, the command line and the freeze). M8a is done
(merged as #14): the store replays both originals' CBST traces in
every recorded field under their presets `escher_gpu()` and `mosp_escher()`, except two
documented divergences (`construct_below_payload` and, for MOSP_ESCHER's constructs without row
counts, `construct_without_row_counts`) and 57 operations whose inputs the traces hold as hashes
only or not at all, and dynG's default compacts first and then grows. M8b is done too (merged as #15):
`dyng::hypergraph` with ESCHER-GPU's and MOSP_ESCHER's ids on every backend,
the line graph equal to ESCHER-GPU's after every batch at the same device memory, `.hg` and
`.dgb` files, the generators, Python (`dyng.Hypergraph`), and the hypergraph headers frozen.
M9a is done (merged as #16, with its review fix for the builds without OpenMP): `triad_count` exact against
ESCHER-GPU on all three backends by key and by content, incremental host engines (an untuned
transliteration of the wedge engine, the author's decision of 2026-10-15) beside the CUDA port,
counts exact up to 2^64 - 1, every paper gate within its bar (update per batch, full count, end
to end, the store accounting and the device-memory peak). M9b is done too (merged as #17): the experimental `count_local_patterns` (count your own pattern, an Op,
with the same engine), `dyng.triad_count` and `dyng triad_count`, the docs, the examples and the
tutorials, and `triad_count` frozen and `stable`, its gates read again on M9b's library. **M9 is
done.** The release preparation of 0.3.0 (R030) makes the hypergraph surface, `triad_count`
(C++, Python and the command line) and the formats `.hg`, `.dgb` and the triad bins at version 1
stable (the author's decision of 2026-10-22), fixes the container's OpenMP allocation-failure
defect (D13, ADR 0061), certifies 0.3.0rc1 and makes its release changes; the final-release pull
request (R031) makes 0.3.0 from the release candidate, with no change to the library.

- The dynamic **hypergraph** with the ESCHER storage, the `sorted_merge` line graph and row
  views, hypergraph generators, and binary batch files (`.dgb`), from C++ and Python (done in
  M8b).
- `triad_count` (hypergraph h-motif triads, ESCHER / ESCHER+), incremental on the sequential,
  OpenMP and CUDA backends.
- Experimental: counting your own local hypergraph patterns (`count_local_patterns`).

Deferred beyond 0.3.0:

- **The alignment flags** (`-falign-functions=64`, `-falign-loops=32`,
  `-Wa,-mbranches-within-32B-boundaries`; about 20 % faster end to end on the development
  machine, R020): the author's decision of 2026-10-08 makes them a performance option with their
  own ADR, a parity run of the release builds, a check with the wheels' toolchain and **a reading
  on a CPU without the jump-conditional-code (JCC) erratum**. The only reference machine, a Xeon
  Gold 6258R (Cascade Lake), has the erratum, and the third flag is its mitigation, so the reading
  that decides the option cannot be taken there: 0.3.0 is built and measured without the flags
  (every gate of its certificate), and the option waits for a machine without the erratum (the
  author agreed to this deferral on 2026-10-23, GOVERNANCE.md).
- **The micro-benchmark suites** (nvbench, Google Benchmark) and `dyng-bench` (PLAN 8.5, item 1):
  planned for a later release, not yet assigned; the paper suites of `benchmarks/paper/` and
  their certificates are the performance record until then (`DYNG_BUILD_BENCHMARKS` has no effect
  yet).

## 0.4.0

Built in four milestones, each merged into `main`: M11a (`hyper_sssp` and the `slack_csr` storage
as-is against MOSP_ESCHER, #22), M10a (`label_propagation` exact against LabelPropagation-CUDA,
#23), M11b (the shared shortest-path engine, Python, the command line and the freeze, #24) and
M10b (the operators engine, slack rows and the streaming mode, Python, the command line and the
freeze, #25), with a fix of the graph's stream-ordered release (#26). The release preparation
R040 records what is stable, carries over the open items, certifies 0.4.0rc1 and makes its
release changes.

- `label_propagation` (DynLP): binary harmonic label propagation under vertex batches. **Done
  (M10, frozen at the end of M10b; ADR 0109):** the port of LabelPropagation-CUDA@a276a3a on the
  sequential and CUDA backends (the damped labels byte for byte the original's, within the gates
  against it), slot rows and vertex batches, the streaming mode on slack rows, the batch files'
  vertex records, Python with zero-copy labels, `dyng label_propagation`, the kNN recipe
  `datasets/preprocess/knn_graph.py` and the tutorial {doc}`tutorials/streaming_label_propagation`.
  **Stable from 0.4.0** (the author's decision of 2026-11-18): the C++ header, `dyng.label_propagation`,
  `dyng label_propagation`, the graph additions (`graph::with_capacity`, slot and slack rows, the
  vertex operations, the float graphs) and their Python counterparts, the vertex-arrival
  generators, the label files and the `.dgt` / `.dgb` vertex records and float weights; its three
  testing oracles stay frozen outside the stable tier. An OpenMP backend stays open to
  contributors (0.5 or later, if contributed).
- `hyper_sssp` (H-SOSP): shortest hyperpaths, with MOSP_ESCHER's kernels as the default CUDA
  engine and the shared shortest-path engine selectable (`engine::operators`; it gives the same
  results, and the gates kept the as-is kernels as the default: ADR 0095). On the sequential and
  CUDA backends (no OpenMP backend in 0.4: HD14), from C++, Python (`dyng.hyper_sssp`) and the
  command line (`dyng hyper_sssp`). **Stable from 0.4.0** (the author's decision of 2026-11-18).
- The hypergraph's `slack_csr` storage and incidence edits (HD9): MOSP_ESCHER's slack CSR line
  graph on every backend, the incidence edits (`+i` / `-i` in `.dgt`, the `.dgb` flag
  `incidence_edits`) on `slack_csr` storage only; `escher` storage refuses them, `triad_count`
  refuses `slack_csr` storage, and `overlaps` stays empty. Part of the stable hypergraph surface
  from 0.4.0, with the format extensions (the author's decision of 2026-11-18).
- Deferred beyond 0.4.0 (M11; HD5, HD9, HD14): MOSP_ESCHER's legacy H-SOSP batch generator (0.4's
  parity and gates replay MOSP_ESCHER's exported batches); incidence edits on `escher` storage and
  `triad_count` on `slack_csr` storage (their composition); the line graph's `overlaps`; an
  OpenMP backend of `hyper_sssp`.
- After 0.4.0 (from M11b's shared engine, ADR 0095): the shared engine's two located update
  costs (the bounded pointer-jumping rounds and the two seed launches on large line graphs);
  an as-is update with the shared static solve (0.69-0.74 of the ported recompute); and the
  rename-only move of `invalidate_subtree` and `packed_min` into `cpp/src/operators/` (two users
  now), with its own SASS record.
- After 0.4.0 (from M10b's per-algorithm kernel loading, ADR 0109; the author's decision of
  2026-11-13): a device allocation order independent of which kernel modules are resident. With
  fewer modules loaded, sssp's persistent kernel reads about 4 % slower on roadNet-PA's 50k
  batches (1.035-1.041 against `ae5fc47`; 1.000-1.001 with every kernel group loaded at start-up).
- After 0.4.0 (from R040's ADR 0024 reading against `v0.3.0`; the author's decision of 2026-11-21):
  the compact-row host apply under set semantics back to 0.3.0's cost. M10's graph additions (the
  vertex-batch and row-layout members of `normalized_batch` and the graph's properties) compile
  `apply_set_batch_host` differently, and cycle_count's OpenMP update on `github_k4_25000_25000_s1`
  reads 1.025 against `v0.3.0` (`graph.apply` +2.2 ms per 50,000-edge batch; not code placement).
  The fix is read again under ADR 0024.
- After 0.4.0 (from M10b's API review, ADR 0109): a vertex-capacity `graph::reserve()`, so that
  the first vertex batch on CUDA need not allocate (a new public name, for the author).
- **Deferred beyond 0.4.0** (the author's decision of 2026-11-18): **conda-forge packages**
  (checkpoint A4, the author's: no submission and no account action for 0.4.0) and **aarch64
  wheels**. 0.4.0 ships the PyPI wheels of 0.3.0's platforms (x86_64 manylinux) only.
- After 0.4.0 (from M11b's review, {doc}`developer/retrospectives/M11`): a
  `hyper_sssp::clone(res, h)` overload that links the copy to its hypergraph at once (0.4.0
  documents that the copy's first `update()` links it).
- Still deferred from 0.3.0, not part of 0.4.0: the alignment flags (they still wait for a reading
  on a CPU without the JCC erratum), the micro-benchmark suites (see "Deferred beyond 0.3.0"),
  and the serial-path remedy of the hypergraph insert's D14 sub-regions (acknowledged by the
  author on 2026-10-23 and kept as they are; a later release may take it with its own reading).

## Towards 1.0

- **0.5:** performance tuning of `triad_count`'s OpenMP backend (0.3.0 ships incremental host
  engines on sequential and OpenMP, an untuned transliteration of the CUDA wedge engine: the
  author's decision of 2026-10-15); more cycle modes (Read-Tarjan, time windows, temporal);
  64-bit hypergraph offsets; ESCHER-GPU's coarse triangles (inner, outer, hybrid and hyperedge
  triangles over the vertex graph; PLAN 11.5's old 0.4 item, one minor up by Appendix F).
- **0.6:** the framework for writing update algorithms becomes public (experimental).
- **1.0:** the API of the core and the stable algorithms is frozen. The release criterion: an
  outside contributor has added an algorithm using only the documentation.

## Not planned

- Distributed memory, multi-GPU, and HIP or SYCL backends: the maintainers will not build them,
  but a contributor-led proposal is welcome (below).
- A general static graph analytics library: static `compute()` exists as the baseline and the
  oracle, not as a competitor to cuGraph or Gunrock.
- Graph databases, persistence and transactions.
- Floating-point weights for shortest paths (integer weights keep results exact).
- Paper claims that the original code never implemented: the kappa-truncated TruCy search,
  incident-vertex and temporal triads, the insert-only MOSP of thesis Chapter 5, open h-motifs.
  They are open to contributors (below); the algorithm pages say plainly what is and is not
  computed.
- kNN graph construction for label propagation in the library (a Python recipe instead:
  `datasets/preprocess/knn_graph.py`, 0.4).
- A C API before 1.0.

## Open to contributors

Each algorithm below starts with the
[new algorithm](https://github.com/dyng-dev/dyng/issues/new?template=new_algorithm.yml) issue
form, and a backend with the
[feature request](https://github.com/dyng-dev/dyng/issues/new?template=feature.yml) form; the
`good first issue` labels mark smaller entry points ({doc}`developer/labels`):

- the kappa-truncated cycle search, as an explicitly approximate mode;
- time-window and temporal cycle *updates*;
- the incremental insert-only MOSP of thesis Chapter 5;
- incident-vertex and temporal triads (THyMe+); open h-motifs;
- new dynamic algorithms: k-core, connected components, PageRank, BFS;
- an OpenMP backend for an algorithm that has only a sequential one (`label_propagation`: 0.5 or
  later, if contributed); new file readers;
- a HIP backend (a new backend tag next to `sequential`, `openmp` and `cuda`), and multi-GPU if a
  research need appears: the maintainers review and maintain a contributor-led backend, but will
  not write one.
