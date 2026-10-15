# Roadmap

What each release of dynG will contain, what is deliberately not planned, and where
contributors can help. Dates are not promised; the order is. The maintainers' detailed plan is
{doc}`developer/plan`.

## Where we are

**0.2.0 is released** (`pip install dyng`; with the CUDA backend `pip install "dyng[cu13]"` or
`"dyng[cu12]"`, upgrading from 0.1.0 with `pip install -U`, {doc}`getting_started/install`; the
release checklist is {doc}`developer/release`). Everything listed for 0.2.0 below is done:
`mosp` is stable from 0.2.0 (the author's decision of 2026-10-07), byte-identical to the
original MOSP-OpenMP and MOSP-CUDA codes and within its own release gates against them, and the
CUDA backend comes from PyPI as the plugin wheels `dyng-cu12` and `dyng-cu13`. The parity
certificate of the release is `benchmarks/results/0.2.0/`, carried over from the release
candidate 0.2.0rc1, which was published on TestPyPI only. The hypergraph and `triad_count`
follow as 0.3.0, `label_propagation` and `hyper_sssp` as 0.4.0.

Before it, **0.1.0** (2026-10-02) made `sssp` and `cycle_count` work on all three backends,
byte-identical to the original codes on their golden corpora and within the performance gates
against them (certificate `benchmarks/results/0.1.0/`), with the Python package, the `dyng`
command line and the documentation. After 0.1.0 the releases come in smaller steps (the
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

After 0.2.0, in 0.2.x releases: the Python tutorials (dynamic SSSP in ten minutes, cycle
counting on a changing graph, MOSP with preferences). A citable DOI per release needs Zenodo,
which the author may connect later (checkpoint A4); it is not part of 0.2.0, and 0.1.0 has none.

## 0.3.0: the dynamic hypergraph

Built in three milestones: M8a (the ESCHER store, internal: one store that reproduces both
ESCHER-GPU's and MOSP_ESCHER's CBST, proven by trace replay against both), M8b (the hypergraph
container on it, its I/O and Python, the header freeze) and M9 (`triad_count`). M8a is done
(pending the hosted CI and the author's merge): the store replays both originals' CBST traces in
every recorded field under their presets `escher_gpu()` and `mosp_escher()`, except two
documented divergences (`construct_below_payload` and, for MOSP_ESCHER's constructs without row
counts, `construct_without_row_counts`) and 57 operations whose inputs the traces hold as hashes
only or not at all, and dynG's default compacts first and then grows. M8b is done too (pending the hosted CI and
the author's merge): `dyng::hypergraph` with ESCHER-GPU's and MOSP_ESCHER's ids on every backend,
the line graph equal to ESCHER-GPU's after every batch at the same device memory, `.hg` and
`.dgb` files, the generators, Python (`dyng.Hypergraph`), and the hypergraph headers frozen.

- The dynamic **hypergraph** with the ESCHER storage, the `sorted_merge` line graph and row
  views, hypergraph generators, and binary batch files (`.dgb`), from C++ and Python (done in
  M8b).
- `triad_count` (hypergraph h-motif triads, ESCHER / ESCHER+).
- Experimental: counting your own local hypergraph patterns (`count_local_patterns`).

## 0.4.0

- `label_propagation` (DynLP): binary harmonic label propagation under vertex batches.
- `hyper_sssp` (H-SOSP): shortest hyperpaths, on the shared shortest-path engine.
- conda-forge packages; aarch64 wheels.

## Towards 1.0

- **0.5:** an OpenMP backend for incremental `triad_count`; more cycle modes (Read-Tarjan,
  time windows, temporal); 64-bit hypergraph offsets.
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
- kNN graph construction for label propagation (a Python recipe instead).
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
- an OpenMP backend for an algorithm that has only a sequential one; new file readers;
- a HIP backend (a new backend tag next to `sequential`, `openmp` and `cuda`), and multi-GPU if a
  research need appears: the maintainers review and maintain a contributor-led backend, but will
  not write one.
