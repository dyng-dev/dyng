# Roadmap

What each release of dynG will contain, what is deliberately not planned, and where
contributors can help. Dates are not promised; the order is. The maintainers' detailed plan is
{doc}`developer/plan`.

## Where we are

**Pre-alpha, before 0.1.0.** `sssp` works on all three backends (sequential, OpenMP and CUDA)
and is byte-identical to the original MOSP-OpenMP and MOSP-CUDA codes on their 495-case golden
corpus, within the performance gates against the originals. The project infrastructure (this
site, CI, community files) is in place. `cycle_count` is being ported (its CPU backends first).

## 0.1.0: the first release

- `sssp` (dynamic single-source shortest paths, DynaMOSP) and `cycle_count` (dynamic k-bounded
  directed cycle counts, TruCy / DynTruCy), each with sequential, OpenMP and CUDA backends and
  parity-tested against the original code.
- Several results kept on one graph with one call: `dyng::update(res, g, batch, r1, r2)`.
- The C++ API of these headers frozen for 0.1 (SemVer from here on).
- `pip install dyng`: a CPU wheel (sequential and OpenMP backends) and a command-line
  interface.
- Documentation: getting started, the update model, the algorithm pages, the API reference.

## 0.1.x: hardening

- CUDA wheels as plugins: `pip install "dyng[cu12]"` or `"dyng[cu13]"`.
- Tutorials on writing your own dynamic algorithm (`dynamic_bfs`, `triangle_delta`).
- Hosted documentation (Read the Docs) and a citable DOI per release (Zenodo).
- Fuzzers for the file readers, mutation checks, sanitizer jobs.

## 0.2.0

- `mosp`: multi-objective shortest paths (DynaMOSP), and a second `sssp` engine for GPUs
  without cooperative launch.
- The dynamic **hypergraph** with the ESCHER storage, and `triad_count` (hypergraph h-motif
  triads, ESCHER / ESCHER+).
- Experimental: counting your own local hypergraph patterns (`count_local_patterns`).

## 0.3.0

- `label_propagation` (DynLP): binary harmonic label propagation under vertex batches.
- `hyper_sssp` (H-SOSP): shortest hyperpaths, on the shared shortest-path engine.
- conda-forge packages; aarch64 wheels.

## Towards 1.0

- **0.4:** an OpenMP backend for incremental `triad_count`; more cycle modes (Read-Tarjan,
  time windows, temporal); 64-bit hypergraph offsets.
- **0.5:** the framework for writing update algorithms becomes public (experimental).
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
