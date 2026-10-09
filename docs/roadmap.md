# Roadmap

What each release of dynG will contain, what is deliberately not planned, and where
contributors can help. Dates are not promised; the order is. The maintainers' detailed plan is
{doc}`developer/plan`.

## Where we are

**0.1.0 is released** (`pip install dyng`; the release checklist is {doc}`developer/release`).
Everything listed for 0.1.0 below is done: `sssp` and `cycle_count` work on all three backends
(sequential, OpenMP and CUDA), are byte-identical to the original MOSP-OpenMP, MOSP-CUDA and
CycleEnumeration-GPU codes on their golden corpora, and are within the performance gates against
the originals (ADR 0018 and ADR 0021 record how the CUDA gates are read; the parity certificate
of the release is `benchmarks/results/0.1.0/`, carried over from the release candidate
0.1.0rc1, which was published on TestPyPI only); the Python package, the `dyng` command line and
the documentation are complete.

After 0.1.0 the releases come in smaller steps (the author's decision of 2026-10-02): the
hardening that was planned as 0.1.x and `mosp` make **0.2.0**, whose release candidate
**0.2.0rc1** is prepared on the branch `release-0.2.0` (TestPyPI only; with the CUDA plugin:
`pip install "numpy>=1.26"`, then `pip install -i https://test.pypi.org/simple/ --no-deps
"dyng==0.2.0rc1" "dyng-cu13==0.2.0rc1"`, {doc}`getting_started/install`).
Everything listed for 0.2.0 below is in it; `mosp` is stable from 0.2.0 (the author's decision of
2026-10-07), with its own release gates against the originals in the certificate
`benchmarks/results/0.2.0rc1/`. As for 0.1.0, the AI assistant pushes the tags and creates the
GitHub Release on the author's behalf; new for 0.2.0, the author merges the release pull
requests personally (0.1.0's were merged by the AI assistant) and approves three PyPI uploads
(`dyng-cu12` and `dyng-cu13`, then `dyng`; 0.1.0 had one, `dyng`; {doc}`developer/release`). The
hypergraph and `triad_count` follow as 0.3.0, `label_propagation` and `hyper_sssp` as 0.4.0.

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

- The dynamic **hypergraph** with the ESCHER storage, and binary batch files (`.dgb`).
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
