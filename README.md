# dynG: dynamic graph and hypergraph updates on GPUs

> **Pre-alpha.** dynG is under active development and has not been released. Nothing here is
> stable yet: APIs, file formats and build options may change without notice until 0.1.0.
> The PyPI package `dyng` 0.0.1 is only a name reservation.

dynG is a C++17/CUDA library (with Python bindings planned) that keeps the results of graph and
hypergraph algorithms up to date while the structure changes in **batches** of insertions,
deletions and weight changes, without recomputing from scratch. It unifies the research codes
of several published systems for dynamic graphs and hypergraphs on GPUs into one library with
one API, one update model, sequential, OpenMP and CUDA backends, and parity tests against the
original implementations.

**Keywords:** dynamic graph, batch update, GPU, CUDA, OpenMP, hypergraph, shortest paths, cycle
counting, triad counting, label propagation.

## Status

| Area | State |
|---|---|
| Core (resources, backends, memory, arrays, errors, logging, profiler) | working on the host (M1a) and on CUDA (M1b: streams, stream-ordered memory resources, device buffers, profiler device times) |
| Graph container (compact rows, MOSP batch semantics) and MOSP-format I/O | working; the updated CSR is byte-identical to MOSP-OpenMP's `applyChangeBatch` |
| `sssp`: dynamic single-source shortest paths (DynaMOSP SOSP update), sequential and OpenMP | working; byte-identical to MOSP-OpenMP@c352151 on its 495-case golden corpus ([parity certificate](parity/results/M1a.md)) |
| `sssp` on CUDA (the fused persistent cooperative kernel) | working (M1b); byte-identical to MOSP-CUDA@e220ee2 on the same corpus, cross-backend equal, performance gates recorded ([M1b certificate](parity/results/M1b.md)) |
| `cycle_count`: exact k-bounded directed cycle histograms (TruCy/DynTruCy update), sequential and OpenMP | working (M2a); bit-identical to CycleEnumeration-GPU@0a976ad on its 24-case golden corpus ([parity certificate](parity/results/M2a.md)) |
| `cycle_count` on CUDA | in progress (M2b) |
| `mosp`, `triad_count` (ESCHER/ESCHER+), hypergraph container | planned (0.2) |
| `label_propagation` (DynLP), `hyper_sssp` (H-SOSP) | planned (0.3) |
| Python package (`pip install dyng`) | planned (0.1) |

## The name

**dynG = dynamic graphs.** The name is written *dynG* (lower-case *dyn*, capital *G*); all code
identifiers are lower case (`namespace dyng`, `#include <dyng/...>`, `import dyng`). dynG is
inspired by [Gunrock](https://github.com/gunrock/gunrock) and follows ideas of RAPIDS cuGraph
and RAFT, but it is **not affiliated** with either project.

## Building from source

Requirements: Linux, a C++17 compiler (GCC >= 11 or Clang >= 15; CI builds with GCC 12/13 and
Clang 17/18), CMake >= 3.30, Ninja, and optionally OpenMP and the CUDA toolkit (>= 12.4; an
sm_75 or newer GPU to run the CUDA backend). No GPU is needed for the CPU backends.

```bash
git clone https://github.com/dyng-dev/dyng.git
cd dyng
conda env create -f environment.yml      # the dyng-dev environment: cmake, ninja, clang-format, ...
source scripts/dev_env.sh                # activates it and sets DYNG_SCRATCH (and nvcc if present)

cmake --preset dev                       # Debug, tests on, warnings as errors
cmake --build --preset dev
ctest --preset dev                       # unit, randomized and fixture-parity tests (no GPU)

ci/check.sh                              # the full local gate: format, cpu-only + dev presets, REUSE, docs
```

Other presets: `cpu-only` (Release, CPU backends), `release`, `relwithdebinfo`, `asan`, `tsan`
and `parity` (the flags of the original research codes, used for parity and performance runs).
`ci/docs.sh` builds the documentation site (Doxygen with warnings as errors, Sphinx, link
check) into `build/docs/html`; `ci/docs.sh --doxygen-only` (or the target `docs-doxygen`) runs
only the Doxygen check of the public headers.

### Build with CUDA

The CUDA backend needs the CUDA toolkit >= 12.4 (CUDA 13.1 is the development toolkit) and, to
run, a GPU of compute capability 7.5 (Turing) or newer. `scripts/dev_env.sh` puts
`/usr/local/cuda-13.1/bin` on `PATH` when it exists (set `DYNG_CUDA_HOME` for another toolkit).
The CPU presets above never build CUDA; the CUDA presets are separate:

```bash
source scripts/dev_env.sh
cmake --preset dev-cuda                  # Debug, this machine's GPUs (native), host code -Werror
cmake --build --preset dev-cuda
ctest --preset dev-cuda -L gpu           # the CUDA tests (each skips when no device is visible)
ctest --preset dev-cuda -L cpu           # the CPU suites in the CUDA build

ci/gpu_local.sh                          # the local GPU gate (below)
ci/build_cuda.sh                         # compile-only release build, as cuda-build.yml does
```

| Preset | Use |
|---|---|
| `dev-cuda` | development: Debug, `native` architectures, tests on |
| `release-cuda` | the release architecture list (sm_75 to sm_120 SASS plus PTX) |
| `parity-cuda` | the flags of MOSP-CUDA (`-O3 -lineinfo`, sm_86): parity and performance runs |
| `sanitize-cuda` | for `compute-sanitizer` runs |
| `ci-cuda13`, `ci-cuda12` | compile-only builds of `.github/workflows/cuda-build.yml` (no GPU needed) |

`DYNG_CUDA_ARCHITECTURES` chooses `native` (the default; without a visible GPU the configure
step warns and uses the release list), `release` or an explicit list (for example
`-DDYNG_CUDA_ARCHITECTURES=86`). `ci/gpu_local.sh` builds `dev-cuda` and then runs
`ctest -L gpu` and `ctest -L cpu`, the sssp golden corpus on the cuda backend (when the goldens
exist, see below), `compute-sanitizer --tool memcheck` over every GPU test executable,
`--tool synccheck` over the CUDA sssp suite and the clang-tidy naming check on the CUDA branches;
it prints a Markdown summary and fails when the test GPU is not visible (the CUDA tests would
skip). On a shared machine
choose the test GPU with `DYNG_TEST_GPU` (default 1: GPU 0 is kept for timing runs); every heavy
step takes the shared lock `$DYNG_SCRATCH/perf.lock` (`DYNG_PERF_LOCK=` disables it) and
`DYNG_GPU_SKIP="memcheck tidy"` skips steps by name. `ci/gpu_local.sh --preset sanitize-cuda`
uses another CUDA preset. `ci/check.sh` takes the same shared lock for its heavy steps.

### Run the example

`examples/cpp/sssp_update.cpp` reads a graph in MOSP's CSR text format, computes a
shortest-path tree, applies a batch with `sssp::update()` and writes the updated tree in MOSP's
file format (CTest checks that the file equals the original's output):

```bash
case=cpp/tests/data/mosp_graph_io/testCase0
mkdir -p out
build/dev/examples/cpp/sssp_update $case/graphCsr $case/insert.txt $case/delete.txt out openmp
# +0 -1 edges, invalidated 0, affected 2, engine fused   (then the profiler's stage CSV)
cmp out/SSSPTreeUpdated.txt cpp/tests/data/mosp_sssp/testCase0/updated/obj0/SSSPTreeUpdated.txt
```

In a CUDA build the last argument can be `cuda` (`build/dev-cuda/examples/cpp/sssp_update ...
out cuda`): the graph and the tree then live on GPU 0 and the fused kernel runs the update, with
the same output file.

`examples/cpp/cycle_count_update.cpp` counts the directed cycles of length 2..k of an edge-list
graph (a TUDataset `*_A.txt` file loads directly), applies a generated batch with
`cycle_count::update()` and prints the histogram in CycleEnumeration-GPU's CSV format;
`dyng-compat-cycle-enum` reproduces the original `cycle-enum` CLI.

### Run the parity check

The parity harness ([parity/README.md](parity/README.md)) builds the pinned originals
MOSP-OpenMP@c352151 and MOSP-CUDA@e220ee2 from `git archive` copies in `$DYNG_SCRATCH`, exports
their golden outputs (495 cases, about 185 MB, outside the repository) and replays them against
dynG byte for byte on every backend:

```bash
git clone https://github.com/SMShovan/MOSP-OpenMP.git ~/Projects/MOSP-OpenMP   # the originals, once
git clone https://github.com/SMShovan/MOSP-CUDA.git ~/Projects/MOSP-CUDA
parity/build_reference.sh MOSP-OpenMP    # scratch copy of the pinned original, built
parity/build_reference.sh MOSP-CUDA      # the CUDA original (needs nvcc)
parity/export_goldens.py                 # the sssp golden corpus in $DYNG_SCRATCH/goldens
ci/check.sh --parity                     # the gate plus `ctest --preset parity -L parity` (CPU)
cmake --preset parity-cuda && cmake --build --preset parity-cuda
parity/compare.py --exe build/parity-cuda/tools/compat/dyng-compat-mosp --configs cuda
```

`cycle_count` has its own corpus from CycleEnumeration-GPU@0a976ad
(`parity/build_reference.sh CycleEnumeration-GPU`, `parity/export_goldens.py cycle_count`,
replayed by `parity/compare.py cycle_count`; see `parity/README.md`).

`parity/perf_ab.py` runs the performance A/B against the unpatched originals under the
exclusive perf lock (`parity/README.md`). The committed records are the
[M1a](parity/results/M1a.md), [M1b](parity/results/M1b.md) and [M2a](parity/results/M2a.md)
parity certificates.

## Using the library from C++

```cmake
find_package(dyng 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE dyng::dyng)
```

```cpp
#include <dyng/dyng.hpp>

int main() {
  auto res = dyng::resources::openmp(8);   // or resources::sequential(), resources::cuda()
  dyng::edge_list<std::int32_t, std::int32_t> edges;
  edges.num_vertices = 4;
  edges.num_weights = 1;
  edges.add_edge(0, 1, {4});
  edges.add_edge(0, 2, {1});
  edges.add_edge(2, 1, {2});
  edges.add_edge(1, 3, {1});
  // graph<> = int32 vertex ids, int32 edge offsets (ADR 0009; int64 past 2^31 - 1 edges), int32
  // weights.
  auto g = dyng::graph<>::from_edges(res, edges.view());

  auto tree = dyng::sssp::compute(res, g, /*source=*/0);    // canonical tree: lowest-id ties
  dyng::edge_batch<std::int32_t, std::int32_t> batch;
  batch.delete_edge(2, 1);
  batch.insert_edge(2, 3, {1});
  dyng::sssp::stats st = dyng::sssp::update(res, g, batch.view(), tree);  // applies the batch
  // tree.distances() == {0, 4, 1, 2}, tree.parents() == {-1, 0, 0, 2}; st.invalidated == 2
}
```

With `resources::cuda()` the graph and the tree live in device memory: read the results with
`dyng::to_vector(res, tree.distances())`. `examples/cpp/sssp_update.cpp` runs the same steps on
MOSP's text files, and
`dyng-compat-mosp` (`tools/compat`, built by the `dev` and `parity` presets and their CUDA
twins) reproduces the output files of MOSP's `mosp` driver for parity runs.

## How to cite

If you use dynG in academic work, please cite the software (`CITATION.cff`; GitHub shows a
"Cite this repository" button) and the paper behind each algorithm you use.
`dyng::citation("sssp")` returns the BibTeX; the entries live in `docs/references.bib`.

| If you use | Please also cite |
|---|---|
| `sssp`, `mosp` | DynaMOSP (IPDPS 2025) and its journal version (IEEE TPDS 2025) |
| `cycle_count` | TruCy / DynTruCy (IEEE Transactions on Computers, submitted) |
| `triad_count`, the ESCHER hypergraph storage | ESCHER (IPDPS 2026) and ESCHER+ (IEEE TKDE 2026) |
| `label_propagation` | DynLP (ACM ICS 2026) |
| `hyper_sssp` | H-SOSP (IA3 workshop at SC 2026) |

## License

dynG is licensed under the [Apache License 2.0](LICENSE). Redistributions must keep the
attribution notices in [NOTICE](NOTICE) (Apache-2.0 section 4(d)).

## Acknowledgements

dynG grew out of doctoral research at the Missouri University of Science and Technology,
advised by Prof. Sajal K. Das, with the co-authors listed in [AUTHORS.md](AUTHORS.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), [MAINTAINERS.md](MAINTAINERS.md), the
[Code of Conduct](CODE_OF_CONDUCT.md), [SUPPORT.md](SUPPORT.md) and [SECURITY.md](SECURITY.md). Design decisions are
recorded as ADRs in [docs/adr/](docs/adr/).
