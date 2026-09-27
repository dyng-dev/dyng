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
| Core (resources, backends, memory, arrays, errors, logging, profiler) | in progress (M1a) |
| `sssp`: dynamic single-source shortest paths (DynaMOSP SOSP update), sequential and OpenMP | working; byte-identical to MOSP-OpenMP@c352151 on its 388-case golden corpus ([parity certificate](parity/results/M1a.md)) |
| `sssp` on CUDA | planned (M1b) |
| `cycle_count`: dynamic k-bounded cycle counts (TruCy/DynTruCy) | planned (0.1) |
| `mosp`, `triad_count` (ESCHER/ESCHER+), hypergraph container | planned (0.2) |
| `label_propagation` (DynLP), `hyper_sssp` (H-SOSP) | planned (0.3) |
| Python package (`pip install dyng`) | planned (0.1) |

## The name

**dynG = dynamic graphs.** The name is written *dynG* (lower-case *dyn*, capital *G*); all code
identifiers are lower case (`namespace dyng`, `#include <dyng/...>`, `import dyng`). dynG is
inspired by [Gunrock](https://github.com/gunrock/gunrock) and follows ideas of RAPIDS cuGraph
and RAFT, but it is **not affiliated** with either project.

## Building from source

Requirements: Linux, a C++17 compiler (GCC >= 11 or Clang >= 15), CMake >= 3.30, Ninja, and
optionally OpenMP. The CUDA backends (CUDA >= 12.4) arrive in a later milestone; no GPU is
needed today.

```bash
git clone https://github.com/dyng-dev/dyng.git
cd dyng
conda env create -f environment.yml      # the dyng-dev environment: cmake, ninja, clang-format, ...
source scripts/dev_env.sh                # activates it and sets DYNG_SCRATCH (and nvcc if present)

cmake --preset dev                       # Debug, tests on, warnings as errors
cmake --build --preset dev
ctest --preset dev

ci/check.sh                              # the full local gate: cpu-only + dev presets, format, REUSE, docs
```

Other presets: `cpu-only` (Release, CPU backends), `release`, `relwithdebinfo`, `asan`, `tsan`
and `parity` (the flags of the original research codes, used for performance comparisons).

## Using the library from C++

```cmake
find_package(dyng 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE dyng::dyng)
```

```cpp
#include <dyng/dyng.hpp>

int main() {
  auto res = dyng::resources::openmp(8);   // or resources::sequential()
  dyng::edge_list<std::int32_t, std::int32_t> edges;
  edges.num_vertices = 4;
  edges.num_weights = 1;
  edges.add_edge(0, 1, {4});
  edges.add_edge(0, 2, {1});
  edges.add_edge(2, 1, {2});
  edges.add_edge(1, 3, {1});
  auto g = dyng::graph<std::int32_t, std::int64_t, std::int32_t>::from_edges(res, edges.view());

  auto tree = dyng::sssp::compute(res, g, /*source=*/0);    // canonical tree: lowest-id ties
  dyng::edge_batch<std::int32_t, std::int32_t> batch;
  batch.delete_edge(2, 1);
  batch.insert_edge(2, 3, {1});
  dyng::sssp::stats st = dyng::sssp::update(res, g, batch.view(), tree);  // applies the batch
  // tree.distances() == {0, 4, 1, 2}, tree.parents() == {-1, 0, 0, 2}; st.invalidated == 2
}
```

`examples/cpp/sssp_update.cpp` runs the same steps on MOSP's text files, and
`dyng-compat-mosp` (`tools/compat`) reproduces the output files of MOSP-OpenMP's `mosp` driver
for parity runs. The parity harness (`parity/`, see its [README](parity/README.md)) builds the
pinned originals from `git archive` copies, exports their golden outputs and replays them against
dynG byte for byte.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) and [GOVERNANCE.md](GOVERNANCE.md). Design decisions are
recorded as ADRs in [docs/adr/](docs/adr/).
