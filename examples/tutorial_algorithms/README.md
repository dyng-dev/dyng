# Tutorial algorithms

The reference solutions of the tutorials of `docs/tutorials`. They are not built with the library.

| Folder | Tutorial | Files |
|---|---|---|
| `my_bfs/` | "Your first dynamic algorithm" (`docs/tutorials/your_first_dynamic_algorithm.md`) | the three files the tutorial has the reader change after `scripts/new_algorithm.py my_bfs --family fixed_point --backends seq,omp`: `problem.hpp` and `my_bfs.cpp` (into `cpp/src/algorithms/my_bfs/`) and `my_bfs_test.cpp` (into `cpp/tests/algorithms/my_bfs/`) |

The tutorial quotes these files between their `// [tutorial: ...]` markers. `ci/scaffold_check.sh`
scaffolds `my_bfs`, copies the files in and runs its conformance kit, so a change of the scaffold
or of the framework that breaks the tutorial fails CI: update these files with it.

The library's own teaching algorithms, `dynamic_bfs` and `triangle_delta` (maturity `tutorial`),
live with the other algorithms in `cpp/src/algorithms/`.
