# The algorithm template

`scripts/new_algorithm.py <name> --family fixed_point|aggregate_delta --backends seq[,omp]`
copies this folder into a new algorithm (PLAN Section 4.8) and runs `scripts/regen.py`. The
result compiles and passes the conformance kit on the first build: its `update()` applies the
batch and recomputes from scratch (`stats.fallback_used = true`, "start green").

Placeholders the script replaces:

| In the template | Becomes |
|---|---|
| `algorithm_template`, `ALGORITHM_TEMPLATE` | the name, in lower and upper case |
| `{{title}}`, `{{computes}}`, `{{family}}`, `{{backends}}`, `{{backends_doc}}`, `{{maintainer}}`, `{{since}}` | the manifest's fields |
| lines `//@@ <selector>` ... `//@@ end` (also `#@@`) | kept when the selector is the family (`fixed_point`, `aggregate_delta`) or a chosen backend (`openmp`), removed otherwise; the marker lines themselves are always removed |

Where the files go:

| Template file | Destination |
|---|---|
| `public/algorithm_template.hpp` | `cpp/include/dyng/<name>.hpp` |
| `manifest.toml`, `CMakeLists.txt`, `algorithm_template.cpp`, `problem.hpp`, `sequential.cpp`, `openmp.cpp` | `cpp/src/algorithms/<name>/` |
| `tests/*` | `cpp/tests/algorithms/<name>/` |
| `docs/algorithm_template.md` | `docs/algorithms/<name>.md` |

This folder is not built (its name starts with an underscore, so `scripts/regen.py` and CMake
skip it); `ci/scaffold_check.sh` generates an algorithm of each family from it, builds it and runs
its conformance kit.
