# The algorithm template (hypergraph)

`scripts/new_algorithm.py <name> --container hypergraph --family fixed_point|aggregate_delta
--backends seq[,omp]` copies this folder into a new hypergraph algorithm (PLAN Section 4.8;
ADR 0043) and runs `scripts/regen.py`. The result compiles and passes the conformance kit's
hypergraph axis on the first build: its `update()` applies the batch and recomputes from scratch
(`stats.fallback_used = true`, "start green"). The graph template is `../_template`; the two share
the placeholders, the markers and the destinations below.

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

The placeholders: the component label of every vertex (the smallest vertex id it reaches through
hyperedges; a fixed-point problem) and the number of incidences, the sum of |e| over the live
hyperedges (an aggregate-delta problem). Both read the hypergraph through its public `h2v()` view
(built at most once per version; the container's own work, not the algorithm's).

This folder is not built (its name starts with an underscore, so `scripts/regen.py` and CMake
skip it); `ci/scaffold_check.sh` generates a hypergraph algorithm of each family from it, builds
it and runs its conformance kit.
