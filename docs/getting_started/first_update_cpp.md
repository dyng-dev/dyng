# A first update in C++

This program builds a four-vertex graph, computes shortest paths from vertex 0, and then applies
a batch with `sssp::update()`, which changes the graph **and** brings the result up to date. It
is `examples/cpp/first_update.cpp`: the `dev` build compiles it, and CTest runs it on every CPU
backend and checks that it prints exactly this:

```text
invalidated 2, affected 2
distances 0 4 1 2
parents -1 0 0 2
```

```{literalinclude} ../../examples/cpp/first_update.cpp
:language: cpp
:start-at: "#include <dyng/dyng.hpp>"
```

What happened:

1. `resources` chose the backend (Explanation: {doc}`../concepts/backends_and_resources`).
2. `compute()` produced the static result, the canonical shortest-path tree.
3. `update()` applied the batch to `g` (its version went up by one) and repaired the tree:
   the deletion of the tree edge (2, 1) invalidated the subtree below vertex 1, and the
   insertion (2, 3) gave vertex 3 a shorter path. The result now matches `g` again, so a
   later `update()` accepts it ({doc}`../concepts/results_and_versions`).

Build it against an installed dynG with `find_package(dyng)` ({doc}`install`). The next step is
the tutorial {doc}`../tutorials/sssp_mosp_files`, which runs the same steps on the file formats
of the original research code.
