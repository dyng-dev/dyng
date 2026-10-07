# Getting started

dynG is a C++17 library with a Python package on top. The Python package (`pip install dyng`)
contains the sequential and OpenMP backends and needs no GPU; the CUDA backend comes with the C++
build and, from 0.2, as the Python plugin wheels `dyng-cu12` / `dyng-cu13`
(`pip install "dyng[cu13]"`, {doc}`install`; until 0.2.0 is released, in the release candidate
0.2.0rc1 on TestPyPI).

A first result in ten lines, in Python:

```{literalinclude} ../../README.md
:language: python
:start-at: "import dyng"
:end-before: "```"
```

and in C++:

```{literalinclude} ../../README.md
:language: cpp
:start-at: "#include <dyng/dyng.hpp>"
:end-before: "```"
```

Both are the README's quickstarts, which the test suite runs (the Python one prints
`[0, 3, 1, 4]` and then `[0, 4, 1, 2] 2`, the C++ one `0 4 1 2 | invalidated 2`).

```{toctree}
:maxdepth: 1

install
first_update_python
first_update_cpp
```
