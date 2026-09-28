# C++ API

The public C++ API lives in `cpp/include/dyng/` (namespace `dyng`) and compiles with a plain
host compiler, also in a build with the CUDA backend (CUDA objects appear only as opaque
handles, such as `stream_ref`); `#include <dyng/dyng.hpp>` includes all of it. These pages are
generated from the Doxygen comments of the public headers (Breathe over Doxygen XML) and are **curated**: one page
per Doxygen group, in the order a new user meets them. `dyng::detail` is private and not shown.

Link against the CMake target `dyng::dyng` (`find_package(dyng)`).

```{toctree}
:maxdepth: 1

core
graph
batch
io
sssp
generators
testing
```

The conventions every entry follows (a one-line brief, every parameter and exception, the
synchronization, backends, determinism and paper paragraphs) are in "Documentation and Doxygen" of
[CONTRIBUTING.md](https://github.com/dyng-dev/dyng/blob/main/CONTRIBUTING.md); Doxygen runs
with warnings as errors and `ci/doxygen_coverage.py` checks the rest.
