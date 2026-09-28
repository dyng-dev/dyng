# Core

Resources and backends (sequential, OpenMP and CUDA), streams (`stream_ref`), memory resources
(host, pinned host, and the stream-ordered CUDA device resource) and buffers, copies between
memory spaces (`to_vector`, `copy`, `to_space`), array views, errors, logging, the profiler (with device times on CUDA),
statistics and the umbrella header `<dyng/dyng.hpp>`.

```{doxygengroup} core
:content-only:
:members:
```
