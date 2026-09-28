# Backends and resources

## One handle, chosen at run time

Every public call takes a `dyng::resources` first. It names the **backend** (`sequential`,
`openmp`, `cuda`), and for CUDA the device and the stream, and it carries the memory resource,
the copy policy and an optional profiler. The backend is a run-time choice: one build of the
library contains every backend it was compiled with, and the program picks per call. Inside the
library a single dispatch maps the choice to a compile-time backend tag once per call, so the
hot loops have no run-time branching.

The design follows RAPIDS' `raft::resources` and keeps the rule that makes it safe to embed:
**no global state**. The library never creates streams on its own, never sets environment
variables and never changes the global OpenMP thread count; two `resources` objects are
independent.

## The sequential backend is the reference

Every algorithm must have a sequential backend. It is the oracle of the tests (in the sense of
Ginkgo's reference executor): the conformance tests compare every other backend with it, and a
contributor without a GPU can develop and test a whole algorithm on it. OpenMP and CUDA are
optimizations of the same algorithm and must give the same deterministic results; each
algorithm declares its **determinism level** (for `sssp`: bit-exact on every backend).

## Copies share settings

`resources` is cheap to copy, and copies share one handle: `attach_profiler`,
`set_memory_resource` and `set_copy_policy` on any copy apply to all. This lets you pass it by
value everywhere. Settings are made during setup; for independent settings (for example two
profilers), create two `resources`.

## Memory and copies

The memory resource follows the shape of CUDA's `cuda::mr` (a non-owning
`memory_resource_ref`), so an RMM pool or a custom allocator can be plugged in without
dependencies in the public headers. The **copy policy** decides what happens when an input sits
in the wrong memory space: copy silently, copy and warn (the default inside timed benchmark
runs, so a hidden copy cannot distort a measurement), or throw.

## Synchronous and asynchronous calls

Host backends complete every call before returning. The CUDA backend (milestone M1b) orders work
on the `resources`' stream; each function's reference entry says whether it is synchronous or
stream-ordered.
