# Choose a backend

Every call takes a `dyng::resources` as its first argument; it decides where the work runs.

```cpp
auto seq = dyng::resources::sequential();   // one host thread: the reference backend
auto omp = dyng::resources::openmp(16);     // 16 OpenMP threads; 0 = omp_get_max_threads()
auto gpu = dyng::resources::cuda(/*device=*/0);   // from milestone M1b
auto dflt = dyng::resources();              // dyng::default_backend()
```

- **sequential** is the reference implementation of every algorithm. Use it for tests, for
  small inputs, and when you need a result to compare against.
- **openmp** uses host threads. `openmp(0)` takes the OpenMP default at the time of the call,
  which honours `OMP_NUM_THREADS`; dynG never changes the global OpenMP thread count.
- **cuda** runs on one GPU and stream. Before M1b it always throws `not_supported_error`.

Ask what this build and machine support before choosing:

```cpp
if (dyng::backend_available(dyng::backend::openmp)) { /* ... */ }
std::printf("default: %s\n", std::string(dyng::to_string(dyng::default_backend())).c_str());
```

A backend that was not compiled in throws `dyng::not_supported_error` when you create its
`resources`, never later in the middle of an algorithm. The deterministic results of an
algorithm (for `sssp`: distances, parents and the deterministic counters) are identical on every
backend; each algorithm page states its determinism level.

Copies of a `resources` object share one handle, so settings made on one copy
(`attach_profiler`, `set_memory_resource`, `set_copy_policy`) apply to all of them. Make those
settings during setup, and create a separate `resources` when you need independent settings
({doc}`../concepts/backends_and_resources`).
