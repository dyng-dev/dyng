# Profile an update

Attach a `dyng::profiler` to the resources; every library call then records its stages and
counters under the names `<algo>.<hook>[.<sub>]` (for example `sssp.identify_affected` or
`graph.apply`), the steps of the update template ({doc}`../concepts/update_model`).

```cpp
dyng::profiler prof;                   // or dyng::profiler(dyng::profiler_options{...})
res.attach_profiler(&prof);            // not owned: prof must outlive the attachment
auto st = dyng::sssp::update(res, g, batch.view(), tree);
res.attach_profiler(nullptr);

prof.write_csv(std::cout);             // kind,name,value (compatible with MOSP's --timing CSV)
prof.write_json(std::cout);            // the same data as JSON
for (const auto& s : prof.stages()) {  // aggregates: name, depth, calls, host_ms, device_ms
  std::printf("%-28s %8.3f ms\n", s.name.c_str(), s.host_ms);
}
```

Options (`dyng::profiler_options`):

| Field | Effect |
|---|---|
| `sync_stages` | synchronize the stream at stage boundaries, so stage times add up to a paper-comparable breakdown (costs synchronization; no effect on host backends) |
| `nvtx` | emit NVTX ranges for Nsight Systems (CUDA builds with NVTX) |
| `cuda_events` | time stages on the device with CUDA events (CUDA backend) |

The profiler is not thread-safe: record from the thread that calls the library. It adds no
global state; two profilers on two `resources` objects are independent.

The counters of one call are also returned directly: `update()` returns the algorithm's `stats`
(for `sssp`: the batch summary, `invalidated`, `affected`, and schedule-dependent counters such
as `pushes`). Each field says whether it is deterministic.
