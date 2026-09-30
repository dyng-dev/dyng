# Results and versions

## `update()` applies the batch

In dynG, `update(res, g, batch, result)` both **changes the graph** and **updates the result**,
in one call. The alternative, where the caller applies the batch first and then asks each
result to catch up, makes it easy to apply a batch twice, to forget a result, or to update a
result against a graph it no longer matches. Keeping the two together makes the common case
one line and the wrong cases detectable (ADR 0006).

## Versions: why a stale result throws

Every graph has a **version** that increases by one with each applied batch, and every result
remembers the graph version it matches. `update()` checks them first:

```cpp
auto a = dyng::sssp::compute(res, g, 0);
auto b = dyng::sssp::compute(res, g, 5);
dyng::sssp::update(res, g, batch.view(), a);   // g: version 0 -> 1, a matches version 1
dyng::sssp::update(res, g, batch2.view(), b);  // throws dyng::stale_result_error: b matches 0
```

To keep several results on one graph, update them together; the batch is applied **once** and
every result is updated:

```cpp
auto [sa, sb] = dyng::update(res, g, batch.view(), a, b);   // a tuple of stats
```

`dyng::update_each()` does the same for a list whose length is known only at run time (for
example one result per objective of a multi-weight graph).

## Validation before the change

Everything that can be checked is checked before the graph changes (ids in range, weights
valid, the result's version), so a rejected batch leaves the graph and every result unchanged.
The one error that can only be detected afterwards comes from a corrupt imported input (for
`sssp`: a tree with a parent cycle adopted through `result::from_arrays()`); it leaves the
result *poisoned*, and every later use of it throws until it is recomputed.

## Results and their workspace

A result is opaque and move-only (`clone()` is the deep copy). It owns its arrays, its options and
the version it matches. The scratch memory of the incremental engine is not part of it: every
`compute()` and `update()` leases a workspace from the `resources` handle it runs with, sizes it
once and returns it, so results updated one after another through one handle share it (ADR 0015).
Updates therefore do not allocate in their algorithm work once the result and the workspace exist
(invariant I9, checked by allocation budgets in debug builds; an excess is logged there, and is an
error under `DYNG_STRICT_BUDGETS=1`, which the conformance kit sets).

## Threads

Graphs and results are not thread-safe:

- Calls that only read a graph or a result (`view()`, the accessors, `to_csr()`, an algorithm's
  `compute()` on a graph nobody changes) may run concurrently.
- A call that changes a graph (`apply()`, an algorithm's `update()`, `dyng::update()`) must not
  overlap any other call on that graph or on a result being updated with it. Two concurrent
  updates of one graph, or reading a result while another thread or stream updates it, are
  undefined unless you order them (a mutex, or stream events on CUDA).
- Distinct graphs and results may be used concurrently, also through copies of one `resources`
  handle: each concurrent call gets its own workspace. The setters of a handle
  (`set_memory_resource()`, `set_copy_policy()`, `attach_profiler()`) affect every copy; call them
  during setup. An attached profiler may record from several threads at once; read its records
  after the calls returned.
