# ADR 0015: Workspace sharing (scratch memory owned by `resources`)

- **Status:** Proposed (M1b); accepted with the 0.1 API freeze (M3), together with ADR 0006.
  Updated by the M1b review (stream order of device workspaces).
- **Date:** 2026-09-27
- **Deciders:** S M Shovan (lead maintainer)

## Context

PLAN Section 4.7.2 put the engines' workspaces inside the `result` objects ("reserve once, reuse
across batches"), and ADR 0006 (M1a) implemented it that way: every `sssp::result` owned a
workspace of about 38 bytes per vertex (the packed words, stamps, flags and six frontier lists of
MOSP-OpenMP's `SospWorkspace`, plus the sequential engine's lists).

The originals do not work like that. `mospUpdate()` (MOSP-OpenMP@c352151, MOSP-CUDA@e220ee2)
creates **one** `SospWorkspace`, reserves it once in its "prepare" stage and runs the K
per-objective SOSP updates one after the other on it. The difference was measured in M1a
(`parity/results/M1a.md`, ADR 0013 "Open M1b blocker"):

- memory: K workspaces instead of one (K = 3: about 230 MB instead of 77 MB on road_usa);
- time: objective 0 of the original pays the first touch of the frontier-list pages it uses
  inside its timed region, objectives 1..K-1 reuse warm pages. dynG either paid the first touch
  of every result's lists inside every objective's region, or (M1a) pre-touched all six lists of
  every result outside the region, which moved about 9 ms per result out of the gated region and
  made objective 0 1.07-1.44x of the original once the moved cost was counted. Objectives 1 and 2
  were close to the 1.05x gate. The per-objective gate (PLAN 6.4.2) could not be met honestly.

`mosp` (0.2) needs the sharing anyway (PLAN 6.4.4: K sssp problems "sharing one workspace"), and
the CUDA backend (M1b) needs a place for device scratch that is allocated once per stream and
memory resource, not once per result.

Requirements: results stay semantically independent (any order of updates of any results gives
the same values as with private workspaces); nothing is allocated inside a steady-state
`update()` (invariant I9); concurrent calls that the threading rules allow (PLAN 4.7.4: read-only
calls such as `compute()` on a graph nobody mutates may run concurrently, also through copies of
one handle) stay safe; the design carries over to device memory.

## Decision

1. **The resources handle owns a workspace pool** (`detail::workspace_pool`,
   `cpp/src/framework/workspace.hpp`), shared by all copies of the handle, like the handle's
   memory resource and profiler, and like the workspace/handle of RAFT and the pooled resources
   of RMM. It maps a workspace *type* (for example `sssp_workspace<int32_t>`) to a list of idle
   workspaces.
2. **Runs lease, results do not own.** `compute()` and every result's part of `update()` lease a
   workspace of their type for the duration of one engine run, size it (`reserve(n)`: a no-op
   once it is large enough; MOSP's semantics: per-vertex arrays assigned, frontier lists only
   reserved) and return it. The K objectives of `dyng::update_each()` run one after the other and
   therefore reuse one workspace, exactly as `mospUpdate()` does. `from_arrays()` and `clone()`
   size the pooled workspace once (profiler stage `sssp.workspace`), so the first update through
   the same handle allocates no scratch either. The per-objective change lists of an update also
   live in the workspace.
3. **Independence.** A workspace holds scratch only. The one piece of state that survives a run,
   the generation counter of the stamps, lives in the same object as the stamps it describes, so
   it stays consistent whichever result runs next. A lease that ends while an exception
   propagates (a failed engine run can leave `in_far` flags set) **discards** its workspace; the
   next lease creates a fresh one. The M1a pre-touch (`sssp.workspace.pretouch`) is removed.
4. **Concurrency.** Leasing is thread-safe (a mutex around the idle list, taken once per run).
   Calls that overlap in time get distinct workspaces; the pool then holds as many workspaces of
   a type as were ever leased at the same time, and reuses the most recently returned one first.
5. **Public surface** (additions to `resources`, PLAN 4.7.1): `release_workspaces()` frees the
   idle workspaces of the handle (and so of every copy); `workspace_bytes()` reports the bytes
   they hold (for the memory accounting of PLAN 8.6). `set_memory_resource()` releases the idle
   workspaces first. Nothing else in the public API changes; `result` no longer mentions a
   workspace.
6. **Device workspaces (CUDA, next step)** use the same pool: a device workspace type allocates
   its buffers from the handle's memory resource on the handle's stream when it is sized and
   frees them in its destructor. Reuse by the next run on the same stream is stream-ordered and
   needs no synchronization. Because the workspaces must be freed while their memory resource is
   alive, the pool is emptied by `set_memory_resource()` and destroyed with the last handle; a
   workspace holds only a `memory_resource_ref` and a `stream_ref`, never a `resources` copy (that
   would keep the handle, and so the pool, alive through itself).

## Consequences

- Memory: one workspace per handle and type (per concurrency level) instead of one per result:
  about 38 bytes per vertex once for K objectives, as the originals.
- The per-objective gate is read as measured on both sides: objective 0 pays the first touch of
  the list pages it uses inside the region, in the original and in dynG. With the pool, the M1b
  OpenMP A/B (`parity/results/M1b.md`) meets the per-objective gate on roadNet-PA, roadNet-CA,
  rgg_n_2_20_s0 and road_usa for all three batch types, and ADR 0013's open blocker is closed.
- Invariant I9 holds for the algorithm phase: after the first run through a handle (or a
  `compute()` / `from_arrays()` / `clone()` through it) a stable workload allocates no workspace
  memory (tested: the pool's created count and bytes stay constant over repeated updates). The
  host `graph::apply()` still builds a new CSR per batch (the straight port of
  `applyChangeBatch`, exempt until the resident apply), and the OpenMP engine's per-thread
  gather lists are the straight port of MOSP's `ListGather`.
- A result used with a different handle than the one it was computed with pays one warm-up
  (the first lease on that handle); documented on `resources` and `result`.
- PLAN Sections 4.7.2 and 5.1 ("workspaces live inside result objects", "`result` holds ... the
  reusable workspace") are superseded by this ADR; ADR 0006 item 3 points here.

## Alternatives considered

- **An explicit workspace argument** of `update_each()` / `update()`: visible in every
  signature and in Python, easy to misuse (a workspace shared between two threads), and it
  would not cover `compute()` or the CUDA stream ordering.
- **Sharing inside `update_each()` only** (one temporary workspace per call): allocates in every
  steady-state update (violates I9) and does not help a user who keeps one result per objective
  and updates them one by one.
- **A pool keyed by graph** (workspace owned by the graph): ties scratch memory to a container
  that is otherwise pure data, needs locking in a `const` graph, and does not match the device
  side, where scratch belongs to a stream and a memory resource, i.e. to `resources`.
- **Keeping per-result workspaces and pre-touching them** (M1a): moves cost out of the timed
  region instead of removing it, and keeps K times the memory.

## Update (M1b Step 2, CUDA core)

The device side is in place (ADR 0016 item 8): a CUDA handle's pool leases workspaces whose
arrays (`detail::scratch_buffer<T>`) live in the handle's device memory resource, ordered on its
stream; the tests `CudaWorkspace.*` check reuse, no allocation in steady state, growth, and the
release through `release_workspaces()`, `set_memory_resource()` and the last copy of the handle.

## Update (M1b Step 5, finish): the per-thread lists on OpenMP

Since M1b Step 4 the OpenMP engine keeps its per-thread lists in the workspace
(`util/thread_list.hpp`) instead of creating them in every parallel region, as MOSP does. A list
keeps its capacity and grows only when its thread takes a larger share of a round than it ever
took before; the dynamic schedule decides the shares, so a steady-state OpenMP update can still
allocate for these lists, rarely and geometrically (like a vector), while everything else in the
workspace stays exactly constant. Bounding the lists up front would need a round's whole size
per thread (on road_usa about 28 x 2 x 24M x 4 bytes), so invariant I9 holds on OpenMP with this
exception, recorded here. The steady-state test of `sssp_workspace_test.cpp` checks the rest of
the workspace exactly (`sssp_workspace::thread_list_bytes()` separates the lists); it had
asserted the total and failed about once in 180 runs.

## Update (M1b review): stream order of device workspaces

Item 6 said that reuse "on the same stream is stream-ordered and needs no synchronization", and
item 4 that concurrent calls get distinct workspaces. Both hold, but item 4 only separates calls
that overlap in *host* time: with the default stream (`cudaStreamPerThread`, a different stream on
every host thread; ADR 0016 item 10) a CUDA call on thread A can return while its kernels still
run, and a call on thread B through a copy of the handle could then lease the same workspace on
another stream. sssp was safe only because every CUDA run ends with a synchronization. Every pooled
workspace now carries a fence: a CUDA lease (`workspace_pool::acquire(res)`) records it on its
stream when it ends, and the next CUDA lease makes its stream wait for it unless it runs on the same
stream (same handle and, for the per-thread stream, same thread; the wait is skipped then, so the
single-stream steady state costs one event record per run). `release_idle()` and the pool's
destructor wait for the fences before the memory is freed. Tested with two threads, a host function
holding one thread's stream and a write behind it (`CudaWorkspace.*`); without the wait the second
thread read the old contents.
