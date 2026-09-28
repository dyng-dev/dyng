# ADR 0017: The CUDA sssp engine, engine selection, placement and the resident device graph

- **Status:** Proposed (M1b); accepted with the 0.1 API freeze (M3)
- **Date:** 2026-09-27
- **Deciders:** S M Shovan (lead maintainer)

## Context

M1b ports MOSP-CUDA@e220ee2's SOSP engine (the persistent cooperative kernel of
`src/sospUpdateGpu.cu`) verbatim behind the fused engine of the CUDA backend (PLAN 4.5.4, 6.4.2),
with a resident device graph whose batch is still applied on the host (PLAN 6.4.1, M1b). The plan
fixes the intent: `engine::automatic` = fused when cooperative launch is available, otherwise
`not_supported_error` (O24: the operators engine arrives in 0.2); a container lives in the memory
of the resources that built it and algorithms never copy a whole graph silently (PLAN 4.6 rule 5);
byte parity with MOSP-CUDA on the golden corpus. It leaves open how a graph whose CSR is applied on
the host is "resident", where results live, when the engine choice is checked, which calls
synchronize, and how the two ported engines (OpenMP and CUDA) may legitimately differ. This ADR
fixes them.

## Decision

1. **The fused engine is the original's code.** `fused.cuh` holds `sospPersistentKernel` with
   mechanical changes only (names, templates on the index types, `sssp_infinity` for
   `DISTANCE_INF`); `cuda.cu` holds the host side of `sospUpdateGpu()` / `sospFromScratchGpu()`
   (packing choice, control block, one cooperative launch, one read-back and synchronization).
   Two additions, both outside the search: the deterministic `affected` counter (the unpack pass
   of an update compares each new pair with the old one and writes only changed pairs, so it moves
   no more bytes than the original's; summed per warp) and a host-side parent-cycle check on the
   control block the kernel already writes (pointer jumping still active in its last round is impossible
   in a forest). The int32 instantiation compiles to the original's 59 registers; the grid is the
   co-resident block count of each kernel instantiation (occupancy API), cached in the workspace.
2. **Engine selection before any change.** `options::cuda_engine` is read in `compute()` and in
   `before_apply()` of `update()`: `automatic` and `fused` need the cooperative-launch capability
   recorded by `resources::cuda()` and otherwise throw `not_supported_error` naming the host
   backends; `operators` throws in 0.1. The graph and the result are unchanged by the failure
   (strong guarantee). `detail::resources_access::force_cooperative_launch()` lets the tests take
   the "unavailable" path on any GPU.
3. **Placement by backend class.** A graph records the backend of the resources that built it
   (`graph_impl::home`, and the device for cuda). Algorithms require resources of the same class
   (host backends are interchangeable: their storage is the same host CSR) and the same device,
   and throw `invalid_argument_error` naming `g.clone(res)` otherwise; `graph::clone(res)` and
   `sssp::result::clone(res)` move a graph or a result to the class of `res`. `graph::space()` is
   `device` for a CUDA graph.
4. **The resident device graph in this release.** A CUDA graph keeps its authoritative CSR in
   host memory (the host apply is the straight port of `applyChangeBatch`) and a device copy of
   the current state (`detail::device_graph`: out- and in-edges, one weight column per
   objective), built on first use with the stream and memory resource of the calling resources
   (stage `graph.upload`), dropped by every applied batch and rebuilt inside the commit of
   `dyng::update` once for all results, exactly as MOSP-CUDA uploads the updated graph once per
   batch and derives its reverse graph on the device (`uploadDeviceGraph`). The in-edge row order
   of the device copy is unspecified (atomic fill, as the original's); the engines take minima
   over in-neighbours, which do not depend on it. The host in-edges of a CUDA graph are never
   built by sssp. A device apply that keeps the arrays across batches replaces the upload later.
5. **Results live on the device.** A result of the CUDA backend holds device buffers
   (`result::space() == device`); `distances()` / `parents()` return device views, which users
   copy with `to_vector()`. `from_arrays()` accepts host or device arrays; the tree is imported,
   canonicalized and validated on the host (with the OpenMP threads available to the handle, see
   6) and uploaded (stage `sssp.upload`), since those checks are O(n) host code shared with the
   CPU backends.
6. **Host threads of a CUDA handle.** The host-side work of a call with CUDA resources (graph
   builds and applies, tree imports and checks, column summaries) uses the OpenMP default thread
   count (`resources_access::host_threads`); `resources::num_threads()` stays 1 for cuda.
7. **Synchronization.** `update()` synchronizes the stream once per result (the control block:
   statistics and error flags), as `sospUpdateGpu()` does. `compute()` synchronizes as well
   (PLAN 5.1 sketches it as asynchronous): the workspace's stamp generation, which the next run on
   the same workspace must continue from, is only known after the kernel. `from_arrays()` and
   `clone()` synchronize so that a CUDA result is complete whenever a call that produced it has
   returned, on any stream.
8. **The two ported engines may differ in one reported flag.** MOSP-CUDA chooses packed
   (distance, parent) words when (n - 1) * max weight fits next to the parent bits and drops
   larger candidates before packing; MOSP-OpenMP also requires one more edge to fit. Right at the
   limit (the n = 2^17 - 1 packing-boundary cases) the CUDA engine packs and the host engines keep
   distances only, so `stats::packed_parents` differs there; the trees are identical for canonical
   inputs (the golden corpus and the cross-backend tests check it). Each backend stays byte-equal
   to its own original.
9. **Timing.** `profiler_options::cuda_events` records CUDA events around each stage on the
   handle's stream (`device_ms`). MOSP-CUDA's per-objective timer is host time up to its
   synchronization, so the gate compares host times of the same scope (`sssp.enact_fused`) and
   reports dynG's device time next to them (`parity/timed_regions/sssp.toml`).

## Consequences

- Byte parity with MOSP-CUDA@e220ee2 on all 495 golden cases, the MOSP fixtures and the
  cross-backend randomized tests (cuda = openmp = sequential) holds with the verbatim kernel.
- Mixing backends is explicit (`clone(res)`); a CUDA graph costs host memory for its CSR as well
  as the device copy until the device apply lands.
- A steady-state update allocates only the upload of the new graph state (the graph's budget,
  exempt as the host apply is); the algorithm phase allocates nothing (tested by counting the
  memory resource's allocations).
- The engine choice and its failure are testable on any machine.
