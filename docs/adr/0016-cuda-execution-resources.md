# ADR 0016: CUDA execution resources (streams, memory, errors, warm-up)

- **Status:** Proposed (M1b); accepted with the 0.1 API freeze (M3). Amended by the M1b review
  (items 10 and 11; 2026-09-28).
- **Date:** 2026-09-27
- **Deciders:** S M Shovan (lead maintainer)

## Context

PLAN Sections 4.6 and 4.7 describe the CUDA side of `resources`: a stream from the caller (the
library never creates streams), a CCCL-shaped memory resource with a `cudaMallocAsync` pool as
the default, pinned host staging owned by `resources`, `DYNG_CUDA_TRY` / `DYNG_CHECK_KERNEL`, a
sticky device error word, and `warm_up()` instead of the originals' `setenv("CUDA_MODULE_LOADING",
"EAGER")`. The engines (the fused sssp kernel next) need the device's cooperative-launch
capability and its multiprocessor count, recorded once. Scratch memory is leased from the
handle's workspace pool (ADR 0015), which must work for device memory too. Several details are
not fixed by the plan; this ADR fixes them.

## Decision

1. **`resources::cuda(device, stream)`** validates the device (`invalid_argument_error`;
   `not_supported_error` if the build has no CUDA or no device is visible), records the device's
   properties once (`detail::cuda_device_properties`: compute capability, multiprocessor count,
   cooperative launch, memory-pool support, ...; read through `detail::resources_access`), and
   takes `default_device_memory_resource(device)` and `default_pinned_host_memory_resource()`.
   Every call that enqueues work makes the handle's device current for its duration and restores
   the caller's device (`detail::scoped_device`), so the per-thread default stream always means
   the handle's device and the caller's current device never matters.
2. **Default device memory: one `cuda_async_memory_resource` per device, process-wide, created on
   first use and never destroyed.** It owns its own `cudaMemPool_t` (never the device's default
   pool, which belongs to the application) with an unlimited release threshold, so freed memory
   stays in the pool and a steady-state workload makes no driver allocation (invariant I9).
   Buffers and workspaces may outlive every `resources` handle because the resource outlives
   them; destroying pools during static destruction would race the runtime's own teardown.
   Alignment up to 256 bytes; `used_bytes()` / `reserved_bytes()` report the pool (for the memory
   gate and the I9 tests).
3. **Pinned staging:** `pinned_host_memory_resource` (portable `cudaHostAlloc`); `deallocate`
   waits for the stream first (`cudaFreeHost` does not wait for pending copies). Engines keep
   staging buffers in their workspaces, never allocating them per call.
4. **The CUDA backend accepts device and managed memory resources only**
   (`set_memory_resource()`); host backends accept host-accessible ones, as before.
5. **Errors.** `DYNG_CUDA_TRY(call)` throws `cuda_error` (`out_of_memory_error` for
   `cudaErrorMemoryAllocation`) naming the error, the call and file:line, and clears the
   runtime's last error. `DYNG_CUDA_TRY_NO_THROW` logs (and is silent during process teardown).
   `DYNG_CHECK_KERNEL(stream)` takes the stream explicitly (PLAN 4.7.3 wrote
   `DYNG_CHECK_KERNEL()`): with `DYNG_CUDA_DEBUG_SYNC` (on in Debug and `sanitize-cuda`) it
   synchronizes that stream, and a stream-less form would have to synchronize the whole device,
   which the backend rules forbid. The macros live in `cpp/src/util/cuda_check.hpp` (not `.cuh`:
   they need only the runtime API, so host translation units use them too).
6. **Device error flags** (`detail::device_error_flags`): one `uint32_t` word in the handle's
   memory and a pinned mirror; kernels `atomicOr` a kind (`capacity`, `invalid_input`,
   `parent_cycle`, `internal`); the host enqueues the read-back after the last kernel and checks
   the mirror after the call's existing synchronization. Mapping: internal or unknown bits ->
   `internal_error`, capacity -> `capacity_error`, the input kinds -> `invalid_argument_error`.
7. **Warm-up.** Every `.cu` file registers its kernels (each explicit instantiation) in the
   kernel registry (`DYNG_REGISTER_KERNEL`); `warm_up()` creates the context, calls
   `cudaFuncGetAttributes` on every registered kernel (which loads its module under lazy loading,
   the CUDA default), primes the stream and the memory pool. The library never sets
   `CUDA_MODULE_LOADING`.
8. **Device workspaces** use the pool of ADR 0015 unchanged; `detail::scratch_buffer<T>` sizes an
   array from the run's `resources` (device memory on its stream) and keeps it while it is large
   enough. `set_memory_resource()` releases idle workspaces first; the pool is the last member of
   the handle's state, so it is destroyed before the resources it allocated from.
9. **Presets.** The CPU presets (`dev`, `release`, `relwithdebinfo`, `parity`, `cpu-only`,
   `asan`, `tsan`) pin `DYNG_ENABLE_CUDA=OFF`; the CUDA builds are separate presets
   (`dev-cuda`, `release-cuda`, `parity-cuda`, `sanitize-cuda`, `ci-cuda12`, `ci-cuda13`). A plain
   `cmake` without a preset enables CUDA when a CUDA compiler is found (PLAN Section 7.3).

10. **The default stream is per thread (M1b review).** `resources::cuda()` with the default
    `stream_ref` uses `cudaStreamPerThread`, which CUDA defines as a different stream on every
    host thread. The handle keeps that value instead of resolving it (no API returns a handle for
    another thread's per-thread stream, and pinning the handle to the creating thread would forbid
    the concurrent read-only calls PLAN 4.7.4 allows). Consequences, documented on `resources`,
    `stream_ref` and `buffer`: copies of a default handle used on two threads run on two streams;
    `synchronize()` waits for the calling thread's stream only; memory is released on the stream
    of the thread that releases it; ordering work across threads is the caller's job, as with two
    explicit streams. PLAN 4.7.4's "a copy refers to the same stream" holds for explicit streams.
    The library itself stays sound: pooled workspaces carry a fence (`detail::cuda_stream_fence`,
    a CUDA event) that each CUDA lease records on its stream when it ends and that the next lease
    waits for when it runs on another stream (another thread's per-thread stream included);
    `release_workspaces()`, `set_memory_resource()` and the pool's destructor wait for the fences
    before freeing (so they are `@sync` now). A result whose arrays were allocated on one stream
    and are replaced through a handle on another (vertex growth) moves them to the updating
    stream before releasing them (`buffer::set_stream()`).

11. **Implicit copies of inputs (M1b review).** PLAN 4.7.1: every function that consumes arrays
    accepts any memory space and copies once when the space does not match; `copy_policy` decides
    whether that is logged at debug (`allow`, the default), at warn (`warn`; also `allow` while a
    profiler is attached) or refused with `invalid_argument_error` (`error`). In this release the
    consumers read their inputs on the host on every backend (the batch is applied on the host,
    graphs are built on the host, trees are imported and checked on the host, ADR 0017 items 4-5),
    so "does not match" means "not host-accessible": `graph::from_edges()`, `graph::from_csr()`,
    `graph::apply()`, `dyng::update()` / `update_each()` / `sssp::update()` and
    `sssp::result::from_arrays()` copy device (or managed-but-not-host-accessible) arrays to the
    host once, each array on its own (`detail::host_input`, `detail::host_batch`,
    `core/staging.hpp`), before anything is changed. Host arrays given to a CUDA consumer are not
    implicit copies: the host is where they are read, and the upload of a result's or graph's own
    state is the function's work (MOSP-CUDA uploads the same data). Before this, M1b rejected
    device batches on CUDA handles and never read the policy; a batch with only some arrays on
    the device crashed `sssp::update()` (it read `insert_src` on the host before checking it).
    When the device apply lands (PLAN 6.4.1), device batches become the matching space and host
    batches the copied one.

## Consequences

- PLAN 7.4 described `dev` as the native-architecture CUDA build. Splitting it keeps the CPU gate
  (`ci/check.sh`, `cpu.yml`) independent of whether a toolkit is installed, and keeps the OpenMP
  performance gates of M1b Step 1 on the unchanged `parity` build; the GPU gate is
  `ci/gpu_local.sh` on `dev-cuda`.
- With CUDA built and a device visible, `default_backend()` is `cuda`, so a default-constructed
  `resources` is a CUDA handle; algorithms without a CUDA backend yet (sssp until its fused
  engine lands) throw `not_supported_error` naming the available backends.
- The default pools hold their peak memory until the process exits. Applications that need the
  memory back use their own `cuda_async_memory_resource` (destroyed with it) through
  `set_memory_resource()`.
