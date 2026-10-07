# MOSP-CUDA

MOSP-CUDA is the GPU implementation of DynaMOSP (IPDPS 2025, IEEE TPDS 2025;
`dyng.citation("sssp")`), originally ported from a SYCL version. Its SOSP update, after the
fixes below one persistent cooperative CUDA kernel per objective, is the CUDA backend of dynG's
`sssp`.

| | |
|---|---|
| Repository | <https://github.com/SMShovan/MOSP-CUDA> |
| Pinned commit | `e220ee20d1b0948ece3df135a02d1b898264c22f` (branch `fix/correctness-perf`) |
| Paper snapshot | tag `baseline-2026-09` = `ac29545` ("Ported from SYCL to CUDA") |
| History | 35 commits, 2025-08 to 2026-09; 34 by S M Shovan, 1 under the placeholder identity `CUDA <user@example.com>` (credited to S M Shovan); 33 AI-assisted |
| Ported in | M1b (the fused kernel, the device graph, the CUDA resources), M7 (the combined graph on the GPU and the MOSP tree: {doc}`../algorithms/mosp`) |
| Parity | byte-identical on the 495-case golden corpus, which MOSP-CUDA's own export reproduces file for file; cross-backend equal; within the performance gates at locked clocks ([M1b certificate](https://github.com/dyng-dev/dyng/blob/main/parity/results/M1b.md), ADR 0018) |

Before the port, the two originals were cross-checked once on the corpus: MOSP-CUDA@e220ee2 and
MOSP-OpenMP@c352151 give identical files (PLAN 6.3 step 2), so a difference could only be the
port's.

## What was ported, from which files

| MOSP-CUDA@e220ee2 | dynG |
|---|---|
| `src/sospUpdateGpu.cu` (`sospPersistentKernel`, `sospUpdateGpu`, `sospFromScratchGpu`, `Packing`, `Control`, `Params`, `SospWorkspace`) | `sssp` on `resources::cuda()`: the kernel verbatim behind the framework's `enact_fused` (`cpp/src/algorithms/sssp/cuda.cu`, `fused.cuh`), the workspace leased from the resources (ADR 0015) |
| `src/deviceGraph.cu`, `headers/deviceGraph.cuh` (`DeviceGraph`, `uploadDeviceGraph`, the reverse CSR built on the device) | the device copy of a CUDA graph (`cpp/src/graph/device_graph.{hpp,cu}`, stage `graph.upload`) |
| `src/csrGraph.cu` (the text readers and writers, the binary graph cache) | the shared readers and writers of `dyng::io`; the binary cache (read by `mosp --cache`) is written by `dyng prep cache` |
| `setenv("CUDA_MODULE_LOADING", "EAGER")` | `resources::warm_up()` |
| the stage timers with `cudaDeviceSynchronize` | profiler stages with optional CUDA-event device times |

## What was fixed in the original before the port

The same correctness fixes as MOSP-OpenMP (**M-a** subtree invalidation instead of counting to
infinity; **M-c** lowest-id ties, which made CUDA and OpenMP byte-identical), plus the GPU work:

- **M-b:** Step 1 moved from a serial host loop to the GPU (roots marked by one kernel, subtrees
  invalidated by pointer jumping, invalidated vertices and insertion heads pulling their best
  in-neighbour).
- **MP1, MP2:** invalidation, pull and near-far push, then one persistent cooperative kernel
  that runs the whole update of one objective (roots, invalidation, pull, near-far push, unpack)
  in one launch instead of a loop of kernels with a host round trip per iteration; **MP4** the
  packed-word overflow; **MP3** the combined graph on the GPU (for `mosp`, 0.2).
- **H-M1 / H-M2:** the in-memory pipeline (inputs read once, the batch applied once).
- **M-d, M-e:** the preference vector and the seeded change generator, as in MOSP-OpenMP.

The fixed code is 6-16x faster per objective than the paper's code on the connectivity-safe 50K
batch, and a batch that disconnects vertices no longer takes minutes (roadNet-PA: 39.1 s per
objective before, the connectivity-safe batch's time after); {doc}`../algorithms/sssp`,
section 8.

## What differs in dynG

- **A data race in the counter** `invalidated`: MOSP-CUDA@e220ee2 reads it from the candidate-list
  counter while other threads already append the insertion heads; dynG's kernel sums per-thread
  counts (ADR 0017 item 1). The trees were never affected.
- **The packing boundary:** at the limit where (distance, parent) still fits one 64-bit word,
  MOSP-CUDA packs one edge weight earlier than MOSP-OpenMP, so `stats::packed_parents` can differ
  between the cuda and the host backends there (n = 2^17 - 1). The trees are equal for canonical
  input trees; for non-canonical imported trees inside the packing window the host backends (as
  MOSP-OpenMP) and cuda (as MOSP-CUDA) return different tie parents (ADR 0029, option A).
- **Placement:** a graph belongs to the resources that built it; results live in device memory and
  are read with `to_vector()` or `dyng.Array` (host copies in Python); the device graph is uploaded
  once per batch inside the commit, as the original's is.
- **Engines:** without cooperative launch, `engine::automatic` runs the multi-kernel operators
  engine (M7, decision O24; ADR 0026), which follows MOSP_ESCHER@4b86159's host loop of kernels
  with this kernel's semantics and returns the same bytes; `engine::fused` raises
  `not_supported_error` there.
- **Registers:** the port keeps the original's 59 registers per thread of the int32 instantiation
  and the same co-resident grid (M1b certificate section 11).
- **Combined graph (M7):** `combinedGraphGpu.cu`'s `combinedEdge`, `countEdgesKernel`, the CUB
  scan and `fillEdgesKernel` are `mosp`'s CUDA combine step (`cpp/src/algorithms/mosp/cuda.cu`);
  `sospFromScratchGpu` on the combined graph is sssp's static solve on a view; `mospUpdate` is
  `mosp::update()`.
- **Not ported (yet):** the file-path API of the library (`parallelCombinedGraph`), `main` writing
  into relative paths, `using namespace std`.

The mapping of every name is section 9 of {doc}`../algorithms/sssp`.
