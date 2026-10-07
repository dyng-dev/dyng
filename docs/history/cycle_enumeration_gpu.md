# CycleEnumeration-GPU

CycleEnumeration-GPU is the code of TruCy / DynTruCy (A. Khanda, S. M. Shovan, A. Satpathy,
S. K. Das, "TruCy: GPU-Accelerated Cycle Enumeration for Large-Scale Static and Dynamic
Networks", submitted to IEEE Transactions on Computers; `dyng.citation("cycle_count")`). It
counts the directed simple cycles of length 2..k of a graph on the CPU (sequential, OpenMP) and
the GPU, and updates the histogram after a batch of edge insertions and deletions. dynG's
`cycle_count` is its exact static count and its update.

| | |
|---|---|
| Repository | <https://github.com/SMShovan/CycleEnumeration-GPU> |
| Pinned commit | `0a976adfa801a712135bf1adb51a228f353a0751` |
| Paper snapshot | tag `baseline-2026-09` = `da2067d` |
| History | 110 commits, 2026-05 to 2026-09, all by S M Shovan (35 AI-assisted); A. Khanda is a co-first author of the paper |
| Ported in | M2a (sequential and OpenMP), M2b (CUDA, the resident device graph) |
| Parity | bit-identical histograms on the 24-case CPU corpus and the 24-case CUDA corpus (DD k = 3..7, GitHub, Twitch, COLLAB, the seed-1 update deltas), the batch generator byte-identical ([M2a](https://github.com/dyng-dev/dyng/blob/main/parity/results/M2a.md) and [M2b](https://github.com/dyng-dev/dyng/blob/main/parity/results/M2b.md) certificates) |

It was the most conventional of the codebases (CMake, GoogleTest, namespaces, exceptions,
Doxygen), so the port changed mostly names and namespaces (`cycle_enum::` to `dyng::cycle_count`,
`dyng::io`, `dyng::testing`), and its build and test patterns became dynG's templates.

## What was ported, from which files

| CycleEnumeration-GPU@0a976ad | dynG |
|---|---|
| `src/sequential/johnson.cpp` (`JohnsonSearch`), `src/openmp/openmp_johnson.cpp` (`count_root`) | `cycle_count::compute` on the host backends (`static_sequential.cpp`, `static_openmp.cpp`) |
| `src/cuda/cuda_static_kernels.cu`, `include/cycle_enum/cuda/cuda_dfs.cuh`, `cuda_work_queue.hpp` | `cycle_count::compute` on `resources::cuda()` (`static_cuda.cu`, `dfs.cuh`, `work_queue.hpp`; the same kernel names) |
| `src/dynamic/update_sequential.cpp`, `update_openmp.cpp`, `cycles_through_edge.cpp` | `cycle_count::update` on the host backends, run by the framework as an aggregate-delta problem |
| `src/dynamic/update_cuda_kernel.cu` (`mark_owners_kernel`, `item_counts_kernel`, `count_owned_cycles_kernel`) | `cycle_count::update` on cuda (`cuda.cu`) |
| `src/dynamic/directed_graph.cpp` (`prepare_batch`, `apply_batch`), `change_rows_kernel`, `next_degree_kernel`, `build_next_rows_kernel` | Step 0 of the framework (`detail::compute_structural_change`) and `graph::apply` under `batch_semantics::set()`; the device merge of a resident graph (`graph/apply_set_device.cu`) |
| `src/core/graph.cpp` (the parallel `from_chars` parser, the Matrix Market banner) | `io::read_edge_list` (identical vertex order and CSR) |
| `src/dynamic/batch_generator.cpp` (`generate_batch`) | `generators::legacy::cycle_enum_batch()`, `dyng generate cycle_enum_batch` |
| `src/core/histogram.cpp` (`CycleHistogram::to_csv`) | `cycle_count::result`, `io::write_histogram_csv`, `dyng.io.histogram_csv` |
| `tests/support/cycle_oracles.hpp` (subset-DP oracles, brute force) | `testing::oracle_simple_cycles`, `testing::brute_force_simple_cycles`, which check dynG's randomized suites as they checked the original's |
| `src/cli/cycle_enum_main.cpp` (the `cycle-enum` driver) | `tools/compat` `dyng-compat-cycle-enum` (a drop-in clone, for parity) and `dyng cycle_count` ({doc}`../api/cli` has the flag table) |

## What was fixed in the original before the port

Recorded in the original's `CHANGES.md` (labels C: correctness, K: kernels, H: host). Every fix
keeps the counts exact, and on valid inputs the histograms of `0a976ad` are bit-identical to the
paper's code on every dataset measured:

- **C1** symmetric Matrix Market files were read in one direction (0 cycles); **C4** invalid
  update batches (deleting a missing edge, inserting an existing one, a self-loop, a new vertex id)
  gave wrong histograms or an out-of-bounds GPU write, now set semantics; **C5** the CUDA update
  allocated a V-byte array per change (out of memory at the paper's batch sizes); **C6** the CPU
  baselines reset O(V) state per root (DD k = 3: 127 s, now 2.5 s); **C9** GPU tests that could
  not fail, replaced by oracles and randomized parity suites with two recorded mutations; **C10**
  a default `-O0` build; **C2, C3** the time-window counters.
- **K1-K3** exact pruned depth-first search kernels with edge and two-hop work items (3.7x to 714x
  faster static kernels), path membership instead of the visited array, the resident G_t with
  G_{t+1} built on the device; **H1, H3** the parallel parser and an O(E + B log B) batch apply.

The table of every item and what dynG takes from it is "Paper vs fixed code" on the
{doc}`../algorithms/cycle_count` page.

## What differs in dynG

- **Exact counts, not the paper's TruCy.** The paper's TruCy is an approximate, kappa-truncated
  Johnson search; that design is in neither the original nor dynG. dynG's counts are the exact
  ("optimal") counts the paper compares TruCy against, and the paper's kappa experiments cannot be
  reproduced with it ({doc}`../algorithms/cycle_count`, section 7).
- **The resident device graph:** the original re-uploads G_t for every update; dynG keeps
  G_{t+1} on the device for the next batch and downloads the host copy only when it is read
  (ADR 0020), an improvement reported separately in the M2b certificate.
- **Step 0 once per update** for every result on the graph (ADR 0020), with a bucket sort of the
  changes; `cycle_count` runs next to `sssp` on one graph with `dyng::update()`.
- **Not ported (yet):** Read-Tarjan, the time-window and temporal modes (0.5), cycle-union,
  branch splitting, the OpenMP task experiment, and the environment tuning
  (`CYCLE_ENUM_CUDA_BLOCK_SIZE`, `CYCLE_ENUM_CUDA_BLOCKS_PER_SM`).

The mapping of every name is section 8 of {doc}`../algorithms/cycle_count`.
