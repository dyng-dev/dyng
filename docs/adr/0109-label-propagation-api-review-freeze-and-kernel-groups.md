# ADR 0109: The label_propagation API review and freeze (0.4), and per-algorithm kernel groups

- **Status:** Accepted (M10b, step docs-api-review-freeze-final-reads). Part A carries out the
  author's decisions AD15 (sheet F1: everything new of M10 is frozen at the end of M10b and becomes
  `stable` at 0.4.0 only on the author's confirmation at the release) and AD12 (sheet E1: the new
  public names), both decided by the author on 2026-10-28 (GOVERNANCE.md, approvals log, P040's
  rows "The 0.4.0 plan ..." and "The new public names of 0.4 (AD12, HD8, HD12; sheet E1)"), by the
  method of ADRs 0023, 0047 and 0060 and the checklist `docs/developer/api_review_checklist.md`.
  **Pending the author:** the stability confirmation at 0.4.0 (AD15), in GOVERNANCE.md's open
  decisions, M10 (AD). Every public name beyond AD12's list (section "The new public names"; the
  author's decision of 2026-10-29 that M10b's API ADR lists them for approval at the freeze) was
  **approved by the author on 2026-11-12** (GOVERNANCE.md, approvals log, the row "M10's public
  names beyond AD12's list"): frozen with the API, a rename is an api-change.
  Part B carries out the author's decisions of 2026-11-03 and 2026-11-05 (the rows "The D14-path
  excursions of M10a step 5" and "The cycle_count module-loading excursions of the merged
  library"); its mechanism is a technical choice under delegation. Neither part changes a rule the
  author approved: the stability policy of PLAN 5.9, the gates and their protocol (AD2's
  `CUDA_MODULE_LOADING=EAGER` on both sides included), the parity rules and the frozen headers
  (additive changes only) are untouched.
- **Date:** 2026-11-12
- **Deciders:** the AI assistant, on the author's behalf (M10b, step 4); the names beyond AD12:
  S M Shovan (approved 2026-11-12); the stability: S M Shovan (pending, at the release)
- **Builds on:** ADR 0023 (the method, the snapshot), ADR 0035 (the freeze enforced against the
  base branch), ADR 0047 (the freeze's Doxygen rules), ADR 0060 (the last algorithm freeze),
  ADRs 0063-0079 and 0100-0108 (label_propagation, its container additions, formats, Python and
  command line).
- **One ADR for two subjects:** M10's numbers are 0063-0079 and 0100-0109 (the reconciliation of
  2026-10-27); 0109 is the last, so the review (part A) and the kernel groups (part B) share it, as
  ADR 0108 shares three subjects.

## Part A: the API review and freeze

### Context

label_propagation arrived in M10a (ADRs 0069-0079) at maturity `experimental`, its header tracked
by the API baseline; M10b added the operators engine (ADRs 0101, 0102), slack rows and the
streaming threshold (0103, 0104), the formats and the label files (0105, 0106), the library
generator (0107), Python, the command line and the kNN recipe (0108). AD15 freezes all of it at
the end of M10b, and makes it `stable` at 0.4.0 only on the author's confirmation, so from the
merge of M10b a change is an api-change (a CHANGELOG entry; a breaking one needs the label too),
and the manifest says `experimental` until the author confirms, as triad_count's did before 0.3.0
was confirmed.

Reviewed: every public entity of `<dyng/label_propagation.hpp>` (`schedule`, `on_limit`,
`options`, `stats`, `result`, `compute()`, `update()`, and the `detail` contract the baseline
lists: `label_propagation_supported_v`, `update_traits<label_propagation::result>`,
`DYNG_LABEL_PROPAGATION_TYPES_MESSAGE`); the graph additions (`graph::with_capacity`,
`row_layout::slotted` and `slack`, `vertex_status`, `graph_view::status` and `row_end`,
`graph::num_active_vertices`, `batch_semantics::net_effect()` and `as_net_effect`,
`edge_batch::insert_vertex` / `delete_vertex` and their read-back, `apply_summary`'s
`dropped_insertions`, `relocated_rows`, `compactions`, the float graph instantiations); the
generators (`<dyng/generators/vertex_arrival.hpp>`); the `dyng::testing` names
(`harmonic_solution.hpp`, `union_find.hpp`, `lp_host_model.hpp`); `io::write_labels` /
`read_labels` and the format tokens (`.dgt` `+v` / `-v` and float weights, `.dgb`
`batch_dtype::float32` and `batch_file_flags::vertex_records`); Python (`dyng.label_propagation`
and the graph, batch, generator, testing and I/O additions of ADR 0108); the command line
(`dyng label_propagation`, `dyng generate vertex_arrival_stream` / `dynlp_vertex_arrival_stream`);
`dyng-compat-dynlp`'s argv; `docs/design/sketches/label_propagation.md` and its open questions.

### Findings and what was done

| # | Area | Finding | Resolution |
|---|---|---|---|
| 1 | Naming | The names follow ADR 0004 (clang-tidy's naming rules pass); they are AD12's, with E1's changes (`dyng-compat-dynlp`, `testing::lp_host_model`, the two generator commands); the algorithm is named for what it computes, the paper is in `@paper`. | none needed |
| 2 | The verb vocabulary | `compute`, `update`, `clone`, `reserve` (a result's workspaces for the largest batch), `set_options` / `get_options` (ADR 0023's accepted exception); accessors are nouns (`labels`, `seed_labels`, `graph_version`, `space`). | none needed |
| 3 | Contract uniformity | The skeleton of ADR 0006: an aggregate `options` (fields appended only), `stats : update_stats` with `batch` first, an opaque move-only pimpl `result<vertex_t>`, `compute()` never mutates the graph, `update(res, g&, batch, r&)` applies the batch, throws `stale_result_error`, and composes through `update_traits` with `dyng::update()`. Differences: the seeds are `compute()`'s arguments (fixed inputs; later seeds arrive with their batch's `insert_vertex_labels`), and `result::reserve()` exists because the published setup sizes the workspaces once for the stream's largest batch. | Accepted |
| 4 | `stats` and AD20 / E2 | The header states AD20's per-field levels (`affected`, `converged`, `fallback_used` deterministic per backend under `damped`, schedule-dependent under `in_place`; `iterations`, `frontier_visits` schedule-dependent) and E2's meanings (`affected` = the vertices whose label bits changed; `initial_frontier` = \|V_aff\|); `components`, `sparsified_edges`, `initial_frontier` and `batch` are deterministic. | none needed |
| 5 | Doxygen | Every entity is documented; `compute()` / `update()` carry `@backends`, `@determinism`, `@paper` (`dynlp2026`) and `@guarantee`. The freeze's rules of ADR 0047 (every entity in the group, `@throws` or `noexcept` on every function, `@sync` / `@async` with a `cuda_error`) were not applied to the header. | `ci/doxygen_coverage.py` holds `dyng/label_propagation.hpp` to them (group `label_propagation`); the header passes |
| 6 | Exceptions | The entry points translate a host `std::bad_alloc` (`out_of_memory_error`; `dyng_label_propagation_allocation_failure_tests`); `update()` is strong before the commit, basic after it (the result poisoned; ADR 0071); OpenMP resources raise `not_supported_error` naming `sequential`, `cuda` and the fix (AD19); `engine::operators` with `schedule::in_place` is refused by both entry points on CUDA (ADR 0101). | none needed |
| 7 | Host-compilability | `<dyng/label_propagation.hpp>` includes no CUDA, CUB or Thrust header; its self-containment unit builds in every preset; `dyng-compat-dynlp` uses public headers only and builds against an install tree (`compat_dynlp.includes`, `ci/cmake_consumer.sh`). | none needed |
| 8 | Ownership, ABI, limits | `result` is a unique pimpl; `options` and `stats` are aggregates (PLAN 5.9 allows appended fields before 1.0). `label_propagation_supported_v`: 32-bit vertex ids, float weights, int32 or int64 offsets; a later release may widen it (additive). | Accepted |
| 9 | The sketch's open questions | (1) `on_limit` stays nested in `label_propagation` until a second algorithm needs the same enumeration (rule of two; M11's `budget_action` / `on_budget` is a different policy and stays separate, AD12); (2) `std::optional<float>` in `options` is accepted (the C++ API is the contract; Python maps it to `None`); (3) the published setup is the documented default and the streaming mode the advanced one (O9, AD16). | Answered; the sketch is marked frozen |
| 10 | The graph additions | All additive (`ci/api_snapshot.py --against origin/main`: additions only; the enumerators `row_layout::slotted` / `slack`, `graph_properties::headroom` and `allow_vertex_growth` exist since 0.1). `graph::reserve()` takes edges only, so the first vertex batch on CUDA allocates the device arrays of its vertex ids (M10a's note); a vertex-capacity reserve would be a new public name. | No change in 0.4; a vertex-capacity reserve is listed for a later release (consequences) |
| 11 | The stable algorithms on slotted and slack graphs | `sssp`, `mosp`, `cycle_count`, `dynamic_bfs`, `triangle_delta` and `dyng::update()`'s composition refuse them with one central `not_supported_error` naming the algorithm, the layout and the fix (AD11; ADR 0070), documented in their `@throws` (M10a); `triad_count` and `hyper_sssp` take hypergraphs. | none needed |
| 12 | Generators and testing | `generate_vertex_arrival_stream()` (bit-exact on every platform within a major version, PLAN 5.9's output stability; ADR 0107) and the legacy stream (byte for byte LabelPropagation-CUDA's); the oracles share no code with the algorithm (ADR 0068). These headers are M10's alone. | Frozen with the algorithm (decision 2) |
| 13 | io and the formats | `io::write_labels` / `read_labels` (O's `i %.6f` bytes; ADR 0106) and AD13's additive codes and flag (ADR 0105) live in headers shared with other milestones (`io/result_io.hpp`, `io/batch_io.hpp`, tracked as a whole until io is frozen; ADR 0035 finding 19). The formats are stable-tier at version 1; old readers refuse the new codes loudly. | Frozen by this ADR as names and bytes; enforced by the format tests (round trips, the 0.3 reader's refusals, the label bytes against O's goldens) |
| 14 | Python | `dyng.label_propagation` mirrors the C++ API (ADR 0108): `Options` with the C++ fields and defaults (string names for the enumerations, `None` for `nullopt`), `Stats`, `Result` (`labels` a read-only float32 `dyng.Array`, zero-copy through DLPack and the array interfaces, copy-on-write while a view is alive), `compute(graph, seed_vertices, seed_labels, *, options, resources, **kwargs)`, `update(graph, batch, result, *, resources)`, `dyng.update()` over its results; AD14's dtype rule; AD19's refusal. | Frozen (decision 3) |
| 15 | Command line | `dyng label_propagation compute|update`: one kebab-case flag per `Options` field (both directions tested), the output files in O's bytes; `dyng generate vertex_arrival_stream` / `dynlp_vertex_arrival_stream`. | Frozen; `test_the_frozen_flags_are_kept` pins every flag |
| 16 | `dyng-compat-dynlp` | The original's argv unchanged (a negative `--tau` keeps the automatic mode; `--tau 0`, a NaN `--tau` and `--maxIter <= 0` refused, C8), plus dynG's diagnostic flags (`--backend`, `--cuda-engine`, `--phases`, `--label-dump`, `--layout`, `--streaming-tau`). A parity tool: not installed, not part of PLAN 5.9's stable tier (ADR 0073). | Reviewed; the diagnostic flags are listed with the names beyond AD12 |
| 17 | Maturity | The manifest stays `experimental` (registry, tables and the algorithm page say so, with "frozen"); AD15's confirmation at 0.4.0 makes it `stable`. | Decision 4 |

### Decision

1. **The label_propagation API is frozen** as reviewed: every declaration of
   `<dyng/label_propagation.hpp>` (its baseline section, the `detail` contract included), the
   Python names of `dyng.label_propagation`, and the flags and output formats of `dyng
   label_propagation` and of the two vertex-arrival generators of `dyng generate`. No signature
   changed in the review, so the freeze needs no migration note.
2. **The baselines.** `ci/api_snapshot.py` gains `FROZEN_BEFORE_STABLE`: the headers frozen at the
   end of their milestone while their algorithm's maturity stays below stable,
   `dyng/label_propagation.hpp` and the headers only it adds (`dyng/generators/vertex_arrival.hpp`,
   `dyng/testing/harmonic_solution.hpp`, `union_find.hpp`, `lp_host_model.hpp`). The baseline
   `cpp/tests/api/api_snapshot/public_api.txt` lists their sections as `frozen` (`ci/docs.sh
   --update-api`); a removed or changed declaration fails `ci/docs.sh` (against the baseline) and
   `ci/api_check.sh` (against the base branch) without the `api-change` label and a CHANGELOG
   entry. The Python half is griffe against the base branch (`ci/api_check.sh`), from the merge of
   M10b; the tests below pin the rule now. The graph additions are in `graph/*`, frozen since 0.1:
   they are checked as additions already.
3. **Both checks fail on a deliberate change** (the outcome below), and the tests keep it:
   `ci/tests/test_api_snapshot.py::test_the_label_propagation_headers_are_frozen_and_a_change_is_breaking`
   (a changed default of `options::tolerance`, a removed `stats::initial_frontier` and a changed
   default of `vertex_arrival_options::num_batches` are breaking changes of their sections; the
   manifest is not stable); `ci/tests/test_api_check.py` (a renamed parameter of
   `dyng.label_propagation.update`, a changed default of `Options.tolerance` and a removed
   `Result.labels` are breaking for griffe); `python/tests/test_label_propagation_cli.py::
   test_the_frozen_flags_are_kept` (the command line).
4. **Maturity.** `experimental` until the author confirms `stable` at 0.4.0 (AD15; GOVERNANCE.md,
   open decisions, M10 (AD), with everything the confirmation covers). On confirmation the manifest
   says `stable`, `stable_algorithm_headers()` lists the header as well, and nothing else changes.
5. **The sketch** `docs/design/sketches/label_propagation.md` is marked frozen, with its questions
   answered (finding 9) and the header as the authority where they differ.

### The new public names (for the author)

**AD12's names, decided by the author on 2026-10-28** (P040's row; ADR 0071 and the step ADRs
record them): label_propagation's `schedule`, `on_limit`, `options`, `stats` (with
`initial_frontier`), `result`, `compute`, `update`; the graph's `with_capacity`,
`num_active_vertices`, `vertex_status`, `graph_view::status` and `row_end`,
`batch_semantics::net_effect()`, `edge_batch::insert_vertex` and `delete_vertex`; the vertex-arrival
generators `generators::generate_vertex_arrival_stream` and
`generators::legacy::dynlp_vertex_arrival_stream`; `testing::harmonic_solution`, `union_find`,
`lp_host_model`; `io::write_labels`, `read_labels`; the `.dgb` names `batch_dtype::float32`,
`batch_file_flags::vertex_records` (with M11's `incidence_edits`, `known_graph`,
`known_hypergraph`); Python's float32 graph classes, `Graph.with_capacity`, `Graph.weight_dtype`,
`semantics='net_effect'`, `dyng.label_propagation`, `VertexArrivalStream`; the commands `dyng
label_propagation`, `dyng generate vertex_arrival_stream`, `dyng generate
dynlp_vertex_arrival_stream`; the tool `dyng-compat-dynlp`; the suite `ics26_dynlp`.

**Beyond AD12's list, approved by the author on 2026-11-12** (the author's decision of 2026-10-29;
GOVERNANCE.md, approvals log, the row "M10's public names beyond AD12's list", which closes the
row "A public name beyond AD12's list" of open decisions, M10 (AD), that keeps the full list by
step): the generators' and oracles' auxiliary types (`vertex_arrival_options`,
`vertex_arrival_batch` with `view()`, `vertex_arrival_stream`; `testing::harmonic_options`,
`harmonic_result`, `label_error`, `label_error_against`, `class_agreement`,
`class_agreement_with`, `lp_model_batch`, `lp_model_report`,
`lp_host_model::vertex_state::{inactive, active, deleted}`); the container's
`batch_semantics::as_net_effect`, `apply_summary::dropped_insertions`, `relocated_rows`,
`compactions`, `edge_batch::insert_vertices()`, `insert_vertex_labels()`, `delete_vertices()`,
`vertex_status::{active, deleted, inactive}`; label_propagation's enumerators, option and stats
fields and result members (`schedule::in_place` / `damped`, `on_limit::error` / `report` /
`fallback_recompute`, `options::similarity_threshold`, `tolerance`, `max_iterations`,
`on_iteration_limit`, `update_schedule`, `cuda_engine`, `stats::components`, `sparsified_edges`,
`result::labels`, `seed_labels`, `reserve`), the dispatch names
(`detail::label_propagation_supported_v`, `DYNG_LABEL_PROPAGATION_TYPES_MESSAGE`,
`detail::make_label_propagation_participant`) and the profiler counters
`label_propagation.irlp_iterations`, `hook_rounds`, `persistent_iterations`; Python's
`dyng.VertexStatus`, `Graph.vertex_status()`, the keyword `weight_dtype` of the readers and
constructors and `Graph.with_capacity`'s keywords, `EdgeBatch`'s vertex properties,
`VertexArrivalStream`'s fields and `write_graph_files()`, `dyng.testing.HarmonicResult` and its
`error(labels)`; the command line's flags beyond the options (`--graph`, `--seeds`,
`--index-base`, `--num-vertices`, `--generate legacy|library`, `--batches`, `--capacity`,
`--true-labels`, `--layout`, `--per-batch-labels`, `--out`, the words `published` / `current`,
`dyng generate`'s `--format` and `--graph-files` with their files); the kNN recipe's flags; the
compat tool's diagnostic flags (`--phases`, `--layout`, `--streaming-tau`); and, from this step,
the build option **`DYNG_DEFER_KERNEL_GROUPS`** (part B).

### Outcome

**The deliberate changes** (a throwaway clone of this branch at the freeze commit, reverted and
deleted; the logs in `$DYNG_SCRATCH/runs/m10b-docs-api-review-freeze-final-reads/demo-*.log`):
recorded in `parity/results/M10b.md` section 4 (the C++ default of `options::tolerance` changed:
`ci/docs.sh --doxygen-only` fails against the baseline, and with the baseline regenerated
`ci/api_check.sh` fails against the base; the Python default of `Options.tolerance` changed:
`ci/api_check.sh` fails; reverted: both pass).

## Part B: kernel groups

### Context

Every `.cu` file registers its fat binary with the CUDA runtime from a static constructor, so
loading libdyng registers every module of the library, and under `CUDA_MODULE_LOADING=EAGER` (the
protocol of every gate, AD2) the context creation of every CUDA process loads every kernel of the
library. M10a measured the cost (`parity/results/M10a-merged/context-probe.json`): the merged
library's context 8.6 ms later than the 0.3 library's, 4.8 ms of it label_propagation's kernels and
3.8 ms those M11a added; it put sssp's CUDA `end_to_end` (roadNet-PA unsafe50k, 1.021) and two
cycle_count CUDA regions (`DD_k4` 1.022 / 1.024) above ADR 0024's bar. The author decided that
M10b loads each algorithm's kernels only when that algorithm runs, every algorithm's (M11a's
included), and reads those regions again (2026-11-03, widened 2026-11-05).

### Options considered

- **One shared object per algorithm, `dlopen`ed on first use.** Every algorithm's host boundary
  (the templates its host code calls in its `.cu` files) would need a function table and
  trampolines; static builds (the wheels) could not do it; and M11's `hyper_sssp`, whose boundary
  M11b is changing, would need edits in its files. Rejected.
- **Launches through `cudaLibraryLoadData` / `cudaLibraryGetKernel`.** Every `<<<...>>>`, every
  CUB and Thrust call would change. Rejected.
- **Deferred registration (chosen).** The registration calls are the only thing that makes a
  module eager. Renaming them in the group's objects, recording them, and replaying them on first
  use changes nothing else: not the device code, not the host code, not the kernels' bindings.

### Decision

1. **A kernel group per algorithm, and one for the hypergraph containers.** `dyng_add_algorithm()`
   makes every algorithm's CUDA sources a group named after the algorithm (no algorithm's own files
   change, M11's `hyper_sssp` included); `dyng_add_module(KERNEL_GROUP hypergraph)` makes the
   hypergraph module's (the ESCHER store, `slack_csr`, the line graph: only hypergraph algorithms
   use them). The core, `util` and `graph` modules (the containers every graph algorithm uses)
   stay registered at load.
2. **The mechanism** (`cmake/kernel_groups.cmake`, `cmake/kernel_group_launcher.sh`,
   `cmake/kernel_group.cpp.in`, `cpp/src/core/kernel_groups.{hpp,cpp}`): the group's
   `CUDA_COMPILER_LAUNCHER` compiles each `.cu` file as before, then renames in the object (objcopy
   `--redefine-syms`) `__cudaRegisterFatBinary`, `__cudaRegisterFatBinaryEnd`,
   `__cudaUnregisterFatBinary`, `__cudaRegisterFunction`, `__cudaRegisterVar`,
   `__cudaRegisterManagedVar`, `__cudaInitModule` and the runtime functions that take a kernel or a
   device symbol (`__cudaGetKernel`, which a launch calls once per site, `cudaFuncGetAttributes`,
   `cudaFuncSetAttribute`, `cudaFuncSetCacheConfig`, `cudaLaunchKernel`,
   `cudaLaunchCooperativeKernel`, `cudaLaunchKernelExC`, the occupancy queries, the symbol
   functions, `cudaGetFuncBySymbol`) to functions of a generated translation unit of the group, and
   refuses (nm) an object that still calls such a function unrenamed, so a new use cannot slip past
   the group. The group records the registrations at load time; the first renamed call (the
   safety net: a group can never be used unloaded), `kernel_groups::load()` or a warm-up replays
   them and makes one runtime call (`cudaFree(nullptr)`) so that the modules are created there, not
   inside the next (timed) call. Thread-safe; the groups are never destroyed (the runtime's
   `atexit` unregistration may run after the static destructors). It works the same in shared and
   static builds (the wheels).
3. **Warm-ups.** `resources::warm_up()` keeps its frozen contract ("loads every kernel of the
   library"): it loads every group first. `detail::warm_up_kernel_groups(res, {names})` warms up
   the kernels outside every group and those of the named groups only; the compatibility tools
   (`dyng-compat-mosp`: sssp, mosp; `dyng-compat-cycle-enum`: cycle_count; `dyng-compat-escher
   run`: triad_count, hypergraph) use it, so their start-up still loads every kernel they run
   before any timed region (the originals' eager loading) and nothing else. `dyng-compat-hsosp`
   (M11's) is unchanged: it still loads everything. label_propagation loads its own group at
   `compute()` and before an update's first CUDA work (its compat tool uses public headers only),
   so the published flow pays for its modules in the untimed `compute()`.
4. **`DYNG_DEFER_KERNEL_GROUPS`** (CMake option, default ON) turns the mechanism off: every module
   is registered at load, as before 0.4. Builds without CUDA have no group.
5. **Tests.** `dyng_kernel_groups_tests` (each test a process of its own under CTest): every
   algorithm with CUDA kernels and the hypergraph containers have a group; creating a context loads
   none; `load()` loads only the named groups; the first `sssp::compute()` loads sssp's group and no
   other; `warm_up()` loads every group and every registered kernel; the narrow warm-up leaves the
   others unloaded; the registered kernels carry their group. Every other GPU suite exercises the
   safety net (no test loads a group explicitly).

### Consequences

- A CUDA process loads the kernels of the core and the graph container, then each algorithm's on
  its first use: a label_propagation process no longer loads cycle_count's 6 MB of fat binaries or
  the hypergraph containers', and an sssp process none of label_propagation's or M11's. The
  re-read regions and the context probe are `parity/results/M10b.md` section 4.
- Under lazy module loading (the CUDA 12.2+ default outside the gates) nothing changes for users.
- A new runtime entry point that takes a kernel or a device symbol must be added to
  `DYNG_KERNEL_GROUP_SYMBOLS` and the template; the launcher's check fails the build until it is.
- For M11 (the integrator): `dyng-compat-hsosp` may use the narrow warm-up as the other tools do
  (its `end_to_end` would then load hyper_sssp's, sssp's and the hypergraph containers' groups
  only); M11's mutant copies of the hypergraph module register at load, as every mutant copy does.
- Later, outside 0.4: a vertex-capacity `graph::reserve()` (finding 10), for the author.
