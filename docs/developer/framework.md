# The framework: the update template in code

**Status:** internal-stable from 0.1 (PLAN Section 4.5). The headers live in `cpp/src/framework/`
and are not installed; they change only with a note in the CHANGELOG and this page. This page is
the guide for writing an algorithm on the framework. The "how to add an algorithm" guide (PLAN
9.4) starts from here.

Every dynamic algorithm in dynG follows one template, the one in Chapter 3 of the thesis plus
ten mechanisms that the corrected research codes leave implicit (PLAN 4.5.1). The framework
turns that template into code:

- a **problem** is a class with *hooks*, one per step of the template;
- an **enactor** calls the hooks in a fixed order, opens one profiler stage per hook, applies the
  convergence policy, checks the device error word and measures the budget;
- **composition** lets several problems share one commit of a batch (`dyng::update(res, g, batch,
  r1, r2, ...)`).

## The template card

```text
normalize → translate → prepare → [before_apply → (AG: count −)] → commit →
identify_affected → seed → { FP: loop until is_converged | AG: count + } → finalize
```

FP is `family::fixed_point` and AG is `family::aggregate_delta`. Everything left of `commit`
reads G_t, the graph before the batch. Everything right of it reads G_{t+1}, the graph after the
batch. `translate` is on the card but has no hook yet: no algorithm in 0.1 uses it (see
[Differences from PLAN 4.5](#differences-from-plan-45)).

## The files

| File | Contents |
|---|---|
| `framework/problem.hpp` | `family`, `not_provided`, `problem_base<derived_t, family>` (the CRTP base with no-op default hooks) |
| `framework/enactor.hpp` | `update_enactor<problem_t>` (update), `static_enactor<problem_t>` (compute), `hook_stages` / `stages_of<problem_t>()` (stage names) |
| `framework/views.hpp` | `old_view<container_t>` (G_t), `new_view<container_t>` (G_{t+1}); invariant I1 |
| `framework/context.hpp` | `context`: the resources, the algorithm name, the workspace pool, the device error word of one run |
| `framework/policies.hpp` | `convergence`, `on_limit`, `sign`, `ownership::min_member` |
| `framework/frontier.hpp` | `internal_frontier`, the `has_empty` trait |
| `framework/budgets.hpp` | `budget`, `budget_scope`, `check_budget`, `last_budget_report` (invariant I9) |
| `framework/conformance.hpp` | the compile-time checks (`static_assert`s with plain-English messages) |
| `framework/composition.hpp` | `requested_batch`, `applied_batch`, `problem_participant<problem_t>`, `make_participant`, `update_one`, `expect_current_result`, `stamp_result` |
| `framework/composition.cpp` | `run_update()`: Step 0 of set semantics once, every before-commit half, one commit, every after-commit half |
| `framework/workspace.hpp`, `scratch_buffer.hpp` | the workspace pool of a `resources` handle (ADR 0015) |
| `core/budget_counters.hpp` | the per-thread allocation and host-synchronization counters behind I9 |

Everything new lives in `dyng::detail::framework`. The M1b/M2 pieces keep their namespace
`dyng::detail`: the workspace pool, `scratch_buffer`, `update_participant` and `run_update`.
PLAN 4.2 names `snapshot.hpp` and `budget.hpp`; this extraction calls them `views.hpp` and
`budgets.hpp`.

## Families

- **`family::fixed_point`**: a value per element, and Step 2 iterates to a fixed point. Examples:
  `sssp`, later `mosp` (by composition), `hyper_sssp`, `label_propagation`. Required hook:
  `loop` (Tier A) or `enact_fused` (Tier B).
- **`family::aggregate_delta`**: one global aggregate.
  P_{t+1} = P_t − Σ contrib(G_t, owned by the deletions) + Σ contrib(G_{t+1}, owned by the
  insertions). Step 2 is one signed count per side under an ownership rule, with no iteration.
  Examples: `cycle_count`, later `triad_count`. Required: `ownership_type`, `count` on the old
  view (the subtraction, before the commit), and `count` on the new view (Tier A) or `enact_fused`
  (Tier B).

## Writing a problem

```cpp
class my_problem : public framework::problem_base<my_problem, framework::family::fixed_point> {
 public:
  static constexpr std::string_view name = "my_algo";  // stages "my_algo.<hook>"
  using container_type = graph<std::int32_t, std::int64_t, float>;
  using stats_type = my_algo::stats;                   // derives from update_stats
  // using frontier_type = ...;                        // optional (internal_frontier)
  // using ownership_type = framework::ownership::min_member;   // aggregate_delta only

  const void* target() const noexcept { return &result_; }
  void identify_affected(framework::context&, framework::new_view<container_type>,
                         const framework::applied_batch<std::int32_t>&,
                         framework::internal_frontier&);
  void loop(framework::context&, framework::new_view<container_type>,
            framework::internal_frontier& in, framework::internal_frontier& out);
  void finalize(framework::context&, my_algo::stats&);
};
```

A default hook of `problem_base` returns `not_provided`. The enactors test the return type of each
hook call at compile time. They neither call a hook that returns `not_provided` nor open its
stage, so the profile of an algorithm lists exactly the hooks it implements. A hook the problem
defines hides the default of the same name (ordinary C++ name hiding). This is why the base has
at most one overload per hook name: `compute_fused` is the static twin of `enact_fused`, and
`seed_static` the static twin of `seed`.

### Hooks

| Hook | Stage | When | Reads |
|---|---|---|---|
| `normalize(ctx, old, batch)` | `<algo>.normalize` | Step 0: the problem's own normalization (the framework's set normalization has already run; `batch.normalized`) | G_t |
| `prepare(ctx, old, batch)` | `<algo>.prepare` | Step 0: classify, summarize, check the batch | G_t |
| `before_apply(ctx, old, batch, f)` | `<algo>.before_apply` | Step 1a: read G_t before it changes | G_t |
| `count(ctx, old, f, sign::minus, o)` | `<algo>.count_minus` | Step 1a (AG): the subtraction | G_t |
| — commit — | `<algo>.commit` / `update.commit` | `run_update()`, once for every result | |
| `identify_affected(ctx, new, applied, f)` | `<algo>.identify_affected` | Step 1b: roots, invalidation, the affected frontier | G_{t+1} |
| `seed(ctx, new, f)` | `<algo>.seed` | Step 1b: initial values of new or invalidated elements | G_{t+1} |
| `loop(ctx, new, in, out)` | `<algo>.loop` (one stage for all of Step 2) | Step 2 (FP), until `is_converged` | G_{t+1} |
| `count(ctx, new, f, sign::plus, o)` | `<algo>.count_plus` | Step 2 (AG): the addition | G_{t+1} |
| `finalize(ctx, stats)` | `<algo>.finalize` | finish: combine, apply deltas, unpack, fill the stats | – |
| `enact_fused(ctx, new, applied, stats)` | `<algo>.enact_fused` | Tier B: replaces `identify_affected` … `finalize` | G_{t+1} |

compute() runs the static enactor: `reset(ctx)` (`<algo>.reset`) → `seed_static(ctx, new, f)`
(`<algo>.seed`) → `loop` until converged (FP, `<algo>.loop`) or `count(ctx, new, f, sign::plus,
o)` (AG, `<algo>.count`) → `finalize` (`<algo>.finalize`). In Tier B it runs
`compute_fused(ctx, new, stats)` (`<algo>.enact_fused`) instead.

**Policy hooks** (no stage):

- `is_converged(ctx, f, iteration)`. Default: `f.empty()` for a frontier with `empty()`. For an
  `internal_frontier` it returns "the loop hook has run once", so that call iterates to the fixed
  point inside the engine.
- `convergence_policy(ctx) -> convergence`. Default: no cap.
- `select_engine(ctx) -> engine`. Default: `engine::automatic`, which resolves to `fused` when
  the problem has a fused engine and `fused_available(ctx)` says it can run in this call, and to
  `operators` otherwise.
- `fused_available(ctx) -> bool`. Whether the fused engine can run in this call, for example
  `ctx.on_cuda()` (and cooperative launch). Default: not provided, read as true. A problem with
  both a Tier A engine and a fused one must provide `fused_available` or `select_engine`
  (a compile-time check), because a fused engine is written for one backend.
- `algorithm_budget(ctx) -> budget`. Default: `budget::unchecked()`.
- `recompute(ctx, new, stats)`. Needed only for `on_limit::fallback_recompute`.

**Lifecycle members**. The enactor and the participant adapter use them; they have no stage, and
they are the "argument validation, version check, result bookkeeping" of `<algo>.cpp` (PLAN 4.8):

- `target()`: the result's address, so that a result passed twice is rejected. Required for
  `run_update()`.
- `begin_update(ctx, old, batch)`: validate the call and bind the result before Step 0. It must
  change nothing visible, because a later participant can still reject the batch. The update
  enactor calls it first in `before_commit`, then chooses the engine.
- `resume(ctx, new, applied)`: re-bind to G_{t+1}. Grow the result for new vertices, lease and
  size the workspace, and build per-run inputs; it may open sub-stages of its own, such as
  `sssp.workspace` and `sssp.changes`.
- `end_update(ctx, new, stats)`: record the graph state the result now matches (`stamp_result`).
- `poison()`: the algorithm phase failed, so the result is unusable until it is recomputed.
- `reads_prepared_graph()`: whether the commit must build the in-edges or the device copy for
  this problem. Default true.

### The common stats

The enactor sets the fields of `update_stats` that it decides:

- `engine_used`: `fused` when `enact_fused` ran, `operators` otherwise. `finalize` may refine it;
  sssp's OpenMP engine is the ported paper engine and reports `fused`.
- `converged`: false only under `on_limit::report`.
- `fallback_used`: true after `on_limit::fallback_recompute`.

The adapter copies the commit's `apply_summary` into `stats.batch` when the stats type has that
field. The problem fills everything else in `finalize` or `enact_fused`: `affected`,
`iterations`, `frontier_visits` and its own counters.

## Tier A and Tier B

- **Tier A (framework-composed).** Steps 1b-2 are written as hooks. This is the default for new
  algorithms and for the tutorials.
- **Tier B (custom engine).** `enact_fused(ctx, new, applied, stats)` replaces `identify_affected`
  … `finalize` for one backend (usually CUDA): the persistent cooperative SOSP kernel, the
  CycleEnum work queue. The framework still owns `normalize`, `prepare`, `before_apply` and the
  AG subtraction, the commit, the stats, the profiler stages, the device error check and the
  tests. Tier B is fully legitimate. A fused paper kernel is not pushed into an abstraction that
  costs 20 %.

The enactor chooses the engine once, in `before_commit`, before the batch is applied:
`select_engine`, with `engine::automatic` resolved through `fused_available`. If the choice is an
engine the problem does not have or one that cannot run in this call (`fused_available` false),
or the convergence policy asks for `on_limit::fallback_recompute` and the problem has no
`recompute` hook, the enactor throws `not_supported_error` there, so the graph and the result stay
unchanged (the strong guarantee of `update()`). The choice is kept for `after_commit` and every
hook can read it through `ctx.chosen_engine()`; cycle_count's `count(-)` on G_t uses it to run
the fused engine's own subtraction.

## Frontiers

The enactor owns two frontiers of `frontier_type` for one run: the input and the output of a
loop round. It swaps them after every `loop` call. A frontier type must be default-constructible
without allocating or throwing (checked), because its storage belongs in the pooled workspace.
In 0.1 both algorithms keep their frontiers inside their engines' workspaces:

- sssp: the affected lists, the near-far piles and the invalidation lists;
- cycle_count: the change lists with their ownership index.

So `internal_frontier` is the only framework frontier. The sparse, dense, bucketed and binned
frontiers of PLAN 4.5.3 and the work items enter `frontier.hpp` with their second user (rule of
two). Until then a problem may define its own frontier type; the framework tests use a list
frontier.

## Policies

- `convergence { kind, tolerance, max_iterations = -1, at_limit = on_limit::error }`. The enactor
  applies the cap. The rule itself is evaluated by `is_converged`.
- `on_limit::error` throws `convergence_error`. `on_limit::report` finalizes the partial result
  with `converged = false`. `on_limit::fallback_recompute` calls `recompute` instead of the rest
  of Step 2 and `finalize`. compute() treats a fallback as an error, because compute() is itself
  the recomputation.
- `sign::minus` and `sign::plus` are the two sides of an aggregate-delta count.
- `ownership::min_member` is cycle_count's rule: a structure that contains several changed
  elements of one phase belongs to the changed element with the smallest id. An
  aggregate-delta problem must name its rule as `ownership_type`, and the enactor passes it to
  every count, so there is no default rule to forget (I2).

The plan's `schedule`, `sync_rule` and `tie_break` policies arrive with their first two users.
The sssp operators engine (M7) needed none of them: it has one schedule (the near-far push) and
its lowest-id rule is the packed minimum itself; label_propagation is the first candidate. The
engine choice is public: `dyng::engine`.

## Composition

`run_update(res, g, batch, participants, n, commit_stage)` (`composition.cpp`) runs these steps:

1. It stages the batch to the host.
2. Under `batch_semantics::as_sets` it normalizes the batch **once** (stage `<algo>.normalize`,
   or `update.normalize` for `dyng::update`; ADR 0020).
3. It runs every participant's before-commit half on G_t.
4. It commits once (`<algo>.commit` or `update.commit`).
5. It runs every participant's after-commit half on G_{t+1}. A participant that fails there is
   poisoned, the others still run, and the first exception is rethrown.

`problem_participant<problem_t>` is the participant of one problem: `begin_update` +
`update_enactor::before_commit`, then `update_enactor::after_commit` + `end_update`. On top of it:

- `update_one<problem_t>(res, g, batch, args...)` is the body of `<algo>::update(res, g, batch,
  r)`: the stage `<algo>.update`, one participant, the commit stage `<algo>.commit`;
- `make_participant<problem_t>(stats, args...)` is what an algorithm's `update_traits` returns
  for `dyng::update(res, g, batch, r1, r2, ...)` and `dyng::update_each`.

Algorithm-level composition is plain C++. `mosp` holds K `sssp` problems and a static `sssp` over
the combined graph; `hyper_sssp` holds one `sssp` problem over a line-graph view.

## Invariants I1-I9

| # | Invariant | Mechanism in the framework |
|---|---|---|
| I1 | Subtract on G_t, add on G_{t+1} | `old_view` and `new_view` are distinct types, and the enactor orders the calls. In Debug builds a view checks the graph's version on every access, so an `old_view` kept past the commit throws `internal_error` (`Views.OldViewAfterTheCommitThrowsInDebugBuilds`). The AG subtraction requires a `count` on the old view (compile-time check). |
| I2 | Exactly-once counting | `ownership_type` is required for `family::aggregate_delta` and has no default. The enactor passes it to every `count`. |
| I3 | Fixed-point termination | `convergence.max_iterations` + `on_limit`, applied by the enactor. sssp needs no cap: its invalidation step routes invalidating changes and its distances only decrease. |
| I4 | No oscillating schedules | arrives with `sync_rule` (label_propagation, 0.3) |
| I5 | Canonical outputs | inside the algorithms (sssp's packed (distance, id) words); `tie_break` arrives with its second user |
| I6 | Race freedom | inside the algorithms (owner-group writes, documented atomics); racecheck in the GPU jobs |
| I7 | 64-bit aggregates | inside the algorithms (`count_t = uint64_t`, checked additions) |
| I8 | An oracle exists | `check_problem` requires a stats type derived from `update_stats`; the conformance kit (`DYNG_CONFORMANCE_SUITE`) checks `compute()`, the sequential backend and the oracle kind |
| I9 | Budgets of the algorithm work | `algorithm_budget` + a `budget_scope` around each half (`before_commit`, `after_commit`); `check_budget` logs an excess in `DYNG_DEBUG_BUDGETS` builds and throws under strict budgets (the conformance kit; see below) |

## Budgets (I9)

With `-DDYNG_DEBUG_BUDGETS=ON` (the default in Debug and in the `dev` preset) the library counts
two things (`core/budget_counters.hpp`, per calling thread):

- every allocation of its own memory resources (host, pinned host, CUDA stream-ordered pool);
- every host synchronization through `detail::cuda_synchronize()`, which `resources::synchronize()`
  uses, and the synchronization of a pinned deallocation.

The update enactor measures both halves of the update: `before_commit` (`begin_update`,
`normalize`, `prepare`, `before_apply` and, for `aggregate_delta`, `count(-)` on G_t) and
`after_commit` (`resume` … `finalize`, or `enact_fused`), and checks their sum. PLAN I9 names only
the phase after the commit; the half before it is measured as well because an `aggregate_delta`
problem does half of its algorithm work there (cycle_count's delete phase on the device), and an
allocation or synchronization regression there is the same class of bug. Once the result and the
workspaces are reserved, the two halves together must stay within `algorithm_budget(ctx)`, for
example `budget::steady_state(1)` for a fused CUDA engine that reads its control block back once.
The commit is not included: `run_update()` measures it separately (`last_commit_counts()`), and
container growth is reported, not failed.
`last_budget_report()` returns the last measurement on the calling thread; conformance check C8
reads it.

**What an excess does.** The check runs after the commit, so a throw there would fail, and poison,
a result that is correct. In a plain `DYNG_DEBUG_BUDGETS` build an excess is therefore logged at
`log_level::warn` and the update succeeds. Under *strict budgets* it throws `internal_error`:
`framework::set_strict_budgets(true)`, the RAII `framework::strict_budgets_scope`, or the
environment variable `DYNG_STRICT_BUDGETS=1`. Conformance check C8 and the framework's own tests
arm strict budgets, so a budget regression still fails the test suites.

**Attribution.** The counters belong to the calling thread, so library calls on other user threads
(which PLAN 4.7.4 allows on distinct containers) are not charged to an update. The worker threads
of an OpenMP region cannot name the thread that opened it; their counts (the growth of their
per-thread lists and, in the conformance executables, host heap allocations) go to one shared
tally that every snapshot includes, which is exact while one thread at a time runs OpenMP regions
of the library.

"Once reserved" is made precise by two more counters:

- **Reservations.** Code that grows a reusable array on purpose calls `detail::note_reservation()`:
  the workspace pool when it creates a workspace, `scratch_buffer::reserve` when it grows, the
  per-thread lists of the OpenMP engines (`util/thread_list.hpp`) when a thread takes a larger
  share than ever before, and the problems where they grow their workspaces or results (sssp's
  workspace and result growth, cycle_count's workspace and histogram). A phase that reserved is a
  *reserving* run: its allocations are reported, not held against the budget; its host
  synchronizations are still checked. A phase that reserved nothing must stay within the whole
  budget. C8 warms the handle up with the same shapes and then requires a run that reserves
  nothing, allocates nothing and synchronizes at most the budget.
- **Container work.** The graph's own materializations inside an update (its device copy uploaded
  on first use after a host commit, the host copy downloaded) run in a `detail::container_scope`:
  counted, recorded as container work (`budget_counters::container_allocations`,
  `container_host_syncs`) and never held against a problem's budget.
- **Instrumentation.** The profiler's synchronizations of `profiler_options::sync_stages` (and
  its CUDA event timers) run in a `detail::instrumentation_scope`: counted, recorded as
  instrumentation (`instrumentation_allocations`, `instrumentation_host_syncs`) and never held
  against a problem's budget.

The budgets of the two algorithms: no allocation once reserved; host synchronizations 0 on the
host backends, 1 for sssp's fused engine on CUDA (the control block), 3 + iterations + epochs for
sssp's operators engine on CUDA (the control block after Step 1, after the first split, after each
near-far round and after the unpack; stated after the phase, when the counts are known) and 4 for
cycle_count on CUDA (the staging
of the change lists when the graph has no set semantics, the item counts of the delete phase on
G_t and of the insert phase on G_{t+1}, and the histogram copy). Before the half before the commit
was measured (M3 review), cycle_count's budget was 2 and covered the insert phase only.

These are not counted: a user-installed memory resource, `std::vector` growth of the host engines
(the conformance executables count host allocations too: `cpp/tests/conformance/
allocation_counter.cpp` replaces the global `operator new` and reports every allocation of the
measured update to `detail::note_allocation()`), and synchronizations an engine makes with the
CUDA runtime directly.
Such an engine calls `detail::note_host_sync()` next to the call. sssp's fused engine does this
at its one `cudaStreamSynchronize` (`run_persistent` in `algorithms/sssp/cuda.cu`), the operators
engine at each read-back of its control block (`read_control` in `algorithms/sssp/operators.cu`);
cycle_count's
CUDA engines at theirs (`cuda.cu`: the item count of a phase, the staging of the change lists, the
histogram copy; `static_cuda.cu`: the scalar read-backs, the end of the count, the histogram
copy), and the graph module's device paths (`graph/apply_set_device.cu`,
`graph/device_graph.cu`) at theirs.

## Device errors

A hook that reads back a device error word records it with `ctx.raise_device_error(bits,
detail)`. Reading the word back must be merged into a synchronization the hook makes anyway, for
example the control block of a persistent kernel. The enactor checks the recorded bits at the end
of each half and throws the precise exception there (`throw_device_errors`: `capacity_error`,
`invalid_argument_error` or `internal_error`). An error recorded before the commit therefore
leaves the graph unchanged. An error after it poisons the result (through `run_update`).

## How sssp and cycle_count map onto the hooks

These are the hooks and stages of the two algorithms. The migration commits keep every stage name
and scope: `parity/timed_regions/*.toml` sum these stages. Both algorithms run through the
enactors since their migration (M3; parity/results/M3.md has the log):

- **sssp** is `sssp_problem` in `algorithms/sssp/problem.hpp`, one problem type for the three
  backends. Its `select_engine` picks the Tier A hooks on the host backends (the engine of the
  call's backend, `sssp_sequential_engine` or `sssp_openmp_engine`, is bound in `resume`) and
  `options::cuda_engine` on CUDA, where `fused_available` (cooperative launch) resolves
  `engine::automatic`: `enact_fused` runs the fused engine, and the Tier A hooks run
  `sssp_cuda_operators_engine` (M7, ADR 0026; bound in `resume`, or in `bind_static` for
  compute()). Both CUDA engines report their control-block errors through
  `ctx.raise_device_error` (the operators engine in `identify_affected`).
- **cycle_count** is `cycle_count_problem` in `algorithms/cycle_count/problem.hpp`
  (`family::aggregate_delta`, `ownership_type = ownership::min_member`). The host backends run the
  Tier A hooks: `count` on the old view is the delete phase on G_t, `count` on the new view the
  insert phase on G_{t+1} (and, in compute(), the static count). On CUDA `select_engine` picks
  Tier B: the subtraction on G_t stays a framework hook (`count(-)` runs the delete phase of the
  ported kernels on the resident device graph), and `enact_fused` runs the insert phase, the copy
  of both histograms and the delta; `compute_fused` runs the static device counters. The ported
  code keeps its own stages inside `cycle_count.enact_fused` (`cycle_count.identify_affected`,
  `cycle_count.count_plus` and `cycle_count.finalize` in an update; `cycle_count.reset`,
  `cycle_count.count` and `cycle_count.finalize` in compute()), so every region of
  `parity/timed_regions/cycle_count.toml` reads the same stages as before.

| Step | sssp (host backends) | sssp (cuda) | cycle_count (host backends) | cycle_count (cuda) |
|---|---|---|---|---|
| framework Step 0 | `sssp.normalize` (as_sets only) | same | `cycle_count.normalize` (as_sets only) | same |
| `begin_update` | backend, stale and poisoned checks, graph requirements, placement, engine | same | backend, stale and poisoned checks, graph requirements, placement; the host workspace lease | same, and the CUDA engine |
| `select_engine` | `operators` (the OpenMP engine reports `fused` in `finalize`, as since M1a) | `options::cuda_engine`: `fused` with cooperative launch, else `operators` (`automatic`) | `operators` | `fused` |
| `normalize` | – | – | `cycle_count.normalize`: own Step 0 (other semantics; `compute_structural_change`) or the copy of the framework's lists (as_sets) | same, plus the check that the bound after the batch fits the device counters |
| `prepare` | `sssp.prepare`: weights, largest weight, delta, 62-bit check | same | – | – |
| `count(-)` | – | – | `cycle_count.count_minus`: the host phase on G_t (ownership index over the deletions) | `cycle_count.count_minus`: the change lists uploaded, the device phase on the resident G_t |
| commit | `sssp.commit` | same | `cycle_count.commit` (no transposition: `reads_prepared_graph` is false) | same (the device merge under set semantics) |
| `resume` | grow, `sssp.workspace` lease, change list, bind the backend's engine | grow, `sssp.workspace` lease, `sssp.changes` upload | the Debug check of the normalized batch, the bound after the batch | same |
| `identify_affected` | `sssp.identify_affected` (roots, subtree invalidation) | Tier B, or the operators engine's kernels (pack, roots, pointer jumping, invalidation, insertion heads; one sync) | `cycle_count.identify_affected` (the ownership index over the insertions) | Tier B |
| `seed` | `sssp.seed` (pull pass) | Tier B, or the pull kernel | – | – |
| `loop` / `count(+)` | `sssp.loop` (internal_frontier: one call runs to the fixed point) | Tier B, or the near-far rounds (one sync each) | `cycle_count.count_plus` (the host phase on G_{t+1}) | Tier B |
| `finalize` | `sssp.finalize` (parent recovery, `affected`) | Tier B, or the unpack kernels (one sync) | `cycle_count.finalize` (histogram delta, `internal_error` for a negative bucket) | Tier B |
| `enact_fused` | – | `sssp.enact_fused` (persistent cooperative kernel; its control-block errors via `raise_device_error`) | – | `cycle_count.enact_fused` > { `cycle_count.identify_affected` (the device graph), `cycle_count.count_plus` (the device phase, the histogram copy), `cycle_count.finalize` } |
| `end_update` | stamp the result, return the workspace | same | stamp the result, return the workspaces | same |
| compute() | `sssp.reset` → `sssp.seed` → `sssp.loop` → `sssp.finalize` | `compute_fused` in `sssp.enact_fused`, or the four Tier A stages (operators engine) | `cycle_count.reset` → `cycle_count.count` → `cycle_count.finalize` | `compute_fused` in `cycle_count.enact_fused` > { `cycle_count.reset`, `cycle_count.count`, `cycle_count.finalize` } |

Under set semantics `cycle_count.normalize` is called twice in `cycle_count::update`: the
framework's Step 0 in `run_update` (once for every result and the commit, ADR 0020) and the
problem's `normalize` hook, which takes the framework's lists as its change lists. The profile
has the same rows as before the migration, with `calls = 2` on that row. Under
`dyng::update(res, g, batch, ...)` the framework's Step 0 is `update.normalize`.

## Testing the framework

`cpp/tests/framework/fake_problems.hpp` has one small problem per family:

- `levels_problem`: BFS levels, fixed point, with a list frontier and a Tier B path;
- `pairs_problem`: reciprocal pairs, aggregate delta, `ownership::min_member`.

`enactor_test.cpp` checks these properties:

- the hook order and the graph version each hook sees;
- one stage per implemented hook;
- update chains equal compute;
- the cap and the three `on_limit` policies;
- Tier B;
- device errors before and after the commit;
- the Debug check of I1;
- the budgets;
- composition: two problems, one commit, equal to separate updates on copies.

`cpp/tests/compile_fail/framework_conformance.cpp` checks the `static_assert` messages of
`conformance.hpp` (CTest `framework.conformance.*`).

The algorithms themselves are checked by the conformance kit (C0-C12,
{doc}`conformance`), for every registered algorithm, backend and graph type.

## Differences from PLAN 4.5

Recorded here, as PLAN 0.3 asks. Each keeps the plan's intent.

| PLAN 4.5.2 sketch | The framework | Why |
|---|---|---|
| `reserve(ctx, capacity)` hook | none | Workspaces are sized by their lease (ADR 0015); no enactor calls a reserve step. |
| `translate` hook | none yet | No 0.1 algorithm uses it; it arrives with hyper_sssp or mosp (rule of two). |
| `prepare(ctx, batch)` | `prepare(ctx, old_view, batch)` | sssp's prepare reads G_t (MOSP computes the weight summary before the batch). |
| – | `normalize(ctx, old_view, batch)` hook | cycle_count reduces batches of other semantics itself; the task's hook order starts with `normalize`. |
| `finalize(ctx)` | `finalize(ctx, stats&)` | The stats are filled at the end of the phase in both algorithms. |
| `count(ctx, snapshot, frontier const&, sign, ownership)` | `count(ctx, view, frontier&, sign, ownership)`, overloaded on the view type | The view type states G_t or G_{t+1} (I1); a `const&` hook also binds. |
| `enact_fused` for compute() | `compute_fused(ctx, new_view, stats&)` | One overload per hook name (name hiding); the stage is still `<algo>.enact_fused`. |
| Validation and bookkeeping | lifecycle members `begin_update`, `resume`, `end_update`, `poison`, `target` | They are the `<algo>.cpp` duties of PLAN 4.8, so the participant adapter can run any problem. |
| `run_update(ctx, g, batch, problems...)` returning a tuple | `run_update(res, g, batch, participants, n, stage)` + `problem_participant` / `update_one` / `make_participant` | The type-erased participants of M1a let `dyng::update` combine results of algorithms compiled in different translation units. |
| frontier table | `internal_frontier` only | Both algorithms keep their frontiers in their workspaces (rule of two). |
| Tier B replaces `identify_affected` … `finalize` | cycle_count's `enact_fused` / `compute_fused` open the ported code's own stages inside `<algo>.enact_fused` | The regions of `parity/timed_regions/cycle_count.toml` (the paper's `kernel_ms` is `cycle_count.count`) keep their stages; the only new row is `cycle_count.enact_fused` on CUDA. |
| the problem's `normalize` after the framework's | under set semantics cycle_count's `normalize` hook takes the framework's lists (a second call of the `cycle_count.normalize` stage) | The hook is Step 0 for every semantics; the stage rows are unchanged. |
| `schedule`, `sync_rule`, `tie_break` | not yet | No two users. |
| `snapshot.hpp`, `budget.hpp` | `views.hpp`, `budgets.hpp` | Names of the extraction task. |
