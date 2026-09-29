# Plan: the short roadmap

This is the living plan of dynG in short form: what the library is, the decisions it rests on,
the milestones with their exit criteria, the risks and the open decisions. The full planning
record (the design of every module, the per-algorithm migration plans, the review history) is
the approved planning document of 2026-09-27, which is not published; as each milestone starts,
the part it needs moves into a design page under `docs/developer/`. Maintainer records (the
ADRs, the retrospectives, the repository-settings guide, comments in the workflows) cite it as
"PLAN Section N"; contributor documentation does not depend on it. Decisions that
change are recorded as ADRs ({doc}`../adr/index`), progress and re-estimates in the milestone
retrospectives ({doc}`retrospectives/index`). The public, user-facing version is
{doc}`../roadmap`.

Status: 2026-09-28. M1a, M1b (CUDA `sssp`) and M4 (this infrastructure) are done and merged
into `main` (integration INT1). M2 (`cycle_count`) is done on the branch `m2b-cycle-cuda`, which
merges its CPU half (M2a) and adds the CUDA backend (M2b): bit-identical to the original's CUDA
backend and within its performance gates; it reaches `main` through a pull request. Next: M3.

## 1. What dynG is

dynG is one C++17/CUDA library, with Python bindings designed in, that keeps the results of
graph and hypergraph algorithms up to date while the structure changes in **batches**, without
recomputing from scratch. It unifies six research repositories of the author (S M Shovan, with
Arindam Khanda, S M Ferdous, Sanjukta Bhowmick, Anurag Satpathy and Sajal K. Das) into one
library:

| Algorithm | Origin (pinned commit) | Paper |
|---|---|---|
| `sssp` | MOSP-OpenMP@c352151, MOSP-CUDA@e220ee2 | DynaMOSP (IPDPS 2025, TPDS 2025) |
| `cycle_count` | CycleEnumeration-GPU@0a976ad | TruCy / DynTruCy (IEEE TC, submitted) |
| `mosp` | MOSP-CUDA, MOSP-OpenMP | DynaMOSP |
| `triad_count`, the hypergraph store | ESCHER-GPU@abcf9b2, MOSP_ESCHER@4b86159 | ESCHER (IPDPS 2026), ESCHER+ (TKDE 2026) |
| `label_propagation` | LabelPropagation-CUDA@a276a3a | DynLP (ICS 2026) |
| `hyper_sssp` | MOSP_ESCHER@4b86159 | H-SOSP (IA3 at SC 2026) |

The shape of the library in one paragraph: the **public side is API-first** in the RAPIDS style
(one `resources` handle, one graph type, one hypergraph type, one batch type per structure, and
the same `compute` / `update` verbs, `options`, `result` and `stats` types for every algorithm);
the **inside is framework-first** in the Gunrock style (every algorithm is an instance of the
thesis Chapter 3 update template, {doc}`../concepts/update_model`; hand-fused paper kernels run
as fused engines inside the same pipeline); the **process is contributor-first** (a mandatory
sequential reference backend, a start-green scaffold, a conformance kit, maturity levels, ADRs).
The originals stay untouched; every port is proved byte-identical to its pinned commit.

### Who it is for

| User | What they need |
|---|---|
| Researchers who *use* dynamic algorithms (Python, notebooks, PyTorch/CuPy) | `pip install`, zero-copy arrays, clear errors, a first result in 10 lines |
| C++ / HPC developers who embed it | `find_package(dyng)`, stream-ordered calls, a pluggable allocator, host-compilable headers, no global state |
| Researchers who *write* update algorithms | a template, a scaffold that is green on day one, a CPU path without a GPU, a documented framework, conformance tests |
| Paper reviewers and readers | reproducible benchmark suites, honest "paper vs code" notes, citable releases (DOI) |

### What makes it one library

One contract (the same five names per algorithm), one container and one batch type per
structure, one update pipeline, one shared engine where algorithms share one (the SSSP engine
serves `sssp`, `mosp` and `hyper_sssp`; the signed recount serves `cycle_count` and
`triad_count`), one vocabulary (errors, profiler stage names, statistics, CLI verbs, page
sections), one style and one test kit. Each is enforced by code, tests or review.

### Non-goals for 0.x

Distributed memory and multi-GPU; HIP or SYCL backends (the maintainers will not build these; a
contributor-led proposal is welcome, see the end of Section 3); a general static analytics
library (Gunrock and cuGraph exist); graph databases or persistence; floating-point weights for
shortest paths; reproducing paper claims the code never implemented (the kappa-truncated TruCy
search, incident-vertex and temporal triads, the insert-only MOSP of thesis Chapter 5, open
h-motifs: these are roadmap items and the docs say plainly that they are missing); kNN graph
construction (a Python recipe instead); a C API before 1.0.

### Success criteria

1. Every port matches its fixed original **exactly** (byte-identical outputs or bit-identical
   counts), or within a documented, calibrated tolerance where the original is
   non-deterministic; approximate algorithms are also checked against a converged reference.
2. Built with the `parity` preset on the reference GPU (RTX A5000, sm_86, CUDA 13.1), every
   port is within **5 %** of the original on paper-timed regions of 10 ms or more, and within
   **10 %** on shorter regions and end to end.
3. `pip install dyng` works without a GPU; `pip install "dyng[cu13]"` adds CUDA.
4. **The 1.0 criterion:** an outside contributor adds an algorithm using only the documentation.

## 2. Decisions

The decisions at a glance (D1-D17 of the plan), with the author's decisions at approval
(2026-09-27), which override the plan's earlier defaults:

| Topic | Decision |
|---|---|
| Name | **dynG**; identifiers `dyng` (`namespace dyng`, `dyng::dyng`, `DYNG_*`, `import dyng`); ADR 0001 |
| Home | GitHub organization **`dyng-dev`**, repository `dyng-dev/dyng`, **public from the start** |
| License | **Apache-2.0** + `NOTICE` + `CITATION.cff` + `dyng::citation()`; a Zenodo DOI per release; ADR 0002 |
| IP and consent | the university IP office asked for the code to be public (cleared); no consent gating of ports: a port is published as soon as it passes its gates |
| History | fresh git history; provenance through file headers and `parity/references.toml` |
| PyPI | `dyng` reserved with a real 0.0.1 through Trusted Publishing (`release.yml`, environments `pypi` / `testpypi`) |
| Contact | `sm.shovan@gmail.com` for the Code of Conduct and security reports, plus GitHub private vulnerability reporting |
| Toolkit | C++17 (CI also C++20); CUDA >= 12.4 for source builds; CMake >= 3.30; GCC >= 11 / Clang >= 15; Python >= 3.12 |
| Layout | RAPIDS-style `cpp/` + `python/`; host-compilable public headers in `cpp/include/dyng/`; framework and operators private in `cpp/src/` |
| Naming | snake_case, no `_t` on types, `_t` only on template parameters, `.hpp` / `.cuh` / `.cpp` / `.cu`, `DYNG_` macros; ADR 0004 |
| Contract | `compute(res, container, inputs, options) -> result`; `update(res, container&, batch, result&) -> stats` applies the batch; stale results throw; per-algorithm oracle kind; ADR 0006 |
| Several results | `dyng::update(res, g, batch, r1, r2, ...)` applies the batch once (0.1) |
| Backends | `sequential` (mandatory reference), `openmp`, `cuda`, chosen at run time through `resources` |
| Framework | the thesis template as a CRTP `problem` + enactors, extracted from `sssp` and `cycle_count` in M3; internal-stable; public (experimental) in 0.5 |
| Performance | straight port first, refactor commit by commit; fused engines allowed; gates 1.05x / 1.10x |
| Python | nanobind + scikit-build-core; CPU wheel `dyng` + CUDA plugin wheels `dyng-cu12` / `dyng-cu13` |
| Docs | Doxygen XML -> Breathe -> Sphinx (MyST, pydata theme), Diataxis; Read the Docs after checkpoint A4 |
| Versioning | SemVer from 0.1.0, one `VERSION` file, Keep a Changelog, `vX.Y.Z` tags |
| Contributions | DCO sign-off for external contributors (no CLA; maintainers exempt, M4 retrospective), Contributor Covenant 3.0, private vulnerability reporting, per-algorithm CODEOWNERS |
| Execution | "execute entirely": milestone by milestone; the author is asked only for account-level actions (GitHub, PyPI, Read the Docs, Zenodo) and real blockers |

The approval checkpoints (ADR 0014): A1 (create the GitHub home) and A2 (public) are approved;
A3 (PyPI name reservation) is approved and done; **A4** (Read the Docs, Zenodo, conda-forge, a
domain) is still to be asked, at M6; A5 (per-port consent) is not needed. Every approval is
logged in `GOVERNANCE.md`.

## 3. Milestones

```text
M0 decisions ─► M1a CPU sssp ─┬─► M1b CUDA sssp ─┐
                              └─► M2 cycle_count ─┴─► M3 framework + 0.1 API freeze ─┐
A1 ─► M4 GitHub infrastructure (parallel with M1-M3) ─────────────────────────────────┴─► M5 Python, CLI, docs ─► 0.1.0
0.1.0 ─► M6 0.1.x hardening │ M7 sssp operators + mosp │ M8 ESCHER store ─► M9 triad_count ─► 0.2.0
0.2.0 ─► M10 label_propagation │ M11 hyper_sssp ─► 0.3.0 ─► 0.4 ─► 0.5 (framework public) ─► 1.0
```

| Milestone | Deliverables | Exit criteria | Status |
|---|---|---|---|
| **M0** Decisions and drafts | LICENSE, NOTICE, CITATION.cff, AUTHORS, README; the 0.0.1 package and its workflow; the provenance record | drafts reviewed; A1 and A3 answered | done (the provenance record, {doc}`provenance`, was written in M4; the NOTICE wording awaits the author's confirmation, O3 below) |
| **M1a** Walking skeleton: CPU `sssp` | repository, presets, minimal core, `graph` (compact rows, MOSP semantics), MOSP I/O, `sssp` sequential + OpenMP, parity harness (first slice), `ci/check.sh`, `cpu.yml`, ADRs 0001/0002/0004/0006/0010/0013 | byte-identical to MOSP-OpenMP@c352151 on the small corpus; updated CSR byte-equal to `applyChangeBatch`; OpenMP A/B recorded; a retrospective with a re-estimate | **done** (495/495 golden cases byte-identical on every CPU backend) |
| **M1b** CUDA `sssp` (fused) + performance harness | CUDA build, streams, CCCL-shaped memory resources, device buffers; resident device graph; the persistent cooperative kernel behind `enact_fused`; `generators::legacy`; `parity/perf_ab.py`; the `edge_t` benchmark (ADR 0009) | byte parity with MOSP-CUDA@e220ee2 and CUDA = OpenMP = sequential; the performance gate on roadNet-PA/CA, rgg_n_2_20_s0, road_usa | **done** (495/495 golden cases byte-identical on the CUDA backend; gates recorded in the M1b parity certificate; ADR 0009 fixes `edge_t` = int32) |
| **M2** `cycle_count` (parallel with M1b) | sorted-rows / set-semantics preset, the CycleEnum parser, static Johnson and the update on 3 backends, work queue, ported tests, randomized parity suites | bit-identical histograms on the golden corpus; generator identity; performance gate (DD, GitHub, Twitch, COLLAB); two recorded mutations fail | **done** (M2a, the CPU backends, and M2b, CUDA: bit-identical to CycleEnumeration-GPU@0a976ad's OpenMP and CUDA backends on their golden corpora, cross-backend equal; every OpenMP and CUDA gate met in both scopes, device memory equal to the original's; on the branch `m2b-cycle-cuda`, merged by pull request) |
| **M3** Framework extraction + 0.1 API freeze | `problem_base`, enactors, views, workspace, policies, `run_update`; both algorithms moved onto it one commit at a time; conformance kit C1-C12; `new_algorithm.py` + template; `regen.py`; API review | both algorithms pass the kit on every backend with parity and performance unchanged; the scaffold is green on first build; the API review recorded | |
| **M4** GitHub repository and infrastructure (parallel) | community files, issue forms, PR template, CODEOWNERS, labels, DCO; workflows `lint`, `cpu`, `docs`; this documentation site; the repository-settings guide | hosted workflows green; a contributor goes from clone to green build with the documented commands | **done** (merged into `main` with M1b in INT1; the hosted runs start with the first push) |
| **M5** Python CPU wheel, CLI, docs for 0.1 | root `pyproject.toml`, nanobind bindings, stubs, pytest; the CPU wheel + sdist; CLI; getting started, the update-model page, the `sssp` and `cycle_count` pages, the API reference, history pages | pytest green against the goldens; `pip install` of the release candidate works on a clean CPU machine; docs build with `-W` | |
| **0.1.0** | | the release checklist, including the parity certificate | |
| **M6** 0.1.x hardening | CUDA plugin wheels; the tutorial algorithms `dynamic_bfs` and `triangle_delta`; Read the Docs and Zenodo (A4); fuzzers; mutation checks; sanitizer jobs | `pip install "dyng[cu13]"` works; tutorials pass the kit; mutations fail as expected | |
| **M7** `sssp` operators engine + `mosp` | the multi-kernel engine as the non-cooperative fallback; composition over K x `sssp`; combined graph; path costs; `dyng prep` | fused == operators byte-identical; byte parity of all MOSP CLI outputs; performance gate | |
| **M8** ESCHER store + hypergraph | the staged CBST merge (ESCHER-GPU first, then the MOSP_ESCHER behaviours one by one); `hypergraph` with `escher` storage; `hyperedge_batch`; `.hg` I/O | CBST trace replay identical for both originals' policies after every merge step; integrity after every batch | |
| **M9** `triad_count` | wedge engine, h-motif patterns, reference backends, experimental `count_local_patterns` | exact counts and deltas on the golden batches; "baseline + deltas == recount"; performance gate; memory <= 1.05x | |
| **0.2.0** | | release checklist | |
| **M10** `label_propagation` | slotted / slack row layouts, vertex batches, the DynLP pipeline, IrLP, the harmonic reference oracle | generator byte identity; damped byte parity; calibrated in-place tolerances; performance gate 50K-5M | |
| **M11** `hyper_sssp` | `slack_csr` storage, incidence edits, the as-is port, then the shared `sssp` engine | distances and parents by id; line-graph delta equality; the seven mutations fail; performance gate | |
| **0.3.0** | conda-forge (`libdyng` + `dyng`), aarch64 wheels | release checklist | |
| **0.4.0** | OpenMP incremental `triad_count`; Read-Tarjan, time-window and temporal cycle modes; `int64` hypergraph offsets | parity and performance for every new piece | |
| **0.5.0** | the framework promoted to `dyng::experimental::framework` | three algorithms share each promoted piece without loss | |
| **1.0.0** | API freeze of core and stable algorithms; ABI checks; optionally a JOSS paper | an outside contributor has added an algorithm using only the docs | |

Later, open to contributors through the `new_algorithm` issue form: the kappa-truncated cycle
search (as an explicitly approximate mode); time-window and temporal cycle updates; the
insert-only MOSP of thesis Chapter 5; incident-vertex and temporal triads; open h-motifs; an
implicit line-graph policy; new dynamic algorithms (k-core, connected components, PageRank,
BFS); a HIP backend; multi-GPU only if a research need appears.

## 4. Estimate (re-estimated after M1a)

The plan estimated 0.1 at 12-17 working weeks. M1a, estimated at 2-3 weeks, was implemented in
five sessions on one day with the AI assistant doing the coding. The remaining milestones will
not scale by the same factor: M1a was the most mechanical milestone (CPU only, one original with
good tests); most of its effort went into work that grows with every later port (the parity
harness, performance investigations, documents); every later port adds GPU performance gates on
a shared, noisy machine; the independent review of M1a found 25 real defects in a milestone that
had passed its own gate, so every milestone needs a review-and-fix step; and author review and
account actions are calendar time the AI cannot shorten. The re-estimate applies a speed-up of
about 3x to the plan's figures:

| Milestone | Plan (working weeks) | Focused effort | Calendar, incl. author gates | Main risk |
|---|---|---|---|---|
| M1b CUDA `sssp` | 2 | 4-6 days | 1.5-2 weeks | the per-objective OpenMP gate (1.05x); end-to-end cost; the cooperative kernel's registers |
| M2 `cycle_count` | 2-3 | 4-6 days | 1-1.5 weeks (parallel with M1b) | the sorted-rows / set-semantics preset; gates on 4 datasets in both scopes |
| M3 framework + API freeze | 2-3 | 5-8 days | 2-3 weeks | design judgment; parity and performance re-run per commit; API sign-off |
| M4 infrastructure | 1-2 | 1-3 days | 3-5 days | first hosted runs; DCO app and rulesets (author) |
| M5 Python, CLI, docs | 2-3 | 5-8 days | 2-3 weeks | nanobind + scikit-build-core, stubs, TestPyPI release candidate |
| **0.1.0** | **12-17** | **about 4-6 weeks** | **about 7-10 weeks** | |
| 0.1.x (M6: CUDA plugin wheels, tutorials, RTD/Zenodo, fuzzers, mutations) | 4-6 | 1.5-2 weeks | 3-4 weeks | the GPU runner decision (O11); wheel sizes |
| 0.2.0 (M7 `mosp` + operators engine, M8 ESCHER store, M9 `triad_count`) | 8-12 | 3-4 weeks | 6-8 weeks | the staged CBST merge; memory within 1.05x; performance on the 2M-hyperedge suites |
| 0.3.0 (M10 `label_propagation`, M11 `hyper_sssp`) | 8-12 | 3-4 weeks | 6-8 weeks | DynLP's calibrated float tolerances and slotted/slack layouts; conda-forge review time |
| **Up to 0.3.0** | **about 32-47** | **about 12-16 weeks** | **about 22-30 weeks** | |

The calendar column assumes the author reviews at each gate within a few days. The estimate is
revisited in the M1b and M3 retrospectives.

## 5. Risks

The risks that shape the plan (the planning record lists 23), with their mitigations:

| Risk | Mitigation |
|---|---|
| **Performance loss from layering** (dispatch, templated hooks, resident apply, version checks, extra syncs) | dispatch once per call; straight port first; fused engines (Tier B); per-commit gates; SASS and register diffs |
| **The framework is bent by later algorithms** (DynLP, H-SOSP in 0.3) | internal-stable until 0.5; extracted from two real, opposite algorithms; the rule of two; changes by ADR |
| **Public API designed before the ports turns out wrong** | only the 0.1 headers are frozen; the rest are reviewed sketches; 0.x allows breaks with migration notes |
| **The CBST merge** (two diverged copies, two policies) | trace replay against both originals before `triad_count`; its own milestone (M8) |
| **`update` applies the batch** surprises users sharing a graph | `stale_result_error` makes misuse loud; the multi-result `dyng::update` is the documented pattern |
| **No GPU CI** (shared lab machine) | manual `ci/gpu_local.sh` before CUDA merges and releases, summary posted on the PR; hosted compile-only CUDA builds |
| **Paper-vs-code gaps** | a mandatory "Differences from the paper" section on every algorithm page; names say what is computed |
| **Maintainer bandwidth** | small tooling surface; release automation; maturity levels; Discussions for support; a 7-day review target, not a promise |
| **Scope creep** | the walking skeleton first; 0.1 is two algorithms and a CPU wheel; exit gates per release; retrospectives re-scope |
| **Parity evidence lost** | persistent `$DYNG_SCRATCH`; SHAs in `references.toml`; the archive-build check; a committed parity certificate per release |

## 6. Open decisions

Each has a recommended default, so work proceeds; it is recorded as an ADR when taken.

| # | Decision | Default | Needed by |
|---|---|---|---|
| O6 | default `edge_t` | decided by the M1b benchmark: int32 with checked construction if int64 costs > 3 % (ADR 0009) | end of M1b |
| O11 | GPU CI | manual `ci/gpu_local.sh` until IT answers about a container runtime; a paid hosted T4 when outside contributors arrive | after IT answers |
| O24 | `sssp` operators engine | 0.2; in 0.1 a clear `not_supported_error` without cooperative launch | M1b |
| O27 | oracle of approximate algorithms | a converged reference within a tolerance derived from the stopping rule | M3 (kit), M10 |
| O18 | governance | lead maintainer until three active maintainers | M4 |
| O15 | changelog mechanics | hand-edited `Unreleased` until merge conflicts hurt | M4 |
| — | milestone merges | through pull requests (the review rules of `CONTRIBUTING.md`), or pushed directly to `main` with a ruleset bypass (recorded in `GOVERNANCE.md`, which notes the interim practice) | M4 |
| O3 | the NOTICE institution line and years | "developed at the Missouri University of Science and Technology", "Copyright 2023-2026 their authors" (the draft in `NOTICE`) | before 0.1.0 |
| O23, O28, O7 | Python floor, wheel scheme, CPU wheel | Python 3.12; CPU `dyng` + CUDA plugins; a CPU wheel from 0.1 | M5 |
| O14 | docs hosting | Read the Docs after A4 (GitHub Pages as the fallback) | M6 |
| O8, O17 | CUDA wheels, oldest GPU | 0.1.x as plugins; sm_75 | M6 |
| O21 | 0.2 if the CBST merge runs late | ship `mosp` as 0.2, triads in 0.2.x or 0.3 | M8 |
| O16 | hypergraph `id_reuse` default | `erase_first`; parity runs set `insert_first` | M8 |
| O9, O10 | DynLP default mode, kNN construction | the published setup; a Python recipe | M10 |
| O12 | H-SOSP engine | the shared `sssp` engine after an as-is port proves parity | M11 |

Closed at approval: the name (O1), the license (O2), the GitHub home (O1′), the local path
(O19), history import (O22), privacy until IP clearance (O26: public now), the contact address
(O13), the algorithm names (O4: `sssp`, `cycle_count`, `hyper_sssp`), type-name style (O5), row
order and multigraph semantics (O25; ADR 0010), publishing before consent (O20: not needed).

## 7. How this plan changes

This page is updated at the end of each milestone, from its retrospective. A changed decision
gets an ADR (old ADRs are marked superseded, never rewritten); a scope change or re-estimate is
made in the retrospective, not in the middle of a port. A version number only guarantees
behaviour once it is released: before 0.1.0 anything can change; after it, SemVer applies.
