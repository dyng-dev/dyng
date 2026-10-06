# Architecture Decision Records

Every decision that shapes the library, or changes an earlier decision, is recorded here as an
ADR: `NNNN-short-title.md` with the sections **Status**, **Context**, **Decision** and
**Consequences**. ADRs are written when their decision is first exercised. An ADR is never
rewritten after it is accepted; a later ADR marks it "Superseded by NNNN".

| ADR | Title | Status |
|---|---|---|
| [0001](0001-name.md) | Name: dynG | Accepted |
| [0002](0002-license-and-credit.md) | License and credit | Accepted |
| [0003](0003-language-and-toolkit-floors.md) | Language and toolkit floors (C++17, CUDA >= 12.4, sm_75, CCCL 3.x memory-resource shape) | Accepted |
| [0004](0004-naming.md) | Naming conventions | Accepted |
| [0006](0006-algorithm-contract.md) | Algorithm contract | Accepted under delegation with the 0.1 API freeze (2026-09-29, ADR 0023) |
| [0009](0009-default-index-types.md) | Default index types (edge_t = int32 with checked construction) | Accepted (M1b) |
| [0010](0010-batch-and-graph-semantics.md) | Batch and graph semantics | Proposed (M1a draft; amended in M2a: set semantics and the unweighted graph) |
| [0011](0011-python-bindings.md) | The Python package (nanobind bindings, the typed layer, dtype dispatch, array lifetimes, locking) | Accepted under delegation (2026-09-30) |
| [0013](0013-parity-and-goldens.md) | Parity and goldens | Proposed (M1a draft) |
| [0014](0014-approval-checkpoints.md) | Approval checkpoints | Accepted |
| [0015](0015-workspace-sharing.md) | Workspace sharing (scratch memory owned by `resources`) | Proposed (M1b) |
| [0016](0016-cuda-execution-resources.md) | CUDA execution resources (streams, memory, errors, warm-up) | Proposed (M1b) |
| [0017](0017-cuda-sssp-engine-and-placement.md) | The CUDA sssp engine, engine selection, placement and the resident device graph | Proposed (M1b); item 2 superseded by 0026 |
| [0018](0018-gpu-clock-state-in-the-cuda-gate.md) | The GPU clock state in the CUDA performance gate | Accepted (2026-09-28: option B, the gate is read at locked clocks; default-clock readings are reported, not gated) |
| [0019](0019-merge-policy.md) | Merge policy (merge commits for milestone and integration work, squash for contributions, no rebase) | Accepted (INT1) |
| [0020](0020-resident-device-graph-and-one-step-0.md) | The resident device graph under set semantics (device set apply, lazily downloaded host copy), and Step 0 once per update | Accepted (2026-09-29, by the author; written in M2b, amended by the M2b review) |
| [0021](0021-power-cap-and-base-lock-in-the-cycle-count-gate.md) | The power cap and the base clock lock in the cycle_count CUDA gate (a case whose GPU cannot hold the boost lock is read at the base lock) | Accepted (2026-09-29, by the author) |
| [0022](0022-conformance-kit-registry-and-scaffold.md) | The conformance kit, the algorithm registry and the scaffold (registry-driven kit, C8's reservations and container work, the scope of new_algorithm.py) | Accepted under delegation (2026-09-29) |
| [0023](0023-api-review-and-freeze-0-1.md) | The 0.1 API review and freeze (the review's findings and fixes, `@guarantee`, the committed public-API listing and its check) | Accepted under delegation (2026-09-29) |
| [0024](0024-reading-a-refactor-against-earlier-dyng.md) | Reading a refactor against an earlier dynG build (rounds over heap layouts, bimodal regions read mode by mode) | Accepted under delegation (2026-09-30) |
| [0025](0025-cli-and-distributions.md) | The command line as a Python console script (generated flags, `dyng prep` with mospPrep's syntax, the originals' output formats), and the build of the distributions (cibuildwheel in CI, the local manylinux_2_28 build, `release.yml` for v0.1.0+) | Accepted under delegation (2026-09-30) |
| [0026](0026-sssp-operators-engine.md) | The sssp operators engine on CUDA (the multi-kernel, non-cooperative engine, byte-identical to the fused one) and the engine choice of 0.2 | Accepted under delegation (2026-09-30) |
| [0027](0027-mosp-api-and-composition.md) | The mosp API (0.2) and mosp as a composition of K sssp problems | Accepted under delegation (2026-09-30) |
| [0028](0028-mosp-python-and-cli.md) | mosp in Python and on the command line (0.2) | Accepted under delegation (2026-10-01) |
| [0029](0029-packing-window-across-backends.md) | The packing window, where the host and CUDA backends follow different originals (non-canonical input trees) | Proposed (2026-10-01, M7 review; a parity rule, for the author) |
| [0030](0030-tutorial-algorithms.md) | The tutorial algorithms: registered with maturity `tutorial`, one source for every backend (executors), undirected graphs in the kit | Accepted under delegation (2026-10-06) |

Planned (PLAN Section 12.3): 0005 layout, 0007 framework,
0008 backends, 0012 versioning and stability.
