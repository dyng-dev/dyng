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
| [0006](0006-algorithm-contract.md) | Algorithm contract | Proposed (M1a draft) |
| [0009](0009-default-index-types.md) | Default index types (edge_t = int32 with checked construction) | Accepted (M1b) |
| [0010](0010-batch-and-graph-semantics.md) | Batch and graph semantics | Proposed (M1a draft; amended in M2a: set semantics and the unweighted graph) |
| [0013](0013-parity-and-goldens.md) | Parity and goldens | Proposed (M1a draft) |
| [0014](0014-approval-checkpoints.md) | Approval checkpoints | Accepted |
| [0015](0015-workspace-sharing.md) | Workspace sharing (scratch memory owned by `resources`) | Proposed (M1b) |
| [0016](0016-cuda-execution-resources.md) | CUDA execution resources (streams, memory, errors, warm-up) | Proposed (M1b) |
| [0017](0017-cuda-sssp-engine-and-placement.md) | The CUDA sssp engine, engine selection, placement and the resident device graph | Proposed (M1b) |
| [0018](0018-gpu-clock-state-in-the-cuda-gate.md) | The GPU clock state in the CUDA performance gate | Accepted (2026-09-28: option B, the gate is read at locked clocks; default-clock readings are reported, not gated) |
| [0019](0019-merge-policy.md) | Merge policy (merge commits for milestone and integration work, squash for contributions, no rebase) | Accepted (INT1) |

Planned (PLAN Section 12.3): 0005 layout, 0007 framework,
0008 backends, 0011 Python, 0012 versioning and stability.
