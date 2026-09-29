# ADR 0021: The power cap and the base clock lock in the cycle_count CUDA gate

- **Status:** Accepted (2026-09-29, by the author; recorded in GOVERNANCE.md). A gate case whose GPU
  cannot hold the boost lock under the power cap is read at the base lock (decision 1); the COLLAB
  update's base-lock reading (0.989x original scope, 0.945x resident) is its gate reading.
- **Date:** 2026-09-29
- **Deciders:** S M Shovan (lead maintainer)

This ADR takes over an "Update (M2b)" section that the M2b work had appended to ADR 0018 after the
author accepted it. An accepted ADR is not rewritten (docs/adr/README.md), so the M2b review
removed that section from ADR 0018 (whose text is again the accepted one) and moved the material
here, to be decided on its own.

## Context

ADR 0018 (accepted 2026-09-28, option B) reads the CUDA performance gate with the GPU clocks locked
for the whole A/B: `perf_ab.py run --backend cuda --lock-clocks boost`, applied to both programs,
with every busy GPU sample checked against the lock. It was decided on the sssp gate, whose
kernels are short.

The `cycle_count` CUDA gate (`parity/cycle_count_perf.py run --backend cuda`, M2b acceptance
criterion 4) is read the same way. Two things differ from sssp:

- **Long counting kernels meet the power cap.** A static count of 40-60 ms (GitHub and Twitch
  k = 4, COLLAB k = 3) or the 6.5 s prior of the COLLAB update draws enough power that the driver
  lowers the SM clock below the locked 1695 MHz ("SW power cap"; the GPU's power limit is its
  default, 230 W). For the static kernels this happens in some rounds and on both sides alike (the
  kernels are the same code with the same registers); the monitor repeats every round in which a
  busy sample is off the lock, and enough rounds hold the lock. The COLLAB update cannot be read at
  `boost` at all: its prior runs under the cap for seconds, so every round is rejected (22 in a row
  in the first M2b campaign before the harness gave up on the case).
- **The monitor samples every 50 ms.** A kernel shorter than a sample can fall between busy
  samples, so the record also gives, per side, the lowest SM clock of any sample and the number of
  busy samples (`gpu_summary`; `parity/results/M2b.md` section 4). Samples below the lock that were
  not busy appear on both sides in the same cases.

## Decision

1. A gate case whose GPU cannot hold the boost lock for the whole process on both sides (every
   round rejected because a busy sample is off the lock under the power cap) is read with
   `--lock-clocks base` (SM 1170 MHz, memory 7601 MHz), where the clocks hold. The lock is still
   applied equally to both programs and checked in every busy sample, which is the rule of option
   B. The record names the lock (`M2b-cuda-perf-cycle_count-collab-update-base.json` and its
   successors) and the default-clock reading of the case is recorded next to it, ungated, like
   every other case.
2. The record keeps, per side, the lowest SM clock of any sample and the number of busy samples.

## Alternatives

- Change the GPU's power limit for the gate so that `boost` holds. Not possible on this machine
  (checked in the second acceptance verification, 2026-09-29): the RTX A5000's current, default
  and maximum power limits are all 230 W (`nvidia-smi -q -d POWER`), so the limit cannot be
  raised even with root, and a lower limit would only lower the clocks further. (The first version
  of this ADR listed "lower the power limit"; that was wrong.)
- Check the lock only in the busy samples of the timed region (the update itself, about 200 ms,
  after the untimed prior). The unpatched original reports no timestamps of its region; the window
  would have to be inferred from the arrival of its unbuffered `update_seconds=` line, and it
  would still change ADR 0018's accepted rule (every busy sample of the process). Not evaluated.
- Gate the COLLAB update only at default clocks (ADR 0018 option A for this case).
- Leave the COLLAB update ungated and report it (a gap in acceptance criterion 4).

## Consequences

- If accepted: the COLLAB update's base-lock reading (`parity/results/M2b.md` sections 4.5 and 8)
  is its gate reading, and criterion 4 of M2b is met for it. If not: the case is reported, and the
  author's alternative is applied in a later milestone.
- ADR 0018 stays as accepted; this ADR only adds the rule for cases that cannot hold the boost lock.
