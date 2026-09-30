# ADR 0024: Reading a refactor against an earlier dynG build (heap layouts and process modes)

- **Status:** Accepted under delegation (2026-09-30; GOVERNANCE.md, "Delegation of technical
  ADRs"). It decides how a dynG-against-dynG A/B is read, which no approved rule covers (M3's
  2 % bar against `019ef13` is a milestone acceptance criterion, not a PLAN 8.6 gate). It does not
  change a rule the author approved: the PLAN 8.6 gates against the unpatched originals, their
  measurement protocol (ADRs 0018, 0021) and the parity rules are untouched, and the readings of
  the fixed protocol are recorded next to the ones this ADR adds.
- **Date:** 2026-09-30
- **Deciders:** the AI assistant, on the author's behalf (M3 acceptance fixes)

## Context

M3's acceptance criterion 3 asks for no regression above 2 % against the pre-M3 dynG (`019ef13`)
on any gated region. `parity/perf_ab.py run --baseline-exe` (and `cycle_count_perf.py
--baseline-kind port`) run the two dynG builds A/B/A/B with the PLAN 8.6 protocol and compare
medians. The M3 review and the acceptance review read several regions above 1.02 in that way and
located them in code M3 did not change (`parity/results/M3.md` sections 5.4 and 6). The
acceptance fixes found two properties of this machine that decide those readings
(`parity/results/M3.md` section 6.3):

1. **The heap layout.** The same two binaries read 0.99-1.00 or 1.07-1.11 on the OpenMP sssp
   local10k update of roadNet-CA depending only on the length of the `--timing` file name (the
   program keeps it on its heap, so every later allocation moves; the environment's length, which
   moves only the stack, has no effect). Over 17 file-name lengths the per-layout ratio ranges
   0.90-1.13 for the same pair, and it goes both ways. The harness's temporary directory has a
   fixed length, so every round of a session reads one layout, and the reading is that layout's.
2. **Process modes.** The short OpenMP updates and the host apply of the large graphs are bimodal
   per process: every sample lands in one of two modes 5-25 % apart, for both builds. When the
   slow-mode fraction is near one half (85 layouts: 44-66 % on every graph), the median of the
   mixture sits in the gap between the modes and moves by the gap when a side has a few more
   slow rounds: the same pair read the CUDA roadNet-PA local10k apply at 1.18x with both modes
   within 1.3 %, and the OpenMP roadNet-PA safe50k apply at 0.87x.

A 2 % bar cannot be read on such regions with one layout and a median; reading them that way
would fail or pass a refactor at random.

## Decision

A dynG-against-dynG A/B (the refactor bar of a milestone) is read as follows.

1. **Layouts.** The rounds cycle through heap layouts: `perf_ab.py run --layouts N` and
   `cycle_count_perf.py run --layouts N` (OpenMP, `--baseline-kind port`) give round r a
   `--timing` file name `8 * (r mod N)` characters longer, the same on both sides. The suite runs
   with 17 layouts (51 rounds; 34 for road_usa and the cycle_count list); the short OpenMP
   updates (sssp local10k, the cycle_count updates) are read over 85 layouts (170 rounds), because
   their per-layout ratios spread over about +-10 % and 17 layouts leave the mean uncertain by
   about 2 %. `--layouts` is refused without a dynG baseline: the originals take no `--timing`
   file.
2. **Modes.** `parity/ab_modes.py` reads every gated region of a record. A unimodal region is its
   ratio of medians. A region is bimodal when Otsu's threshold over both sides' samples separates
   two classes whose medians differ by more than 4 % of the median, each with at least 3 samples
   and 5 % of them; it is read as the ratio of medians inside each mode both sides reached, the
   slow-mode fraction of each side, and the two-sided Fisher exact test of equal fractions. The
   95 % bootstrap interval of the ratio of medians is printed for every region.
3. **The bar.** A region is within the bar when its ratio of medians is, or, bimodal, when every
   within-mode ratio is and the fractions do not differ (p >= 0.05). A region outside it, or with
   a bootstrap interval wider than about 3 %, is read again with more rounds (170 over 85
   layouts on OpenMP; 102 on CUDA), and the longer reading decides. A region still outside is a
   regression to be located (stage by stage: `parity/experiments/sssp_stage_ab.py`,
   `cycle_count_stage_ab.py`) and fixed.
4. **What is recorded.** The fixed-layout readings of the protocol stay in the record next to the
   layout readings, with the regions above the bar named; nothing is dropped.

## Consequences

- The M3 refactor bar is read with these rules in `parity/results/M3.md` section 6.5.
- The PLAN 8.6 gates against the originals are read as before, with one layout. The same effects
  reach them (a port's reading can move by several percent with the layout), which the margins of
  M3 absorb (`parity/results/M3.md` section 6.6: 0.64-1.02x against 1.05-1.10x gates).
  Whether the gates should also move to layouts, or the parity preset control code placement
  (retrospective open item 1), is the author's decision before the 0.2 gates.
- A later milestone's refactor bar uses the same rules; its record states the number of layouts
  and rounds.
