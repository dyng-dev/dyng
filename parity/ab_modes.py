#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Mode-aware reading of a dynG-against-dynG A/B (the refactor bar of M3; not a PLAN 8.6 gate).

    parity/ab_modes.py [--bar 1.02] <perf_ab.py / cycle_count_perf.py JSON>...

Some regions are bimodal per process: every sample lands in one of two modes several percent
apart (the short OpenMP sssp updates over heap layouts, the host apply of large graphs, whose
slow mode is the process that does not get huge pages). The median of such a mixture is the
median of the slow-mode fraction: with a fraction near one half it jumps between the modes when
one side has a few more slow rounds, whatever the code does. This tool reads every region of
every record (the regions the gates read, whatever the side A is), with the 95 % bootstrap interval
of its ratio of medians, as follows:

- **unimodal** (the two classes of Otsu's threshold over the samples of both sides are less than
  --gap of the median apart, or one holds fewer than 3 samples or 5 %): the ratio of medians, as
  perf_ab.py reports it;
- **bimodal**: the ratio of medians inside each mode that both sides reached, the slow-mode
  fraction of each side, and the two-sided Fisher exact test of equal fractions.

A region is within the bar when its ratio of medians is, or, bimodal, when every within-mode ratio
is and the fractions do not differ (p >= --alpha). The output is a table per record; the exit
status is 1 if a region is outside the bar.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

# Regions that no gate reads (perf_ab.py and cycle_count_perf.py).
UNGATED = (
    "sosp_total",
    "static_count",
    "static_read",
    "static_memcpy",
    "static_total",
    "update_device",
)


def fisher_p(a_slow: int, a_n: int, b_slow: int, b_n: int) -> float:
    """Two-sided Fisher exact test of equal proportions a_slow / a_n and b_slow / b_n."""
    k = a_slow + b_slow
    n = a_n + b_n

    def prob(x: int) -> float:
        return math.comb(a_n, x) * math.comb(b_n, k - x) / math.comb(n, k)

    observed = prob(a_slow)
    lo, hi = max(0, k - b_n), min(k, a_n)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def split(samples: list[float], gap: float) -> float | None:
    """The threshold between two modes, or None for one mode.

    The threshold is Otsu's (the cut of the sorted samples with the largest between-class
    variance; a few samples between the modes do not hide them, as they hide the largest gap).
    The samples are bimodal when the medians of the two classes differ by more than gap * median
    and the smaller class holds at least 3 samples and 5 % of them."""
    s = sorted(samples)
    n = len(s)
    total = sum(s)
    best, cut, left = -1.0, None, 0.0
    for j in range(1, n):
        left += s[j - 1]
        w0, w1 = j / n, (n - j) / n
        m0, m1 = left / j, (total - left) / (n - j)
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best:
            best, cut = between, j
    if cut is None:
        return None
    low, high = s[:cut], s[cut:]
    if min(len(low), len(high)) < max(3, 0.05 * n):
        return None
    if statistics.median(high) - statistics.median(low) <= gap * statistics.median(s):
        return None
    return (s[cut - 1] + s[cut]) / 2


def bootstrap_ci(a: list[float], b: list[float], rounds: int = 2000) -> tuple[float, float]:
    """95 % percentile-bootstrap interval of median(b) / median(a) (seeded: reproducible)."""
    rng = random.Random(20260930)
    ratios = sorted(
        statistics.median(rng.choices(b, k=len(b))) / statistics.median(rng.choices(a, k=len(a)))
        for _ in range(rounds)
    )
    return ratios[int(0.025 * rounds)], ratios[int(0.975 * rounds) - 1]


def read_region(a: list[float], b: list[float], bar: float, gap: float, alpha: float) -> dict:
    median_ratio = statistics.median(b) / statistics.median(a)
    ci = bootstrap_ci(a, b)
    t = split(a + b, gap)
    if t is None:
        return {"modes": 1, "ratio": median_ratio, "ci": ci, "within": median_ratio <= bar}
    modes = {}
    for name, keep in (("fast", lambda x: x < t), ("slow", lambda x: x >= t)):
        ma, mb = [x for x in a if keep(x)], [x for x in b if keep(x)]
        if ma and mb:
            modes[name] = statistics.median(mb) / statistics.median(ma)
    a_slow, b_slow = sum(x >= t for x in a), sum(x >= t for x in b)
    p = fisher_p(a_slow, len(a), b_slow, len(b))
    within = all(r <= bar for r in modes.values()) and p >= alpha
    return {
        "modes": 2,
        "ratio": median_ratio,
        "ci": ci,
        "mode_ratios": modes,
        "slow": (a_slow, len(a), b_slow, len(b)),
        "p": p,
        "within": median_ratio <= bar or within,
    }


def regions_of(doc: dict):
    for case, result in doc["results"].items():
        for region in result.get("regions", []):
            if region["region"].startswith(UNGATED):
                continue
            if "original_samples" not in region or "port_samples" not in region:
                continue
            yield case, region


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--bar", type=float, default=1.02, help="the ratio a region must not pass")
    parser.add_argument("--gap", type=float, default=0.04, help="mode split: gap / median")
    parser.add_argument("--alpha", type=float, default=0.05, help="Fisher test level")
    parser.add_argument("--all", action="store_true", help="print every region, not only bimodal")
    args = parser.parse_args()
    outside = 0
    for path in args.records:
        doc = json.loads(path.read_text())
        rows = []
        for case, region in regions_of(doc):
            r = read_region(
                region["original_samples"], region["port_samples"], args.bar, args.gap, args.alpha
            )
            outside += not r["within"]
            if not args.all and r["modes"] == 1 and r["within"]:
                continue
            lo, hi = r["ci"]
            text = f"  {case:32s} {region['region']:28s} median {r['ratio']:.3f}"
            text += f" [{lo:.3f}, {hi:.3f}]"
            if r["modes"] == 2:
                a_slow, a_n, b_slow, b_n = r["slow"]
                modes = ", ".join(f"{m} {v:.3f}" for m, v in r["mode_ratios"].items())
                text += (
                    f"; bimodal: {modes}; slow A {a_slow}/{a_n} B {b_slow}/{b_n} (p {r['p']:.2f})"
                )
            text += "" if r["within"] else "  OUTSIDE"
            rows.append(text)
        count = sum(1 for _ in regions_of(doc))
        print(f"{path.name}: {count} regions")
        print("\n".join(rows) if rows else "  every region unimodal and within the bar")
    return 1 if outside else 0


if __name__ == "__main__":
    raise SystemExit(main())
