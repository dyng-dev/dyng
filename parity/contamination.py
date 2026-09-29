#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Contamination monitor of the performance harness (PLAN Section 8.6, "contamination monitor").

The load average cannot tell foreign load from the measured process's own threads (a 56-thread
run alone raises it to about 56). The monitor measures, over the lifetime of one measured
process, the CPU time the whole machine spent busy (/proc/stat: user, nice, system, irq,
softirq, steal) minus the CPU time of the harness itself and of the processes it waited for
(getrusage RUSAGE_SELF and RUSAGE_CHILDREN). The rest is foreign CPU time: other users' processes,
other agents' builds, kernel threads. Divided by the wall time it is the number of cores the
foreign work kept busy on average during that run.

    with Monitor() as m:
        subprocess.run(...)
    m.result  -> {"wall_s": ..., "busy_cores": ..., "own_cores": ..., "foreign_cores": ...}

The /proc/stat counters tick in jiffies (1/100 s per CPU), so a short run carries a rounding error
of a few jiffies over all CPUs; FLAG_CORES leaves room for it. A run is flagged when the foreign
work averaged more than FLAG_CORES cores; flagged runs are recorded, not dropped (PLAN 8.6: a
contaminated measurement is flagged, not failed).
"""

from __future__ import annotations

import os
import resource
import statistics
import time

FLAG_CORES = 2.0
CLOCK_TICKS = os.sysconf("SC_CLK_TCK")


def busy_seconds() -> float:
    """CPU seconds the machine has spent busy since boot (all CPUs, /proc/stat)."""
    with open("/proc/stat") as f:
        fields = f.readline().split()
    if fields[0] != "cpu":
        raise RuntimeError("/proc/stat: unexpected first line")
    user, nice, system, _idle, _iowait, irq, softirq, steal = (int(x) for x in fields[1:9])
    return (user + nice + system + irq + softirq + steal) / CLOCK_TICKS


def own_seconds() -> float:
    """CPU seconds of this process and of the children it has waited for."""
    total = 0.0
    for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
        r = resource.getrusage(who)
        total += r.ru_utime + r.ru_stime
    return total


class Monitor:
    """Measures the foreign CPU use during a block (see the module docstring)."""

    def __init__(self) -> None:
        self.result: dict = {}

    def __enter__(self) -> Monitor:
        self._busy = busy_seconds()
        self._own = own_seconds()
        self._wall = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        wall = time.perf_counter() - self._wall
        busy = busy_seconds() - self._busy
        own = own_seconds() - self._own
        foreign = max(0.0, busy - own)
        self.result = {
            "wall_s": wall,
            "busy_cores": busy / wall if wall > 0 else 0.0,
            "own_cores": own / wall if wall > 0 else 0.0,
            "foreign_cores": foreign / wall if wall > 0 else 0.0,
        }


def summarize(runs: list[dict]) -> dict:
    """The contamination record of a series of runs: median and max of the foreign cores, and how
    many runs exceeded FLAG_CORES."""
    foreign = [r["foreign_cores"] for r in runs]
    return {
        "method": "(/proc/stat busy CPU time - RUSAGE_SELF - RUSAGE_CHILDREN) / wall, per run",
        "flag_cores": FLAG_CORES,
        "foreign_cores_median": statistics.median(foreign) if foreign else 0.0,
        "foreign_cores_max": max(foreign) if foreign else 0.0,
        "flagged_runs": sum(1 for f in foreign if f > FLAG_CORES),
        "runs": len(foreign),
    }
