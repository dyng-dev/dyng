#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The budget of the committed test data (ADR 0013 and its amendment of M9a, the author's decision
D5 of 2026-10-15): the tracked files under cpp/tests/data total at most 1.25 MB (1,250,000 bytes).

    python3 ci/test_data_budget.py            # exit 1 when the tracked data exceed the budget

Larger goldens live outside the repository ($DYNG_SCRATCH/goldens, parity/goldens.toml); the
fixtures committed here are the small cases hosted CI replays.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = "cpp/tests/data"
BUDGET = 1_250_000  # bytes: 1.25 MB (decision D5; 1 MB before M9a)


def tracked_sizes(repo: Path = REPO, data: str = DATA) -> dict[str, int]:
    """The size of every tracked file under `data` (every file when git is not available)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z", "--", data],
            capture_output=True,
            check=True,
        ).stdout
        names = [n for n in out.decode().split("\0") if n]
    except (OSError, subprocess.CalledProcessError):
        names = [str(p.relative_to(repo)) for p in (repo / data).rglob("*") if p.is_file()]
    return {n: (repo / n).stat().st_size for n in names if (repo / n).is_file()}


def check(sizes: dict[str, int], budget: int = BUDGET) -> tuple[bool, str]:
    """Whether the sizes fit the budget, and a one-line report."""
    total = sum(sizes.values())
    verdict = "OK" if total <= budget else "OVER BUDGET"
    return total <= budget, (
        f"test data: {total:,} bytes in {len(sizes)} tracked files under {DATA} "
        f"(budget {budget:,} bytes): {verdict}"
    )


def main() -> int:
    ok, line = check(tracked_sizes())
    print(line)
    if not ok:
        largest = sorted(tracked_sizes().items(), key=lambda kv: -kv[1])[:10]
        for name, size in largest:
            print(f"  {size:>10,}  {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
