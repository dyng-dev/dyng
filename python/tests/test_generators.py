# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.generators.legacy: bit-exact with the originals' generators (the committed fixtures)."""

from __future__ import annotations

import shlex
from pathlib import Path

import dyng
import pytest
from dyng.generators import legacy


def _changes_cases(data: Path) -> list[tuple[str, dict[str, object]]]:
    out = []
    flags = {
        "--changes": ("num_changes", int),
        "--ins": ("insertion_percentage", float),
        "--seed": ("seed", int),
        "--mode": ("mode", str),
        "--wmin": ("weight_min", int),
        "--wmax": ("weight_max", int),
        "--local": ("local_hops", int),
        "--source": ("source", int),
    }
    for line in (data / "mosp_changes" / "cases.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, *args = shlex.split(line)
        kw: dict[str, object] = {}
        i = 0
        while i < len(args):
            if args[i] == "--safe":
                kw["safe_deletions"] = True
                i += 1
                continue
            key, conv = flags[args[i]]
            kw[key] = conv(args[i + 1])
            i += 2
        out.append((name, kw))
    return out


def test_mosp_changes_equal_mosp_prep(data: Path, tmp_path: Path) -> None:
    g = dyng.io.read_csr_triplet(
        data / "mosp_changes" / "graph" / "graphCsr", properties="mosp_compatible"
    )
    cases = _changes_cases(data)
    assert len(cases) >= 10
    for name, kw in cases:
        batch, report = legacy.mosp_changes(g, **kw)  # type: ignore[arg-type]
        dyng.io.write_legacy_batch(tmp_path / f"{name}.i", tmp_path / f"{name}.d", batch)
        expected = data / "mosp_changes" / name
        assert (tmp_path / f"{name}.i").read_bytes() == (expected / "insert.txt").read_bytes(), name
        assert (tmp_path / f"{name}.d").read_bytes() == (expected / "delete.txt").read_bytes(), name
        assert report.summary == (expected / "report.txt").read_text().strip(), name


def test_cycle_enum_batch(data: Path) -> None:
    f = data / "cycle_enum" / "parser" / "reference_sample.txt"
    g = dyng.io.read_edge_list(f, properties="cycle_enum_compatible")
    b = legacy.cycle_enum_batch(g, num_deletions=3, num_insertions=4, seed=5)
    assert (b.num_deletions, b.num_insertions) == (3, 4)
    again = legacy.cycle_enum_batch(g, num_deletions=3, num_insertions=4, seed=5)
    assert b.insert_src.tolist() == again.insert_src.tolist()
    assert b.delete_dst.tolist() == again.delete_dst.tolist()
    h = dyng.cycle_count.compute(g, max_length=4)
    dyng.cycle_count.update(g, b, h)
    assert h.counts.tolist() == dyng.cycle_count.compute(g, max_length=4).counts.tolist()
    with pytest.raises(dyng.InvalidArgumentError):
        legacy.cycle_enum_batch(g, num_deletions=10**6)
