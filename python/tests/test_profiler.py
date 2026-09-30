# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.profile: stages and counters of the library's calls."""

from __future__ import annotations

import json

import dyng
import pytest


def test_profile_records_stages(res: dyng.Resources) -> None:
    g = dyng.Graph.from_edges([0, 1, 2], [1, 2, 0], [1, 1, 1], resources=res)
    tree = dyng.sssp.compute(g, 0)
    with dyng.profile(res) as p:
        dyng.sssp.update(g, dyng.EdgeBatch(delete=([1], [2])), tree)
    names = [s.name for s in p.stages]
    assert any(n.startswith("sssp.") for n in names)
    assert all(s.calls >= 1 and s.host_ms >= 0 for s in p.stages)
    assert p.samples and p.samples[0].name in names
    assert isinstance(p.counters, dict)
    assert p.to_csv().startswith("stage") or "," in p.to_csv()
    json.loads(p.to_json())
    assert p.total_host_ms(names[0]) >= 0
    assert "stages" in repr(p)
    # detached on exit: nothing more is recorded
    count = len(p.samples)
    dyng.sssp.compute(g, 0)
    assert len(p.samples) == count
    p.reset()
    assert p.stages == []


def test_nested_profiles_restore_the_outer_one() -> None:
    r = dyng.Resources.sequential()
    g = dyng.Graph.from_edges([0], [1], [1], resources=r)
    with dyng.profile(r) as outer:
        with dyng.profile(r) as inner:
            dyng.sssp.compute(g, 0)
        dyng.cycle_count.compute(g)
    assert any(s.name.startswith("sssp.") for s in inner.stages)
    assert not any(s.name.startswith("cycle_count.") for s in inner.stages)
    assert any(s.name.startswith("cycle_count.") for s in outer.stages)


def test_to_dataframe() -> None:
    pd = pytest.importorskip("pandas")
    r = dyng.Resources.sequential()
    g = dyng.Graph.from_edges([0], [1], [1], resources=r)
    with dyng.profile(r) as p:
        dyng.sssp.compute(g, 0)
    df = p.to_dataframe()
    assert isinstance(df, pd.DataFrame) and "host_ms" in df.columns
