# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared by test_triad_count.py and test_triad_count_cli.py: the patterns, ESCHER-GPU's
executables, and the committed make-test fixture of decision D5
(cpp/tests/data/goldens/triad_counts/escher_gpu_make_test.txt, written by parity/triad_goldens.py
fixture) with the regeneration of its groups."""

from __future__ import annotations

import typing
from pathlib import Path

import dyng

KINDS = ["h_motif", "stathyper_type1", "stathyper_type2", "stathyper_type3"]
#: ESCHER-GPU's executables and the pattern each counts.
EXES = dict(zip(["main", "type1", "type2", "type3"], KINDS, strict=True))

FIXTURE = Path("goldens") / "triad_counts" / "escher_gpu_make_test.txt"


def read_fixture(data: Path) -> tuple[list[dict[str, typing.Any]], dict[tuple[str, str], dict]]:
    """The groups and A's counts of cpp/tests/data/goldens/triad_counts (parity/triad_goldens.py
    fixture): per (group, exe) the full count and per batch (deletions, insertions, delta)."""
    groups: list[dict[str, typing.Any]] = []
    cases: dict[tuple[str, str], dict] = {}
    for line in (data / FIXTURE).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if f[0] == "group":
            keys = ["name", "seed", "n", "max_card", "r1", "r2", "capacity", "batches"]
            g: dict[str, typing.Any] = {
                k: (v if k == "name" else int(v)) for k, v in zip(keys, f[1:9], strict=True)
            }
            g.update(deletions=int(f[9]), insertions=int(f[10]), mode=f[11])
            groups.append(g)
        elif f[0] == "full":
            cases.setdefault((f[1], f[2]), {"batches": []})["full"] = [int(x) for x in f[4:]]
        elif f[0] == "batch":
            c = cases.setdefault((f[1], f[2]), {"batches": []})
            c["batches"].append((int(f[4]), int(f[5]), [int(x) for x in f[7:]]))
    return groups, cases


def regenerate(
    g: dict[str, typing.Any], res: dyng.Resources
) -> tuple[dyng.Hypergraph, dyng.generators.legacy.EscherBatchGenerator]:
    """A make-test group's hypergraph and batch generator, as A drew them (one engine)."""
    engine = dyng.generators.legacy.EscherEngine(g["seed"])
    h = dyng.generators.legacy.escher_hypergraph(
        engine,
        num_hyperedges=g["n"],
        max_cardinality=g["max_card"],
        min_vertex=g["r1"] - 1,
        max_vertex=g["r2"] - 1,
        line_graph="sorted_merge",
        capacity=dyng.CapacityPolicy.escher_gpu(g["capacity"]),
        resources=res,
    )
    gen = dyng.generators.legacy.EscherBatchGenerator(
        engine,
        num_deletions=g["deletions"],
        num_insertions=g["insertions"],
        insert_mode=g["mode"],
        max_cardinality=g["max_card"],
        min_vertex=g["r1"] - 1,
        max_vertex=g["r2"] - 1,
    )
    return h, gen
