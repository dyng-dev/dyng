# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""``dyng hyper_sssp compute|update`` (PLAN Sections 5.6 and 5.7; ADR 0097).

- the option flags are the fields of ``dyng.hyper_sssp.Options`` one to one (kebab case);
- the outputs are sssp's distance and parent files by hyperedge id (``dyng.io.read_distances`` /
  ``read_parents`` read them back), equal to the library's result;
- ``.dgt`` and ``.dgb`` batches with incidence edits; ``--source``; the documented exit statuses
  (0, 1 when the work fails, 2 for a usage error).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import dyng
import numpy as np
import pytest
from dyng.cli import build_parser, main

SEQ = dyng.Resources("sequential")
HG = "1 2 ; 2\n2 3 ; 3\n3 4 ; 1\n1 4 ; 9\n5 6 ; 4\n"  # 1-based ids, as both originals' files
BATCH = "%dgt 1 hypergraph\n%batch 0\n+i 0 3\n%batch 1\n-h 3\n-i 0 1\n+h 3 4 ; 1\n"


def run(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, str, str]:
    """Run ``dyng argv...`` in this process; (status, stdout, stderr)."""
    try:
        status = main([str(a) for a in argv])
    except SystemExit as e:
        status = int(e.code or 0)
    out, err = capsys.readouterr()
    return status, out, err


def _subparser(*path: str) -> argparse.ArgumentParser:
    p = build_parser()
    for name in path:
        action = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
        p = action.choices[name]  # type: ignore[attr-defined]
    return p


def write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def reference(
    source: int, batches: list[dyng.HyperedgeBatch]
) -> tuple[dyng.Hypergraph, dyng.hyper_sssp.Result]:
    """The library's hypergraph and result of the same input."""
    lists = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5]]
    hg = dyng.Hypergraph.from_lists(
        lists, [2, 3, 1, 9, 4], storage="slack_csr", line_graph="slack_csr", resources=SEQ
    )
    sp = dyng.hyper_sssp.compute(hg, source)
    for b in batches:
        dyng.hyper_sssp.update(hg, b, sp)
    return hg, sp


def read_result(out: Path, n: int) -> tuple[list[int], list[int]]:
    d = dyng.io.read_distances(out / "hyperedge_distances.txt", n)
    p = dyng.io.read_parents(out / "hyperedge_parents.txt", n)
    return d.tolist(), p.tolist()


def test_the_flags_are_the_option_fields() -> None:
    for verb in ("compute", "update"):
        p = _subparser("hyper_sssp", verb)
        flags = {s for a in p._actions for s in a.option_strings}
        for f in dataclasses.fields(dyng.hyper_sssp.Options):
            assert "--" + f.name.replace("_", "-") in flags, f.name
        for flag in ("--input", "--source", "--out", "--read-options", "--random-weights"):
            assert flag in flags
        assert ("--batches" in flags) == (verb == "update")
    on_budget = next(
        a for a in _subparser("hyper_sssp", "update")._actions if "--on-budget" in a.option_strings
    )
    assert tuple(on_budget.choices) == ("recompute", "error")  # type: ignore[arg-type]


def test_compute(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    hg = write(tmp_path, "h.hg", HG)
    out = tmp_path / "out"
    status, text, _ = run(
        capsys, "hyper_sssp", "compute", "--input", hg, "--out", out, "--vertex-distances"
    )
    assert status == 0, text
    assert "compute" in text and "reachable 4 of 5 hyperedges" in text
    _hg, sp = reference(0, [])
    assert read_result(out, 5) == (sp.hyperedge_distances.tolist(), sp.hyperedge_parents.tolist())
    assert (out / "hyperedge_distances.txt").read_text().splitlines() == [
        "0 2",
        "1 5",
        "2 6",
        "3 9",
        "4 INF",
    ]
    assert (out / "hyperedge_parents.txt").read_text().splitlines()[:2] == ["0 -1", "1 0"]
    v = dyng.io.read_distances(out / "vertex_distances.txt", 6).tolist()
    assert v == sp.vertex_distances().tolist()


@pytest.mark.parametrize("binary", [False, True])
def test_update_with_incidence_edits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], binary: bool
) -> None:
    hg = write(tmp_path, "h.hg", HG)
    dgt = write(tmp_path, "b.dgt", BATCH)
    batches = dyng.io.read_hyperedge_batches(dgt, weighted=True)
    path = dgt
    if binary:
        path = tmp_path / "b.dgb"
        dyng.io.write_binary_batches(path, batches)
    out = tmp_path / "out"
    status, text, err = run(
        capsys,
        "hyper_sssp",
        "update",
        "--input",
        hg,
        "--batches",
        path,
        "--source",
        5,
        "--work-budget",
        "inf",
        "--on-budget",
        "error",
        "--out",
        out,
        "--compare-recompute",
    )
    assert status == 0, err
    assert "batch 1:" in text and "incidences -0 +1" in text and "batch 2:" in text
    assert "match=yes" in err and "update_seconds=" in err
    _hg, sp = reference(5, batches)
    assert read_result(out, len(sp.hyperedge_distances)) == (
        sp.hyperedge_distances.tolist(),
        sp.hyperedge_parents.tolist(),
    )


def test_exit_statuses(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    hg = write(tmp_path, "h.hg", HG)
    out = tmp_path / "out"
    # usage errors: 2
    assert run(capsys, "hyper_sssp", "compute", "--input", hg)[0] == 2  # no --out
    assert run(capsys, "hyper_sssp", "update", "--input", hg, "--out", out)[0] == 2  # no batches
    status, _, err = run(
        capsys, "hyper_sssp", "compute", "--input", hg, "--out", out, "--on-budget", "x"
    )
    assert status == 2 and "--on-budget" in err
    status, _, err = run(
        capsys, "hyper_sssp", "compute", "--input", hg, "--out", out, "--random-weights", "1,2"
    )
    assert status == 2 and "MIN,MAX,SEED" in err
    # the work fails: 1, with the library's message
    bad = write(tmp_path, "bad.hg", "1 2 ; x\n")
    status, _, err = run(capsys, "hyper_sssp", "compute", "--input", bad, "--out", out)
    assert status == 1 and err.startswith("dyng: error:") and "bad.hg:1" in err
    unweighted = write(tmp_path, "u.hg", "1 2\n2 3\n")
    status, _, err = run(capsys, "hyper_sssp", "compute", "--input", unweighted, "--out", out)
    assert status == 1 and "--random-weights" in err
    status, _, _ = run(
        capsys,
        "hyper_sssp",
        "compute",
        "--input",
        unweighted,
        "--random-weights",
        "1,10,7",
        "--out",
        out,
    )
    assert status == 0
    status, _, err = run(
        capsys, "hyper_sssp", "compute", "--input", hg, "--out", out, "--source", 6
    )
    assert status == 1 and "--source 6 is out of range" in err
    status, _, err = run(
        capsys, "hyper_sssp", "compute", "--input", hg, "--out", out, "--storage", "escher"
    )
    assert status == 1 and "slack_csr" in err
    if dyng.config()["backends"]["openmp"]:
        status, _, err = run(
            capsys, "hyper_sssp", "compute", "--input", hg, "--out", out, "--backend", "openmp"
        )
        assert status == 1 and 'dyng.Resources("sequential")' in err
    malformed = write(tmp_path, "m.dgt", "%dgt 1 hypergraph\n+i 0\n")
    status, _, err = run(
        capsys, "hyper_sssp", "update", "--input", hg, "--batches", malformed, "--out", out
    )
    assert status == 1 and "m.dgt" in err


def test_hsosp_loader(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # MOSP_ESCHER's loader (relabelled by first appearance, seeded weights U[1, 100])
    hg = write(tmp_path, "h.hg", "10 20\n20 30\n30 40 50\n")
    out = tmp_path / "out"
    status, _, err = run(
        capsys, "hyper_sssp", "compute", "--input", hg, "--read-options", "hsosp", "--out", out
    )
    assert status == 0, err
    h = dyng.io.read_hypergraph(
        hg, options="hsosp", storage="slack_csr", line_graph="slack_csr", resources=SEQ
    )
    sp = dyng.hyper_sssp.compute(h, 0)
    d, p = read_result(out, 3)
    assert d == sp.hyperedge_distances.tolist() and p == sp.hyperedge_parents.tolist()
    assert np.all(np.asarray(h.weights) >= 1)
