# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""``dyng triad_count compute|update`` (PLAN Sections 5.6 and 5.7; ADR 0059).

- the flags are the fields of ``dyng.triad_count.Options`` and the keywords of
  ``dyng.io.read_hypergraph``;
- ``--format json`` is the triad-bins JSON (read back by ``dyng.io.read_triad_bins``);
- ``--format escher`` prints ESCHER-GPU@abcf9b2's lines byte for byte: against A's counts of the
  committed make-test fixture (hosted CI; the groups whose input the legacy generators
  regenerate), and, where the golden set ``triad_count_escher_gpu`` is present
  ($DYNG_SCRATCH/goldens/triad_count/escher_gpu, parity/goldens.toml), against A's UNPATCHED
  standard output of every make-test case (15 groups x 4 executables): the diff must be empty
  (A's generator lines aside, which belong to its generated inputs).
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import shutil
import subprocess
from pathlib import Path

import dyng
import numpy as np
import pytest
from dyng.cli import build_parser, main
from dyng.cli._triad_count import _READ_SKIP
from triad_fixture import EXES, read_fixture, regenerate

BACKENDS = ["sequential"] + (["openmp"] if dyng.config()["backends"]["openmp"] else [])
GOLDENS = (
    Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
    / "goldens"
    / "triad_count"
    / "escher_gpu"
)


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


def _flags(parser: object) -> set[str]:
    return {s for a in parser._actions for s in a.option_strings}  # type: ignore[attr-defined]


@pytest.mark.parametrize("verb", ["compute", "update"])
def test_every_option_field_and_read_keyword_has_a_flag(verb: str) -> None:
    flags = _flags(_subparser("triad_count", verb))
    for f in dataclasses.fields(dyng.triad_count.Options):
        assert "--" + f.name.replace("_", "-") in flags, f.name
    for name, p in inspect.signature(dyng.io.read_hypergraph).parameters.items():
        if p.kind is inspect.Parameter.KEYWORD_ONLY and name not in _READ_SKIP:
            assert "--" + name.replace("_", "-") in flags, name
    assert {"--input", "--read-options", "--min-initial-payload", "--format", "--output"} <= flags
    kind = next(a for a in _subparser("triad_count", verb)._actions if a.dest == "opt_kind")
    assert list(kind.choices) == [p.value for p in dyng.triad_count.Pattern]


SMALL_HG = "1 2 3\n2 3 4\n3 4 5\n1 3 5\n1 2 3 4 5 6\n6 7\n2 3\n"
SMALL_DGT = "%dgt 1 hypergraph\n%batch 0\n-h 1\n-h 4\n+h 1 2 4 5\n+h 3 6\n%batch 1\n+h 2 3 4\n"


def _small(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "s.hg").write_text(SMALL_HG)
    (tmp_path / "s.dgt").write_text(SMALL_DGT)
    return tmp_path / "s.hg", tmp_path / "s.dgt"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", list(EXES.values()))
def test_compute_and_update_json(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, backend: str, kind: str
) -> None:
    hg, dgt = _small(tmp_path)
    h = dyng.io.read_hypergraph(hg, resources=dyng.Resources(backend))
    want = dyng.triad_count.compute(h, kind=kind)
    status, out, _ = run(
        capsys, "triad_count", "compute", "--input", hg, "--kind", kind, "--backend", backend
    )
    assert status == 0 and out == dyng.io.triad_bins_json(want)
    doc = json.loads(out)
    assert doc["format"] == "dyng.triad_bins" and doc["version"] == 1 and doc["kind"] == kind
    deltas = []
    for b in dyng.io.read_hyperedge_batches(dgt):
        deltas.append(dyng.triad_count.update(h, b, want).delta)
    status, out, err = run(
        capsys,
        "triad_count",
        "update",
        "--input",
        hg,
        "--batches",
        dgt,
        "--kind",
        kind,
        "--backend",
        backend,
        "--output",
        tmp_path / "o" / "bins.json",
        "--compare-recompute",
    )
    assert status == 0 and out == "" and "match=yes" in err and "update_seconds=" in err
    back = dyng.io.read_triad_bins(tmp_path / "o" / "bins.json")
    assert back == dyng.io.TriadBins.of(want, deltas)
    assert back.graph_version == 2


def test_escher_lines_of_a_small_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    hg, dgt = _small(tmp_path)
    status, out, _ = run(
        capsys, "triad_count", "update", "--input", hg, "--batches", dgt, "--format", "escher"
    )
    lines = out.splitlines()
    assert status == 0 and lines[0] == "Hypergraph: 7 hyperedges, max vertex id 7, 22 pins"
    assert lines[1].startswith("Motif counts (30 bins): ") and len(lines[1].split()) == 34
    assert lines[2] == "Batch 1 (2 deletions, 2 insertions)"
    assert lines[3].startswith("Motif delta counts (30 bins): ")
    assert lines[4] == "Batch 2 (0 deletions, 1 insertions)" and len(lines) == 6
    status, out, _ = run(
        capsys,
        "triad_count",
        "compute",
        "--input",
        hg,
        "--format",
        "escher",
        "--kind",
        "stathyper_type1",
    )
    assert out.splitlines()[1] == "[GPU] Type1 baseline motif count (sum_e C(|e|, 3)): 24"


def test_errors(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    hg, dgt = _small(tmp_path)
    status, _, err = run(capsys, "triad_count", "compute", "--input", hg, "--line-graph", "none")
    assert status == 1 and "dyng: error:" in err and "line_graph" in err
    status, _, err = run(
        capsys, "triad_count", "compute", "--input", hg, "--no-store-v2h", "--kind", "h_motif"
    )
    assert status == 1 and "store_v2h" in err
    status, out, _ = run(
        capsys, "triad_count", "compute", "--input", hg, "--line-graph", "none", "--kind",
        "stathyper_type1", "--format", "escher",
    )  # fmt: skip
    assert status == 0 and out.endswith("): 24\n")
    status, _, err = run(
        capsys, "triad_count", "compute", "--input", hg, "--min-initial-payload", 9
    )
    assert status == 2 and "--capacity escher_gpu" in err
    status, _, err = run(capsys, "triad_count", "update", "--input", hg)
    assert status == 2 and "--batches" in err
    status, _, err = run(capsys, "triad_count", "compute", "--input", tmp_path / "missing.hg")
    assert status == 1 and "dyng: error:" in err
    status, _, err = run(capsys, "triad_count", "compute", "--input", hg, "--kind", "motif")
    assert status == 2 and "invalid choice" in err


def test_expected_ids_are_checked(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    # A .dgb recording with its expected ids replays under the recording's id order only.
    rows = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 2, 4]]
    h = dyng.Hypergraph.from_lists(rows, id_reuse="erase_first")
    batch = dyng.HyperedgeBatch(delete_hyperedges=[1], insert_hyperedges=[[0, 1, 3]])
    ids = h.apply(batch).inserted_ids
    dyng.io.write_hypergraph(tmp_path / "h.hg", dyng.Hypergraph.from_lists(rows))
    dyng.io.write_binary_batches(tmp_path / "b.dgb", [batch], expected_ids=[ids])
    argv = ["triad_count", "update", "--input", tmp_path / "h.hg", "--batches", tmp_path / "b.dgb"]
    status, _, err = run(capsys, *argv, "--id-reuse", "erase_first")
    assert status == 0, err
    status, _, err = run(capsys, *argv, "--id-reuse", "insert_first")
    assert status == 1 and "expected ids" in err


# -------------------------------------------------------------------------------------------------
# --format escher against ESCHER-GPU: the committed make-test counts (hosted CI)
# -------------------------------------------------------------------------------------------------


def _expected_lines(kind: str, case: dict) -> list[str]:
    """A's count lines of a fixture case (the full count, then per batch the Batch and delta
    lines; A prints the Batch line only when a run has more than one batch)."""
    from dyng.cli._triad_count import escher_count_line

    out = [escher_count_line(kind, case["full"], delta=False)]
    for i, (dels, ins, delta) in enumerate(case["batches"], 1):
        if len(case["batches"]) > 1:
            out.append(f"Batch {i} ({dels} deletions, {ins} insertions)\n")
        out.append(escher_count_line(kind, delta, delta=True))
    return out


@pytest.mark.parametrize("backend", BACKENDS)
def test_escher_format_equals_the_make_test_fixture(
    capsys: pytest.CaptureFixture[str], data: Path, tmp_path: Path, backend: str
) -> None:
    if not dyng.generators.legacy.standard_library_matches_legacy_streams():
        pytest.skip("this standard library does not reproduce the legacy generator streams")
    groups, cases = read_fixture(data)
    res = dyng.Resources(backend)
    for g in groups:
        # Record the group as the parity harness does: A's input as a 1-based .hg file and its
        # batches (drawn from the live state) as a .dgb file with the expected ids.
        h, gen = regenerate(g, res)
        d = tmp_path / g["name"]
        dyng.io.write_hypergraph(d / "input.hg", h)
        batches, ids = [], []
        for _ in range(g["batches"]):
            b = gen.next(h)
            batches.append(b)
            ids.append(h.apply(b).inserted_ids)
        if batches:
            dyng.io.write_binary_batches(d / "batches.dgb", batches, expected_ids=ids)
        heavy = g["n"] > 3000
        for exe in ["type1", "type2"] if heavy else list(EXES):
            argv = [
                "triad_count",
                "update" if batches else "compute",
                "--input",
                d / "input.hg",
                *(["--batches", d / "batches.dgb"] if batches else []),
                "--num-vertices",
                g["r2"],
                "--capacity",
                "escher_gpu",
                "--min-initial-payload",
                g["capacity"],
                "--kind",
                EXES[exe],
                "--format",
                "escher",
                "--backend",
                backend,
            ]
            status, out, err = run(capsys, *argv)
            assert status == 0, err
            lines = out.splitlines(keepends=True)
            assert lines[0].startswith(
                f"Hypergraph: {g['n']} hyperedges, max vertex id {g['r2']}, "
            )
            assert lines[1:] == _expected_lines(EXES[exe], cases[(g["name"], exe)]), (g, exe)


# -------------------------------------------------------------------------------------------------
# --format escher against A's unpatched standard output (the golden set, when present)
# -------------------------------------------------------------------------------------------------


def _golden_groups() -> list[str]:
    tests = GOLDENS / "cases" / "test"
    return sorted(p.name for p in tests.iterdir()) if tests.is_dir() else []


def _unzstd(src: Path, dst: Path) -> Path:
    with open(dst, "wb") as out:
        subprocess.run(["zstd", "-q", "-dc", str(src)], stdout=out, check=True)
    return dst


def _without_generator_lines(text: str) -> str:
    """A's standard output without the block its generator prints for a generated input
    ("Generating hypergraph with:", three "  - " lines and a blank line)."""
    lines = text.splitlines(keepends=True)
    if lines and lines[0] == "Generating hypergraph with:\n":
        end = lines.index("\n")
        lines = lines[end + 1 :]
    return "".join(lines)


@pytest.mark.parametrize("group", _golden_groups() or ["(no golden set)"])
def test_escher_format_equals_escher_gpus_stdout(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, group: str
) -> None:
    if not _golden_groups():
        pytest.skip(f"the golden set triad_count_escher_gpu is not at {GOLDENS}")
    if shutil.which("zstd") is None:
        pytest.skip("zstd is not installed")
    gdir = GOLDENS / "cases" / "test" / group
    meta = json.loads((gdir / "group.json").read_text())
    inp = json.loads((GOLDENS / "inputs" / f"{meta['input']}.json").read_text())
    hg = _unzstd(GOLDENS / inp["stored"], tmp_path / f"{meta['input']}.hg")
    batches = [_unzstd(gdir / b["file"], tmp_path / f"b{b['index']}.dgb") for b in meta["batches"]]
    backend = BACKENDS[-1]  # openmp where built: the syn100k groups count 100,000 hyperedges
    for exe, kind in EXES.items():
        stdout = gdir / f"{exe}.stdout"
        if not stdout.is_file():
            continue
        argv = [
            "triad_count",
            "update" if batches else "compute",
            "--input",
            hg,
            *(["--batches", *batches] if batches else []),
            "--num-vertices",
            meta["vertices"],
            "--id-reuse",
            "erase_first",
            "--capacity",
            "escher_gpu",
            "--min-initial-payload",
            meta["payload_capacity"],
            "--kind",
            kind,
            "--format",
            "escher",
            "--backend",
            backend,
        ]
        status, out, err = run(capsys, *argv)
        assert status == 0, err
        want = _without_generator_lines(stdout.read_text())
        assert out == want, f"{group}/{exe}: the escher lines differ from A's stdout"
        np.testing.assert_equal(len(out.splitlines()), len(want.splitlines()))
