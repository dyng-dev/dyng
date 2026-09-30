# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The ``dyng`` command line (PLAN Section 5.6; ADR 0025) against the originals' outputs.

The committed fixtures (cpp/tests/data) were written by the pinned originals: MOSP-OpenMP
c352151's ``mospPrep`` and ``mosp`` (the files of ``dyng prep`` and ``dyng sssp``) and
CycleEnumeration-GPU 0a976ad's ``cycle-enum`` (the standard output of ``dyng cycle_count``) and
its batch generator (``dyng generate cycle_enum_batch``). Every comparison is byte for byte.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import struct
import subprocess
import sys
from pathlib import Path

import dyng
import numpy as np
import pytest
from dyng.cli import build_parser, main
from dyng.cli._prep import CACHE_MAGIC, cache_identity

BACKENDS = ["sequential"] + (["openmp"] if dyng.config()["backends"]["openmp"] else [])


def run(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, str, str]:
    """Run ``dyng argv...`` in this process; (status, stdout, stderr)."""
    try:
        status = main([str(a) for a in argv])
    except SystemExit as e:  # argparse errors and --help / --version
        status = int(e.code or 0)
    out, err = capsys.readouterr()
    return status, out, err


def same_tree(a: Path, b: Path) -> None:
    """The two directories hold the same files with the same bytes."""
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert files_a == files_b
    for f in files_a:
        assert (a / f).read_bytes() == (b / f).read_bytes(), f


def sssp_cases(data: Path) -> list[tuple[str, Path, int]]:
    out = []
    for line in (data / "mosp_sssp" / "cases.txt").read_text().splitlines():
        name, rel, k = line.split()
        out.append((name, data / rel, int(k)))
    return out


# -------------------------------------------------------------------------------------------------
# The parser
# -------------------------------------------------------------------------------------------------


def _subparser(*path: str) -> object:
    p = build_parser()
    for name in path:
        action = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
        p = action.choices[name]  # type: ignore[attr-defined]
    return p


def _flags(parser: object) -> set[str]:
    return {s for a in parser._actions for s in a.option_strings}  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("path", "cls"),
    [
        (("sssp", "compute"), dyng.sssp.Options),
        (("sssp", "update"), dyng.sssp.Options),
        (("cycle_count", "compute"), dyng.cycle_count.Options),
        (("cycle_count", "update"), dyng.cycle_count.Options),
    ],
)
def test_every_option_field_has_a_flag(path: tuple[str, ...], cls: type) -> None:
    flags = _flags(_subparser(*path))
    for f in dataclasses.fields(cls):
        assert "--" + f.name.replace("_", "-") in flags, f.name


@pytest.mark.parametrize(
    ("path", "fn"),
    [
        (("generate", "mosp_changes"), dyng.generators.legacy.mosp_changes),
        (("generate", "cycle_enum_batch"), dyng.generators.legacy.cycle_enum_batch),
        (("cycle_count", "update"), dyng.generators.legacy.cycle_enum_batch),
    ],
)
def test_every_generator_argument_has_a_flag(path: tuple[str, ...], fn: object) -> None:
    import inspect

    flags = _flags(_subparser(*path))
    for name, p in inspect.signature(fn).parameters.items():  # type: ignore[arg-type]
        if p.kind is inspect.Parameter.KEYWORD_ONLY:
            assert "--" + name.replace("_", "-") in flags, name


def test_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    status, out, _ = run(capsys, "--version")
    assert status == 0 and out.strip() == f"dyng {dyng.__version__}"
    for cmd in (["sssp", "update"], ["cycle_count", "compute"], ["prep", "mtx2csr"], ["convert"]):
        status, out, _ = run(capsys, *cmd, "--help")
        assert status == 0 and "usage: dyng" in out
    status, _, err = run(capsys)
    assert status == 2 and "COMMAND" in err
    status, out, _ = run(capsys, "config")
    assert status == 0 and "dyng" in out


def test_console_script_and_module(tmp_path: Path) -> None:
    env = dict(os.environ)
    out = subprocess.run(
        [sys.executable, "-m", "dyng", "--version"], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0 and out.stdout.strip() == f"dyng {dyng.__version__}"
    script = Path(sys.executable).parent / "dyng"
    if not script.exists():
        pytest.skip("the console script is not installed next to this interpreter")
    out = subprocess.run([str(script), "--version"], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.strip() == f"dyng {dyng.__version__}"


def test_cuda_backend_is_reported(capsys: pytest.CaptureFixture[str], data: Path) -> None:
    if dyng.config()["backends"]["cuda"]:
        pytest.skip("this build has the CUDA backend")
    g = data / "cycle_enum" / "parser" / "reference_sample.txt"
    status, _, err = run(capsys, "cycle_count", "compute", "--graph", g, "--backend", "cuda")
    assert status == 1 and "dyng: error:" in err and "dyng-cu1" in err


# -------------------------------------------------------------------------------------------------
# cycle_count against cycle-enum's standard output
# -------------------------------------------------------------------------------------------------

# The original's flags -> dynG's (the options' names and the generator's arguments).
_CYCLE_FLAGS = {
    "--input": "--graph",
    "--max-cycle-length": "--max-length",
    "--max-length": "--max-length",
    "--openmp-threads": "--threads",
    "--threads": "--threads",
    "--deletes": "--num-deletions",
    "--inserts": "--num-insertions",
    "--batch-seed": "--seed",
    "--batch-locality": "--locality-window",
    "--backend": "--backend",
    "--algorithm": "--method",
    "--mode": "--mode",
}


def _translate_cycle_enum(line: str, data: Path) -> list[str] | None:
    """The dyng command of a cycle-enum case line (None for a task dynG does not have)."""
    argv = shlex.split(line.replace("@DATA@", str(data / "cycle_enum")))
    task = "compute"
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--compare-recompute":
            out.append(a)
            i += 1
            continue
        if a not in _CYCLE_FLAGS and a != "--task":
            return None
        v = argv[i + 1]
        if a == "--task":
            if v not in ("count", "update"):
                return None
            task = "compute" if v == "count" else "update"
        elif a == "--backend":
            out += ["--backend", "openmp" if v == "omp" else v]
        else:
            out += [_CYCLE_FLAGS[a], v]
        i += 2
    return ["cycle_count", task, *out]


def _cycle_enum_cases(data: Path) -> list[tuple[str, str, int]]:
    d = data / "cycle_enum" / "cli"
    lines = (d / "cases.txt").read_text().splitlines()
    return [
        (f"c{i:02d}", line, int((d / f"c{i:02d}.status").read_text()))
        for i, line in enumerate(lines)
    ]


@pytest.mark.parametrize("index", range(19))
def test_cycle_count_prints_the_originals_output(
    capsys: pytest.CaptureFixture[str], data: Path, index: int
) -> None:
    name, line, status = _cycle_enum_cases(data)[index]
    argv = _translate_cycle_enum(line, data)
    if status != 0:
        # The original's error cases: dynG fails too where its rules agree (a bound below 2, more
        # deletions than edges, a malformed or missing file); the others are no error in dynG
        # (`--openmp-threads 0` is the OpenMP default, `--task update` without counts an empty
        # batch) or are usage errors (status 2).
        if argv is not None and name in ("c12", "c14", "c15", "c16"):
            got, out, err = run(capsys, *argv)
            assert got == 1 and out == "" and "dyng: error:" in err, (name, err)
        return
    assert argv is not None, line
    if "--backend" in argv and argv[argv.index("--backend") + 1] == "openmp":
        if "openmp" not in BACKENDS:
            pytest.skip("no OpenMP backend")
    else:
        argv += ["--backend", "sequential"]
    got, out, err = run(capsys, *argv)
    assert got == 0, err
    assert out == (data / "cycle_enum" / "cli" / f"{name}.out").read_text(), (name, argv)
    if "--compare-recompute" in argv:
        assert "match=yes" in err


@pytest.mark.parametrize("backend", BACKENDS)
def test_cycle_count_update_from_batch_files(
    capsys: pytest.CaptureFixture[str], data: Path, tmp_path: Path, backend: str
) -> None:
    g = data / "cycle_enum" / "generator" / "graph300.txt"
    gen = ["--num-deletions", "50", "--num-insertions", "60", "--seed", "12345"]
    common = ["--graph", g, "--max-length", "5", "--backend", backend]
    s1, generated, _ = run(
        capsys, "cycle_count", "update", *common, *gen, "--write-batch", tmp_path / "b.txt"
    )
    s2, text, _ = run(capsys, "generate", "cycle_enum_batch", "--graph", g, *gen)
    assert s1 == 0 and s2 == 0
    assert (tmp_path / "b.txt").read_text() == text
    s3, from_file, _ = run(capsys, "cycle_count", "update", *common, "--batch", tmp_path / "b.txt")
    assert s3 == 0 and from_file == generated
    # the same batch as MOSP files (insert.txt, delete.txt)
    ins = [ln.split()[1:] for ln in text.splitlines() if ln.startswith("+")]
    dele = [ln.split()[1:] for ln in text.splitlines() if ln.startswith("-")]
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "insert.txt").write_text("".join(f"{u} {v}\n" for u, v in ins))
    (tmp_path / "m" / "delete.txt").write_text("".join(f"{u} {v}\n" for u, v in dele))
    s4, from_mosp, _ = run(capsys, "cycle_count", "update", *common, "--changes", tmp_path / "m")
    assert s4 == 0 and from_mosp == generated
    # the recomputed histogram of the updated graph
    cg = dyng.io.read_edge_list(g, properties="cycle_enum_compatible")
    b = dyng.generators.legacy.cycle_enum_batch(cg, num_deletions=50, num_insertions=60, seed=12345)
    cg.apply(b)
    assert generated == dyng.io.histogram_csv(dyng.cycle_count.compute(cg, max_length=5))
    s5, _, err = run(capsys, "cycle_count", "update", *common, *gen, "--batch", tmp_path / "b.txt")
    assert s5 == 2 and "one batch" in err


def test_text_batch_errors(capsys: pytest.CaptureFixture[str], data: Path, tmp_path: Path) -> None:
    g = data / "cycle_enum" / "parser" / "reference_sample.txt"
    for text in ("* 1 2\n", "+ 1\n", "- 1 x\n"):
        (tmp_path / "bad.txt").write_text(text)
        status, _, err = run(
            capsys, "cycle_count", "update", "--graph", g, "--batch", tmp_path / "bad.txt"
        )
        assert status == 1 and "bad.txt:1" in err, (text, err)
    (tmp_path / "far.txt").write_text(f"+ 0 {2**40}\n")
    status, _, err = run(
        capsys, "cycle_count", "update", "--graph", g, "--batch", tmp_path / "far.txt"
    )
    assert status == 1 and "dyng: error:" in err  # range-checked, never narrowed


# -------------------------------------------------------------------------------------------------
# sssp and prep init / expected against mospPrep and mosp
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_sssp_compute_and_update_equal_mosp(
    capsys: pytest.CaptureFixture[str], data: Path, tmp_path: Path, backend: str
) -> None:
    cases = sssp_cases(data)
    assert len(cases) >= 30
    for name, inp, k in cases:
        expected = data / "mosp_sssp" / name
        graph = ["--graph", inp / "graphCsr", "--num-weights", k, "--backend", backend]
        out = tmp_path / name
        status, _, err = run(capsys, "sssp", "compute", *graph, "--out", out / "c", "--quiet")
        assert status == 0, (name, err)
        same_tree(out / "c", expected / "init")
        # from the original's initial trees (the `mosp` driver), then from computed trees
        for init in (["--init", expected / "init"], []):
            status, _, err = run(
                capsys,
                "sssp",
                "update",
                *graph,
                "--changes",
                inp,
                *init,
                "--out",
                out / "u",
                "--quiet",
            )
            assert status == 0, (name, err)
            same_tree(out / "u", expected / "updated")


def test_sssp_single_objective_and_types(
    capsys: pytest.CaptureFixture[str], data: Path, tmp_path: Path
) -> None:
    inp = data / "mosp_sssp" / "c2i_1" / "input"
    expected = data / "mosp_sssp" / "c2i_1"
    for types in (["--vertex-type", "int32", "--edge-type", "int64"], ["--vertex-type", "int64"]):
        out = tmp_path / "-".join(types)
        status, text, _ = run(
            capsys,
            "sssp",
            "update",
            "--graph",
            inp / "graphCsr",
            *types,
            "--insert",
            inp / "insert.txt",
            "--delete",
            inp / "delete.txt",
            "--objective",
            "2",
            "--delta",
            "3",
            "--out",
            out,
            "--write-graph",
            out / "g",
        )
        assert status == 0
        assert "obj2: invalidated" in text and "obj0" not in text
        assert sorted(p.name for p in out.iterdir()) == [
            "gColInd.txt",
            "gRowPtr.txt",
            "gValues.txt",
            "obj2",
        ]
        same_tree(out / "obj2", expected / "updated" / "obj2")
    applied = data / "mosp_graph_io" / "testCase9"
    status, _, _ = run(
        capsys,
        "sssp",
        "update",
        "--graph",
        applied / "graphCsr",
        "--changes",
        applied,
        "--out",
        tmp_path / "t9",
        "--write-graph",
        tmp_path / "t9" / "g",
        "--quiet",
    )
    assert status == 0
    for f in ("RowPtr", "ColInd", "Values"):
        want = (applied / "applied" / f"graphCsr{f}.txt").read_bytes()
        assert (tmp_path / "t9" / f"g{f}.txt").read_bytes() == want


def test_sssp_errors(capsys: pytest.CaptureFixture[str], data: Path, tmp_path: Path) -> None:
    inp = data / "mosp_sssp" / "c2i_0" / "input"
    status, _, err = run(capsys, "sssp", "update", "--graph", inp / "graphCsr", "--out", tmp_path)
    assert status == 2 and "a batch is required" in err
    status, _, err = run(
        capsys, "sssp", "compute", "--graph", inp / "graphCsr", "--source", 99, "--out", tmp_path
    )
    assert status == 1 and "dyng: error:" in err
    status, _, err = run(
        capsys, "sssp", "compute", "--graph", tmp_path / "nothing", "--out", tmp_path
    )
    assert status == 1 and "dyng: error:" in err


@pytest.mark.parametrize("backend", BACKENDS)
def test_prep_init_and_expected(
    capsys: pytest.CaptureFixture[str], data: Path, tmp_path: Path, backend: str
) -> None:
    for name, inp, k in sssp_cases(data):
        expected = data / "mosp_sssp" / name
        status, out, err = run(
            capsys, "prep", "init", inp / "graphCsr", tmp_path / name / "i", "-k", k,
            "--backend", backend,
        )  # fmt: skip
        assert status == 0, err
        assert out.splitlines()[-1].startswith("init done in ") and out.endswith("(rc=0)\n")
        assert out.count("dijkstra obj") == len(list((expected / "init").iterdir()))
        same_tree(tmp_path / name / "i", expected / "init")
        status, _, err = run(
            capsys, "prep", "expected", inp / "graphCsr", inp, tmp_path / name / "e", "-k", k,
            "--backend", backend,
        )  # fmt: skip
        assert status == 0, err
        same_tree(tmp_path / name / "e", expected / "updated")


# -------------------------------------------------------------------------------------------------
# prep mtx2csr, widen, cache, changes
# -------------------------------------------------------------------------------------------------


def _same_csr(a: Path | str, b: Path | str) -> None:
    for f in ("RowPtr", "ColInd", "Values"):
        assert Path(f"{a}{f}.txt").read_bytes() == Path(f"{b}{f}.txt").read_bytes(), f


@pytest.mark.parametrize(
    ("mtx", "args", "prefix", "report"),
    [
        ("m0_symmetric.mtx", (3, 1, 100, 12345), "m0_k3_seed12345_", "n=6 directed edges=14"),
        ("m1_general.mtx", (1, 1, 100, 12345), "m1_k1_seed12345_", "n=5 directed edges=9"),
        ("m1_general.mtx", (4, 1, 2**31 - 1, 7), "m1_k4_seed7_", "n=5 directed edges=9"),
    ],
)
def test_prep_mtx2csr(
    capsys: pytest.CaptureFixture[str],
    data: Path,
    tmp_path: Path,
    mtx: str,
    args: tuple[int, ...],
    prefix: str,
    report: str,
) -> None:
    d = data / "mosp_graph_io" / "mtx"
    status, out, _ = run(capsys, "prep", "mtx2csr", d / mtx, f"{tmp_path}/g_", *args)
    assert status == 0
    symmetric = int("symmetric" in mtx)
    assert out.splitlines()[0] == f"mtx2csr: {report} symmetric={symmetric}"
    _same_csr(f"{tmp_path}/g_", d / prefix)
    # dyng convert does the same
    rw = ",".join(str(x) for x in args[1:])
    status, _, _ = run(
        capsys, "convert", d / mtx, f"{tmp_path}/c_", "--to", "csr", "--num-weights", args[0],
        "--random-weights", rw,
    )  # fmt: skip
    assert status == 0
    _same_csr(f"{tmp_path}/c_", d / prefix)


@pytest.mark.parametrize(
    ("source", "args", "expected"),
    [
        ("m1_k1_seed12345_", (3, 1, 100, 9), "m1_k1_seed12345_widen_k3_seed9_"),
        ("m0_k3_seed12345_", (5, 1, 2**31 - 1, 4), "m0_k3_seed12345_widen_k5_seed4_"),
        ("m0_k3_seed12345_", (3, 1, 100, 4), "m0_k3_seed12345_"),  # K unchanged: a copy
    ],
)
def test_prep_widen(
    capsys: pytest.CaptureFixture[str],
    data: Path,
    tmp_path: Path,
    source: str,
    args: tuple[int, ...],
    expected: str,
) -> None:
    d = data / "mosp_graph_io" / "mtx"
    status, out, _ = run(capsys, "prep", "widen", d / source, f"{tmp_path}/w_", *args)
    assert status == 0 and out.startswith("widen done in ") and out.endswith("(rc=0)\n")
    _same_csr(f"{tmp_path}/w_", d / expected)


def test_prep_errors(capsys: pytest.CaptureFixture[str], data: Path, tmp_path: Path) -> None:
    d = data / "mosp_graph_io" / "mtx"
    status, out, err = run(
        capsys, "prep", "widen", d / "m0_k3_seed12345_", f"{tmp_path}/w_", 2, 1, 100, 4
    )
    assert status == 1 and "graph already has 3 objectives" in err and "(rc=1)" in out
    for bad in ((0, 1, 100, 1), (33, 1, 100, 1), (1, 0, 100, 1), (1, 50, 49, 1), ("x", 1, 2, 3)):
        status, _, err = run(capsys, "prep", "mtx2csr", d / "m1_general.mtx", tmp_path / "x", *bad)
        assert status == 2 and "must be an integer in" in err, bad
    status, _, _ = run(capsys, "prep", "changes", d / "m1_k1_seed12345_", tmp_path, "--bogus", 1)
    assert status == 2
    status, out, err = run(capsys, "prep", "init", tmp_path / "missing_", tmp_path / "o")
    assert status == 1 and "dyng: error:" in err and out.endswith("(rc=1)\n")


def _changes_cases(data: Path) -> list[tuple[str, list[str]]]:
    out = []
    for line in (data / "mosp_changes" / "cases.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, *args = shlex.split(line)
            out.append((name, args))
    return out


def test_prep_changes_equal_mosp_prep(
    capsys: pytest.CaptureFixture[str], data: Path, tmp_path: Path
) -> None:
    graph = data / "mosp_changes" / "graph" / "graphCsr"
    cases = _changes_cases(data)
    assert len(cases) >= 10
    for name, args in cases:
        expected = data / "mosp_changes" / name
        status, out, err = run(capsys, "prep", "changes", graph, tmp_path / name, *args)
        assert status == 0, (name, err)
        for f in ("insert.txt", "delete.txt"):
            assert (tmp_path / name / f).read_bytes() == (expected / f).read_bytes(), (name, f)
        report = (expected / "report.txt").read_text().strip()
        assert out.splitlines()[0] == f"changes: {report}", name


def test_generate_mosp_changes(
    capsys: pytest.CaptureFixture[str], data: Path, tmp_path: Path
) -> None:
    graph = data / "mosp_changes" / "graph" / "graphCsr"
    # `mospPrep changes --mode targeted --changes 80 --ins 50 --seed 9 --local 5 --source 7
    # --safe` with the generator's argument names
    status, out, _ = run(
        capsys, "generate", "mosp_changes", "--graph", graph, "--mode", "targeted",
        "--num-changes", 80, "--insertion-percentage", 50, "--seed", 9, "--local-hops", 5,
        "--source", 7, "--safe-deletions", "--out", tmp_path,
    )  # fmt: skip
    assert status == 0
    expected = data / "mosp_changes" / "targeted_local_safe"
    for f in ("insert.txt", "delete.txt"):
        assert (tmp_path / f).read_bytes() == (expected / f).read_bytes(), f
    assert out.strip() == f"changes: {(expected / 'report.txt').read_text().strip()}"


def test_generate_cycle_enum_batch_equals_generate_batch(
    capsys: pytest.CaptureFixture[str], data: Path, tmp_path: Path
) -> None:
    d = data / "cycle_enum" / "generator"
    lines = (d / "cases.txt").read_text().splitlines()
    assert len(lines) == 22
    for i, line in enumerate(lines):
        file, dels, ins, seed, *window = line.split()
        argv = ["generate", "cycle_enum_batch", "--graph", d / file, "--num-deletions", dels]
        argv += ["--num-insertions", ins, "--seed", seed]
        if window:
            argv += ["--locality-window", window[0]]
        expected = (d / f"g{i:02d}.expected").read_text()
        status, out, err = run(capsys, *argv, "--out", tmp_path / "b.txt")
        if expected == "error\n":
            assert status == 1 and "dyng: error:" in err, line
            continue
        assert status == 0, (line, err)
        assert (tmp_path / "b.txt").read_text() == expected, line
        status, out, _ = run(capsys, *argv)
        assert status == 0 and out == expected


def _reference_mosp_prep() -> Path | None:
    scratch = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
    exe = scratch / "ref" / "MOSP-OpenMP@c352151" / "unpatched" / "bin" / "mospPrep"
    return exe if exe.is_file() and os.access(exe, os.X_OK) else None


def test_prep_cache(capsys: pytest.CaptureFixture[str], data: Path, tmp_path: Path) -> None:
    src = data / "mosp_sssp" / "c2i_1" / "input"
    for f in ("RowPtr", "ColInd", "Values"):
        (tmp_path / f"graphCsr{f}.txt").write_bytes((src / f"graphCsr{f}.txt").read_bytes())
    prefix = f"{tmp_path}/graphCsr"
    status, out, _ = run(capsys, "prep", "cache", prefix, tmp_path / "c" / "dyng.bin")
    assert status == 0 and out.startswith("cache done in ")
    blob = (tmp_path / "c" / "dyng.bin").read_bytes()
    # MOSPCSR2 | identity length | identity | n, K (int32) | m (int64) | rowPtr | colInd | weights
    assert blob[:8] == CACHE_MAGIC
    (length,) = struct.unpack_from("<I", blob, 8)
    identity = blob[12 : 12 + length]
    assert identity == cache_identity(prefix)
    lines = identity.decode().splitlines()
    assert lines[0] == os.path.realpath(f"{prefix}RowPtr.txt") and len(lines) == 9
    n, k, m = struct.unpack_from("<iiq", blob, 12 + length)
    g = dyng.io.read_csr_triplet(prefix, properties="mosp_compatible")
    csr = g.to_csr()
    assert (n, k, m) == (g.num_vertices, g.num_weights, g.num_edges)
    body = np.frombuffer(blob, dtype="<i4", offset=12 + length + 16)
    assert body.size == n + 1 + m * (1 + k)
    assert body[: n + 1].tolist() == csr.row_ptr.tolist()
    assert body[n + 1 : n + 1 + m].tolist() == csr.col_ind.tolist()
    assert body[n + 1 + m :].tolist() == np.asarray(csr.weights).reshape(-1).tolist()
    ref = _reference_mosp_prep()
    if ref is None:
        pytest.skip("the reference mospPrep (parity/build_reference.sh) is not built here")
    subprocess.run(
        [str(ref), "cache", prefix, str(tmp_path / "c" / "orig.bin")],
        check=True,
        capture_output=True,
    )
    assert (tmp_path / "c" / "orig.bin").read_bytes() == blob


# -------------------------------------------------------------------------------------------------
# convert
# -------------------------------------------------------------------------------------------------


def test_convert_round_trips(
    capsys: pytest.CaptureFixture[str], data: Path, tmp_path: Path
) -> None:
    prefix = data / "mosp_sssp" / "c2i_1" / "input" / "graphCsr"
    status, out, _ = run(capsys, "convert", prefix, tmp_path / "g.txt")
    assert status == 0 and out.startswith("convert: csr -> edges:")
    status, _, _ = run(
        capsys, "convert", tmp_path / "g.txt", f"{tmp_path}/back_", "--to", "csr",
        "--num-weights", 3,
    )  # fmt: skip
    assert status == 0
    _same_csr(f"{tmp_path}/back_", prefix)
    status, _, _ = run(capsys, "convert", prefix, tmp_path / "g.mtx", "--weight-column", 1)
    assert status == 0
    g = dyng.io.read_csr_triplet(prefix)
    m = dyng.io.read_matrix_market(tmp_path / "g.mtx")
    src, dst, w = g.edges()
    ms, md, mw = m.edges()
    assert sorted(zip(src.tolist(), dst.tolist(), w[:, 1].tolist(), strict=True)) == sorted(
        zip(ms.tolist(), md.tolist(), mw[:, 0].tolist(), strict=True)
    )
    (tmp_path / "u.txt").write_text("0 1\n1 2\n")
    status, _, err = run(capsys, "convert", tmp_path / "u.txt", f"{tmp_path}/x_", "--to", "csr")
    assert status == 2 and "needs weights" in err


def test_every_flag_is_documented() -> None:
    """docs/api/cli.md mentions every flag of every command (the reference stays complete)."""
    doc = Path(__file__).resolve().parents[2] / "docs" / "api" / "cli.md"
    if not doc.is_file():
        pytest.skip("docs/api/cli.md is not in this tree")
    text = doc.read_text()

    def walk(parser: object) -> set[str]:
        out = set()
        for a in parser._actions:  # type: ignore[attr-defined]
            out |= {s for s in a.option_strings if s not in ("-h", "--help")}
            for sub in getattr(a, "choices", None) or {}:
                if a.__class__.__name__ == "_SubParsersAction":
                    out |= walk(a.choices[sub])
        return out

    import re

    def documented(flag: str) -> bool:
        if flag.startswith("--no-"):  # the negative form of a boolean flag
            flag = "--" + flag[len("--no-") :]
        return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text) is not None

    missing = sorted(f for f in walk(build_parser()) if not documented(f))
    assert not missing, f"flags missing from docs/api/cli.md: {missing}"
