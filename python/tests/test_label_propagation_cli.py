# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""``dyng label_propagation compute|update`` and ``dyng generate vertex_arrival_stream`` /
``dynlp_vertex_arrival_stream`` (M10b step 3; ADR 0108; AD12 with E1).

The CLI's damped labels on LabelPropagation-CUDA's 5k5 stream equal dyng-compat-dynlp's
sequential output byte for byte (the committed digest, and the tool itself when a build tree of
this checkout has it); one flag per Options field, both ways; the generators' files.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path

import dyng
import numpy as np
import pytest
from dyng.cli import build_parser, main

REPO = Path(__file__).resolve().parents[2]
DIGESTS = REPO / "examples" / "data" / "label_propagation_5k5_damped.sha256"
GENERATOR_DIGESTS = (
    REPO / "cpp" / "tests" / "data" / "goldens" / "label_propagation"
    / "generator_5k5_d1e-4_s42.sha256"
)  # fmt: skip
STREAM = ("--total-vertices", "5000", "--num-batches", "5")


def run(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, str, str]:
    try:
        status = main([str(a) for a in argv])
    except SystemExit as e:
        status = int(e.code or 0)
    out, err = capsys.readouterr()
    return status, out, err


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(path: Path, name: str) -> str:
    for line in path.read_text().splitlines():
        if line and not line.startswith("#") and line.split()[1] == name:
            return line.split()[0]
    raise KeyError(name)


def _compat_dynlp() -> Path | None:
    """dyng-compat-dynlp of a build tree of this checkout (DYNG_COMPAT_DYNLP wins)."""
    env = os.environ.get("DYNG_COMPAT_DYNLP")
    candidates = (
        [Path(env)]
        if env
        else [
            REPO / "build" / p / "tools" / "compat" / "dyng-compat-dynlp"
            for p in ("dev", "cpu-only", "dev-cuda", "parity")
        ]
    )
    for exe in candidates:
        if exe.is_file() and os.access(exe, os.X_OK):
            return exe
    return None


def test_update_on_the_legacy_stream_equals_the_compat_tool(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    status, out, err = run(
        capsys,
        "label_propagation",
        "update",
        "--generate",
        "legacy",
        *STREAM,
        "--update-schedule",
        "damped",
        "--backend",
        "sequential",
        "--out",
        tmp_path / "cli",
    )
    assert status == 0, err
    assert out.splitlines()[1].startswith("batch 1: ") and "1217 iterations" in out
    labels = tmp_path / "cli" / "dynlp_labels.txt"
    assert sha256(labels) == digest(DIGESTS, "sequential")
    rows = (tmp_path / "cli" / "stats.csv").read_text().splitlines()
    assert rows[0].startswith("batch,time_ms,iterations,frontier_visits,components,accuracy,")
    assert [r.split(",")[2] for r in rows[1:]] == ["0", "1217", "548", "327", "65"]
    exe = _compat_dynlp()
    if exe is None:
        pytest.skip("dyng-compat-dynlp is not built in this checkout (the digest was compared)")
    subprocess.run(
        [
            str(exe),
            "--totalNodes",
            "5000",
            "--numBatches",
            "5",
            "--update",
            "damped",
            "--backend",
            "sequential",
            "--noValidate",
            "--noGraphFiles",
            "--quiet",
            "--output",
            str(tmp_path / "compat"),
        ],
        check=True,
        capture_output=True,
        timeout=600,
    )
    assert labels.read_bytes() == (tmp_path / "compat" / "dynlp_labels.txt").read_bytes()
    compat = (tmp_path / "compat" / "results.csv").read_text().splitlines()
    # the original's columns (time and eval time apart) are the tool's
    for ours, theirs in zip(rows[1:], compat[1:], strict=True):
        a, b = ours.split(","), theirs.split(",")
        assert [a[0], *a[2:6]] == [b[0], *b[2:6]]


def test_generate_writes_the_stream_and_update_reads_it(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    s = tmp_path / "s"
    status, out, err = run(
        capsys, "generate", "dynlp_vertex_arrival_stream", *STREAM, "--out", s, "--graph-files"
    )
    assert status == 0, err
    assert "similarity_threshold=" in out
    meta = json.loads((s / "stream.json").read_text())
    stream = dyng.generators.legacy.dynlp_vertex_arrival_stream(
        total_vertices=5000, num_batches=5, resources=dyng.Resources("sequential")
    )
    assert meta["similarity_threshold"] == stream.similarity_threshold
    assert meta["num_vertices"] == 5000 and meta["num_batches"] == 5
    assert np.array_equal(np.loadtxt(s / "row_capacity.txt", dtype=np.int64), stream.row_capacity)
    for name in ("csrRowPtr.txt", "csrColInd.txt", "csrValues.txt", "csrTrueLabels.txt"):
        assert sha256(s / "graph" / name) == digest(GENERATOR_DIGESTS, name), name
    assert (s / "true_labels.txt").read_bytes() == (s / "graph" / "csrTrueLabels.txt").read_bytes()
    want = digest(DIGESTS, "sequential")
    for extra in (
        ("--capacity", s / "row_capacity.txt", "--true-labels", s / "true_labels.txt"),
        ("--capacity", "auto", "--similarity-threshold", str(meta["similarity_threshold"])),
    ):
        out_dir = tmp_path / f"run{len(extra)}{extra[1] == 'auto'}"
        status, _, err = run(
            capsys,
            "label_propagation",
            "update",
            "--batches",
            s / "batches.dgb",
            *extra,
            "--update-schedule",
            "damped",
            "--backend",
            "sequential",
            "--out",
            out_dir,
            "--quiet",
        )
        assert status == 0, err
        assert sha256(out_dir / "dynlp_labels.txt") == want
    # the text batch file and the library's generator
    status, _, err = run(
        capsys, "generate", "vertex_arrival_stream", *STREAM, "--out", tmp_path / "t",
        "--format", "dgt", "--quiet",
    )  # fmt: skip
    assert status == 0, err
    lib = dyng.generators.vertex_arrival_stream(
        total_vertices=5000, num_batches=5, resources=dyng.Resources("sequential")
    )
    batches = dyng.io.read_batches(tmp_path / "t" / "batches.dgt", weight_dtype="float32")
    assert len(batches) == 5
    for a, b in zip(lib.batches, batches, strict=True):
        assert np.array_equal(a.insert_vertices, b.insert_vertices)
        assert a.insert_weights.tobytes() == b.insert_weights.tobytes()


def test_compute_reads_an_edge_list_and_seeds(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    (tmp_path / "g.txt").write_text("0 1 1.0\n1 2 1.0\n3 4 0.5\n")
    (tmp_path / "s.txt").write_text("# v class\n0 1\n2 0\n")
    status, out, err = run(
        capsys,
        "label_propagation",
        "compute",
        "--graph",
        tmp_path / "g.txt",
        "--seeds",
        tmp_path / "s.txt",
        "--tolerance",
        "1e-7",
        "--backend",
        "sequential",
        "--out",
        tmp_path / "o",
    )
    assert status == 0, err
    assert "seeds=2" in out
    lines = (tmp_path / "o" / "dynlp_labels.txt").read_text().splitlines()
    assert lines[0] == "0 1.000000" and lines[2] == "2 0.000000" and lines[1] == "1 0.500000"
    assert lines[3:] == ["3 0.500000", "4 0.500000"]
    status, _, err = run(
        capsys, "label_propagation", "compute", "--graph", tmp_path / "g.txt", "--seeds",
        tmp_path / "s.txt", "--similarity-threshold", "published", "--out", tmp_path / "x",
    )  # fmt: skip
    assert status == 2 and "current" in err


def _subparser(parser: argparse.ArgumentParser, *path: str) -> argparse.ArgumentParser:
    for name in path:
        action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        parser = action.choices[name]
    return parser


def test_one_flag_per_options_field_both_ways() -> None:
    fields = {f.name for f in dataclasses.fields(dyng.label_propagation.Options)}
    for verb in ("compute", "update"):
        p = _subparser(build_parser(), "label_propagation", verb)
        flags = {
            a.dest[len("opt_") :]: a.option_strings for a in p._actions if a.dest.startswith("opt_")
        }
        assert set(flags) == fields, verb  # every field has a flag, every option flag a field
        for name, strings in flags.items():
            assert strings == ["--" + name.replace("_", "-")], (verb, name)


def test_usage_errors_and_help(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    base = ("label_propagation", "update", "--out", tmp_path / "o")
    status, _, err = run(capsys, *base, "--backend", "openmp")
    assert status == 2 and "invalid choice" in err  # no OpenMP backend (AD19)
    status, _, err = run(capsys, *base, "--backend", "sequential")
    assert status == 2 and "--generate" in err
    status, _, err = run(
        capsys, *base, "--generate", "legacy", "--batches", "x.dgb", "--backend", "sequential"
    )
    assert status == 2
    status, _, err = run(
        capsys, *base, "--generate", "legacy", "--similarity-threshold", "nan",
        "--backend", "sequential",
    )  # fmt: skip
    assert status == 2 and "finite" in err
    status, out, _ = run(capsys, "generate", "--help")
    assert status == 0
    assert "vertex_arrival_stream" in out and "dynlp_vertex_arrival_stream" in out
    assert "legacy generators of" not in out  # the group is no longer legacy-only
    status, out, _ = run(capsys, "--help")
    assert "label_propagation" in out


def test_streaming_mode_runs(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    status, _, err = run(
        capsys, "label_propagation", "update", "--generate", "library", "--total-vertices",
        "2000", "--num-batches", "3", "--layout", "slack", "--similarity-threshold", "current",
        "--update-schedule", "damped", "--backend", "sequential", "--per-batch-labels",
        "--quiet", "--out", tmp_path,
    )  # fmt: skip
    assert status == 0, err
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "dynlp_labels.txt",
        "dynlp_labels_b0.txt",
        "dynlp_labels_b1.txt",
        "dynlp_labels_b2.txt",
        "stats.csv",
    ]
    assert (tmp_path / "dynlp_labels_b2.txt").read_bytes() == (
        tmp_path / "dynlp_labels.txt"
    ).read_bytes()


# The command line frozen at the end of M10b (ADR 0109; AD15: stable from 0.4.0 only on the
# author's confirmation): every flag below stays, with its name (PLAN 5.9: CLI flags are stable-tier
# API); a flag may be added (an additive change with a CHANGELOG entry), never removed or renamed.
FROZEN_FLAGS = {
    ("label_propagation", "compute"): (
        "--backend --cuda-engine --device --graph --help --index-base --max-iterations "
        "--num-vertices --on-iteration-limit --out --quiet --seeds --similarity-threshold "
        "--tolerance --update-schedule -h"
    ),
    ("label_propagation", "update"): (
        "--average-degree --backend --batches --capacity --cuda-engine --deleted-fraction --device "
        "--generate --ground-truth-fraction --help --layout --max-iterations --num-batches "
        "--on-iteration-limit --out --per-batch-labels --quiet --seed --similarity-threshold "
        "--tolerance --total-vertices --true-labels --update-schedule -h"
    ),
    ("generate", "vertex_arrival_stream"): (
        "--average-degree --deleted-fraction --edge-type --format --graph-files "
        "--ground-truth-fraction --help --num-batches --out --quiet --seed --total-vertices -h"
    ),
    ("generate", "dynlp_vertex_arrival_stream"): (
        "--average-degree --deleted-fraction --edge-type --format --graph-files "
        "--ground-truth-fraction --help --num-batches --out --quiet --seed --total-vertices -h"
    ),
}


# And each frozen flag's choices, argparse default and required-ness (ADR 0109 decision 1: the
# flags are frozen with their values; a flag whose default is None takes the Options field's or
# the generator argument's default, which the griffe baseline guards). Adding a choice is
# additive (update the table with the CHANGELOG entry); removing one or changing a default fails.
FROZEN_FLAG_SPECS = {
    ("label_propagation", "compute"): {
        "--graph": (None, None, True),
        "--seeds": (None, None, True),
        "--index-base": (None, 0, False),
        "--num-vertices": (None, None, False),
        "--tolerance": (None, None, False),
        "--max-iterations": (None, None, False),
        "--on-iteration-limit": (("error", "report", "fallback_recompute"), None, False),
        "--update-schedule": (("in_place", "damped"), None, False),
        "--cuda-engine": (("automatic", "fused", "operators"), None, False),
        "--similarity-threshold": (None, None, False),
        "--out": (None, None, True),
        "--quiet": (None, False, False),
        "--backend": (("sequential", "cuda"), None, False),
        "--device": (None, 0, False),
    },
    ("label_propagation", "update"): {
        "--generate": (("legacy", "library"), None, False),
        "--total-vertices": (None, None, False),
        "--average-degree": (None, None, False),
        "--num-batches": (None, None, False),
        "--seed": (None, None, False),
        "--ground-truth-fraction": (None, None, False),
        "--deleted-fraction": (None, None, False),
        "--batches": (None, None, False),
        "--capacity": (None, "auto", False),
        "--true-labels": (None, None, False),
        "--layout": (("slotted", "slack"), "slotted", False),
        "--tolerance": (None, None, False),
        "--max-iterations": (None, None, False),
        "--on-iteration-limit": (("error", "report", "fallback_recompute"), None, False),
        "--update-schedule": (("in_place", "damped"), None, False),
        "--cuda-engine": (("automatic", "fused", "operators"), None, False),
        "--similarity-threshold": (None, None, False),
        "--out": (None, None, True),
        "--per-batch-labels": (None, False, False),
        "--quiet": (None, False, False),
        "--backend": (("sequential", "cuda"), None, False),
        "--device": (None, 0, False),
    },
    ("generate", "vertex_arrival_stream"): {
        "--total-vertices": (None, None, False),
        "--average-degree": (None, None, False),
        "--num-batches": (None, None, False),
        "--seed": (None, None, False),
        "--ground-truth-fraction": (None, None, False),
        "--deleted-fraction": (None, None, False),
        "--edge-type": (("int32", "int64"), None, False),
        "--out": (None, None, True),
        "--format": (("dgb", "dgt"), "dgb", False),
        "--graph-files": (None, False, False),
        "--quiet": (None, False, False),
    },
    ("generate", "dynlp_vertex_arrival_stream"): {
        "--total-vertices": (None, None, False),
        "--average-degree": (None, None, False),
        "--num-batches": (None, None, False),
        "--seed": (None, None, False),
        "--ground-truth-fraction": (None, None, False),
        "--deleted-fraction": (None, None, False),
        "--edge-type": (("int32", "int64"), None, False),
        "--out": (None, None, True),
        "--format": (("dgb", "dgt"), "dgb", False),
        "--graph-files": (None, False, False),
        "--quiet": (None, False, False),
    },
}


def test_the_frozen_flags_are_kept() -> None:
    for path, frozen in FROZEN_FLAGS.items():
        p = _subparser(build_parser(), *path)
        flags = {o for a in p._actions for o in a.option_strings}
        missing = set(frozen.split()) - flags
        assert not missing, (path, sorted(missing))


def test_the_frozen_flags_keep_their_choices_and_defaults() -> None:
    for path, specs in FROZEN_FLAG_SPECS.items():
        p = _subparser(build_parser(), *path)
        actions = {a.option_strings[0]: a for a in p._actions if a.option_strings}
        for flag, (choices, default, required) in specs.items():
            a = actions[flag]
            got = tuple(a.choices) if a.choices is not None else None
            if choices is None:  # a new restriction narrows what the flag accepts
                assert got is None, (path, flag, got)
            else:  # a choice may be added, never removed
                assert got is not None and set(choices) <= set(got), (path, flag, got)
            assert a.default == default, (path, flag, a.default)
            assert a.required == required, (path, flag, a.required)
