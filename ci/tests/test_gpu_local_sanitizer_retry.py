# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""ci/gpu_local.sh's bounded retry of a racecheck run whose process died (ADR 0112; R031 open
item 6): the functions `sanitize` and `sanitizer_death_only`, cut from the script and run in bash
with a fake compute-sanitizer that prints a given output per attempt. Only the outcome "process
didn't terminate successfully" with 0 hazards, no other error or warning and no failed test is
run once more; everything else fails at once, and a second death fails."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = (REPO / "ci" / "gpu_local.sh").read_text()

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

DEATH = (
    "========= COMPUTE-SANITIZER\n"
    "========= Error: process didn't terminate successfully\n"
    "=========     The application may have hit an error when dereferencing Unified Memory "
    "from the host. Please rerun the application under cuda-gdb or a host debugger to catch "
    "host side errors.\n"
    "========= Target application returned an error\n"
    "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
)
PASS = (
    "========= COMPUTE-SANITIZER\n"
    "[  PASSED  ] 3 tests.\n"
    "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
)
HAZARD = (
    "========= COMPUTE-SANITIZER\n"
    "========= Error: Race reported between Write access at 0x10 in k() and Read access\n"
    "========= Error: process didn't terminate successfully\n"
    "========= Target application returned an error\n"
    "========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)\n"
)
WARNING = DEATH.replace(
    "========= Target", "========= Warning: Race reported between Write and Read\n========= Target"
)
FAILED_TEST = DEATH.replace("========= Target", "[  FAILED  ] Suite.Case\n========= Target")
TEST_FAILURE = (
    "========= COMPUTE-SANITIZER\n"
    "[  FAILED  ] Suite.Case\n"
    "========= Target application returned an error\n"
    "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
)


def functions() -> str:
    start = SCRIPT.index("sanitizer_retries=()\n")
    end = SCRIPT.index("# A step's result: passed, FAILED")
    return SCRIPT[start:end]


def run(tmp_path: Path, outputs: list[tuple[str, int]], tool: str = "racecheck") -> tuple:
    """Run `sanitize <fake> --tool <tool> ...` with the fake printing outputs[i] and exiting with
    its status on its i-th call; return (status, calls, retries, stdout)."""
    for i, (text, rc) in enumerate(outputs):
        (tmp_path / f"out{i}").write_text(text)
        (tmp_path / f"rc{i}").write_text(str(rc))
    fake = tmp_path / "compute-sanitizer"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'd="{tmp_path}"\n'
        'n=$(cat "$d/calls" 2>/dev/null || echo 0)\n'
        'echo $((n + 1)) >"$d/calls"\n'
        'cat "$d/out$n"\n'
        'exit "$(cat "$d/rc$n")"\n'
    )
    fake.chmod(0o755)
    driver = (
        "set -euo pipefail\n"
        'heavy() { "$@"; }\n'
        + functions()
        + f'if sanitize "{fake}" --tool {tool} --error-exitcode 1 exe --gtest_brief=1; then\n'
        "  status=0\nelse\n  status=$?\nfi\n"
        'echo "STATUS=${status} RETRIES=${#sanitizer_retries[@]}"\n'
    )
    out = subprocess.run(
        ["bash", "-c", driver],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "TMPDIR": str(tmp_path)},
        check=True,
    ).stdout
    last = out.strip().splitlines()[-1]
    status = int(last.split()[0].split("=")[1])
    retries = int(last.split()[1].split("=")[1])
    calls = int((tmp_path / "calls").read_text())
    assert not list(tmp_path.glob("dyng-racecheck.*")), "the temporary logs are removed"
    return status, calls, retries, out


def test_a_pass_runs_once(tmp_path: Path) -> None:
    assert run(tmp_path, [(PASS, 0)])[:3] == (0, 1, 0)


def test_a_death_with_nothing_reported_runs_once_more(tmp_path: Path) -> None:
    status, calls, retries, out = run(tmp_path, [(DEATH, 1), (PASS, 0)])
    assert (status, calls, retries) == (0, 2, 1)
    assert "running it once more (ADR 0112)" in out
    # The log keeps both runs' output: the death, then the passing run.
    assert out.index("process didn't terminate successfully") < out.index("[  PASSED  ]")


def test_a_second_death_fails(tmp_path: Path) -> None:
    status, calls, retries, _ = run(tmp_path, [(DEATH, 1), (DEATH, 1), (PASS, 0)])
    assert status != 0 and calls == 2 and retries == 1


@pytest.mark.parametrize("output", [HAZARD, WARNING, FAILED_TEST, TEST_FAILURE])
def test_anything_else_fails_at_once(tmp_path: Path, output: str) -> None:
    status, calls, retries, _ = run(tmp_path, [(output, 1), (PASS, 0)])
    assert status != 0 and calls == 1 and retries == 0


def test_a_hazard_after_the_retry_fails(tmp_path: Path) -> None:
    status, calls, retries, _ = run(tmp_path, [(DEATH, 1), (HAZARD, 1)])
    assert status != 0 and calls == 2 and retries == 1


@pytest.mark.parametrize("tool", ["memcheck", "synccheck", "initcheck"])
def test_the_other_tools_are_never_retried(tmp_path: Path, tool: str) -> None:
    death = DEATH.replace("RACECHECK", tool.upper())
    status, calls, retries, _ = run(tmp_path, [(death, 1), (PASS, 0)], tool=tool)
    assert status != 0 and calls == 1 and retries == 0


def test_every_sanitizer_call_of_the_script_goes_through_sanitize() -> None:
    assert 'heavy "${sanitizer}"' not in SCRIPT
    assert SCRIPT.count('sanitize "${sanitizer}"') >= 20
    # The retried runs are listed outside the summary's table (certify.py reads its rows only).
    assert "Sanitizer runs retried after a process death (ADR 0112)" in SCRIPT
