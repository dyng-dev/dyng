# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The ``dyng`` command line (PLAN Section 5.6; ADR 0025).

``dyng <algorithm> <verb> [--kebab-case options]`` with the verbs ``compute`` and ``update``, and
the utility commands ``prep`` (MOSP's ``mospPrep`` subcommands), ``convert`` and ``generate``::

    dyng sssp compute --graph roadNet-CA_ --out init              # = mospPrep init
    dyng sssp update --graph roadNet-CA_ --changes b1 --init init --out updated   # = mosp
    dyng cycle_count compute --graph DD_A.txt --max-length 6      # = cycle-enum --task count
    dyng cycle_count update --graph DD_A.txt --max-length 4 --num-deletions 40 \\
        --num-insertions 40 --seed 1                              # = cycle-enum --task update
    dyng prep mtx2csr roadNet-CA.mtx roadNet-CA_ 1 1 100 12345    # = mospPrep mtx2csr
    dyng convert g.mtx g.txt
    dyng generate cycle_enum_batch --graph DD_A.txt --num-deletions 5 --num-insertions 5

Every option flag is the kebab-case name of a field of the algorithm's ``Options``
(``max_length`` -> ``--max-length``); the outputs are the originals' formats (MOSP's distance and
tree files, CycleEnumeration-GPU's histogram CSV, MOSP's batch files), so they can be compared
byte for byte with the originals and the goldens. The console script ``dyng`` and
``python -m dyng`` call :func:`main`.

Exit status: 0 on success, 1 when the work fails (the message is printed as
``dyng: error: ...``), 2 for a usage error.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import dyng

from ._algorithms import add_cycle_count, add_sssp
from ._common import CliError, UsageError
from ._prep import add_prep, run_prep
from ._tools import add_convert, add_generate

__all__ = ["build_parser", "main"]


def _config(args: argparse.Namespace) -> int:
    dyng.show_config()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the ``dyng`` command (for documentation and tests)."""
    p = argparse.ArgumentParser(
        prog="dyng",
        description="dynG: dynamic graph algorithms that update their results under batches of "
        "changes. Commands: the algorithms (sssp, cycle_count) with the verbs compute and "
        "update, and the utilities prep, convert and generate.",
        epilog="Run 'dyng <command> --help' for the flags of a command.",
    )
    p.add_argument("--version", action="version", version=f"dyng {dyng.__version__}")
    p.add_argument(
        "--log-level",
        choices=("trace", "debug", "info", "warn", "error", "off"),
        default=None,
        help="the library's log level (default: its current level)",
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND", required=True)
    add_sssp(sub)
    add_cycle_count(sub)
    add_prep(sub)
    add_convert(sub)
    add_generate(sub)
    c = sub.add_parser("config", help="the build and runtime configuration (show_config)")
    c.set_defaults(func=_config)
    return p


def _message(e: BaseException) -> str:
    # The library's messages start with "dyng: " (C++ io_error, ...); the CLI adds its own.
    return str(e).removeprefix("dyng: ")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``dyng`` command with ``argv`` (default: ``sys.argv[1:]``); returns the exit
    status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.log_level is not None:
            dyng.set_log_level(args.log_level)
        if getattr(args, "prep", False):
            return run_prep(args)
        return int(args.func(args))
    except UsageError as e:
        print(f"dyng: error: {_message(e)}", file=sys.stderr)
        return 2
    except (dyng.Error, OSError, CliError, ValueError) as e:
        print(f"dyng: error: {_message(e)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
