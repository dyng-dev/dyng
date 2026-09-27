#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Check the repository metadata under .github/ that GitHub only validates after a push.

Checked (PLAN Sections 8.9 and 10.1-10.2; docs/developer/labels.md):

* ``.github/labels.yml``: a list of labels with a unique ``name`` (at most 50 characters), a
  ``color`` ``#rrggbb`` and a ``description`` of at most 100 characters; the labels the plan
  names (the good-first-issue categories, ``ci:gpu``, ``ci:bench``, ...) are present.
* ``.github/ISSUE_TEMPLATE/*.yml``: the GitHub issue-form schema (top-level keys, element
  types and attributes), plus the rules GitHub enforces only when rendering a form: unique
  element ids and labels, unique non-empty dropdown options, no reserved option "None", and
  every label a form applies exists in labels.yml. ``config.yml``: the chooser configuration.
* ``.github/CODEOWNERS``: every rule is ``<pattern> <owner>...`` with owners ``@user``,
  ``@org/team`` or an e-mail address, no syntax GitHub does not support, a catch-all ``*``
  rule, and every anchored path (``/dir/``, ``/file``) exists in the repository.
* ``.github/PULL_REQUEST_TEMPLATE.md``: the review checklist of PLAN Section 8.9 is present.
* ``.github/workflows/*.yml`` (least privilege and pinning, PLAN Section 8.8): a top-level
  ``permissions`` block that grants nothing beyond ``read`` (write permissions only per job);
  ``pull_request_target`` only in a workflow that neither checks out nor runs anything (the
  first-interaction greeter); every job with a ``timeout-minutes``; every ``uses:`` of a remote
  action pinned to a full 40-character commit SHA with a ``# vX.Y.Z`` comment naming the tag;
  every ``actions/checkout`` step with ``persist-credentials: false``; no ``${{ }}`` expression
  in a ``run:`` script other than ``matrix.*``, ``env.*`` and ``runner.*`` (pass values through
  ``env:`` instead). actionlint and zizmor (pre-commit hooks) check the rest.
* Tool versions that dependabot cannot update: every ``micromamba-version`` of the workflows is
  the same, and every CMake requirement a workflow installs with pip equals the one of
  ``environment.yml``.

Usage::

    python3 ci/github_meta_check.py                # check the repository
    python3 ci/github_meta_check.py --self-test    # first prove that broken inputs are rejected
    python3 ci/github_meta_check.py --verify-pins  # also check each action SHA against its
                                                   # version tag (git ls-remote; needs network)

Needs PyYAML (in the dyng-dev environment; the pre-commit hook installs it).
"""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Labels named by the plan (Section 10.2) and applied by the issue forms or dependabot.
REQUIRED_LABELS = {
    "good first issue",
    "good-first-issue:op",
    "good-first-issue:reader",
    "good-first-issue:backend",
    "good-first-issue:docs",
    "help wanted",
    "ci:gpu",
    "ci:bench",
    "api-change",
    "port",
    "parity",
    "triage",
    "dependencies",
    "github_actions",
}

# The issue forms of PLAN Section 10.1 plus the two the M4 milestone adds.
REQUIRED_FORMS = {
    "bug.yml",
    "feature.yml",
    "new_algorithm.yml",
    "port_research_code.yml",
    "parity_regression.yml",
    "api_change.yml",
    "performance_regression.yml",
    "docs.yml",
}

# The review checklist of PLAN Section 8.9, rule 5: each entry must appear in the PR template.
PR_CHECKLIST = [
    "**Tests:**",
    "**Parity:**",
    "**Docs:**",
    "**CHANGELOG:**",
    "**DCO:**",
    "**Benchmarks:**",
    "**Library rules:**",
    "**Headers:**",
]

FORM_TOP_KEYS = {"name", "description", "title", "labels", "assignees", "projects", "type", "body"}
ELEMENT_ATTRIBUTES = {
    "markdown": {"value"},
    "textarea": {"label", "description", "placeholder", "value", "render"},
    "input": {"label", "description", "placeholder", "value"},
    "dropdown": {"label", "description", "multiple", "options", "default"},
    "checkboxes": {"label", "description", "options"},
}
ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
OWNER_RE = re.compile(
    r"^(@[A-Za-z0-9](?:-?[A-Za-z0-9])*(?:/[A-Za-z0-9._-]+)?|[^@\s]+@[^@\s]+\.[^@\s]+)$"
)
# A remote action pinned by SHA with its version tag in a comment:
#   uses: owner/repo[/path]@<40 hex>  # vX.Y.Z
USES_RE = re.compile(r"^\s*(?:-\s+)?uses:\s*(\S+?)\s*(?:#\s*(\S+).*)?$")
PINNED_RE = re.compile(r"^([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(/[^@]+)?@([0-9a-f]{40})$")
VERSION_RE = re.compile(r"^v\d+(\.\d+)*$")
EXPRESSION_RE = re.compile(r"\$\{\{\s*(.*?)\s*\}\}")
# Contexts that an attacker cannot set and that are safe to expand inside a shell script.
SAFE_RUN_CONTEXTS = ("matrix.", "env.", "runner.")


def check_labels(labels: object) -> tuple[list[str], set[str]]:
    """Return the errors of a parsed labels.yml and the set of label names."""
    errors: list[str] = []
    names: set[str] = set()
    if not isinstance(labels, list) or not labels:
        return ["labels.yml: expected a non-empty list of labels"], names
    seen: set[str] = set()
    for i, label in enumerate(labels):
        where = f"labels.yml[{i}]"
        if not isinstance(label, dict):
            errors.append(f"{where}: expected a mapping")
            continue
        unknown = set(label) - {"name", "color", "description", "from_name"}
        if unknown:
            errors.append(f"{where}: unknown keys {sorted(unknown)}")
        name = label.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 50:
            errors.append(f"{where}: name must be a non-empty string of at most 50 characters")
            continue
        where = f"labels.yml '{name}'"
        if name.lower() in seen:
            errors.append(f"{where}: duplicate name (label names are case-insensitive)")
        seen.add(name.lower())
        names.add(name)
        if not isinstance(label.get("color"), str) or not COLOR_RE.match(label["color"]):
            errors.append(f"{where}: color must be '#rrggbb'")
        desc = label.get("description")
        if not isinstance(desc, str) or not desc.strip() or len(desc) > 100:
            errors.append(
                f"{where}: description must be a non-empty string of at most 100 characters"
            )
    for missing in sorted(REQUIRED_LABELS - names):
        errors.append(f"labels.yml: missing the label '{missing}' (PLAN Section 10.2)")
    return errors, names


def _check_element(where: str, element: object, ids: set[str], labels: set[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(element, dict):
        return [f"{where}: expected a mapping"]
    kind = element.get("type")
    if kind not in ELEMENT_ATTRIBUTES:
        return [f"{where}: unknown type {kind!r}"]
    unknown = set(element) - {"type", "id", "attributes", "validations"}
    if unknown:
        errors.append(f"{where}: unknown keys {sorted(unknown)}")
    attrs = element.get("attributes")
    if not isinstance(attrs, dict):
        return errors + [f"{where}: missing attributes"]
    unknown = set(attrs) - ELEMENT_ATTRIBUTES[kind]
    if unknown:
        errors.append(f"{where}: unknown attributes {sorted(unknown)} for type {kind}")
    if kind == "markdown":
        if "id" in element or "validations" in element:
            errors.append(f"{where}: markdown elements take neither id nor validations")
        if not isinstance(attrs.get("value"), str) or not attrs["value"].strip():
            errors.append(f"{where}: markdown needs a non-empty value")
        return errors
    elem_id = element.get("id")
    if elem_id is not None:
        if not isinstance(elem_id, str) or not ID_RE.match(elem_id):
            errors.append(f"{where}: id {elem_id!r} may only contain letters, digits, '-' and '_'")
        elif elem_id in ids:
            errors.append(f"{where}: duplicate id {elem_id!r}")
        else:
            ids.add(elem_id)
    label = attrs.get("label")
    if not isinstance(label, str) or not label.strip():
        errors.append(f"{where}: needs a label")
    elif label in labels:
        errors.append(f"{where}: duplicate label {label!r}")
    else:
        labels.add(label)
    validations = element.get("validations", {})
    if not isinstance(validations, dict) or set(validations) - {"required"}:
        errors.append(f"{where}: validations may only contain 'required'")
    elif "required" in validations and not isinstance(validations["required"], bool):
        errors.append(f"{where}: validations.required must be true or false")
    if kind == "dropdown":
        options = attrs.get("options")
        if not isinstance(options, list) or not options:
            errors.append(f"{where}: dropdown needs a non-empty list of options")
        else:
            if not all(isinstance(o, str) and o.strip() for o in options):
                errors.append(f"{where}: dropdown options must be non-empty strings (quote yes/no)")
            elif len(set(options)) != len(options):
                errors.append(f"{where}: duplicate dropdown options")
            elif any(o.strip().lower() == "none" for o in options):
                errors.append(f"{where}: 'None' is a reserved dropdown option")
            default = attrs.get("default")
            if default is not None and not (
                isinstance(default, int) and 0 <= default < len(options)
            ):
                errors.append(f"{where}: default must be an index into options")
        if "multiple" in attrs and not isinstance(attrs["multiple"], bool):
            errors.append(f"{where}: multiple must be true or false")
    if kind == "checkboxes":
        options = attrs.get("options")
        if not isinstance(options, list) or not options:
            errors.append(f"{where}: checkboxes need a non-empty list of options")
        else:
            for j, option in enumerate(options):
                if not isinstance(option, dict) or not isinstance(option.get("label"), str):
                    errors.append(f"{where}.options[{j}]: needs a label")
                elif set(option) - {"label", "required"}:
                    errors.append(f"{where}.options[{j}]: unknown keys")
    if kind == "textarea" and "render" in attrs and not isinstance(attrs["render"], str):
        errors.append(f"{where}: render must be a language name")
    return errors


def check_form(name: str, form: object, known_labels: set[str]) -> list[str]:
    """Return the errors of one parsed issue form."""
    if not isinstance(form, dict):
        return [f"{name}: expected a mapping"]
    errors: list[str] = []
    unknown = set(form) - FORM_TOP_KEYS
    if unknown:
        errors.append(f"{name}: unknown top-level keys {sorted(unknown)}")
    for key in ("name", "description"):
        if not isinstance(form.get(key), str) or not form[key].strip():
            errors.append(f"{name}: '{key}' must be a non-empty string")
    applied = form.get("labels", [])
    if isinstance(applied, str):
        applied = [s.strip() for s in applied.split(",")]
    if not isinstance(applied, list):
        errors.append(f"{name}: labels must be a list")
        applied = []
    for label in applied:
        if label not in known_labels:
            errors.append(f"{name}: applies the label '{label}', which is not in labels.yml")
    body = form.get("body")
    if not isinstance(body, list) or not body:
        return errors + [f"{name}: body must be a non-empty list"]
    if all(isinstance(e, dict) and e.get("type") == "markdown" for e in body):
        errors.append(f"{name}: body needs at least one input element")
    ids: set[str] = set()
    labels: set[str] = set()
    for i, element in enumerate(body):
        errors += _check_element(f"{name} body[{i}]", element, ids, labels)
    return errors


def check_chooser(config: object) -> list[str]:
    """Return the errors of a parsed ISSUE_TEMPLATE/config.yml."""
    if not isinstance(config, dict):
        return ["config.yml: expected a mapping"]
    errors: list[str] = []
    if set(config) - {"blank_issues_enabled", "contact_links"}:
        errors.append("config.yml: unknown keys")
    if not isinstance(config.get("blank_issues_enabled", False), bool):
        errors.append("config.yml: blank_issues_enabled must be true or false")
    for i, link in enumerate(config.get("contact_links", []) or []):
        if not isinstance(link, dict) or set(link) != {"name", "url", "about"}:
            errors.append(f"config.yml contact_links[{i}]: needs exactly name, url and about")
        elif not str(link["url"]).startswith("https://"):
            errors.append(f"config.yml contact_links[{i}]: url must be https")
    return errors


def check_codeowners(text: str, root: Path) -> list[str]:
    """Return the errors of a CODEOWNERS file."""
    errors: list[str] = []
    has_catch_all = False
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        pattern, owners = parts[0], parts[1:]
        where = f"CODEOWNERS:{n}"
        if not owners:
            errors.append(f"{where}: '{pattern}' has no owner")
        for owner in owners:
            if not OWNER_RE.match(owner):
                errors.append(f"{where}: invalid owner '{owner}'")
        if pattern.startswith("!") or "[" in pattern or pattern.startswith("\\#"):
            errors.append(f"{where}: GitHub does not support '!', '[ ]' or '\\#' in patterns")
        if pattern == "*":
            has_catch_all = True
        elif pattern.startswith("/") and not any(c in pattern for c in "*?"):
            if not (root / pattern.lstrip("/")).exists():
                errors.append(f"{where}: '{pattern}' does not exist in the repository")
    if not has_catch_all:
        errors.append("CODEOWNERS: no catch-all '*' rule")
    return errors


def check_pr_template(text: str) -> list[str]:
    """Return the errors of the pull request template."""
    return [
        f"PULL_REQUEST_TEMPLATE.md: the checklist item {item} is missing (PLAN Section 8.9)"
        for item in PR_CHECKLIST
        if f"- [ ] {item}" not in text
    ]


def _triggers(workflow: dict) -> set[str]:
    # PyYAML (YAML 1.1) reads the key `on` as the boolean True.
    on = workflow.get("on", workflow.get(True))
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return {str(x) for x in on}
    if isinstance(on, dict):
        return {str(x) for x in on}
    return set()


def check_workflow(name: str, text: str) -> list[str]:
    """Return the least-privilege and pinning errors of one workflow file."""
    errors: list[str] = []
    try:
        workflow = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [f"{name}: not valid YAML ({exc})"]
    if not isinstance(workflow, dict):
        return [f"{name}: expected a mapping"]
    jobs = workflow.get("jobs")
    steps = [
        step
        for job in (jobs.values() if isinstance(jobs, dict) else [])
        if isinstance(job, dict)
        for step in job.get("steps") or []
        if isinstance(step, dict)
    ]
    runs_code = any(
        "run" in step or str(step.get("uses", "")).startswith("actions/checkout@") for step in steps
    )
    if "pull_request_target" in _triggers(workflow) and (runs_code or not steps):
        errors.append(
            f"{name}: pull_request_target runs with a privileged token (PLAN Section 8.8); it is "
            "allowed only in a workflow without checkout and without run steps; use pull_request"
        )
    perms = workflow.get("permissions")
    if perms is None:
        errors.append(f"{name}: no top-level 'permissions' (the default token may write)")
    elif isinstance(perms, str):
        errors.append(f"{name}: top-level permissions '{perms}'; list the scopes explicitly")
    elif isinstance(perms, dict):
        for scope, level in perms.items():
            if level not in ("read", "none"):
                errors.append(
                    f"{name}: top-level permission {scope}: {level}; grant write "
                    "permissions only to the job that needs them"
                )
    for n, line in enumerate(text.splitlines(), start=1):
        match = USES_RE.match(line)
        if not match or line.lstrip().startswith("#"):
            continue
        ref, comment = match.group(1).strip("'\""), match.group(2)
        if ref.startswith("./") or ref.startswith("docker://"):
            continue
        if not PINNED_RE.match(ref):
            errors.append(f"{name}:{n}: '{ref}' is not pinned to a full commit SHA")
        elif not comment or not VERSION_RE.match(comment):
            errors.append(f"{name}:{n}: '{ref}' needs a '# vX.Y.Z' comment naming its tag")
    if not isinstance(jobs, dict) or not jobs:
        return errors + [f"{name}: no jobs"]
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
        if "uses" not in job and "timeout-minutes" not in job:
            errors.append(
                f"{name} jobs.{job_id}: no 'timeout-minutes' (a hung step would hold the runner "
                "for GitHub's 6-hour default)"
            )
        for i, step in enumerate(job.get("steps") or []):
            if not isinstance(step, dict):
                continue
            where = f"{name} jobs.{job_id}.steps[{i}]"
            uses = str(step.get("uses", ""))
            if uses.startswith("actions/checkout@"):
                with_ = step.get("with") or {}
                if with_.get("persist-credentials") is not False:
                    errors.append(
                        f"{where}: actions/checkout needs 'persist-credentials: false' "
                        "(the token would stay in .git/config)"
                    )
            run = step.get("run")
            if isinstance(run, str):
                for expr in EXPRESSION_RE.findall(run):
                    if not expr.startswith(SAFE_RUN_CONTEXTS):
                        errors.append(
                            f"{where}: '${{{{ {expr} }}}}' is expanded into the script; "
                            "pass it through 'env:' instead"
                        )
    return errors


MICROMAMBA_RE = re.compile(r"^\s*micromamba-version:\s*(\S+)\s*$", re.MULTILINE)
CMAKE_SPEC_RE = re.compile(r"[\"']?(cmake[<>=!~][^\"'\s]*)[\"']?")


def check_tool_pins(workflows: dict[str, str], environment: str) -> list[str]:
    """Return the errors of tool versions pinned in several files (dependabot updates none)."""
    errors: list[str] = []
    micromamba = {
        (name, version)
        for name, text in workflows.items()
        for version in MICROMAMBA_RE.findall(text)
    }
    if len({version for _, version in micromamba}) > 1:
        errors.append(
            "tool pins: different micromamba-version values: "
            + ", ".join(f"{n}={v}" for n, v in sorted(micromamba))
        )
    env_cmake = re.findall(r"^\s*-\s*(cmake[<>=!~]\S*)\s*$", environment, re.MULTILINE)
    if len(env_cmake) != 1:
        return errors + ["tool pins: environment.yml has no single 'cmake' requirement"]
    for name, text in sorted(workflows.items()):
        for line in text.splitlines():
            if "pip install" not in line:
                continue
            for spec in CMAKE_SPEC_RE.findall(line):
                if spec != env_cmake[0]:
                    errors.append(
                        f"tool pins: {name} installs '{spec}', environment.yml has '{env_cmake[0]}'"
                    )
    return errors


def pinned_actions(root: Path) -> dict[tuple[str, str], set[str]]:
    """Return {(owner/repo, version tag): {pinned SHAs}} over every workflow."""
    pins: dict[tuple[str, str], set[str]] = {}
    for path in sorted((root / ".github" / "workflows").glob("*.y*ml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = USES_RE.match(line)
            if not match or not match.group(2):
                continue
            pinned = PINNED_RE.match(match.group(1).strip("'\""))
            if pinned:
                pins.setdefault((pinned.group(1), match.group(2)), set()).add(pinned.group(3))
    return pins


def verify_pins(root: Path) -> list[str]:
    """Check with git ls-remote that each pinned SHA is the commit its version tag names."""
    errors: list[str] = []
    for (repo, tag), shas in sorted(pinned_actions(root).items()):
        url = f"https://github.com/{repo}"
        try:
            out = subprocess.run(
                ["git", "ls-remote", url, f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"pins: git ls-remote {url} failed ({exc})")
            continue
        refs = dict(reversed(line.split("\t")) for line in out.splitlines() if "\t" in line)
        # An annotated tag is peeled ('^{}') to the commit it names.
        commit = refs.get(f"refs/tags/{tag}^{{}}", refs.get(f"refs/tags/{tag}"))
        if commit is None:
            errors.append(f"pins: {repo} has no tag {tag}")
            continue
        for sha in sorted(shas):
            if sha != commit:
                errors.append(f"pins: {repo}@{sha} is not {tag} ({commit})")
            else:
                print(f"github_meta_check: {repo} {tag} = {sha}")
    return errors


def check_repository(root: Path) -> list[str]:
    """Run every check on the files of the repository at `root`."""
    gh = root / ".github"
    labels_doc = yaml.safe_load((gh / "labels.yml").read_text(encoding="utf-8"))
    errors, known = check_labels(labels_doc)
    forms_dir = gh / "ISSUE_TEMPLATE"
    present = {p.name for p in forms_dir.glob("*.yml")} - {"config.yml"}
    for extra in sorted(p.name for p in forms_dir.glob("*.yaml")):
        errors.append(f"ISSUE_TEMPLATE/{extra}: use the .yml extension")
    for missing in sorted(REQUIRED_FORMS - present):
        errors.append(f"ISSUE_TEMPLATE/{missing}: missing issue form")
    for name in sorted(present):
        form = yaml.safe_load((forms_dir / name).read_text(encoding="utf-8"))
        errors += check_form(name, form, known)
    errors += check_chooser(yaml.safe_load((forms_dir / "config.yml").read_text(encoding="utf-8")))
    errors += check_codeowners((gh / "CODEOWNERS").read_text(encoding="utf-8"), root)
    errors += check_pr_template((gh / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8"))
    workflows = sorted((gh / "workflows").glob("*.y*ml"))
    if not workflows:
        errors.append("workflows: no workflow files")
    texts = {f"workflows/{path.name}": path.read_text(encoding="utf-8") for path in workflows}
    for name, text in texts.items():
        errors += check_workflow(name, text)
    errors += check_tool_pins(texts, (root / "environment.yml").read_text(encoding="utf-8"))
    return errors


def self_test(root: Path) -> list[str]:
    """Mutate the real files in memory and make sure every mutation is rejected."""
    gh = root / ".github"
    labels = yaml.safe_load((gh / "labels.yml").read_text(encoding="utf-8"))
    bug = yaml.safe_load((gh / "ISSUE_TEMPLATE" / "bug.yml").read_text(encoding="utf-8"))
    _, known = check_labels(labels)
    failures: list[str] = []

    def expect(what: str, errors: list[str]) -> None:
        if not errors:
            failures.append(f"self-test: {what} was accepted")

    def mutated_labels(fn):
        x = copy.deepcopy(labels)
        fn(x)
        return check_labels(x)[0]

    def mutated_form(fn):
        x = copy.deepcopy(bug)
        fn(x)
        return check_form("bug.yml", x, known)

    def first(form, kind):
        return next(e for e in form["body"] if e["type"] == kind)

    expect(
        "a duplicate label",
        mutated_labels(lambda x: x.append(dict(x[0], name=x[0]["name"].upper()))),
    )
    expect("a bad colour", mutated_labels(lambda x: x[0].update(color="d73a4a")))
    expect("a long description", mutated_labels(lambda x: x[0].update(description="x" * 101)))
    expect(
        "a missing plan label",
        mutated_labels(lambda x: x.remove(next(y for y in x if y["name"] == "ci:gpu"))),
    )
    expect("a form without name", mutated_form(lambda x: x.pop("name")))
    expect("an unknown element type", mutated_form(lambda x: x["body"][1].update(type="textbox")))
    expect("an invalid id", mutated_form(lambda x: x["body"][1].update(id="what happened!")))
    expect("a duplicate id", mutated_form(lambda x: x["body"][2].update(id=x["body"][1]["id"])))
    expect(
        "a duplicate element label",
        mutated_form(
            lambda x: x["body"][2]["attributes"].update(label=x["body"][1]["attributes"]["label"])
        ),
    )
    expect(
        "a reserved dropdown option",
        mutated_form(lambda x: first(x, "dropdown")["attributes"]["options"].append("None")),
    )
    expect(
        "a boolean dropdown option",
        mutated_form(lambda x: first(x, "dropdown")["attributes"]["options"].append(True)),
    )
    expect(
        "an unknown attribute",
        mutated_form(lambda x: x["body"][1]["attributes"].update(options=["a"])),
    )
    expect("an unknown label", mutated_form(lambda x: x["labels"].append("no-such-label")))
    expect(
        "a chooser http link",
        check_chooser({"contact_links": [{"name": "a", "url": "http://x", "about": "b"}]}),
    )
    expect("an owner without @", check_codeowners("* SMShovan\n", root))
    expect("a negated pattern", check_codeowners("* @a\n!/docs/ @a\n", root))
    expect("a missing path", check_codeowners("* @a\n/no/such/dir/ @a\n", root))
    expect("no catch-all rule", check_codeowners("/docs/ @a\n", root))
    expect("a PR template without the DCO item", check_pr_template("- [ ] **Tests:**\n"))
    docs = (gh / "workflows" / "docs.yml").read_text(encoding="utf-8")
    checkout = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1"
    assert checkout in docs and "persist-credentials: false" in docs, "self-test fixture moved"
    expect(
        "a workflow without permissions", check_workflow("w", docs.replace("permissions:", "x:"))
    )
    expect(
        "a top-level write permission",
        check_workflow("w", docs.replace("contents: read", "contents: write", 1)),
    )
    expect(
        "write-all permissions",
        check_workflow(
            "w", re.sub(r"(?m)^permissions:\n  contents: read", "permissions: write-all", docs)
        ),
    )
    expect(
        "a pull_request_target trigger",
        check_workflow("w", docs.replace("  pull_request:\n", "  pull_request_target:\n")),
    )
    expect(
        "an action pinned to a tag",
        check_workflow("w", docs.replace(checkout, "actions/checkout@v7.0.1")),
    )
    expect(
        "an abbreviated SHA",
        check_workflow("w", docs.replace(checkout, "actions/checkout@3d3c42e  # v7.0.1")),
    )
    expect(
        "a pin without its version comment",
        check_workflow("w", docs.replace(checkout, checkout.split("  #")[0])),
    )
    expect(
        "a checkout that keeps the token",
        check_workflow(
            "w", docs.replace("persist-credentials: false", "persist-credentials: true")
        ),
    )
    expect(
        "an untrusted expression in a script",
        check_workflow(
            "w",
            docs.replace(
                "run: ci/docs.sh\n", "run: echo ${{ github.event.pull_request.title }}\n", 1
            ),
        ),
    )
    expect(
        "a job without timeout-minutes",
        check_workflow("w", docs.replace("    timeout-minutes: 30\n", "", 1)),
    )
    welcome = (gh / "workflows" / "welcome.yml").read_text(encoding="utf-8")
    greeter = "      - uses: actions/first-interaction@"
    assert greeter in welcome and not check_workflow("w", welcome), "self-test fixture moved"
    expect(
        "pull_request_target with a checkout",
        check_workflow(
            "w",
            welcome.replace(
                greeter,
                f"      - uses: {checkout}\n        with:\n          persist-credentials: false\n"
                + greeter,
            ),
        ),
    )
    expect(
        "pull_request_target with a run step",
        check_workflow("w", welcome.replace(greeter, "      - run: make\n" + greeter)),
    )
    environment = (root / "environment.yml").read_text(encoding="utf-8")
    cpu = (gh / "workflows" / "cpu.yml").read_text(encoding="utf-8")
    assert "micromamba-version: 2.9.0-0" in docs and "cmake>=" in cpu, "self-test fixture moved"
    expect(
        "two micromamba versions",
        check_tool_pins(
            {"a": docs, "b": docs.replace("version: 2.9.0-0", "version: 2.8.0-0")},
            environment,
        ),
    )
    expect(
        "a CMake requirement that differs from environment.yml",
        check_tool_pins({"cpu": re.sub(r"cmake>=[^\"']*", "cmake>=3.28", cpu)}, environment),
    )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--self-test", action="store_true", help="first check that broken inputs are rejected"
    )
    parser.add_argument(
        "--verify-pins",
        action="store_true",
        help="also check each pinned action SHA against its version tag (network)",
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="repository root (default: this checkout)"
    )
    args = parser.parse_args()
    errors = self_test(args.root) if args.self_test else []
    errors += check_repository(args.root)
    if args.verify_pins:
        errors += verify_pins(args.root)
    for error in errors:
        print(f"github_meta_check: {error}", file=sys.stderr)
    if errors:
        return 1
    print(
        "github_meta_check: OK (labels, issue forms, chooser, CODEOWNERS, PR template, workflows"
        + (", action pins" if args.verify_pins else "")
        + ")"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
