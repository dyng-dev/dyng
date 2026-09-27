# Reserving `dyng` on PyPI (checkpoint A3)

Approved on 2026-09-27 (see `GOVERNANCE.md`). **Done on 2026-09-27:** `dyng` 0.0.1 is on
[PyPI](https://pypi.org/project/dyng/0.0.1/) and [TestPyPI](https://test.pypi.org/project/dyng/0.0.1/)
(uploaded 16:26 UTC by `release.yml`, run for tag `v0.0.1` at commit `15a6051`). The first run
of the workflow failed in the TestPyPI publish step (the reviewers read `invalid-publisher` in
its log: the pending trusted publisher did not match yet); a second run five minutes later
published to both indexes. The steps below are kept as the record of what was set up.

## What is published

`tools/name_reservation/` holds a real, minimal pure-Python package `dyng` 0.0.1 (`__version__`,
`status()`, a README, Apache-2.0 metadata, project URLs pointing to
<https://github.com/dyng-dev/dyng>). It is kept out of the repository root so that it never
conflicts with the real package, whose `pyproject.toml` (scikit-build-core) arrives at the root
in milestone M5.

Local check (uploads nothing):

```bash
source scripts/dev_env.sh
python -m build --outdir /tmp/dyng-dist tools/name_reservation
twine check --strict /tmp/dyng-dist/*
```

## One-time setup (lead maintainer; done 2026-09-27)

1. On GitHub, in `dyng-dev/dyng` → Settings → Environments, create the environments
   **`testpypi`** and **`pypi`** (optionally with "required reviewers" for `pypi`).
2. On <https://test.pypi.org/manage/account/publishing/> add a *pending trusted publisher*:
   project `dyng`, owner `dyng-dev`, repository `dyng`, workflow `release.yml`,
   environment `testpypi`.
3. On <https://pypi.org/manage/account/publishing/> add the same pending publisher with
   environment `pypi`.
4. Push the tag: `git tag -a v0.0.1 -m "dyng 0.0.1: name reservation" && git push origin v0.0.1`.
   `release.yml` builds the package, checks that the tag equals its version, publishes to
   TestPyPI and then to PyPI.

## Later releases

`release.yml` is bound to the trusted publishers by its file name and the environment names, so
both stay unchanged. From M5 on, a tag equal to the `VERSION` file builds the real
distributions (the `build` job's selection step); pre-release tags (`a`, `b`, `rc`, `dev`) are
published to TestPyPI only.
