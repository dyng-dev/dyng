# Security policy

## Supported versions

dynG is alpha: 0.1.0rc1, the release candidate of the first release, is on TestPyPI only. The
PyPI package `dyng` 0.0.1 is only a name reservation and contains no library code. From 0.1.0 on, security fixes go into the **latest
minor release** (as a new patch release) and into `main`; older minor releases are not
patched.

| Version | Supported |
|---|---|
| `main` and the 0.1.0 release candidates | yes |
| 0.0.1 (name reservation, no library code) | no code to fix |
| latest minor release (from 0.1.0 on) | yes |
| older minor releases | no; please upgrade |

## Reporting a vulnerability

Please **do not open a public issue, discussion or pull request** for a security problem.
Report it privately, by either of these ways:

- **GitHub private vulnerability reporting** (preferred): open
  <https://github.com/dyng-dev/dyng/security/advisories/new>, or go to the repository's
  **Security** tab and click **Report a vulnerability**. The report is visible only to you and
  the maintainers.
- **E-mail** to the lead maintainer: sm.shovan@gmail.com, with "dynG security" in the subject.

Please include:

- the affected version or commit, and how you installed dynG (source build, wheel);
- the backend (sequential, OpenMP, CUDA), the operating system, the compiler, and for CUDA the
  GPU, the driver and the CUDA version;
- a minimal reproducer (input files and the call or command) and what you observed (crash,
  out-of-bounds access, memory growth, sanitizer report);
- whether the problem is already known to others, and how you would like to be credited.

## What happens next (disclosure process)

1. **Acknowledgement within 7 days.** A maintainer confirms that the report was received and
   who handles it.
2. **Assessment.** We reproduce the problem, decide whether it is a vulnerability (see
   *Scope*), and estimate its severity (CVSS). We tell you the outcome and keep you informed at
   least every 14 days until the report is closed.
3. **Fix in private.** The fix is prepared in a GitHub security advisory's temporary private
   fork, with a regression test. You are invited to review it.
4. **Coordinated disclosure.** We agree on a disclosure date with you, normally when the fixed
   release is available and no later than **90 days** after the report (shorter if the problem
   is being exploited, longer only with your agreement). On that date we publish the patch
   release, the GitHub security advisory (with a CVE requested through GitHub where one is
   warranted) and a `Security` entry in `CHANGELOG.md`.
5. **Credit.** Reporters are credited in the advisory unless they ask not to be.

If a report turns out not to be a vulnerability, we say why and, with your agreement, move it
to a normal issue.

## Scope

In scope:

- memory-safety problems (crash, out-of-bounds read or write, use after free, unbounded memory
  use) in the library, the command-line tools or the Python package, including on **malformed
  or hostile input files** (Matrix Market, edge lists, the MOSP text formats, batch files);
- problems in the build, release and CI configuration of this repository that could let
  someone tamper with published artifacts (for example the GitHub workflows or the PyPI
  Trusted Publishing setup).

Out of scope:

- results that are merely wrong on valid input: these are bugs, please report them as normal
  issues (they matter to us just as much);
- resource use that is proportional to a legitimately large input;
- vulnerabilities in third-party software (the CUDA toolkit, the driver, compilers, Python),
  which should be reported to their vendors; tell us if dynG needs to react.
