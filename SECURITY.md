# Security policy

## Supported versions

dynG is pre-alpha: nothing has been released yet (the PyPI package `dyng` 0.0.1 is only a name
reservation and contains no library code). From 0.1.0 on, security fixes go into the latest
minor release only.

| Version | Supported |
|---|---|
| `main` (pre-alpha) | yes |
| 0.0.1 (name reservation) | no code to fix |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it privately:

- through GitHub's private vulnerability reporting for `dyng-dev/dyng`
  ("Security" tab -> "Report a vulnerability"), or
- by e-mail to the lead maintainer, sm.shovan@gmail.com.

Include the affected commit or version, the backend (sequential, OpenMP, CUDA), a minimal
reproducer (input files and the call or command) and what you observed. We aim to acknowledge a
report within 7 days and to agree on a disclosure date with you once a fix is ready.

## Scope

dynG reads graph, batch and result files (Matrix Market, the MOSP text formats); a crash, an
out-of-bounds access or unbounded memory use on a malformed input file is in scope. Results that
are merely wrong on valid input are bugs: please report them as normal issues.
