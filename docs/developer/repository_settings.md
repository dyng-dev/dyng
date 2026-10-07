# Repository settings (lead maintainer)

Some parts of the project infrastructure are **account-level settings** that no file in the
repository can make: they have to be clicked by an owner of the `dyng-dev` organization. This
page lists every such setting, in the order in which they should be made, each with the exact
click path (GitHub web interface as of September 2026; GitHub occasionally renames an entry, and
each section links the GitHub documentation). PLAN Sections 8.8, 8.9 and 10.1 define what is
required; Appendix E and `GOVERNANCE.md` record the decisions.

"Settings" always means the **repository** settings of `dyng-dev/dyng`
(<https://github.com/dyng-dev/dyng/settings>) unless the step says **organization** settings
(<https://github.com/organizations/dyng-dev/settings>).

## Checklist

"Done" means applied by the lead maintainer (through the GitHub API, 2026-09-27) and checked
against the read-only API on 2026-09-28 (`gh api repos/dyng-dev/dyng`, its `actions/permissions`,
`rulesets`, `environments` and security endpoints, and `gh api orgs/dyng-dev`). The state of
2026-09-30 (the `DCO` check required, the signing key) and of 2026-10-01 (M5 merged, the 24
required checks, `pypa/cibuildwheel` on the Actions allow-list) was reported by the author and is
recorded in the approvals log of `GOVERNANCE.md`.

| # | Setting | When | Status |
|---|---|---|---|
| 1 | Organization: two-factor authentication, base permissions | now | **done** (2FA required, base permission Read; the optional restriction of repository creation, 1.3, is not set) |
| 2 | About: description, website, topics | now | **done** (description and the 14 topics; no website until step 13) |
| 3 | Features: Issues, Discussions, no wiki | now | **done** (Issues and Discussions on; wiki and projects off) |
| 4 | Pull requests: merge commits and squash merges, no rebase (ADR 0019) | now | **done** |
| 5 | Actions: permissions, SHA pinning, fork approval | now | **done** (allow-list, `pypa/cibuildwheel@*` included since 2026-09-30; SHA pinning required, approval for all external contributors, read-only workflow token) |
| 6 | Security: private vulnerability reporting, Dependabot, secret scanning | now | **done** (private vulnerability reporting; Dependabot alerts and security updates; secret scanning with push protection; CodeQL, optional, not set up) |
| 7 | The DCO app and web sign-off | now | **done** (DCO app installed on `dyng-dev/dyng` by the author, integration id 1861; web sign-off required); `DCO` is a **required check** of the `main` ruleset since 2026-09-30. The author's membership of `dyng-dev` is public and the maintainer's commits are SSH-signed (the signing key "dynG commit signing", registered on the author's account on 2026-09-29), so the app exempts them as signed commits of an organization member |
| 8 | Labels | after the M4 merge | pending: the `labels` workflow applies `.github/labels.yml` on the first push of `main` with the M4 merge |
| 9 | Ruleset for `main`: pull requests and required checks | after every required workflow has run once on a pull request | **done** after pull request #1 (INT1, merged with the merge commit `eda8b8b`): ruleset `main` (id 24131378), strict; 17 required checks, **18** with `DCO` from 2026-09-30, and **24** after pull request #4 (M5, merge commit `8842acf`): `scaffold`, the four checks of `python.yml` and `Python API (griffe)` added (2026-10-01) |
| 10 | Ruleset for release tags | now | **done** (`release tags`: `v*`, creation, update, deletion and force pushes restricted; repository admins bypass; the tags `v0.1.0rc1` and `v0.1.0` may be pushed by the AI assistant on the author's behalf, GOVERNANCE.md 2026-09-30) |
| 11 | Environments `pypi` and `testpypi`: protection rules | now | **done** (`pypi`: required reviewer SMShovan, tags `v*` only, no administrator bypass; `testpypi`: tags `v*` only) |
| 11b | Environments of the CUDA plugins `testpypi-cu12`, `pypi-cu12`, `testpypi-cu13`, `pypi-cu13`: protection rules (release 0.2.0; ADR 0032) | before the first `v0.2.0*` tag | the pending trusted publishers of `dyng-cu12` / `dyng-cu13` exist (set up by the author for 0.2.0, PLAN Appendix F; workflow `release.yml`, one environment pair per project); the protection rules of the four environments, as in section 11, are for the author to set and confirm |
| 12 | Discussions: categories and the pinned roadmap | after step 3 | open (the default categories, Polls included; no pinned roadmap yet) |
| 13 | Read the Docs | later: checkpoint A4, milestone M6 | later |
| 14 | Zenodo | later: checkpoint A4, milestone M6 (PLAN 11.2); **not connected for 0.1.0**, so 0.1.0 has no DOI (the author, 2026-10-01; section 14) | later |

## 1. Organization security

PLAN Section 10.1 requires two-factor authentication for the organization.

1. Organization settings → sidebar **Security** → **Authentication security** → tick
   **Require two-factor authentication for everyone in the dyng-dev organization** → **Save**.
   (Members without 2FA are removed, so enable 2FA on your own account first:
   your avatar → **Settings** → **Password and authentication**.)
2. Organization settings → sidebar **Access** → **Member privileges** → **Base permissions**:
   **Read** (members get write access per repository, through teams or invitations).
3. Organization settings → **Member privileges** → **Repository creation**: untick
   **Public** and **Private** (only owners create repositories).

Docs: <https://docs.github.com/en/organizations/keeping-your-organization-secure>.

## 2. About: description, website, topics

1. Open <https://github.com/dyng-dev/dyng>. In the right column, click the gear icon next to
   **About**.
2. **Description:**
   `Dynamic graph and hypergraph algorithms on GPUs: keep shortest paths, cycle counts, triads and labels up to date under batch updates (C++17/CUDA, OpenMP; Python planned).`
3. **Website:** leave empty until Read the Docs is connected (step 13), then
   `https://dyng.readthedocs.io`.
4. **Topics** (type each and press Enter): `dynamic-graphs`, `graph-algorithms`,
   `hypergraphs`, `gpu`, `cuda`, `openmp`, `cpp17`, `hpc`, `shortest-paths`,
   `cycle-counting`, `label-propagation`, `graph-analytics`, `batch-updates`, `python`.
5. Under **Include in the home page**, keep **Releases** ticked; untick **Packages** and
   **Deployments** (the PyPI environment deployments would otherwise be shown).
6. **Save changes**.

The citation button ("Cite this repository") appears automatically from `CITATION.cff`; the
security policy, Code of Conduct, contributing guide and support file are detected from the
root files (check the **Insights** → **Community standards** page: every item should be ticked).

## 3. Features: Issues, Discussions, wiki

Settings → **General** → section **Features**:

1. **Wikis:** untick (the documentation lives in `docs/`).
2. **Issues:** keep ticked. The issue forms and `config.yml` in `.github/ISSUE_TEMPLATE/` are
   used automatically.
3. **Sponsorships:** leave unticked.
4. **Discussions:** tick, then click **Set up discussions**. GitHub opens a draft welcome
   post: write two sentences (what dynG is, where the roadmap is, that questions go here) and
   click **Start discussion**. The categories are set in step 12.
5. **Projects:** optional (untick if not used).

Docs: <https://docs.github.com/en/discussions/quickstart>.

## 4. Pull requests: merge commits and squash merges, no rebase

The merge policy (ADR 0019, `GOVERNANCE.md`): pull requests of **milestone and integration
work** are merged with a **merge commit**, so the commit SHAs that parity certificates,
retrospectives and ADRs cite stay reachable from `main`; **external contributions** are
**squash-merged**, and their title (Conventional Commits) becomes the commit subject; **rebase
merging** is disabled. (This replaces the plan's "squash merges only", PLAN Section 8.9.)

Settings → **General** → section **Pull Requests**:

1. Tick **Allow merge commits**; **Default commit message**: **Pull request title and
   description**.
2. Tick **Allow squash merging**; **Default commit message**: **Pull request title and
   description**.
3. Untick **Allow rebase merging**.
4. Tick **Always suggest updating pull request branches**.
5. Tick **Allow auto-merge** (optional; merges once the required checks pass).
6. Tick **Automatically delete head branches**.

Each change is saved immediately.

## 5. Actions

Settings → sidebar **Actions** → **General**:

1. **Actions permissions:** select **Allow dyng-dev, and select non-dyng-dev, actions and
   reusable workflows**; tick **Allow actions created by GitHub**, untick **Allow actions by
   Marketplace verified creators** (least privilege: every other action is listed by name);
   in **Allow specified actions and reusable workflows** enter:

   ```text
   pypa/gh-action-pypi-publish@*,
   pypa/cibuildwheel@*,
   crazy-max/ghaction-github-labeler@*,
   mamba-org/setup-micromamba@*
   ```

   (`pypa/cibuildwheel` is new in M5, for `wheels.yml`; added to the setting by the maintainer
   through the API on 2026-09-30, before the M5 pull request ran.)

   and tick **Require actions to be pinned to a full-length commit SHA** (every workflow in
   `.github/workflows/` already is; dependabot keeps the pins current). **Save**.
   When a workflow adds an action from another owner, add it to this list in the same pull
   request.
2. **Fork pull request workflows from outside collaborators:** select **Require approval for
   all external contributors** (PLAN Sections 8.8 and 10.1). **Save**.
3. **Workflow permissions:** select **Read repository contents and packages permissions**, and
   untick **Allow GitHub Actions to create and approve pull requests**. **Save**. (Every
   workflow sets its own least-privilege `permissions:`; jobs that need more, like
   `labels.yml` or `release.yml`, ask for it per job.)

Docs: <https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository>.

## 6. Security

Settings → sidebar section **Security and quality** → **Advanced Security**:

1. **Private vulnerability reporting:** click **Enable** (Appendix E, decision O13;
   `SECURITY.md` points reporters to it). Reporters then see **Report a vulnerability** on the
   repository's **Security** tab.
2. **Dependency graph:** on (always on for public repositories).
3. **Dependabot alerts:** **Enable**. **Dependabot security updates:** **Enable**. (Version
   updates are configured by `.github/dependabot.yml`.)
4. **Secret Protection** (free for public repositories): **Enable**, and **Push protection**:
   **Enable**.
5. **Code scanning** → **CodeQL analysis:** optional; **Set up** → **Default** → **Enable
   CodeQL** scans C/C++ without a build, Python and the workflows.

To receive the reports, make sure your notification settings deliver security alerts: your
avatar → **Settings** → **Notifications** → **Security alerts**.

Docs: <https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/configure-vulnerability-reporting/configure-for-a-repository>.

## 7. The DCO app and web sign-off

External contributors sign off their commits (CONTRIBUTING.md); the DCO app checks every pull
request and reports a check named **DCO**. `.github/dco.yml` exempts members of `dyng-dev`.

1. Open <https://github.com/apps/dco> → **Install** (or **Configure** if it is installed
   elsewhere).
2. Choose the **dyng-dev** organization.
3. Select **Only select repositories** → choose **dyng-dev/dyng** → **Install**.
4. After the next pull request, the check **DCO** appears; add it to the required checks
   (step 9) once the maintainers' membership of `dyng-dev` is public. The app exempts only
   *public* members (`require.members: false`), and only for **signed** commits, so a
   maintainer's commits carry no `Signed-off-by` line but must be signed (SSH or GPG, with the
   key registered as a signing key on their GitHub account).
   **Status (2026-09-30): done.** `DCO` is a required check of the `main` ruleset (id 24131378;
   the check's source is the DCO app, integration id 1861), the 18th. The author's membership is
   public (2026-09-29) and every commit of the maintainer is SSH-signed with the key "dynG commit
   signing", registered as a signing key on the author's account on 2026-09-29; the local
   repository signs automatically (`commit.gpgsign=true`, `gpg.format=ssh`). External
   contributors sign off (`git commit -s`), as CONTRIBUTING.md says.
5. So that contributors who fix a typo in GitHub's web editor pass the check: Settings →
   **General** → in the first section (below **Template repository**) tick **Require
   contributors to sign off on web-based commits**. GitHub then adds the `Signed-off-by` line
   to every commit made in the web interface (CONTRIBUTING.md says so).

`.github/dco.yml` also allows *remediation commits*: a contributor who forgot the sign-off
during a review adds an empty signed-off commit instead of rewriting the branch
(CONTRIBUTING.md, "Fixing a missing sign-off").

Docs: <https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-the-commit-signoff-policy-for-your-repository>.

## 8. Labels

`.github/labels.yml` defines the labels (docs/developer/labels.md). The `labels` workflow
applies it automatically on every push to `main` that changes the file, so after the M4 merge
the labels exist without any click. To apply it by hand: **Actions** tab → workflow **labels**
→ **Run workflow** → branch `main` → **Run workflow**. Do not edit labels in the web
interface: the next run resets listed labels to the file.

## 9. Ruleset for `main`

PLAN Section 8.9: everything goes through a pull request and the required checks must be green;
the merge methods follow ADR 0019 (merge commits for milestone and integration work, squash
merges for contributions). While the lead maintainer is the only maintainer, **zero approvals**
are required (the author merges their own pull request once the checks pass); from the day a
second maintainer exists, raise it to one.

**Status: done (checked 2026-09-28; `DCO` added 2026-09-30; the M5 checks added 2026-10-01).**
The ruleset `main` (id 24131378)
was created after pull request #1 (INT1, merged with the merge commit `eda8b8b`) had run every
workflow: pull request required (0 approvals, stale approvals dismissed, conversation resolution
required, merge methods merge and squash), the required checks of the table below (4 `lint`,
10 `cpu` (the 9 matrix jobs and `scaffold`), 3 `cuda-build`, 1 `docs`, 4 `python`, 1 `api-check`
and `DCO`: **24** in all; 17 at creation, 18 with `DCO` from 2026-09-30, 24 with the M5 checks
from 2026-10-01) with **Require branches to be up to date** (strict), deletions and force pushes blocked. The **Repository admin** role is
on the bypass list in the mode **For pull requests only**: an administrator can merge a pull
request whose checks are not green, but cannot push to `main` directly. GitHub offers a
required check in the ruleset only after it has run in the last 7 days, which is why the
ruleset had to wait for the first pull request. The steps below record how it was made.

1. Settings → sidebar section **Code and automation** → **Rules** → **Rulesets** →
   **New ruleset** → **New branch ruleset**.
2. **Ruleset name:** `main`. **Enforcement status:** **Active**.
3. **Bypass list:** **Add bypass** → **Repository admin** → mode **For pull requests only** (see
   the note below).
4. **Target branches** → **Add target** → **Include default branch**.
5. Tick these rules:
   - **Restrict deletions**.
   - Leave **Require linear history** unticked: milestone merges are merge commits (ADR 0019).
   - **Require a pull request before merging**, and open its options:
     - **Required approvals:** `0` now; `1` once there is a second maintainer.
     - **Dismiss stale pull request approvals when new commits are pushed:** tick.
     - **Require review from Code Owners:** untick now; tick with the second maintainer.
     - **Require approval of the most recent reviewable push:** untick now; tick with the
       second maintainer.
     - **Require conversation resolution before merging:** tick.
     - **Allowed merge methods:** **Merge** and **Squash** (not **Rebase**).
   - **Require status checks to pass**, and open its options:
     - **Require branches to be up to date before merging:** tick.
     - **Add checks** → type each name below and pick the entry whose source is
       **GitHub Actions** (for **DCO**, the source is the DCO app).
   - **Block force pushes**.
6. **Create**.

Required checks (the job names as they appear in a pull request's checks list):

| Workflow | Check names |
|---|---|
| `lint.yml` | `pre-commit`, `tidy`, `harness`, `name-reservation` |
| `cpu.yml` | `cpu-only / gcc-12 / openmp=ON`, `cpu-only / gcc-13 / openmp=ON`, `cpu-only / clang-18 / openmp=ON`, `cpu-only / gcc-13 / openmp=OFF`, `cpu-only / clang-17 / openmp=ON`, `dev / gcc-12 / openmp=ON`, `dev / gcc-13 / openmp=ON`, `dev / clang-18 / openmp=ON`, `dev / clang-18 / openmp=OFF`, `scaffold` (required since 2026-10-01) |
| `cuda-build.yml` | `CUDA 13.1.1 (ci-cuda13)`, `CUDA 13.4.1 (ci-cuda13)`, `CUDA 12.9.2 (ci-cuda12)` (the job `compile`, named `CUDA <toolkit> (<preset>)` per matrix entry; compile-only, no GPU) |
| `docs.yml` | `site` (not `external-links`, which runs only weekly and on demand) |
| `python.yml` (M5; required since 2026-10-01) | `Editable install, stubs, mypy, pytest (gcc)`, `Editable install, stubs, mypy, pytest (clang-18)`, `From the sdist (Python 3.12)`, `From the sdist (Python 3.13)` |
| `api-check.yml` (M5; required since 2026-10-01) | `Python API (griffe)` |
| DCO app (integration id 1861) | `DCO` (required since 2026-09-30; step 7) |

Do **not** require checks of workflows that run only for some files (`labels.yml`, and
`wheels.yml`, which runs on pull requests only when they touch the packaging or the library), only on
tags (`release.yml`), only on some events (`welcome.yml`, the first-interaction greeting) or
only on a schedule (the `external-links` job of `docs.yml`): a required check that never runs
blocks every pull request. When a
workflow is added (`python`, `api-check`; PLAN Section 8.8), a job of the `cpu` matrix is
renamed, or a toolkit of the `cuda-build` matrix is bumped (its version is part of the check
name), update this list and the ruleset in the same pull request.

**Note on bypassing.** With the Repository admin role allowed to bypass **for pull requests
only**, nobody can push to `main` directly, including the organization owners: milestone and
integration work reaches `main` through pull requests, merged with a merge commit (ADR 0019;
the AI assistant can open them). The bypass only lets an administrator merge a pull request
whose required checks are not all green (for example when a hosted runner is down); record
every such merge in the pull request. An empty bypass list would remove that escape hatch; mode
**Always allow** would re-open direct pushes and must then be recorded in `GOVERNANCE.md`.

Docs: <https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository>.

## 10. Ruleset for release tags

A tag `v*` publishes to TestPyPI and PyPI (`release.yml`), so only the lead maintainer may
create one, or, for the tags `v0.1.0rc1` and `v0.1.0`, the AI assistant on the lead maintainer's
behalf (GOVERNANCE.md, approvals log, 2026-09-30). The PyPI upload still waits for the lead
maintainer's approval of the `pypi` environment (step 11).

1. Settings → **Rules** → **Rulesets** → **New ruleset** → **New tag ruleset**.
2. **Ruleset name:** `release tags`. **Enforcement status:** **Active**.
3. **Bypass list** → **Add bypass** → **Repository admin** → **Always allow**.
4. **Target tags** → **Add target** → **Include by pattern** → `v*`.
5. Tick **Restrict creations**, **Restrict updates**, **Restrict deletions** and **Block force
   pushes**.
6. **Create**.

## 11. Environments `pypi` and `testpypi`

Both environments exist (created for the name reservation, `docs/developer/pypi_name_reservation.md`);
their names are bound to the PyPI trusted publishers: **never rename them**.

1. Settings → sidebar **Environments** → **pypi**.
2. **Deployment protection rules:** tick **Required reviewers** → add **SMShovan**; leave
   **Prevent self-review** unticked while you are the only maintainer → **Save protection
   rules**. Every PyPI upload then waits for your approval in the workflow run (**Review
   deployments** → **Approve and deploy**).
3. **Deployment branches and tags:** select **Selected branches and tags** → **Add deployment
   branch or tag rule** → **Ref type:** **Tag** → **Name pattern:** `v*` → **Add rule**.
4. Untick **Allow administrators to bypass configured protection rules**.
5. Back to **Environments** → **testpypi**: repeat step 3 (tag rule `v*`); required reviewers
   are optional here (release candidates go to TestPyPI only).

**The CUDA plugins (from 0.2.0; ADR 0032).** `release.yml` publishes `dyng-cu12` through the
environments `testpypi-cu12` and `pypi-cu12`, and `dyng-cu13` through `testpypi-cu13` and
`pypi-cu13`: each project's pending trusted publisher (on TestPyPI and on PyPI: repository
`dyng-dev/dyng`, workflow `release.yml`, that environment) is bound to its own pair, because PyPI
refuses two identical pending publishers. Give them the rules of `pypi` / `testpypi`:

6. **Environments** → **New environment** (if GitHub has not created it at the first run) →
   `pypi-cu12` → steps 2 to 4 (required reviewer **SMShovan**, tag rule `v*`, no administrator
   bypass); the same for `pypi-cu13`.
7. `testpypi-cu12` and `testpypi-cu13`: step 3 (tag rule `v*`).

A final release with the plugins then waits for three approvals (`pypi`, `pypi-cu12`,
`pypi-cu13`); the **Review deployments** dialog of the run lists the waiting environments with a
checkbox each, so one **Approve and deploy** can approve all three. Never rename the six
environments.

Docs: <https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments>.

## 12. Discussions: categories and the pinned roadmap

1. **Discussions** tab → next to **Categories**, click the pencil icon.
2. Keep **Announcements** (maintainers only), **General**, **Ideas**, **Q&A** (answerable)
   and **Show and tell**; delete **Polls** if unused. **Save** after each change.
3. The public roadmap (PLAN Section 10.5): **New discussion** → category **Announcements** →
   title `dynG roadmap` → a short summary with a link to `docs/roadmap.md` → **Start
   discussion**; then, in the discussion's right column, **Pin discussion** → **Pin
   discussion**.

## 13. Read the Docs (later: checkpoint A4)

The documentation is built locally and in CI until then; `.readthedocs.yaml` is ready. Do this
only after approving checkpoint A4 (PLAN Section 0.5; milestone M6).

1. Open <https://app.readthedocs.org> → **Log in** → **Using GitHub** (or **Sign up** →
   **Using GitHub**) and authorize Read the Docs.
2. **Add project** → **Configure your GitHub App** if `dyng-dev/dyng` is not listed: install
   the Read the Docs GitHub App on **dyng-dev** with **Only select repositories** →
   **dyng-dev/dyng**.
3. Back in **Add project**, search `dyng-dev/dyng` → **Continue** → **Name:** `dyng`,
   **Default branch:** `main` → **Next** → Read the Docs finds `.readthedocs.yaml` → **Next**
   → the first build starts.
4. Project → **Settings** → **Pull request builds** → tick **Build pull requests for this
   project** → **Update** (previews of the documentation on every pull request).
5. Project → **Settings** → **Automation rules** → **Add rule** → **Match:** **SemVer
   versions**, **Version type:** **Tag**, **Action:** **Activate version** → **Save**
   (every release tag gets its documentation version).
6. Set the repository website to `https://dyng.readthedocs.io` (step 2), and record A4 in
   `GOVERNANCE.md`.

Docs: <https://docs.readthedocs.com/platform/stable/intro/add-project.html>.

## 14. Zenodo (later: checkpoint A4)

A DOI per release (PLAN Section 10.4). Zenodo archives each **GitHub Release** (not a bare tag)
and takes the metadata from `CITATION.cff`.

1. Open <https://zenodo.org> → **Log in** → **Log in with GitHub** → authorize Zenodo.
2. Give Zenodo access to the organization: on GitHub, your avatar → **Settings** →
   **Applications** → **Authorized OAuth Apps** → **Zenodo** → under **Organization access**,
   click **Grant** next to **dyng-dev**.
3. On Zenodo: your account menu (top right) → **GitHub** → **Sync now** → find
   `dyng-dev/dyng` → switch it **On**.
4. The next published GitHub Release (step 11 of {doc}`release`, created after the tag) is
   archived and gets a DOI. Add the DOI badge to `README.md` and the `doi` field to
   `CITATION.cff` in the next pull request, and record A4 in `GOVERNANCE.md`.

Zenodo's GitHub integration archives only the releases **published after** the repository is
switched on; it does not go back to earlier ones. **0.1.0 has no DOI:** the author decided on
2026-10-01 not to connect Zenodo for 0.1.0 (GOVERNANCE.md, approvals log), so its GitHub
Release is not archived. Zenodo may be connected later (checkpoint A4, steps 1-3, planned with
M6, PLAN 11.2); the first DOI then comes with the first GitHub Release published after that
(0.1.1 or later), and each later release gets its own. Archiving 0.1.0 afterwards would need an
upload by hand (**New upload**, the release's source archive and the metadata of
`CITATION.cff`), which would be a new decision of the author.

Docs: <https://help.zenodo.org/docs/github/enable-repository/>.
