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

| # | Setting | When | Status |
|---|---|---|---|
| 1 | Organization: two-factor authentication, base permissions | now | |
| 2 | About: description, website, topics | now | |
| 3 | Features: Issues, Discussions, no wiki | now | |
| 4 | Pull requests: squash merges only | now | |
| 5 | Actions: permissions, SHA pinning, fork approval | now | |
| 6 | Security: private vulnerability reporting, Dependabot, secret scanning | now | |
| 7 | The DCO app | now | |
| 8 | Labels | after the M4 merge | |
| 9 | Ruleset for `main`: pull requests and required checks | after every required workflow has run once on a pull request | |
| 10 | Ruleset for release tags | now | |
| 11 | Environments `pypi` and `testpypi`: protection rules | now | |
| 12 | Discussions: categories and the pinned roadmap | after step 3 | |
| 13 | Read the Docs | later: checkpoint A4, milestone M6 | |
| 14 | Zenodo | later: checkpoint A4, before 0.1.0 is published | |

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

## 4. Pull requests: squash merges only

PLAN Section 8.9: squash merges, the pull request title (Conventional Commits) becomes the
commit subject.

Settings → **General** → section **Pull Requests**:

1. Untick **Allow merge commits**.
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
   crazy-max/ghaction-github-labeler@*,
   mamba-org/setup-micromamba@*
   ```

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

Docs: <https://docs.github.com/en/code-security/security-advisories/working-with-repository-security-advisories/configuring-private-vulnerability-reporting-for-a-repository>.

## 7. The DCO app

External contributors sign off their commits (CONTRIBUTING.md); the DCO app checks every pull
request and reports a check named **DCO**. `.github/dco.yml` exempts members of `dyng-dev`.

1. Open <https://github.com/apps/dco> → **Install** (or **Configure** if it is installed
   elsewhere).
2. Choose the **dyng-dev** organization.
3. Select **Only select repositories** → choose **dyng-dev/dyng** → **Install**.
4. After the next pull request, the check **DCO** appears; add it to the required checks
   (step 9).

## 8. Labels

`.github/labels.yml` defines the labels (docs/developer/labels.md). The `labels` workflow
applies it automatically on every push to `main` that changes the file, so after the M4 merge
the labels exist without any click. To apply it by hand: **Actions** tab → workflow **labels**
→ **Run workflow** → branch `main` → **Run workflow**. Do not edit labels in the web
interface: the next run resets listed labels to the file.

## 9. Ruleset for `main`

PLAN Section 8.9: everything goes through a pull request, the required checks must be green,
squash merges only. While the lead maintainer is the only maintainer, **zero approvals** are
required (the author merges their own pull request once the checks pass); from the day a
second maintainer exists, raise it to one.

GitHub offers a required check in the ruleset only after it has run in the last 7 days, so
open one pull request first (for example the M4 merge) and let every workflow finish.

1. Settings → sidebar section **Code and automation** → **Rules** → **Rulesets** →
   **New ruleset** → **New branch ruleset**.
2. **Ruleset name:** `main`. **Enforcement status:** **Active**.
3. **Bypass list:** leave empty (see the note below).
4. **Target branches** → **Add target** → **Include default branch**.
5. Tick these rules:
   - **Restrict deletions**.
   - **Require linear history**.
   - **Require a pull request before merging**, and open its options:
     - **Required approvals:** `0` now; `1` once there is a second maintainer.
     - **Dismiss stale pull request approvals when new commits are pushed:** tick.
     - **Require review from Code Owners:** untick now; tick with the second maintainer.
     - **Require approval of the most recent reviewable push:** untick now; tick with the
       second maintainer.
     - **Require conversation resolution before merging:** tick.
     - **Allowed merge methods:** only **Squash**.
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
| `cpu.yml` | `cpu-only / gcc-12 / openmp=ON`, `cpu-only / gcc-13 / openmp=ON`, `cpu-only / clang-18 / openmp=ON`, `cpu-only / gcc-13 / openmp=OFF`, `cpu-only / clang-17 / openmp=ON`, `dev / gcc-12 / openmp=ON`, `dev / gcc-13 / openmp=ON`, `dev / clang-18 / openmp=ON`, `dev / clang-18 / openmp=OFF` |
| `docs.yml` | `site` (not `external-links`, which runs only weekly and on demand) |
| DCO app | `DCO` |

Do **not** require checks of workflows that run only for some files (`labels.yml`), only on
tags (`release.yml`), only on some events (`welcome.yml`, the first-interaction greeting) or
only on a schedule (the `external-links` job of `docs.yml`): a required check that never runs
blocks every pull request. When a
workflow is added (`cuda-build`, `python`, `api-check`; PLAN Section 8.8) or a job of the
`cpu` matrix is renamed, update this list and the ruleset in the same pull request.

**Note on bypassing.** With an empty bypass list, nobody can push to `main` directly, including
the organization owners: milestone work is merged through pull requests (the AI assistant can
open them). If you decide to keep pushing directly for a while, **Add bypass** →
**Repository admin** → mode **Always allow**, and record the decision in `GOVERNANCE.md`.

Docs: <https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository>.

## 10. Ruleset for release tags

A tag `v*` publishes to TestPyPI and PyPI (`release.yml`), so only the lead maintainer may
create one.

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
4. The next published GitHub Release (from 0.1.0; `release.yml` creates it) is archived and
   gets a DOI. Add the DOI badge to `README.md` and the `doi` field to `CITATION.cff` in the
   next pull request, and record A4 in `GOVERNANCE.md`.

Docs: <https://help.zenodo.org/docs/github/enable-repository/>.
