# Contributing to TIDE 2.0

Thank you for your interest in contributing to TIDE 2.0! This guide describes how
to set up your environment, the branching and commit conventions we follow, and
what to check before opening a pull request.

For a project overview and feature list, see the [README](README.md). To report a
security vulnerability, follow [SECURITY.md](SECURITY.md) — **do not** open a
public issue. This project is released under the [MIT License](LICENSE-MIT); by
contributing you agree your contributions are licensed under the same terms.

## Development environment

The fastest way to get started is the **Dev Container**, which provisions Python
3.12, [`uv`](https://docs.astral.sh/uv/), all dependencies, and the pre-commit
hooks automatically. See [Dev Container (recommended)](README.md#dev-container-recommended)
in the README.

If you develop **locally** instead (see
[Local Installation](README.md#local-installation-without-dev-container)):

```bash
uv python install 3.12.8
uv sync
make setup-hooks          # REQUIRED: installs pre-commit AND the commit-msg hook
```

> **`make setup-hooks` is the critical bootstrap step.** It installs both the
> pre-commit hooks and the `commit-msg` hook. Without it, your commits will not be
> validated locally and may fail CI or get rejected later. The Dev Container runs
> this for you.

Notes:

- This project uses **`uv`** — not `pip` or `poetry`.
- Python **3.12 only** (`>=3.12,<3.13`), constrained by the `spacy`/`thinc`
  C-extension stack.

## Branching & workflow (Gitflow-style)

TIDE 2.0 uses a [Gitflow-style branching model](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
with two long-lived branches — `main` and `development` — plus short-lived
working branches:

| Branch | Role | Releases |
|---|---|---|
| `main` | Stable production history | Stable releases (e.g. `v1.0.0`) |
| `development` | Integration / the pre-release branch | The prerelease |
| `feat/*`, `fix/*`, etc. | Working branches | None |
| `hotfix/*` | Urgent production fix | Patch |

Contributor rules:

- **Branch features from `development`** and open your **PR against `development`**.
- `main` is stable production history — **never** PR a feature straight to `main`.
- **Hotfixes** branch from `main`. Open **two PRs** for a hotfix: one into `main`
  (to ship the fix) and one into `development` (to back-merge it), so the fix
  lands on both lines and history stays linear.
- **Branch naming:** use `<type>/<short-desc>`, where `<type>` is one of the
  allowed [commit types](#commit-messages--conventional-commits) (or `hotfix`) —
  e.g. `feat/regex-recognizer`. A repository ruleset enforces this prefix and
  **rejects** branches that don't match. If you have a tracking ticket you may
  include it (e.g. `fix/STAR-12269-changelog-order`), but it is **optional** —
  outside contributors won't have one. Keep names short, lowercase, and descriptive.

You do **not** bump versions or edit `CHANGELOG.md` — see
[Releases](#releases-how-versioning-works).

### Day-to-day Git workflow

**Start a feature branch** from an up-to-date `development`:

```bash
git checkout development
git pull --rebase          # fast-forward to remote; should not produce conflicts
git checkout -b feat/<short-desc>
```

**Rebase onto `development` regularly** (do this daily, and always before opening
or updating a PR — `development` changes often):

```bash
git checkout development
git pull --rebase          # update local development to match remote
git checkout feat/<short-desc>
git rebase development     # resolve any conflicts, then `git rebase --continue`
git push --force-with-lease # your branch history was rewritten; safe-force the update
```

Keep your branch rebased on the latest `development` until it merges, so the PR
diff shows **only your changes**. PRs are **squash-merged**, so intermediate
commits are collapsed — keep commit messages [conventional](#commit-messages--conventional-commits)
since the squash message is what lands in history.

## Commit messages — Conventional Commits

Commit messages **must** follow [Conventional Commits v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).
A `commit-msg` hook validates every commit in `--strict` mode and **rejects
non-conforming messages locally**.

**Format:**

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Allowed types** (exactly these ten):

| Type | Meaning | Version bump | In changelog |
|---|---|---|---|
| `feat` | A new feature | minor | yes |
| `fix` | A bug fix | patch | yes |
| `perf` | A performance improvement | patch | yes |
| `docs` | Documentation only | none | yes |
| `build` | Build system / dependencies | none | only `build(deps)` |
| `refactor` | Code change that neither fixes a bug nor adds a feature | none | no |
| `style` | Formatting, whitespace (no code-meaning change) | none | no |
| `test` | Adding or fixing tests | none | no |
| `ci` | CI configuration / scripts | none | no |
| `chore` | Routine maintenance | none | no |

**Description style:** a lowercase, imperative-mood summary with **no trailing
period**. Add a **scope** (a noun in parentheses) to localize the change, e.g.
`feat(anonymizer):`.

Good examples (real entries from this repo's history):

```
fix(anonymizer): resolve patient_uid name collision causing silent 0-row output
perf(pipeline): auto-tune CPU resource allocation for Mac/CPU runs
feat: add OSSF Scorecard support
docs: correct README and runner docstrings to match code
```

Bad examples:

```
Fixed a bug.                  # not a valid type, capitalized, past tense, trailing period
update                        # no type, not descriptive
feature: add new recognizer   # "feature" is not an allowed type (use "feat")
```

### Repo-specific rule: no breaking-change notation

> **This is a deliberate deviation from the Conventional Commits v1.0.0 spec.**

Breaking-change notation is **not permitted**:

- No `BREAKING CHANGE:` footer.
- No `!` after the type/scope (e.g. `feat!:` or `feat(api)!:`).

A local `forbid-breaking-change` hook **rejects** these. For example, this commit
is **rejected**:

```
feat!: drop support for the legacy config format    # REJECTED by the hook
```

**Why:** major-version progression is managed deliberately by maintainers, **not**
driven by commit syntax. Commits only ever bump minor (`feat`) or patch
(`fix`/`perf`); a major release is a maintainer decision. Keeping breaking-change
syntax out of commits avoids accidental major bumps.

## Code style & quality

All style and quality checks run automatically via pre-commit. Key tools:

- **[ruff](https://docs.astral.sh/ruff/)** — linting (with autofix) and
  formatting. Line length 120, double quotes, single-line imports.
- **[`ty`](https://github.com/astral-sh/ty)** — type checking (Python 3.12).
- **[bandit](https://bandit.readthedocs.io/)** — security linting.
- **[nbstripout](https://github.com/kynan/nbstripout)** — strips output from
  Jupyter notebooks before commit.

Run all hooks manually at any time:

```bash
uv run pre-commit run --all-files
```

## Testing

```bash
uv run pytest                              # all tests (coverage runs automatically)
uv run pytest --no-cov                     # faster, no coverage
uv run pytest tests/test_masking_anonymizer.py   # a single file
uv run pytest -m "not integration"         # skip slow integration tests
```

Add tests for new behavior and make sure the suite passes before opening a PR.
See the [Testing](README.md#testing) section of the README for details on the
coverage reports.

## Pull request checklist

Before opening a PR, confirm:

- [ ] Branched from and rebased on `development`.
- [ ] PR targets **`development`** (not `main`).
- [ ] Commit messages follow [Conventional Commits](#commit-messages--conventional-commits)
      (no breaking-change notation).
- [ ] `uv run pre-commit run --all-files` passes clean.
- [ ] `uv run pytest` passes; new behavior has tests.
- [ ] Relevant docs updated. **Do not** hand-edit `CHANGELOG.md` — it is automated.

## Releases (how versioning works)

Releases are produced by maintainers using
[`python-semantic-release`](https://python-semantic-release.readthedocs.io/)
along the two long-lived branches: `development` is the single pre-release line,
while `main` produces stable releases. The next version and the `CHANGELOG.md`
entries are derived **from your commit types** — `feat` → minor, `fix`/`perf` →
patch. That's *why* commit hygiene matters.

Contributors do **not** bump versions or hand-edit `CHANGELOG.md`; the automation
owns both.

## Reporting issues & security

- **Bugs & feature requests:** [GitHub Issues](https://github.com/susom/tide2/issues).
- **Questions & discussion:** [GitHub Discussions](https://github.com/susom/tide2/discussions).
- **Security vulnerabilities:** follow [SECURITY.md](SECURITY.md) — please do
  **not** file a public issue for a vulnerability.
