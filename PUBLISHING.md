# Publishing `tide2` to PyPI

This project is published to [PyPI](https://pypi.org/p/tide2) with a **trunk-based,
two-phase** release process:

- **Phase 1 — Release PR (automatic).**
  [`release-please`](https://github.com/googleapis/release-please)
  ([`.github/workflows/release-please.yml`](.github/workflows/release-please.yml))
  watches `main` and keeps **one** open release PR current. It computes the next
  version from your Conventional-Commit PR titles and writes `CHANGELOG.md` +
  `.release-please-manifest.json`. It does **not** tag or publish.
- **Phase 2 — Publish Release (manual).**
  ([`.github/workflows/publish.yml`](.github/workflows/publish.yml)) is a
  `workflow_dispatch` workflow the maintainer triggers deliberately *after*
  merging the release PR. From that click everything is automatic: read the
  version → tag → build → PyPI → GitHub Release → verify.

  > A `verify-published` **warning** ("PyPI propagation lag") does not mean the
  > release failed. PyPI is immutable and the upload is already confirmed by
  > `publish-pypi`; the public index can take 10+ minutes to list a new version.
  > Confirm with `uv pip install "tide2==<ver>"` after a few minutes. Only a
  > `verify-published` *error after the version is listed* indicates a real problem.

Uploads use **Trusted Publishing (OIDC)** and writes back to the repo (tag,
Release) use the built-in `GITHUB_TOKEN` — **no API tokens are stored anywhere**
(no `PYPI_API_TOKEN`, no PAT, no GitHub App token).

```
feature PRs (Conventional-Commit titles) ──▶ main      (always via PR, squash-merge)
        │
        │  Phase 1: release-please maintains ONE release PR on main (auto, continuous)
        ▼
  release PR bumps CHANGELOG.md + manifest  (version decided ONCE)   ◀── a real PR
        │  maintainer reviews + merges          ◀── HUMAN GATE 1
        ▼
  main carries the updated CHANGELOG.md + .release-please-manifest.json
     (release-please does NOT tag — skip-github-release: true)
        │
        │  Phase 2: maintainer clicks Actions ▸ Publish Release ▸ Run workflow  ◀── HUMAN GATE 2
        ▼
  resolve: read version from manifest (NOT recomputed) ─▶ guard ─▶ tag: create vX.Y.Z
        ▼
  build (once, checked out at tag) ─▶ twine check ─▶ install-smoke-test wheel
        ▼
  [pypi environment: approve]   ◀── HUMAN GATE 3 ─▶ PyPI (OIDC + PEP 740 attestations)
        ▼
  GitHub Release vX.Y.Z (notes from CHANGELOG + dist/* artifacts)
        ▼
  verify-published: install from PyPI in clean venv + import smoke test
        │
        └─ (optional, same workflow with dry_run: true) TestPyPI dry-run — off the prod path
```

**Three human touch-points, everything between them automatic:** merge the release
PR → click Run workflow → approve the `pypi` environment.

The single build-time source of truth for the version is the **git tag**:
`hatch-vcs` derives the package version from it. There is no static version string
to edit by hand. The version is **decided once** by release-please (persisted to
`.release-please-manifest.json`); Phase 2 only *reads* that number, creates the
matching tag, and `hatch-vcs` bakes it into the wheel — it is never recomputed.

---

## The release PR (Phase 1)

release-please runs on every push to `main` and maintains a single, stable release
PR. You don't create or update it by hand — it reflects everything merged since the
last release.

- **Squash-merge every PR** with a **Conventional-Commit PR title** — the title is
  the squash subject release-please parses. A `feat:` title bumps minor, `fix:` /
  `perf:` bumps patch (see [CONTRIBUTING.md](CONTRIBUTING.md)). A CI check
  (`pr-title-lint.yml`) rejects non-conforming titles and breaking-change notation.
- **Attribution is preserved:** GitHub adds `Co-authored-by:` trailers for every
  author in a squashed PR.
- The release PR updates **only** `CHANGELOG.md` + `.release-please-manifest.json`.
  Because the version is `dynamic` (hatch-vcs), release-please is configured with
  `release-type: simple`, which has no static version file to rewrite.

---

## Cutting a release

> **Versions are immutable on (Test)PyPI.** A given `X.Y.Z` can be uploaded only
> once to each index — even a deleted/yanked version's filename cannot be reused.
> release-please picks the next number for you; if you ever publish by hand, pick
> the **next unused** version. Check what already exists:
>
> ```bash
> # production PyPI
> curl -s https://pypi.org/pypi/tide2/json | python3 -c \
>   "import sys,json;print(sorted(json.load(sys.stdin)['releases']))"
> # TestPyPI
> curl -s https://test.pypi.org/pypi/tide2/json | python3 -c \
>   "import sys,json;print(sorted(json.load(sys.stdin)['releases']))"
> ```

### How the version number is chosen

Versions follow [SemVer](https://semver.org/) and are derived from commit/PR types
(see [CONTRIBUTING.md](CONTRIBUTING.md)):

- `feat:` → **minor** bump (e.g. `1.1.0` → `1.2.0`)
- `fix:` / `perf:` → **patch** bump (e.g. `1.1.0` → `1.1.1`)
- Major bumps are a **deliberate maintainer decision** (breaking-change notation
  is forbidden in commits/PR titles, so majors are never inferred automatically).

### The release sequence

1. Merge feature PRs into `main` (squash, Conventional titles).
2. Review and **merge the release-please PR** — this decides the version and writes
   `CHANGELOG.md` + the manifest. *(Human gate 1.)*
3. Go to **Actions ▸ Publish Release ▸ Run workflow**, run from `main`, and leave
   **`dry_run` unchecked**. This is the deliberate "ship it." *(Human gate 2.)*
4. **Approve the `pypi` environment** when prompted. *(Human gate 3.)*
5. Confirm the GitHub Release was created and `verify-published` is green.

That's it — merging the release PR and clicking Run workflow are decoupled on
purpose, so you can merge the notes now and ship later (e.g. a coordinated
announcement).

### TestPyPI dry-run (optional, non-blocking)

Run **Actions ▸ Publish Release ▸ Run workflow** with **`dry_run` checked** to
build from `main`, `twine check`, and upload to TestPyPI only — **no tag, no PyPI,
no GitHub Release**. TestPyPI is never a hard prerequisite for a real release; the
real quality gate is build-once + `twine check` + the wheel install-smoke-test in
the `build` job.

> The dry-run sets `SETUPTOOLS_SCM_PRETEND_VERSION` to the manifest version so the
> wheel built from `main` carries a clean, uploadable number (hatch-vcs would
> otherwise stamp a `…+g<sha>`/`.devN` local version that PyPI rejects).

### Pre-releases (`rcN`)

To ship a pre-release, let the release PR carry an `X.Y.Z-rcN` version (via
release-please's pre-release support). Phase 2 detects the `rc`/`a`/`b`/`dev`
segment and routes it to **TestPyPI only**, tags it, and creates the GitHub
Release with **`--prerelease`** — production PyPI is skipped.

### Releasing with the CLI

Equivalent to the UI steps once the release PR is merged:

```bash
# Production release (after merging the release-please PR on main)
gh workflow run publish.yml --ref main

# TestPyPI dry-run
gh workflow run publish.yml --ref main -f dry_run=true

# Watch the run
gh run watch "$(gh run list --workflow=publish.yml -L1 --json databaseId --jq '.[0].databaseId')"
```

> `workflow_dispatch` runs the workflow file **as it exists on `main`**, and the
> `resolve` job reads the version from `main`'s manifest. Always dispatch from
> `main` after the release PR has merged.

---

## Maintainer setup (verify; already done)

`tide2` already exists on PyPI, so the items below are a **checklist to confirm**,
not first-time setup. They live outside the repo.

### 1. Trusted Publishers (verify the existing config)

The Trusted Publishers on
[PyPI](https://pypi.org/manage/account/publishing/) and
[TestPyPI](https://test.pypi.org/manage/account/publishing/) are already
configured. For reference, the matching fields are:

| Field             | Value           |
| ----------------- | --------------- |
| PyPI Project Name | `tide2`         |
| Owner             | `susom`         |
| Repository name   | `tide2-core`    |
| Workflow name     | `publish.yml`   |
| Environment name  | `pypi` (on PyPI) / `testpypi` (on TestPyPI) |

> **The publish workflow keeps the filename `publish.yml`**, so the OIDC match
> still holds — **no Trusted Publisher change is needed.** If the file were ever
> renamed, the "Workflow name" on **both** PyPI and TestPyPI would have to be
> updated to match, or OIDC fails with `invalid-publisher`.

### 2. GitHub Environments (confirm)

In **Settings ▸ Environments**, confirm both exist:

- `testpypi`
- `pypi` — confirm it has a **required reviewer** so production uploads need an
  approval click in the Actions UI (this is human gate 3).

No secrets are needed — OIDC handles authentication.

### 3. Repo settings the automation depends on (confirm)

- **Settings ▸ General ▸ Pull Requests:** enable **"Default to PR title for squash
  merge commits."** Without it, GitHub uses the *first commit* as the squash
  subject and release-please bumps from the wrong text — the PR-title lint passes
  anyway, so this fails silently.
- **Settings ▸ Rules:** confirm no `v*` tag-protection rule/ruleset blocks
  `GITHUB_TOKEN` from pushing the Phase-2 tag. If one exists, grant the workflow's
  token a bypass or scope the rule to exclude release tags — otherwise Phase 2
  halts at the `tag` job.

---

## Verifying a release

Build locally exactly as CI does:

```bash
uv build
uvx twine check dist/*
```

Inspect what the wheel will declare as its dependencies (sanity-check core vs
extras before publishing):

```bash
unzip -p dist/*.whl '*/METADATA' | grep -E '^(Requires-Dist|Provides-Extra):'
```

Install and smoke-test from TestPyPI in a throwaway environment (the
`--extra-index-url` lets the many real dependencies resolve from production PyPI):

```bash
uv venv /tmp/tide2-check --python 3.12
uv pip install --python /tmp/tide2-check/bin/python \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  tide2
/tmp/tide2-check/bin/python -c "import tide2; print(tide2.__version__)"
```

To test a specific version (e.g. a pre-release) explicitly:

```bash
uv pip install --python /tmp/tide2-check/bin/python \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  'tide2==1.1.0'
```

After the production release:

```bash
uv pip install tide2          # full install: includes the ML stack (spaCy / torch / transformers)
uv pip install 'tide2[llm]'   # add LLM provider SDKs for the optional LLM-based recognizer
```

The ML inference stack (`spacy`, `torch`, `transformers`) is **required** and
ships in the base install — no model can run without it. The optional `llm`
extra adds the LLM provider SDKs (`anthropic`, `openai`, `google-genai`,
`google-cloud-aiplatform`) for the LLM-based recognizer.

---

## Rollback / bad release

PyPI versions are **immutable and non-reusable**:

1. **Yank** the broken version on PyPI (existing pins still resolve; new installs
   skip it).
2. **Never** delete + re-upload the same number — that filename is burned forever.
3. **Bump** to the next patch via a normal release PR with the fix, then publish
   that.
4. Optionally annotate the GitHub Release and open a regression issue.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| No release PR appears | No releasable commits since the last release | Nothing to release |
| Guard fails: *CHANGELOG ≠ manifest* | Release PR not merged yet | Merge the release-please PR, then re-run Phase 2 |
| Guard fails: *tag already exists* | That version was already published | Let release-please open the next release PR |
| Wrong / no version bump | Squash subject wasn't a Conventional Commit | Fix the PR title (the `pr-title-lint` check enforces it); confirm "Default to PR title for squash" is enabled |
| Empty / generic release notes | No matching `## …X.Y.Z` section in `CHANGELOG.md` | Confirm the release PR merged; the job falls back to `--generate-notes` |
| `release already exists` | Re-running a finished release | `gh release delete vX.Y.Z`, or `gh release upload vX.Y.Z dist/*` |
| Build guard: *not a clean release version* | Built ref produced a `…+g<sha>`/`.devN` local version | Real releases build at the `vX.Y.Z` tag; dry-runs set `SETUPTOOLS_SCM_PRETEND_VERSION` — re-run from `main` after merging the release PR |
| `400 File already exists` on upload | That version was already uploaded (immutable) | Bump to the next unused `X.Y.Z` via a new release PR |
| TestPyPI install can't resolve deps | TestPyPI doesn't mirror all of PyPI | Always pass `--extra-index-url https://pypi.org/simple/` |
| OIDC `invalid-publisher` / environment not found | Trusted Publisher or GitHub Environment misconfigured, or workflow renamed | Verify the [maintainer setup](#maintainer-setup-verify-already-done); keep the filename `publish.yml` |
