# Publishing `tide2` to PyPI

This project is published to [PyPI](https://pypi.org/p/tide2) with a manual
GitHub Actions workflow ([`.github/workflows/publish.yml`](.github/workflows/publish.yml))
that builds with `uv` and uploads via **Trusted Publishing (OIDC)** — no API
tokens are stored anywhere.

The release flow is **TestPyPI first, then production PyPI**:

```
conventional commits ──▶ python-semantic-release ──▶ vX.Y.Z tag + CHANGELOG
                                                          │
                                          run "Publish to PyPI" workflow
                                                          │
                                  target: testpypi ──▶ TestPyPI  (dry run)
                                                          │
                                  target: both ─────▶ TestPyPI ▶ PyPI
```

The single source of truth for the version is the **git tag**: `hatch-vcs`
derives the package version from it at build time. There is no version string to
edit by hand.

---

## How the workflow works

`publish.yml` is a `workflow_dispatch` (manual) workflow with two inputs:

| Input     | Values                | Meaning                                                                 |
| --------- | --------------------- | ----------------------------------------------------------------------- |
| `target`  | `testpypi` / `both`   | `testpypi` = upload to TestPyPI only (dry run). `both` = TestPyPI ▶ PyPI |
| `version` | blank / `X.Y.Z`       | First-release bootstrap override only. Leave blank for normal releases.  |

It runs three jobs:

1. **`build`** — checks out the selected ref with full history (needed for
   `hatch-vcs`), runs `uv build`, then a **guard** that fails fast if the built
   version is not a clean `X.Y.Z` (PyPI rejects PEP 440 local versions like
   `+g<sha>` / `.devN`), then `uvx twine check`, and uploads the `dist/` artifact.
2. **`publish-testpypi`** — downloads the artifact and uploads to TestPyPI via
   OIDC (environment `testpypi`).
3. **`publish-pypi`** — only when `target: both`; uploads to production PyPI via
   OIDC (environment `pypi`). Runs after TestPyPI succeeds.

> **Key behaviour:** `workflow_dispatch` runs the workflow file **as it exists on
> the selected ref**. The ref you pick must therefore contain `publish.yml` (and,
> if you want the new code published, the code changes too). This is why the very
> first release needs the bootstrap script (see below), and why you tag the branch
> that already carries the workflow.

---

## One-time setup (maintainer, once per project)

These steps live outside the repo and only need to be done once.

### 1. Configure Trusted Publishers

On **both** [PyPI](https://pypi.org/manage/account/publishing/) and
[TestPyPI](https://test.pypi.org/manage/account/publishing/), add a *pending*
publisher (for a project that does not exist yet) with:

| Field             | Value           |
| ----------------- | --------------- |
| PyPI Project Name | `tide2`         |
| Owner             | `susom`         |
| Repository name   | `tide2-core`    |
| Workflow name     | `publish.yml`   |
| Environment name  | `pypi` (on PyPI) / `testpypi` (on TestPyPI) |

### 2. Create GitHub Environments

In **Settings ▸ Environments**, create two environments matching the names above:

- `testpypi`
- `pypi` — recommended: add a **required reviewer** so production uploads need an
  approval click in the Actions UI.

No secrets are needed — OIDC handles authentication.

---

## Cutting a release

> **Versions are immutable on (Test)PyPI.** A given `X.Y.Z` can be uploaded only
> once to each index — even a deleted/yanked version's filename cannot be reused.
> Always pick the **next unused** version. Check what already exists:
>
> ```bash
> # production PyPI
> curl -s https://pypi.org/pypi/tide2/json | python3 -c \
>   "import sys,json;print(sorted(json.load(sys.stdin)['releases']))"
> # TestPyPI
> curl -s https://test.pypi.org/pypi/tide2/json | python3 -c \
>   "import sys,json;print(sorted(json.load(sys.stdin)['releases']))"
> ```

### Choosing the version number

Versions follow [SemVer](https://semver.org/) and are derived from commit types
(see [CONTRIBUTING.md](CONTRIBUTING.md)):

- `feat:` → **minor** bump (e.g. `1.0.0` → `1.1.0`)
- `fix:` / `perf:` → **patch** bump (e.g. `1.0.0` → `1.0.1`)
- Major bumps are a **deliberate maintainer decision** (breaking-change notation
  is forbidden in commits, so majors are never inferred automatically).

For TestPyPI-only dry runs you may also use a PEP 440 pre-release suffix
(`1.1.0rc1`, `1.1.0a1`) to avoid burning a real version number.

### Normal releases (after the first one)

1. Land your changes on the default branch using
   [conventional commits](CONTRIBUTING.md). `python-semantic-release` computes
   the next version, updates `CHANGELOG.md`, and creates the `vX.Y.Z` tag.
2. Go to **Actions ▸ Publish to PyPI ▸ Run workflow**.
   - **Use workflow from**: select the `vX.Y.Z` tag.
   - **target**: `testpypi` for a dry run, or `both` to release.
3. For a `both` run, approve the `pypi` environment when prompted.

The workflow refuses to publish anything that is not a clean `X.Y.Z` version
(hatch-vcs emits a `+g<sha>`/`.devN` local version on untagged commits, which PyPI
rejects), so you cannot accidentally publish a dev build.

### Releasing manually with the CLI

Equivalent to the UI steps above, once a clean `vX.Y.Z` tag exists on the ref you
want to publish:

```bash
# 1. Create and push the annotated tag (from the branch/commit you want to ship)
git tag -a v1.1.0 -m "Release 1.1.0"
git push origin v1.1.0

# 2. Dispatch the workflow against that tag
gh workflow run publish.yml --ref v1.1.0 -f target=testpypi   # dry run
# ...or release to production after the dry run looks good:
gh workflow run publish.yml --ref v1.1.0 -f target=both

# 3. Watch the run
gh run watch "$(gh run list --workflow=publish.yml -L1 --json databaseId --jq '.[0].databaseId')"
```

> **Tagging a working branch (not yet merged):** because the workflow publishes
> the ref you select, you *can* tag a `feat/*` branch directly to get its code
> onto TestPyPI before the PR merges. This is fine for **TestPyPI dry runs**. For
> a production (`both`) release, prefer tagging `main`/`development` after merge so
> the published version corresponds to mainline history.

### The very first release (bootstrap)

The first `vX.Y.Z` tag must be created **after** `publish.yml` is merged so the tag
contains the workflow (workflow_dispatch runs the workflow as it exists on the
selected ref). Use the helper script:

```bash
# TestPyPI only (safe dry run)
scripts/first_release.sh 1.0.0

# TestPyPI then production PyPI
scripts/first_release.sh 1.0.0 both
```

It validates the build locally, creates and pushes the annotated tag, and
dispatches the workflow against it. The `version` input on the workflow exists
solely as a fallback override (`SETUPTOOLS_SCM_PRETEND_VERSION`) and is not needed
once the tag carries the workflow.

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

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Guard step fails: *"not a clean release version"* | The selected ref has no `vX.Y.Z` tag, so `hatch-vcs` produced a `…+g<sha>` / `.devN` version. | Run the workflow from an actual `vX.Y.Z` **tag**, not a branch. |
| `400 File already exists` on upload | That version was already uploaded to the index (versions are immutable). | Bump to the next unused `X.Y.Z` (or a new `rcN` for TestPyPI) and re-tag. |
| TestPyPI install can't resolve dependencies | TestPyPI doesn't mirror all of PyPI. | Always pass `--extra-index-url https://pypi.org/simple/` when installing from TestPyPI. |
| `publish-pypi` job is skipped | `target` was `testpypi`. | Re-run with `target: both` to promote to production. |
| OIDC auth error / environment not found | Trusted Publisher or GitHub Environment not configured. | Complete the [one-time setup](#one-time-setup-maintainer-once-per-project). |
| Workflow doesn't show the new tag in the ref picker | The tag predates `publish.yml`, or wasn't pushed. | Push the tag; ensure it was created on a commit that contains `publish.yml`. |
