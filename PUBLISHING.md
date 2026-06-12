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

After the production release:

```bash
uv pip install tide2          # base install (CPU)
uv pip install 'tide2[gpu]'   # add spaCy / torch / transformers for ML recognizers
```

The base install is CPU-only; the heavy ML stack (`spacy`, `torch`,
`transformers`) lives in the `gpu` extra so `pip install tide2` stays lean.
