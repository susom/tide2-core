# CLAUDE.md

Guidance for AI agents working in **tide2-core** (TIDE 2.0). This file is the fast
path to editing the repo correctly; it deliberately links to the canonical docs
instead of copying them, because copied content drifts.

## Orientation

TIDE 2.0 is a healthcare PII/PHI de-identification and anonymization toolkit built
on **Microsoft Presidio**. The core is a two-stage pipeline — entity recognition
(regex + transformer NER) → anonymization (HIPS crypto, FPE, faker, date jitter) —
run distributed via **Ray** (`ray.data.map_batches` over CPU/GPU actors).

- Full overview + feature list + **module/architecture map**: [`README.md`](README.md).
  For the package layout, read README's *Architecture* section — **do not** keep a
  second copy of that tree here.
- Published API reference: <https://susom.github.io/tide2-core/> (pdoc-generated;
  see *Docs* gotcha below).

## Gotchas (read first — these are the non-obvious rules)

- **uv only** — never `pip` or `poetry`. **Python 3.12 only** (`>=3.12,<3.13`),
  constrained by the `spacy`/`thinc` C-extension stack.
- **`make setup-hooks` is mandatory** — installs the pre-commit hooks **and** the
  `commit-msg` hook. Without it, commits aren't validated locally and fail later.
  (The Dev Container runs it for you.)
- **Never hand-edit `CHANGELOG.md` or bump the version.** release-please owns both.
  There is **no static version string** — `hatch-vcs` derives the wheel version
  from the git tag (`src/tide2/_version.py` is generated at build time).
- **No breaking-change notation.** `!` after the type (`feat!:`) and
  `BREAKING CHANGE:` footers are **rejected** by the local `forbid-breaking-change`
  commit-msg hook. Major bumps are a deliberate maintainer decision, never inferred
  from commit syntax.
- **The PR title must be a valid Conventional Commit.** PRs are squash-merged and
  the **PR title becomes the squash subject** that release-please parses to compute
  the version bump (enforced by CI: `pr-title-lint.yml`).
- **Docs = docstrings + README.** The GitHub Pages site is pdoc-generated: its
  landing page *is* `README.md` (included via `.. include:: ../../README.md` in
  `src/tide2/__init__.py`) and its API reference comes from **Google-style
  docstrings** (`__docformat__ = "google"`). So:
  - public-API changes → update the Google-style docstrings;
  - feature/CLI/architecture changes → update `README.md`.
  Both auto-publish on every push to `main` via `.github/workflows/docs.yml`.
- **Don't remove the `pyarrow` pre-import** in the `docs` / `docs-serve` Makefile
  targets — it prevents a native-init segfault in the Presidio→transformers→pyarrow
  import chain. Every module must import cleanly, since pdoc imports them all (the
  docs build uses `uv sync --extra docs --extra llm`).
- **Small-box / Colab Ray deadlock:** on ≲4-CPU boxes the pipeline hangs at `0/1`
  unless you pass **fractional CPUs *and* `--no-checkpoint` together** — both fixes
  are required. See README → *"Why small boxes deadlock"* for the knob table; don't
  reproduce it here.

## Setup & everyday commands

```bash
uv sync                      # install (add --extra llm / --all-extras as needed)
make setup-hooks             # REQUIRED once: pre-commit + commit-msg hooks
source .venv/bin/activate    # Dev Container activates this automatically

# Tests (coverage runs automatically)
uv run pytest
uv run pytest --no-cov                            # faster
uv run pytest -m "not integration"               # skip slow integration tests
uv run pytest tests/test_masking_anonymizer.py   # single file

# Quality — ruff (line-length 120, double quotes, single-line imports),
# ty (type check), bandit (security), nbstripout
uv run pre-commit run --all-files

# Docs preview (hot reload at localhost:8080)
uv sync --extra docs --extra llm && make docs-serve
```

## Making changes

Trunk-based workflow (**no `development` branch**). Branch from an up-to-date `main`
as `<type>/<short-desc>` (e.g. `feat/regex-recognizer`); a repo ruleset **rejects**
branch names whose prefix isn't an allowed commit type. Open the PR against `main`;
it is **squash-merged** with the PR title as subject. Rebase on `main` regularly and
`git push --force-with-lease` after a rebase.

The **10 allowed commit types** and their effect:

| Type | Bump | In changelog |
|---|---|---|
| `feat` | minor | yes |
| `fix`, `perf` | patch | yes |
| `docs` | none | yes |
| `build` | none | only `build(deps)` |
| `refactor`, `style`, `test`, `ci`, `chore` | none | no |

Description: lowercase imperative, no trailing period; add a scope noun where it
helps (`feat(anonymizer):`).

**Before opening a PR:** branched/rebased on `main`; PR targets `main`; valid
Conventional PR title (no breaking-change notation); `uv run pre-commit run
--all-files` clean; `uv run pytest` passes with tests for new behavior; relevant
docs updated; **do not** hand-edit `CHANGELOG.md`.

Full detail: [`CONTRIBUTING.md`](CONTRIBUTING.md). Vulnerabilities:
[`SECURITY.md`](SECURITY.md) (never a public issue).

## Releasing / publishing (pointers)

Two-phase, trunk-based (full flow: [`PUBLISHING.md`](PUBLISHING.md)):

1. **release-please** keeps one open **release PR** on `main` that bumps
   `CHANGELOG.md` + `.release-please-manifest.json` (no tag).
2. A maintainer dispatches **Actions ▸ Publish Release** (`publish.yml`), which
   tags → builds → publishes to PyPI → creates the GitHub Release.

Three human gates: merge release PR → click Run workflow → approve the `pypi`
environment. Uploads use **Trusted Publishing (OIDC)** — no stored tokens.

Agent-relevant rules: the version is decided **once** by release-please and only
*read* in phase 2 (never recompute or edit it); PyPI versions are
**immutable/non-reusable** (bad release → yank + bump, never delete-and-reupload);
**keep the workflow filename `publish.yml`** — renaming it breaks the OIDC
Trusted-Publisher match. Local build sanity: `uv build && uvx twine check dist/*`.

## Entry points

Behavior and flags live in README's *CLI Usage* — here are just the mappings
(`pyproject.toml` → `[project.scripts]`):

- `tide2-runner` → `tide2.runner.cli:main`
- `tide2-visualizer` → `tide2.cli.main_visualizer:main`
