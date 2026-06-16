#!/usr/bin/env bash
# First-release bootstrap for publishing tide2 to PyPI.
#
# WHY THIS EXISTS
# ---------------
# The publish.yml workflow is triggered with workflow_dispatch and is meant to be
# run *from a vX.Y.Z tag*. But workflow_dispatch runs the workflow file as it
# exists on the selected ref, so any tag created BEFORE publish.yml was merged
# does not contain the workflow and cannot be selected. This script handles that
# one-time chicken-and-egg for the very first release.
#
# After the first release, you never need this script again: cut tags normally
# (e.g. via python-semantic-release) once publish.yml is on the default branch,
# then just run the "Publish to PyPI" workflow from the new tag.
#
# WHAT IT DOES
# ------------
# 1. Validates you are on a clean checkout of the branch that contains publish.yml.
# 2. Creates an annotated vX.Y.Z tag at HEAD and pushes it.
# 3. Triggers the publish workflow against that tag via `gh workflow run`.
#
# Because the tag now contains publish.yml (it was created after the workflow was
# merged), no version override is needed and the normal tag-based path is used.
#
# Requirements: git, gh (authenticated: `gh auth status`), uv.
#
# Usage:
#   scripts/first_release.sh 1.0.0            # TestPyPI only (default, safe)
#   scripts/first_release.sh 1.0.0 both       # TestPyPI then production PyPI
set -euo pipefail

VERSION="${1:-}"
TARGET="${2:-testpypi}"

if [ -z "$VERSION" ]; then
  echo "Usage: $0 <version> [testpypi|both]" >&2
  echo "Example: $0 1.0.0 both" >&2
  exit 1
fi

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]]; then
  echo "error: '$VERSION' is not a clean X.Y.Z release version (optionally aN/bN/rcN, e.g. 1.0.0 or 1.0.0rc1)." >&2
  echo "       .devN and local versions are rejected because the publish workflow refuses them." >&2
  exit 1
fi

if [ "$TARGET" != "testpypi" ] && [ "$TARGET" != "both" ]; then
  echo "error: target must be 'testpypi' or 'both' (got '$TARGET')." >&2
  exit 1
fi

TAG="v$VERSION"

# Refuse to run on a dirty tree — the build must reflect committed source.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "error: working tree has uncommitted changes. Commit or stash first." >&2
  exit 1
fi

# The tag must contain publish.yml, otherwise the dispatched run can't see it.
if [ ! -f ".github/workflows/publish.yml" ]; then
  echo "error: .github/workflows/publish.yml not found at HEAD." >&2
  echo "Merge the publish workflow before bootstrapping the first release." >&2
  exit 1
fi

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  echo "error: tag $TAG already exists." >&2
  exit 1
fi

echo "==> Sanity-check the build produces a clean version locally"
rm -rf dist
SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION" uv build >/dev/null
uvx twine check dist/* >/dev/null
echo "    OK: dist/ built and metadata valid for $VERSION"

echo "==> Creating and pushing annotated tag $TAG"
git tag -a "$TAG" -m "Release $VERSION"
git push origin "$TAG"

echo "==> Dispatching publish workflow from $TAG (target: $TARGET)"
gh workflow run publish.yml --ref "$TAG" -f target="$TARGET"

echo
echo "Done. Track the run with:  gh run watch \$(gh run list --workflow=publish.yml -L1 --json databaseId --jq '.[0].databaseId')"
echo "Or open the Actions tab in the browser."
