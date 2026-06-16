"""Helpers for optional dependency handling.

Kept dependency-free so it is safe to import from package ``__init__`` modules
that guard heavy/optional imports.
"""

from __future__ import annotations

# Top-level distributions provided by the optional ``[llm]`` extra. A missing
# module is treated as "the extra is not installed" only when it belongs to one
# of these import roots; anything else is a genuine internal import bug and must
# propagate unchanged.
_LLM_SDK_ROOTS = (
    "openai",
    "anthropic",
    "google.genai",
    "google.cloud.aiplatform",
)


def _is_llm_sdk_module(module_name: str | None) -> bool:
    if not module_name:
        return False
    return any(module_name == root or module_name.startswith(root + ".") for root in _LLM_SDK_ROOTS)


def reraise_missing_llm_sdk(feature: str, exc: ModuleNotFoundError) -> None:
    """Raise LLM-extra install guidance when ``exc`` is a missing optional SDK.

    If ``exc`` reports a module that is part of the ``[llm]`` extra, raise a
    clear ``ModuleNotFoundError`` telling the user to install it. Otherwise the
    error is a real internal import failure (e.g. a missing ``tide2.*`` module):
    this function returns without raising so the caller can re-raise the
    original exception with ``raise`` in its own ``except`` block, leaving the
    traceback unchanged (no helper frame) so the bug is not masked.
    """
    if _is_llm_sdk_module(exc.name):
        raise ModuleNotFoundError(
            f"{feature} requires the optional LLM provider SDKs (missing {exc.name!r}). "
            "Install them with `pip install 'tide2[llm]'` (or `uv sync --extra llm`).",
            name=exc.name,
        ) from exc
