"""Unit tests for ``TransformerCore._special_token_affixes`` validation (CPU, CI).

The method derives the tokenizer's ``(prefix, suffix)`` special-token wrap by
diffing a probe encoded with vs. without special tokens. Before the fix it cached
whatever the diff produced, so a **probe miss** silently returned empty affixes
(caching them, sticky) and windowed inference would forward every window without
the model's ``[CLS]``/``[SEP]`` — degraded, silent under-detection.

These tests drive the real method (via ``__new__`` so no model loads) with stub
tokenizers, asserting:
    (a) a probe miss raises ``RuntimeError`` (before caching, so not sticky);
    (b) a count mismatch (right shape, wrong number) raises;
    (c) a clean match returns the correct affixes and caches them;
    (d) a zero-special-token tokenizer passes with empty affixes.
"""

import pytest

from tide2.transformers.core import TransformerCore


class _CleanTokenizer:
    """[CLS] <content> [SEP]; content identical with/without special tokens."""

    name_or_path = "stub/clean"

    def __call__(self, text, add_special_tokens=True):
        content = [200, 201]
        ids = [101, *content, 102] if add_special_tokens else content
        return {"input_ids": ids}

    def num_special_tokens_to_add(self, pair=False):
        return 2


class _MissTokenizer:
    """Content tokens differ between the two calls, so the diff finds no match."""

    name_or_path = "stub/miss"

    def __call__(self, text, add_special_tokens=True):
        if add_special_tokens:
            return {"input_ids": [101, 200, 201, 102]}
        return {"input_ids": [7592]}  # unrelated id -> no contiguous match

    def num_special_tokens_to_add(self, pair=False):
        return 2


class _CountMismatchTokenizer:
    """Diff derives 1 special token but the tokenizer declares 2."""

    name_or_path = "stub/mismatch"

    def __call__(self, text, add_special_tokens=True):
        content = [200, 201]
        ids = [101, *content] if add_special_tokens else content  # prefix only
        return {"input_ids": ids}

    def num_special_tokens_to_add(self, pair=False):
        return 2


class _NoSpecialTokenizer:
    """A tokenizer that adds no special tokens: empty affixes are correct."""

    name_or_path = "stub/none"

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [200, 201]}

    def num_special_tokens_to_add(self, pair=False):
        return 0


def _core_with(tokenizer):
    core = TransformerCore.__new__(TransformerCore)  # skip __init__: no model load
    core._tokenizer = tokenizer
    return core


def test_clean_match_returns_and_caches_affixes():
    core = _core_with(_CleanTokenizer())
    assert core._special_token_affixes() == ([101], [102])
    # Second call returns the cached value even if the tokenizer disappears.
    core._tokenizer = None
    assert core._special_token_affixes() == ([101], [102])


def test_probe_miss_raises_and_is_not_sticky():
    core = _core_with(_MissTokenizer())
    with pytest.raises(RuntimeError, match="special-token affixes"):
        core._special_token_affixes()
    # Nothing cached on the failure path -> not sticky.
    assert getattr(core, "_special_affixes", None) is None


def test_count_mismatch_raises():
    core = _core_with(_CountMismatchTokenizer())
    with pytest.raises(RuntimeError, match="special token"):
        core._special_token_affixes()


def test_zero_special_tokens_passes():
    core = _core_with(_NoSpecialTokenizer())
    assert core._special_token_affixes() == ([], [])
