"""Property tests for token-accurate windowing in the GPU actor (CPU-only, CI).

The actor's job is to cover **every token of every chunk** — the old path
tokenized with ``truncation=True`` and silently dropped the tail of any chunk
that tokenized past ``model_max_length``, so PHI there was never redacted. These
tests prove the replacement without a real model or GPU by driving the actor with
an injectable fake tokenizer whose chars/token ratio is tunable, so token-dense
chunks (which trigger windowing) are reproducible in CI.

Asserted (mapping to the plan's acceptance criteria):
    (a) no window exceeds the per-window token budget;
    (b) windows cover all of a chunk's tokens with no gaps (overlapping, never
        truncating);
    (c) merged predictions carry correct chunk-relative char offsets covering the
        whole chunk;
    (d) under a simulated CUDA OOM, results stay complete/ordered and the
        tokenizer is called exactly once (no re-tokenize);
    plus edge cases: within-budget parity (single window), exactly-at-budget,
    empty/zero-token, degenerate long-but-few-tokens, and multi-chunk bucketing.
"""

import json
from itertools import pairwise

from tide2.actors.transformer import TransformerInferenceActor


class _FakeTokenizerCore:
    """Stand-in TransformerCore with a tunable chars/token tokenizer.

    ``tokenize_ragged`` slices each text into contiguous tokens of
    ``chars_per_token`` characters (the last may be shorter), emitting exact char
    offsets that tile the whole string with no gaps. ``forward_windows`` returns
    one prediction per content token carrying that token's ``(start, end)`` offset,
    so a test can reconstruct exactly which characters were covered. It optionally
    raises a CUDA-OOM when a window-batch exceeds ``oom_over``.
    """

    num_special_tokens = 2

    def __init__(self, chars_per_token: int = 3, oom_over: int | None = None) -> None:
        self.chars_per_token = chars_per_token
        self.oom_over = oom_over
        self.tokenize_calls = 0
        self.forward_sizes: list[int] = []

    def tokenize_ragged(self, texts):
        self.tokenize_calls += 1
        cpt = self.chars_per_token
        input_ids, offset_mapping = [], []
        for t in texts:
            ids, offs = [], []
            k = 0
            tid = 0
            while k < len(t):
                end = min(k + cpt, len(t))
                ids.append(tid)
                offs.append((k, end))
                tid += 1
                k = end
            input_ids.append(ids)
            offset_mapping.append(offs)
        return {"input_ids": input_ids, "offset_mapping": offset_mapping}

    def forward_windows(self, windows):
        self.forward_sizes.append(len(windows))
        if self.oom_over is not None and len(windows) > self.oom_over:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 MiB.")
        return [
            [
                {"entity": "B-TOK", "score": 1.0, "start": s, "end": e, "word": text[s:e], "index": i}
                for i, (s, e) in enumerate(offs)
            ]
            for _ids, offs, text in windows
        ]


def _make_actor(
    core: _FakeTokenizerCore, *, budget: int, overlap: int, cap=lambda _t: 10**9
) -> TransformerInferenceActor:
    actor = TransformerInferenceActor.__new__(TransformerInferenceActor)
    actor._core = core
    actor._token_budget = budget
    actor._window_overlap = overlap
    actor._num_special_tokens = core.num_special_tokens
    actor._batch_cap_for_tokens = cap
    actor._handled_oom_count = 0
    return actor


def _covered_chars(preds: list[dict]) -> set[int]:
    """Set of character positions covered by a list of token predictions."""
    covered: set[int] = set()
    for p in preds:
        covered |= set(range(p["start"], p["end"]))
    return covered


class TestWindowPlanning:
    """(a) budget respected and (b) full token coverage with no gaps."""

    def test_no_window_exceeds_budget(self):
        core = _FakeTokenizerCore(chars_per_token=1)  # 1 char == 1 token
        actor = _make_actor(core, budget=10, overlap=3)
        enc = core.tokenize_ragged(["x" * 97])

        windows = actor._plan_windows(["x" * 97], enc["input_ids"], enc["offset_mapping"])

        assert windows  # a 97-token chunk over a 10-token budget must be windowed
        assert all(len(w.content_ids) <= 10 for w in windows)

    def test_windows_cover_all_tokens_no_gaps(self):
        core = _FakeTokenizerCore(chars_per_token=1)
        n = 97
        actor = _make_actor(core, budget=10, overlap=3)
        enc = core.tokenize_ragged(["x" * n])

        windows = actor._plan_windows(["x" * n], enc["input_ids"], enc["offset_mapping"])

        covered: set[int] = set()
        for w in windows:
            covered |= set(range(w.token_start, w.token_end))
        assert covered == set(range(n))  # every token, no gaps

    def test_adjacent_windows_overlap_by_configured_amount(self):
        core = _FakeTokenizerCore(chars_per_token=1)
        budget, overlap = 10, 3
        actor = _make_actor(core, budget=budget, overlap=overlap)
        enc = core.tokenize_ragged(["x" * 50])

        windows = actor._plan_windows(["x" * 50], enc["input_ids"], enc["offset_mapping"])

        # Consecutive windows step forward by budget - overlap.
        step = budget - overlap
        for prev, curr in pairwise(windows):
            assert curr.token_start - prev.token_start == step

    def test_within_budget_is_single_window(self):
        core = _FakeTokenizerCore(chars_per_token=1)
        actor = _make_actor(core, budget=10, overlap=3)
        enc = core.tokenize_ragged(["x" * 7])  # 7 tokens < budget 10

        windows = actor._plan_windows(["x" * 7], enc["input_ids"], enc["offset_mapping"])

        assert len(windows) == 1
        assert (windows[0].token_start, windows[0].token_end) == (0, 7)

    def test_exactly_at_budget_is_single_window(self):
        core = _FakeTokenizerCore(chars_per_token=1)
        actor = _make_actor(core, budget=10, overlap=3)
        enc = core.tokenize_ragged(["x" * 10])  # exactly budget

        windows = actor._plan_windows(["x" * 10], enc["input_ids"], enc["offset_mapping"])

        assert len(windows) == 1
        assert windows[0].token_end == 10

    def test_zero_token_chunk_produces_no_window(self):
        core = _FakeTokenizerCore(chars_per_token=1)
        actor = _make_actor(core, budget=10, overlap=3)
        enc = core.tokenize_ragged([""])

        windows = actor._plan_windows([""], enc["input_ids"], enc["offset_mapping"])

        assert windows == []

    def test_degenerate_long_but_few_tokens_single_window(self):
        # A 1e6-char note that tokenizes to a handful of tokens (the corpus has one)
        # must be a single within-budget window, not thousands.
        core = _FakeTokenizerCore(chars_per_token=100_000)  # 1e6 chars -> 10 tokens
        text = "y" * 1_000_000
        actor = _make_actor(core, budget=510, overlap=40)
        enc = core.tokenize_ragged([text])

        windows = actor._plan_windows([text], enc["input_ids"], enc["offset_mapping"])

        assert len(windows) == 1
        assert len(windows[0].content_ids) == 10


class TestEndToEndCoverage:
    """(c) merged predictions carry correct offsets covering the whole chunk."""

    def test_dense_chunk_predictions_cover_every_char(self):
        # 2.64 chars/token dense text: a 2048-char chunk tokenizes past a 510-token
        # budget, so the old truncating path would drop the tail. Windowing must
        # cover every character.
        core = _FakeTokenizerCore(chars_per_token=2)  # dense: 2 chars/token
        text = "a" * 1200  # 600 tokens > budget 510 -> multiple windows
        actor = _make_actor(core, budget=510, overlap=40)

        results = actor._run_inference_raw_with_oom_recovery([text])

        assert len(results) == 1
        assert _covered_chars(results[0]) == set(range(len(text)))  # no dropped tail
        # Every prediction's word matches its own offsets (offsets are correct).
        assert all(p["word"] == text[p["start"] : p["end"]] for p in results[0])

    def test_within_budget_parity_matches_single_forward(self):
        core = _FakeTokenizerCore(chars_per_token=3)
        text = "abcdefghij"  # ~4 tokens, well within budget
        actor = _make_actor(core, budget=510, overlap=40)

        results = actor._run_inference_raw_with_oom_recovery([text])

        # A within-budget chunk is one window == one forward over all its tokens.
        assert core.forward_sizes == [1]
        assert _covered_chars(results[0]) == set(range(len(text)))

    def test_zero_token_chunk_yields_empty_result(self):
        core = _FakeTokenizerCore(chars_per_token=1)
        actor = _make_actor(core, budget=10, overlap=3)

        results = actor._run_inference_raw_with_oom_recovery([""])

        assert results == [[]]

    def test_call_emits_contract_with_correct_offsets(self):
        # Drives the public __call__ to prove the emitted predictions_raw_json
        # round-trips through JSON with offsets intact.
        core = _FakeTokenizerCore(chars_per_token=2)
        text = "a" * 1200
        actor = _make_actor(core, budget=510, overlap=40)

        batch = {
            "chunk_text": [text],
            "text_hash": ["h0"],
            "chunk_id": [0],
            "char_offset_start": [0],
        }
        out = actor(batch)

        assert out["predictions_raw_json"]  # one JSON blob for the one chunk
        preds = json.loads(out["predictions_raw_json"][0])
        assert _covered_chars(preds) == set(range(len(text)))


class TestOomKeepsCoverageAndTokenizesOnce:
    """(d) OOM stays complete/ordered and the tokenizer runs exactly once."""

    def test_forced_oom_recovers_with_full_coverage_and_one_tokenize(self):
        core = _FakeTokenizerCore(chars_per_token=2, oom_over=1)  # OOM on >1 window/forward
        text = "a" * 1200  # 600 tokens -> several windows
        actor = _make_actor(core, budget=510, overlap=40)

        results = actor._run_inference_raw_with_oom_recovery([text])

        assert _covered_chars(results[0]) == set(range(len(text)))  # no lost windows
        assert core.tokenize_calls == 1  # never re-tokenized during recovery
        assert actor._handled_oom_count > 0  # the OOM path actually fired


class TestMultiChunkBucketing:
    """Bucketing groups windows by length and merges every chunk correctly."""

    def test_multiple_chunks_each_fully_covered(self):
        core = _FakeTokenizerCore(chars_per_token=1)
        actor = _make_actor(core, budget=8, overlap=2, cap=lambda _t: 2)  # small cap -> many groups
        texts = ["x" * 20, "y" * 3, "z" * 15, "w" * 9]

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert len(results) == len(texts)
        for text, preds in zip(texts, results, strict=True):
            assert _covered_chars(preds) == set(range(len(text)))
        # Every forward respected the window-batch cap of 2.
        assert all(size <= 2 for size in core.forward_sizes)

    def test_buckets_are_length_homogeneous(self):
        core = _FakeTokenizerCore(chars_per_token=1)
        actor = _make_actor(core, budget=100, overlap=0, cap=lambda _t: 3)
        # Distinct within-budget lengths -> one window each, bucketed by length.
        texts = ["a" * n for n in [7, 1, 3, 9, 2, 5, 4, 8, 6]]
        enc = core.tokenize_ragged(texts)
        windows = actor._plan_windows(texts, enc["input_ids"], enc["offset_mapping"])

        groups = actor._bucket_windows(windows)

        # Concatenating groups reproduces the fully length-sorted window order.
        flat = [windows[i] for g in groups for i in g]
        assert [len(w.content_ids) for w in flat] == sorted(len(w.content_ids) for w in windows)
        # Within each group lengths are non-decreasing (homogeneous slice).
        for g in groups:
            lengths = [len(windows[i].content_ids) for i in g]
            assert lengths == sorted(lengths)
