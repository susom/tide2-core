"""Correctness, format-fidelity and context tests for DateJitterAnonymizer."""

import datetime as datetime_module
import importlib.util
import random
import re
import sys
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pytest

import tide2.anonymizers.date_jitter as date_jitter_module
from tide2.anonymizers.date_jitter import DateJitterAnonymizer
from tide2.anonymizers.guarded import end_note
from tide2.anonymizers.guarded import start_note

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN_PATH = FIXTURES / "date_jitter_golden.json"
GENERATOR_PATH = FIXTURES / "make_date_jitter_golden.py"

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_date_jitter_golden", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def anonymizer():
    return DateJitterAnonymizer()


def shift(anonymizer, text, jitter, **params):
    return anonymizer.operate(text, {"entity_type": "DATE", "jitter": jitter, **params})


def _suffix(day):
    return DateJitterAnonymizer._get_day_suffix(day)


def _random_dates(count, seed, start=date(1900, 1, 1), end=date(2099, 12, 31)):
    rng = random.Random(seed)  # noqa: S311
    span = (end - start).days
    return [start + timedelta(days=rng.randrange(span + 1)) for _ in range(count)]


# Each entry: (render a date, strptime format of the rendered text, keep-date predicate).
FULL_DATE_FORMATS = {
    "iso": (lambda d: d.isoformat(), "%Y-%m-%d", lambda _: True),
    "iso_slash": (lambda d: f"{d:%Y/%m/%d}", "%Y/%m/%d", lambda _: True),
    "iso_unpadded": (lambda d: f"{d.year}-{d.month}-{d.day}", "%Y-%m-%d", lambda _: True),
    "us_padded": (lambda d: f"{d:%m/%d/%Y}", "%m/%d/%Y", lambda _: True),
    "us_unpadded": (lambda d: f"{d.month}/{d.day}/{d.year}", "%m/%d/%Y", lambda _: True),
    "us_dash": (lambda d: f"{d:%m-%d-%Y}", "%m-%d-%Y", lambda _: True),
    "compact": (lambda d: f"{d:%Y%m%d}", "%Y%m%d", lambda _: True),
    "day_first": (lambda d: f"{d.day}/{d.month}/{d.year}", "%d/%m/%Y", lambda d: d.day >= 13),
    "day_first_padded": (lambda d: f"{d:%d-%m-%Y}", "%d-%m-%Y", lambda d: d.day >= 13),
    "month_name": (lambda d: f"{MONTHS[d.month - 1]} {d.day}, {d.year}", "%B %d, %Y", lambda _: True),
    "month_abbrev": (lambda d: f"{MONTHS[d.month - 1][:3]} {d.day} {d.year}", "%b %d %Y", lambda _: True),
    "day_month_name": (lambda d: f"{d.day} {MONTHS[d.month - 1]} {d.year}", "%d %B %Y", lambda _: True),
    "dd_mon_yyyy": (lambda d: f"{d.day:02d}-{MONTHS[d.month - 1][:3]}-{d.year}", "%d-%b-%Y", lambda _: True),
    "ordinal": (
        lambda d: f"{MONTHS[d.month - 1]} {d.day}{_suffix(d.day)}, {d.year}",
        "%B %d, %Y",
        lambda _: True,
    ),
}


def _strptime(text, fmt):
    text = re.sub(r"(?<=\d)(st|nd|rd|th)", "", text)
    try:
        return datetime.strptime(text, fmt).date()  # noqa: DTZ007
    except ValueError:
        # "May" in a %b format is read as the full name, which then needs %B.
        return datetime.strptime(text, fmt.replace("%b", "%B")).date()  # noqa: DTZ007


@pytest.mark.parametrize("name", sorted(FULL_DATE_FORMATS))
def test_date_shift_property(anonymizer, name):
    """The parsed output date is exactly ``jitter`` days after the parsed input date."""
    render, fmt, keep = FULL_DATE_FORMATS[name]
    rng = random.Random(f"shift-{name}")  # noqa: S311
    dates = [d for d in _random_dates(2500, seed=name) if keep(d)][:2000]
    for d in dates:
        jitter = rng.randint(-180, 180)
        text = render(d)
        out = shift(anonymizer, text, jitter)
        assert out != "[DATE]", text
        assert (_strptime(out, fmt) - _strptime(text, fmt)).days == jitter, (text, jitter, out)


@pytest.mark.parametrize("name", ["iso", "us_padded", "compact", "month_name", "ordinal"])
def test_date_shift_is_invertible(anonymizer, name):
    """Shifting by ``-jitter`` undoes shifting by ``jitter`` (the anonymizer's own parser)."""
    render, _fmt, _keep = FULL_DATE_FORMATS[name]
    rng = random.Random(f"invert-{name}")  # noqa: S311
    for d in _random_dates(500, seed=f"inv-{name}", start=date(1905, 1, 1), end=date(2095, 12, 31)):
        jitter = rng.randint(-180, 180)
        text = render(d)
        assert shift(anonymizer, shift(anonymizer, text, jitter), -jitter) == text


def test_date_golden():
    """Output is byte-identical to the checked-in golden file."""
    generator = _load_generator()
    assert generator.serialize(generator.build()) == GOLDEN_PATH.read_text(encoding="utf-8"), (
        "date output changed; if intended, run tests/fixtures/make_date_jitter_golden.py and review the diff"
    )


def test_date_golden_covers_every_pattern(anonymizer):
    generator = _load_generator()
    first_match = {
        next(index for index, pattern in enumerate(anonymizer.patterns) if pattern.search(text))
        for text in generator.corpus()
        if any(pattern.search(text) for pattern in anonymizer.patterns)
    }
    assert first_match == set(range(len(anonymizer.patterns)))


_MONTH_TOKEN = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(MONTHS) + r"|Sept|" + "|".join(m[:3] for m in MONTHS) + r")(?![A-Za-z])", re.I
)
_WEEKDAY_TOKEN = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(WEEKDAYS) + r"|" + "|".join(w[:3] for w in WEEKDAYS) + r")(?![A-Za-z])", re.I
)


_MAY = re.compile(r"(?<![A-Za-z])may(?![A-Za-z])", re.I)


def _case_class(token):
    return "U" if token.isupper() else "l" if token.islower() else "T"


def _skeleton(text):
    """Return the style of a date text: digits are masked, names keep form and case."""
    text = _WEEKDAY_TOKEN.sub(lambda m: f"<W{'F' if len(m.group()) > 3 else 'A'}{_case_class(m.group())}>", text)
    text = _MONTH_TOKEN.sub(
        lambda m: f"<M{'F' if len(m.group()) > 3 and m.group().lower() != 'sept' else 'A'}{_case_class(m.group())}>",
        text,
    )
    text = re.sub(r"\d+", "N", text)
    return re.sub(r"(?<=N)(st|nd|rd|th)", "<S>", text, flags=re.I)


def test_date_format_roundtrip(anonymizer):
    """Output keeps the style of the input: month form, case, suffix, ``of``, separators, context."""
    cases = _load_generator().build()
    checked = 0
    for text, outputs in cases.items():
        if _MAY.search(text) or not re.search(r"\d", text):
            continue
        for out in outputs:
            if out in ("[DATE]", text) or _MAY.search(out):
                continue
            assert _skeleton(out) == _skeleton(text), (text, out)
            checked += 1
    assert checked > 1000


@pytest.mark.parametrize(
    ("text", "jitter", "expected"),
    [
        # zero padding follows the input
        ("3/5/2023", 5, "3/10/2023"),
        ("03/05/2023", 5, "03/10/2023"),
        ("03/5/2023", 5, "3/10/2023"),
        ("10/15/2023", 112, "02/04/2024"),
        ("2023-3-5", 5, "2023-3-10"),
        ("05-Mar-2023", 20, "25-Mar-2023"),
        ("15 March 2023", 112, "5 July 2023"),
        # month form and case
        ("mar 15, 2023", 5, "mar 20, 2023"),
        ("MARCH 15, 2023", 5, "MARCH 20, 2023"),
        ("March 15, 2023", 20, "April 4, 2023"),
        ("Dec 31 2023", 5, "Jan 5 2024"),
        ("Sept 15, 2023", 3, "Sept 18, 2023"),
        ("Sept 28 2023", 3, "Oct 1 2023"),
        ("May 5, 2023", 3, "May 8, 2023"),
        ("15-May-2023", 20, "4-Jun-2023"),
        # ordinal suffix and "of" only when present
        ("March 5th, 2023", 5, "March 10th, 2023"),
        ("March 1st", 3, "March 4th"),
        ("MARCH 1ST, 2023", 3, "MARCH 4TH, 2023"),
        ("March 5, 2023", 5, "March 10, 2023"),
        ("March of 2023", 112, "July of 2023"),
        ("March 2023", 112, "July 2023"),
        ("March2023", 112, "July2023"),
        # separators
        ("2023/03/15", 5, "2023/03/20"),
        ("15-03-2023", 5, "20-03-2023"),
        ("March/15/2023", 5, "March/20/2023"),
        ("15/03", 5, "20/03"),
        # two-digit years
        ("12/31/99", 5, "01/05/00"),
        ("1/1/30", 5, "1/6/30"),
        ("12/31/29", -3, "12/28/29"),
        # month-only dates are shifted from the 15th, year-only passes through
        ("03/2023", 112, "07/2023"),
        ("2023-03", 112, "2023-07"),
        ("2023", 5, "2023"),
    ],
)
def test_date_style_examples(anonymizer, text, jitter, expected):
    assert shift(anonymizer, text, jitter) == expected


@pytest.mark.parametrize("tz", ["", "Z", "-07:00", "+05:30", "-0700"])
@pytest.mark.parametrize("clock", ["T10:30:00", "T23:59:59.123", " 10:30", " 9:05 PM", "T00:00"])
def test_date_context_time_and_timezone_are_kept(anonymizer, clock, tz):
    out = shift(anonymizer, f"2023-03-15{clock}{tz}", 112)
    assert out == f"2023-07-05{clock}{tz}"


def test_date_context_surrounding_text_is_kept(anonymizer):
    assert shift(anonymizer, "Visit on 2023-03-15 at noon", 5) == "Visit on 2023-03-20 at noon"
    assert shift(anonymizer, "(03/15/2023)", 5) == "(03/20/2023)"
    assert shift(anonymizer, "on March 15, 2023.", 5) == "on March 20, 2023."


def test_date_context_offset_is_not_read_as_a_date(anonymizer):
    assert shift(anonymizer, "2023-03-15T10:30-05:30", 5) == "2023-03-20T10:30-05:30"


def test_date_context_every_date_in_a_span_is_shifted(anonymizer):
    assert shift(anonymizer, "03/15/2023 - 03/20/2023", 5) == "03/20/2023 - 03/25/2023"
    assert shift(anonymizer, "2023-03-15 to 2023-04-02", 5) == "2023-03-20 to 2023-04-07"
    assert shift(anonymizer, "Mar 15 - Mar 20", 5) == "Mar 20 - Mar 25"


def _weekday_forms(d):
    full = WEEKDAYS[d.weekday()]
    return [full, full[:3], full.upper(), full[:3].lower()]


def test_date_context_weekday_follows_the_shifted_date(anonymizer):
    """Weekday tokens are rotated: the output weekday is the true weekday of the output date."""
    rng = random.Random(99)  # noqa: S311
    for d in _random_dates(300, seed=7):
        jitter = rng.randint(-180, 180)
        shifted = d + timedelta(days=jitter)
        for form in _weekday_forms(d):
            expected = _weekday_forms(shifted)[_weekday_forms(d).index(form)]
            assert shift(anonymizer, f"{form} {d.isoformat()}", jitter) == f"{expected} {shifted.isoformat()}"
            named = f"{form}, {MONTHS[d.month - 1]} {d.day}, {d.year}"
            expected_named = f"{expected}, {MONTHS[shifted.month - 1]} {shifted.day}, {shifted.year}"
            assert shift(anonymizer, named, jitter) == expected_named


def test_date_context_weekday_is_not_copied(anonymizer):
    out = shift(anonymizer, "Wed 2023-03-15", 5)
    assert out == "Mon 2023-03-20"
    assert not out.startswith("Wed")


def test_date_no_clock(anonymizer, monkeypatch):
    """The output never depends on the current date, including for yearless Feb 29."""
    source = Path(date_jitter_module.__file__).read_text(encoding="utf-8")
    assert "now(" not in source
    assert "today(" not in source

    corpus = [*_load_generator().corpus(), "Feb 29", "February 29th", "02/29", "29/02", "2/29", "02/29/24", "02/29/00"]

    def run_at(year):
        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(year, 6, 15, tzinfo=tz)

            @classmethod
            def today(cls):
                return cls(year, 6, 15)

        class FakeDate(date):
            @classmethod
            def today(cls):
                return cls(year, 6, 15)

        with monkeypatch.context() as patch:
            patch.setattr(datetime_module, "datetime", FakeDatetime)
            patch.setattr(datetime_module, "date", FakeDate)
            patch.setattr(date_jitter_module, "date", FakeDate)
            return {text: [shift(anonymizer, text, j) for j in (-178, 5, 112)] for text in corpus}

    in_2025, in_2028 = run_at(2025), run_at(2028)
    assert in_2025 == in_2028
    for text in ("Feb 29", "February 29th", "02/29", "29/02", "2/29"):
        assert "[DATE]" not in in_2025[text]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Feb 29", "Mar 5"), ("February 29th", "March 5th"), ("02/29", "03/05"), ("29/02", "05/03")],
)
def test_date_yearless_uses_a_leap_reference_year(anonymizer, text, expected):
    assert shift(anonymizer, text, 5) == expected


def test_date_two_digit_year_pivot(anonymizer):
    """00-29 is 20xx and 30-99 is 19xx by default; the pivot is a parameter."""
    # 2000 is a leap year, 1900 is not.
    assert shift(anonymizer, "02/28/00", 2) == "03/01/00"
    assert shift(anonymizer, "02/28/00", 2, two_digit_year_pivot=0) == "03/02/00"
    assert shift(anonymizer, "02/28/40", 365) == "02/27/41"
    for bad in (-1, 101):
        with pytest.raises(ValueError, match="two_digit_year_pivot"):
            shift(anonymizer, "02/28/00", 2, two_digit_year_pivot=bad)


@pytest.mark.parametrize(
    ("text", "jitter", "expected"),
    [
        ("20231105", 5, "20231110"),
        ("20231231", 5, "20240105"),
        ("20230315", 5, "20230320"),
        ("20230315", 112, "20230705"),
        ("20230315103000", 5, "20230320103000"),
        ("15 March 2023", 112, "5 July 2023"),
        ("15 Mar 2023", 5, "20 Mar 2023"),
        ("15-MAR-2023", 112, "5-JUL-2023"),
        ("2023-03-15 10:30", 112, "2023-07-05 10:30"),
        ("2023-03-15T23:30:00-07:00", 5, "2023-03-20T23:30:00-07:00"),
        ("Wed 2023-03-15", 112, "Wed 2023-07-05"),
        ("Visit on 2023-03-15 at noon", 112, "Visit on 2023-07-05 at noon"),
        ("Mar 15, 2023", 5, "Mar 20, 2023"),
        ("MARCH 15, 2023", 5, "MARCH 20, 2023"),
        ("March 2023", 112, "July 2023"),
        ("Feb 29", 5, "Mar 5"),
        ("February 29th", 5, "March 5th"),
        ("2023-03", 112, "2023-07"),
        ("2023", 5, "2023"),
        ("3/5/2023", 5, "3/10/2023"),
    ],
)
def test_date_examples(anonymizer, text, jitter, expected):
    assert shift(anonymizer, text, jitter) == expected


@pytest.mark.parametrize(
    "text",
    [
        "invalid date text",
        "2023-13-32",
        "13/45/2020",
        "02/30/2023",
        "2023-02-30",
        "Feb 29 2023",
        "20231305",
        "15 March",
    ],
)
def test_date_fallbacks_are_recorded(anonymizer, text):
    events = start_note()
    try:
        assert shift(anonymizer, text, 5) == "[DATE]"
    finally:
        end_note()
    assert events == [("fallback", "DATE")]


@pytest.mark.parametrize("text", ["2023", "1990s", "March", "Mar", "w", "-", "5", "21w 1d", "36w", "4d", "Monday"])
def test_date_pass_throughs_record_no_fallback(anonymizer, text):
    events = start_note()
    try:
        out = shift(anonymizer, text, 5)
    finally:
        end_note()
    assert events == []
    assert out != "[DATE]"
    if text != "Monday":
        assert out == text


def test_date_iso_unchanged(anonymizer):
    """Bare ISO dates are shifted exactly by ``(date + jitter).isoformat()``."""
    rng = random.Random(2026)  # noqa: S311
    for d in _random_dates(20000, seed=2026):
        jitter = rng.randint(-180, 180)
        assert shift(anonymizer, d.isoformat(), jitter) == (d + timedelta(days=jitter)).isoformat()
