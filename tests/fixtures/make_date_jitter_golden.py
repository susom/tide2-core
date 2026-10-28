"""Regenerate tests/fixtures/date_jitter_golden.json (not collected by pytest).

Run after an intentional change to the date output, then review the diff of the fixture:

    uv run python tests/fixtures/make_date_jitter_golden.py
"""

import json
import random
from datetime import date
from datetime import timedelta
from pathlib import Path

from tide2.anonymizers.date_jitter import DateJitterAnonymizer

GOLDEN_PATH = Path(__file__).with_name("date_jitter_golden.json")
JITTERS = [-178, -3, 3, 5, 112, 178]
_MONTHS = [
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
_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_SAMPLE_DATES = [
    date(2023, 3, 15),
    date(2023, 3, 5),
    date(2023, 11, 5),
    date(2023, 12, 31),
    date(2024, 2, 29),
    date(1999, 12, 31),
    date(2008, 8, 9),
    date(2001, 1, 1),
    date(1935, 7, 4),
    date(1900, 1, 1),
    date(2099, 12, 31),
    date(2012, 10, 22),
    date(2029, 6, 30),
    date(1930, 2, 28),
]
# Inputs that are not dates, or that must pass through, or that cannot be parsed.
_STATIC_INPUTS = [
    "Monday",
    "Mon",
    "monday",
    "MONDAY",
    "Monday,",
    "  Friday  ",
    "2023",
    "1990s",
    "March",
    "Mar",
    "w",
    "-",
    "5",
    "21w 1d",
    "36w",
    "4d",
    "invalid date text",
    "2023-13-32",
    "13/45/2020",
    "02/30/2023",
    "2023-02-30",
    "20231305",
    "15 March",
    "Feb 29",
    "February 29th",
    "02/29",
    "2/29",
    "29/02",
    "Feb 29 2023",
    "February 29, 2023",
    "02/29/2023",
    "02/29/24",
    "02/29/00",
    "03/05/99",
    "12/31/29",
    "1/1/30",
    "00/00/0000",
    "99999999",
    "03/15/2023 - 03/20/2023",
    "2023-03-15 to 2023-04-02",
    "2023-03-15T10:30-05:30",
]


def _formats(d: date) -> list[str]:
    month, day, year = d.month, d.day, d.year
    abbrev, name = _MONTHS[month - 1][:3], _MONTHS[month - 1]
    suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    weekday = _WEEKDAYS[d.weekday()]
    return [
        f"{year}-{month:02d}-{day:02d}",
        f"{year}/{month:02d}/{day:02d}",
        f"{year}-{month}-{day}",
        f"{month}/{day}/{year}",
        f"{month:02d}/{day:02d}/{year}",
        f"{month}-{day}-{year}",
        f"{month:02d}-{day:02d}-{year}",
        f"{day}/{month}/{year}",
        f"{day:02d}/{month:02d}/{year}",
        f"{day}-{month}-{year}",
        f"{month}/{day}/{year % 100:02d}",
        f"{month:02d}/{day:02d}/{year % 100:02d}",
        f"{month:02d}-{day:02d}-{year % 100:02d}",
        f"{day:02d}/{month:02d}/{year % 100:02d}",
        f"{day:02d}-{month:02d}-{year % 100:02d}",
        f"{year}{month:02d}{day:02d}",
        f"{year}{month:02d}{day:02d}103000",
        f"{name} {day}, {year}",
        f"{abbrev} {day}, {year}",
        f"{abbrev} {day} {year}",
        f"{name.upper()} {day}, {year}",
        f"{name} {day}{suffix}, {year}",
        f"{name} {day}{suffix} {year}",
        f"{name}/{day}/{year}",
        f"{name}-{day}-{year}",
        f"{day}-{abbrev.upper()}-{year}",
        f"{day}-{abbrev}-{year}",
        f"{day:02d}-{abbrev}-{year}",
        f"{day} {name} {year}",
        f"{day} {abbrev} {year}",
        f"{day}{suffix} {name} {year}",
        f"{day} {name.lower()} {year}",
        f"{name} {year}",
        f"{abbrev} {year}",
        f"{name} of {year}",
        f"{name}{year}",
        f"{name.upper()} {year}",
        f"{month:02d}/{year}",
        f"{month}/{year}",
        f"{year}-{month:02d}",
        f"{year}/{month:02d}",
        f"{name} {day}",
        f"{abbrev} {day}",
        f"{name} {day}{suffix}",
        f"{month:02d}/{day:02d}",
        f"{month}/{day}",
        f"{day}/{month:02d}",
        f"{month:02d}-{day:02d}",
        f"{year}-{month:02d}-{day:02d} 10:30",
        f"{year}-{month:02d}-{day:02d} 10:30:15",
        f"{year}-{month:02d}-{day:02d}T10:30:00",
        f"{year}-{month:02d}-{day:02d}T23:30:00-07:00",
        f"{year}-{month:02d}-{day:02d}T23:30:00Z",
        f"{month:02d}/{day:02d}/{year} 9:05 PM",
        f"{month:02d}/{day:02d}/{year} 14:30",
        f"{name} {day}, {year} 10:30",
        f"{day}/{month}/{year} 08:15",
        f"{weekday} {year}-{month:02d}-{day:02d}",
        f"{weekday.upper()} {name} {day}, {year}",
        f"{weekday.lower()}, {year}-{month:02d}-{day:02d}",
        f"{year}-{month:02d}-{day:02d} ({weekday})",
        f"Visit on {year}-{month:02d}-{day:02d} at noon",
        f"on {name} {day}, {year}.",
    ]


def corpus() -> list[str]:
    """Return the deterministic list of inputs covered by the golden file."""
    inputs = list(_STATIC_INPUTS)
    for sample in _SAMPLE_DATES:
        inputs.extend(_formats(sample))
    rng = random.Random(20260101)  # noqa: S311
    for _ in range(40):
        sample = date(1900, 1, 1) + timedelta(days=rng.randrange(365 * 200))
        inputs.extend([sample.isoformat(), f"{sample.month:02d}/{sample.day:02d}/{sample.year}"])
    return list(dict.fromkeys(inputs))


def build() -> dict[str, list[str]]:
    """Run the anonymizer over the corpus; each value lists the outputs for ``JITTERS``."""
    anonymizer = DateJitterAnonymizer()
    return {
        text: [anonymizer.operate(text, {"entity_type": "DATE", "jitter": jitter}) for jitter in JITTERS]
        for text in corpus()
    }


def serialize(golden: dict[str, list[str]]) -> str:
    """Return the fixture text: one input per line so diffs stay reviewable."""
    lines = [
        f"{json.dumps(text, ensure_ascii=False)}: {json.dumps(outputs, ensure_ascii=False)}"
        for text, outputs in golden.items()
    ]
    return '{"jitters": ' + json.dumps(JITTERS) + ',\n"cases": {\n' + ",\n".join(lines) + "\n}}\n"


if __name__ == "__main__":
    GOLDEN_PATH.write_text(serialize(build()), encoding="utf-8")
    print(f"wrote {GOLDEN_PATH} ({len(corpus())} inputs x {len(JITTERS)} jitters)")
