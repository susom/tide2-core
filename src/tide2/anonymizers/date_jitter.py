"""Privacy-preserving date shifting operator.

Adds a deterministic per-patient jitter (whole days) to every date in a value. The
date tokens are rewritten in place, so the input style survives: numeric or named
month, full or abbreviated name, case, zero padding, ordinal suffix, the word
``of`` and the separators. Everything that is not part of the date (time of day,
the ``T`` separator, timezone, surrounding text) is kept verbatim, except weekday
names, which are rotated by the same jitter so they never reveal the original
weekday. The result depends only on the input and the jitter, never on the clock.

Known limit: a named-month date written directly before a colon-separated clock
(``Dec 31, 99:10:30``, ``March 31:10:30``) is not protected from clock masking, so
the clock takes the two-digit year or the day. The date is then shifted without its
year (``Jan 5, 99:10:30``) or masked as ``[DATE]``. Numeric dates followed by a
clock (``2023-12-31:10:30``) are handled. Write the clock after a space, or a ``T``
for ISO values, to avoid this.
"""

import functools
import re
from datetime import date
from datetime import datetime
from datetime import timedelta
from re import Match
from re import Pattern
from typing import Any

from presidio_anonymizer.operators import Operator
from presidio_anonymizer.operators import OperatorType

from tide2.anonymizers.guarded import record_fallback
from tide2.cryptographic.date_jitter import ABSOLUTE_MIN_JITTER_DAYS
from tide2.cryptographic.date_jitter import DEFAULT_MAX_AGE
from tide2.cryptographic.date_jitter import validate_max_age
from tide2.utils.nulls import BLANK_NULL_STRINGS
from tide2.utils.nulls import is_null

# Ordinal suffix: 11th-13th (and the 10-20 band) always take "th".
_ORDINAL_TEEN_LOW = 10
_ORDINAL_TEEN_HIGH = 20
_DAYS_IN_WEEK = 7
_PADDED_LEN = 2
_TWO_DIGIT_YEAR_LEN = 2
_ABBREV_LEN = 3
_SEPTEMBER = 9
# Month-only dates (e.g. "2023-03") are shifted from the middle of the month.
_MONTH_ONLY_DAY = 15
# Leap year used for dates with no year, so that "Feb 29" always parses. It never
# appears in the output because a yearless input produces a yearless output.
_REFERENCE_LEAP_YEAR = 2000
# Two-digit years below the pivot are 20xx, the rest 19xx (00-29 -> 20xx, 30-99 -> 19xx).
_DEFAULT_TWO_DIGIT_YEAR_PIVOT = 30
_MAX_TWO_DIGIT_YEAR_PIVOT = 100
# Results are cached per process by (text, jitter, pivot); long values (mis-tagged spans) are not.
_RESULT_CACHE_SIZE = 65536
_MAX_CACHED_TEXT_LEN = 64
_FEBRUARY = 2
_LEAP_DAY = 29
_FOUR_DIGIT_YEAR_LEN = 4
# Entity types for a day, month or year held in its own column, mapped to the part they carry.
_PART_ENTITY_TYPES = {"DATE_DAY": "day", "DATE_MONTH": "month", "DATE_YEAR": "year"}

# Alternations are ordered longest-first so backtracking never settles for a prefix.
_MON = r"(?P<month>1[0-2]|0[1-9]|[1-9])"
_MONTH = (
    r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Sept|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
)
_DAY = r"(?P<day>3[01]|[12][0-9]|0[1-9]|[1-9])"
# Unambiguous day-first value (13-31): too large to be a month, so a numeric
# date starting with it must be dd/mm, not mm/dd. Ambiguous leading values
# (01-12) keep the US mm/dd interpretation handled by the _MON patterns.
_DAY_FIRST = r"(?P<day>1[3-9]|2[0-9]|3[01])"
_DAY_SUFFIX = r"(?P<suffix>st|nd|rd|th)?"
_YEAR = r"(?P<year>(?:19|20)[0-9][0-9])"
_SEP = r"(?P<sep>[/-])"
_START = r"\b"
# A date may be directly followed by the "T" of an ISO timestamp.
_END = r"(?:\b|(?=T))"
# Compact dates may be followed by an HHMMSS time (yyyymmddHHMMSS); any other trailing digit is not allowed.
_COMPACT_END = r"(?:(?![0-9])|(?=(?:[01][0-9]|2[0-3])[0-5][0-9][0-5][0-9](?![0-9])))"

_DAY_OF_WEEK = r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b"
_WEEKDAY_PATTERN = re.compile(_DAY_OF_WEEK, re.IGNORECASE)

# Times of day and UTC offsets are hidden from date matching so that, for
# example, the "30-05" in "10:30-05:30" is never read as dd-mm. A time never
# starts inside a number, and a numeric or dd-Mon-yy date directly followed by
# ":" and a clock ("2023-12-31:10:30") is kept whole, so the clock cannot take
# the day or the last year digits.
_TIME_PATTERN = re.compile(
    r"(?P<date>(?:[0-9]{4}[/-][0-9]{1,2}[/-][0-9]{1,2}|[0-9]{1,2}[/-](?:[0-9]{1,2}|[A-Za-z]{3,9})[/-][0-9]{2,4})"
    r"(?=:[0-9]))"
    r"|(?<![0-9])[0-9]{1,2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]+)?)?(?:[+-][0-9]{2}:[0-9]{2}|[+-][0-9]{4}(?![0-9]))?"
)

_MONTH_NUMBERS = {
    name: number
    for number, name in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1
    )
}

# Once every date is blanked out, a scan with no word character left cannot match any pattern.
_WORD_CHAR = re.compile(r"\w")

# Non-PHI values returned unchanged: a year or decade ("2016", "1990s") or a gestational age ("21w 1d", "36w", "4d").
_NON_DATE_VALUE = re.compile(r"(?:19|20)\d{2}s?|\d{1,2}w(?:\s+\d{1,2}d)?|\d{1,2}d")


def _blank_clock(match: Match) -> str:
    """Return the text to scan for ``match`` of ``_TIME_PATTERN``: a date is kept, a clock is blanked."""
    return match.group() if match.group("date") else " " * len(match.group())


class DateJitterAnonymizer(Operator):
    """Anonymizer that shifts every date in a value by a jitter, keeping its format.

    Dates are found with an ordered list of patterns (the first pattern that matches
    wins; further dates in the same value are shifted too). Month-only dates are
    shifted from the 15th of the month, and year-only values pass through unchanged
    (Safe Harbor allows the year). Values that hold no valid date become ``[DATE]``.

    Two-digit years follow a fixed pivot (00-29 is 20xx, 30-99 is 19xx) that can be
    changed with ``two_digit_year_pivot``. A shifted two-digit year keeps two digits,
    so a result that crosses the pivot (Dec 2029 to Jan 2030) reads in the other
    century. Dates without a year are shifted in a leap year, so a yearless result
    can differ by one day from a shift in a non-leap year across 29 February.

    With a ``birth_date``, the jitter is added first and the result is then capped:
    a shifted date later than the shifted ``max_age``-th birthday (default 90,
    birthday plus jitter) becomes that date, so every event after it gets the same
    date and no shifted date reveals an age over ``max_age``. Yearless and year-only
    values are not capped. An invalid ``max_age`` (outside 1 to 150, or not a number)
    turns the cap off instead of raising: the anonymizer does not check it, and the
    job runners reject it before they start.

    Weekday names are rotated by the jitter alone. When a date is capped its shift is
    larger than the jitter, so a weekday written next to a capped date can differ from
    the weekday of the shifted date. This is a known limit and is not corrected.

    A day, month or year held on its own (entity types ``DATE_DAY``, ``DATE_MONTH``,
    ``DATE_YEAR``) is replaced by the same part of the shifted ``full_date`` of its
    row, in the format of the input. A value that is not that part of ``full_date``,
    or a null ``full_date``, becomes ``[DATE]``. The parts of a row only agree with
    each other and with the full-date column when every call gets the same jitter
    and ``birth_date``; this operator takes the jitter as given and does not derive it.
    A four-digit year below 1000 is not zero padded (``999``, not ``0999``): the date
    patterns cover the years 1900 to 2099, so such years are outside the supported range.
    """

    # Presidio builds a new operator for every span, so per-instance setup is kept minimal.
    default_replacement = "[DATE]"
    supported_entity_types = ["DATE_TIME", "DATE", *_PART_ENTITY_TYPES]  # noqa: RUF012

    month_names = {  # noqa: RUF012
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }

    month_abbrev = {  # noqa: RUF012
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    days_of_week_full = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]  # noqa: RUF012
    days_of_week_abbrev = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]  # noqa: RUF012

    _month_words = frozenset(month_names.values()) | frozenset(month_abbrev.values())

    def __init__(self):
        """Initialize the date jitter anonymizer."""
        super().__init__()
        self.patterns = self._compile_patterns()

    @staticmethod
    def _get_day_suffix(day: int) -> str:
        """Return the ordinal suffix for a day of the month (st, nd, rd, th)."""
        if _ORDINAL_TEEN_LOW <= day % 100 <= _ORDINAL_TEEN_HIGH:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    @staticmethod
    @functools.cache
    def _compile_patterns() -> list[Pattern]:
        """Compile the date patterns in priority order (the first pattern that matches wins).

        The patterns only cover the date itself; the formatting is read back from the
        named groups (``month``, ``day``, ``year``, ``suffix``, ``sep``).
        """
        patterns = [
            # dd/Month/yyyy, dd-Mon-yyyy
            f"{_START}{_DAY}{_SEP}{_MONTH}(?P=sep){_YEAR}{_END}",
            # dd Month yyyy, dd Mon yyyy, 1st March, 2023
            f"{_START}{_DAY}{_DAY_SUFFIX}\\s+{_MONTH},?\\s+{_YEAR}{_END}",
            # dd/mm/yyyy or dd-mm-yyyy (numeric, day-first): only when the leading number is 13-31
            f"{_START}{_DAY_FIRST}{_SEP}{_MON}(?P=sep){_YEAR}{_END}",
            # yyyy-mm-dd or yyyy/mm/dd, also followed by "T" and a time
            f"{_START}{_YEAR}{_SEP}{_MON}(?P=sep){_DAY}{_END}",
            # yyyymmdd (fixed width) and yyyymmddHHMMSS
            f"{_START}{_YEAR}(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12][0-9]|3[01]){_COMPACT_END}",
            # Month/Day/Year (month name)
            f"{_START}{_MONTH}{_SEP}{_DAY}(?P=sep){_YEAR}{_END}",
            # mm/dd/yyyy
            f"{_START}{_MON}{_SEP}{_DAY}(?P=sep){_YEAR}{_END}",
            # mm/dd/yy
            f"{_START}{_MON}{_SEP}{_DAY}(?P=sep)(?P<year>[0-9]{{2}}){_END}",
            # dd/mm/yy (numeric, day-first): only when the leading number is 13-31
            f"{_START}{_DAY_FIRST}{_SEP}{_MON}(?P=sep)(?P<year>[0-9]{{2}}){_END}",
            # January 1st, 2010
            f"{_START}{_MONTH} +{_DAY}{_DAY_SUFFIX}[,\\s]+{_YEAR}{_END}",
            # Dec 31, 99 / January 1st 05
            f"{_START}{_MONTH} +{_DAY}{_DAY_SUFFIX}[,\\s]+(?P<year>[0-9]{{2}}){_END}",
            # mm/yyyy
            f"{_START}{_MON}{_SEP}{_YEAR}{_END}",
            # March of 2010 / March 2010 / March2010
            f"{_START}{_MONTH}\\s*(?:of\\s+)?{_YEAR}{_END}",
            # March 10, March 10th
            f"{_START}{_MONTH}\\s+{_DAY}{_DAY_SUFFIX}\\b",
            # dd/mm (day-first, no year): only when the leading number is 13-31
            f"{_START}{_DAY_FIRST}{_SEP}{_MON}\\b",
            # mm/dd (no year)
            f"{_START}{_MON}{_SEP}{_DAY}\\b",
            # yyyy/mm (no day)
            f"{_START}{_YEAR}{_SEP}{_MON}\\b",
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def operate(self, text: str, params: dict) -> str:
        """Anonymize a date string by shifting every date in it by a deterministic jitter.

        Args:
            text: The original date text in any supported format. It may carry a time,
                timezone, weekday or other surrounding text.
            params: Operator parameters. Required/supported keys:
                - jitter (int): Number of days to shift the date. Required.
                - entity_type (str): Entity type label. Required: must be one of
                  the supported types ("DATE_TIME", "DATE", "DATE_DAY",
                  "DATE_MONTH", "DATE_YEAR"). Any other value (including the
                  absent default) raises ValueError.
                - full_date (date, datetime, numpy.datetime64 or ISO string): The
                  real full date of the row. Required for the part types, where
                  null gives ``[DATE]``; ignored for the others.
                - two_digit_year_pivot (int): Two-digit years below this value are
                  20xx, the rest 19xx. Optional, default 30.
                - birth_date (date, datetime, numpy.datetime64 or ISO string): The
                  real birth date. Optional; null or absent turns the age cap off.
                - max_age (int): The age at which dates are capped: after the jitter,
                  a date later than the shifted ``max_age``-th birthday becomes that
                  date. Optional, default 90. A value outside 1 to 150 (or not a
                  number) turns the cap off instead of raising; the job runners
                  reject it before they start.

        Raises:
            ValueError: If ``jitter`` is missing, ``entity_type`` is not a
                supported type, ``two_digit_year_pivot`` is out of range,
                ``birth_date`` or ``full_date`` is not a date, or a part
                type has no ``full_date`` key.

        Returns:
            The text with each date shifted and its format kept, time and
            surrounding text unchanged and weekday names rotated by the same
            jitter. Non-date values (standalone months, years, gestational ages,
            single characters) are returned unchanged. A value with no valid date
            becomes ``[DATE]``.
        """
        entity_type = params.get("entity_type", "DEFAULT")

        try:
            # Always convert to Python int (handles str, numpy.int64, etc.)
            jitter = int(params["jitter"])
        except KeyError as err:
            raise ValueError("Jitter parameter is required for date anonymization.") from err

        if entity_type not in self.supported_entity_types:
            raise ValueError(f"Entity type '{entity_type}' is not supported for date anonymization.")

        pivot = int(params.get("two_digit_year_pivot", _DEFAULT_TWO_DIGIT_YEAR_PIVOT))
        if not 0 <= pivot <= _MAX_TWO_DIGIT_YEAR_PIVOT:
            raise ValueError("two_digit_year_pivot must be between 0 and 100.")

        birth_date = _parse_date_param(params.get("birth_date"), "birth_date")
        cap = _age_cap_or_none(birth_date, params.get("max_age", DEFAULT_MAX_AGE)) if birth_date is not None else None

        part = _PART_ENTITY_TYPES.get(entity_type)
        if part is not None:
            result, fell_back = self._operate_part(text, part, params, jitter, cap)
        else:
            shift = _shift_cached if len(text) <= _MAX_CACHED_TEXT_LEN else self._shift
            result, fell_back = shift(text, jitter, pivot, cap)
        if fell_back:
            record_fallback(entity_type)
        return result

    def _operate_part(self, text: str, part: str, params: dict, jitter: int, cap: date | None) -> tuple[str, bool]:
        """Return the shifted day, month or year ``text`` and whether it fell back to ``[DATE]``."""
        if "full_date" not in params:
            raise ValueError("full_date parameter is required for date part anonymization.")
        full_date = _parse_date_param(params["full_date"], "full_date")
        if full_date is None:
            return self.default_replacement, True
        shift = _shift_part_cached if len(text) <= _MAX_CACHED_TEXT_LEN else self._shift_part
        return shift(text, part, full_date, jitter, cap)

    def _shift_part(
        self, text: str, part: str, full_date: date, jitter: int, cap: date | None = None
    ) -> tuple[str, bool]:
        """Return ``text`` as the same part of the shifted ``full_date``, and whether it fell back to ``[DATE]``.

        ``text`` must be that part of ``full_date`` (the real, unshifted date); the
        result keeps its zero padding, month name form and case, and year width.
        """
        token = text.strip()
        try:
            shifted = _shifted_date(full_date, jitter, cap)
        except OverflowError:
            return self.default_replacement, True

        result = None
        if part == "month":
            result = self._shift_month_token(token, full_date.month, shifted.month)
        elif part == "day":
            result = self._shift_day_token(token, full_date.day, shifted.day)
        elif part == "year":
            result = self._shift_year_token(token, full_date.year, shifted.year)
        if result is None:
            return self.default_replacement, True
        return result, False

    @staticmethod
    def _shift_day_token(token: str, day: int, shifted_day: int) -> str | None:
        """Return ``shifted_day`` in the format of ``token``, or None when ``token`` is not ``day``."""
        if not _is_number(token, 1, _PADDED_LEN) or int(token) != day:
            return None
        return f"{shifted_day:02d}" if _is_zero_padded(token) else str(shifted_day)

    def _shift_month_token(self, token: str, month: int, shifted_month: int) -> str | None:
        """Return ``shifted_month`` in the format of ``token``, or None when ``token`` is not ``month``."""
        if _is_number(token, 1, _PADDED_LEN):
            if int(token) != month:
                return None
            return f"{shifted_month:02d}" if _is_zero_padded(token) else str(shifted_month)
        lowered = token.lower()
        if token.title() not in self._month_words and lowered != "sept":
            return None
        if _MONTH_NUMBERS[lowered[:_ABBREV_LEN]] != month:
            return None
        return self._format_month_name(token, shifted_month, False)

    @staticmethod
    def _shift_year_token(token: str, year: int, shifted_year: int) -> str | None:
        """Return ``shifted_year`` in the width of ``token``, or None when ``token`` is not ``year``."""
        if _is_number(token, _FOUR_DIGIT_YEAR_LEN):
            return str(shifted_year) if int(token) == year else None
        if _is_number(token, _TWO_DIGIT_YEAR_LEN):
            return f"{shifted_year % 100:02d}" if int(token) == year % 100 else None
        return None

    def _shift(self, text: str, jitter: int, pivot: int, cap: date | None = None) -> tuple[str, bool]:
        """Return the shifted ``text`` and whether it fell back to ``[DATE]``.

        Dates with a year are shifted by ``jitter`` and then limited to ``cap + jitter``,
        the shifted age cap. The result depends only on the arguments, so ``operate``
        caches it.
        """
        if self._is_passthrough(text):
            return text, False

        matches = self._find_dates(text)
        if not matches:
            weekday = self._standalone_weekday(text)
            if weekday is not None:
                return self._rotate_weekday(weekday, jitter), False
            return self.default_replacement, True

        try:
            edits = [edit for match in matches for edit in self._date_edits(match, jitter, pivot, cap)]
        except (ValueError, OverflowError):
            return self.default_replacement, True

        edits.extend(
            (weekday.start(), weekday.end(), self._rotate_weekday(weekday.group(), jitter))
            for weekday in _WEEKDAY_PATTERN.finditer(text)
        )
        for start, end, replacement in sorted(edits, reverse=True):
            text = text[:start] + replacement + text[end:]
        return text, False

    def _find_dates(self, text: str) -> list[Match]:
        """Return the non-overlapping date matches in ``text``, highest-priority pattern first."""
        scan = _TIME_PATTERN.sub(_blank_clock, text) if ":" in text else text
        found: list[Match] = []
        while _WORD_CHAR.search(scan):
            match = next((m for m in (pattern.search(scan) for pattern in self.patterns) if m), None)
            if match is None:
                break
            found.append(match)
            scan = scan[: match.start()] + " " * (match.end() - match.start()) + scan[match.end() :]
        return found

    def _date_edits(self, match: Match, jitter: int, pivot: int, cap: date | None = None) -> list[tuple[int, int, str]]:
        """Return the (start, end, replacement) edits that shift one matched date.

        A date with a year is limited to ``cap + jitter`` after the shift.

        Raises:
            ValueError: If the matched tokens are not a real calendar date.
            OverflowError: If the shifted date is out of range.
        """
        tokens = match.groupdict()
        month_text = tokens["month"]
        day_text = tokens.get("day")
        year_text = tokens.get("year")

        month_numeric = month_text.isdigit()
        month = int(month_text) if month_numeric else _MONTH_NUMBERS[month_text[:_ABBREV_LEN].lower()]
        day = int(day_text) if day_text is not None else _MONTH_ONLY_DAY
        if year_text is None:
            year = _REFERENCE_LEAP_YEAR
        elif len(year_text) == _TWO_DIGIT_YEAR_LEN:
            year = 2000 + int(year_text) if int(year_text) < pivot else 1900 + int(year_text)
        else:
            year = int(year_text)
        shifted = _shifted_date(date(year, month, day), jitter, cap if year_text is not None else None)

        # Numeric dates keep zero padding only when every numeric part was padded.
        pad = len(month_text) == _PADDED_LEN and (day_text is None or len(day_text) == _PADDED_LEN)

        replacements = {}
        if month_numeric:
            replacements["month"] = f"{shifted.month:02d}" if pad else str(shifted.month)
        else:
            replacements["month"] = self._format_month_name(month_text, shifted.month, "sep" in tokens)
        if day_text is not None:
            padded = pad if month_numeric else day_text.startswith("0")
            replacements["day"] = f"{shifted.day:02d}" if padded else str(shifted.day)
        if year_text is not None:
            two_digit = len(year_text) == _TWO_DIGIT_YEAR_LEN
            replacements["year"] = f"{shifted.year % 100:02d}" if two_digit else str(shifted.year)
        if tokens.get("suffix"):
            suffix = self._get_day_suffix(shifted.day)
            replacements["suffix"] = suffix.upper() if tokens["suffix"].isupper() else suffix

        return [(*match.span(group), value) for group, value in replacements.items()]

    def _format_month_name(self, token: str, month: int, separated: bool) -> str:
        """Return ``month`` as a name in the full/abbreviated form and case of ``token``.

        "May" is both a full name and an abbreviation, so it counts as abbreviated
        only inside separator-delimited dates such as ``15-May-2023``.
        """
        lowered = token.lower()
        full = len(lowered) > _ABBREV_LEN and lowered != "sept"
        if lowered == "may":
            full = not separated
        if full:
            name = self.month_names[month]
        elif lowered == "sept" and month == _SEPTEMBER:
            name = "Sept"
        else:
            name = self.month_abbrev[month]

        if token.isupper():
            return name.upper()
        if token.islower():
            return name.lower()
        return name

    def _rotate_weekday(self, token: str, jitter: int) -> str:
        """Rotate a weekday name by ``jitter`` days, keeping full/abbreviated form and case.

        Python's ``%`` returns a non-negative result for negative operands, so a
        negative jitter needs no special case. The weekday maps to itself when
        ``jitter % 7 == 0``, which is intended.
        """
        normalized = token.title()
        table = self.days_of_week_abbrev if len(normalized) == _ABBREV_LEN else self.days_of_week_full
        replacement = table[(table.index(normalized) + jitter) % _DAYS_IN_WEEK]

        if token.isupper():
            return replacement.upper()
        if token.islower():
            return replacement.lower()
        return replacement

    @staticmethod
    def _standalone_weekday(text: str) -> str | None:
        """Return the weekday when it is the only alphanumeric content of ``text``.

        Whitespace and punctuation around it (e.g. "Monday," or " Monday ") are
        allowed; a weekday inside a longer value is not standalone.
        """
        weekday = _WEEKDAY_PATTERN.search(text)
        if weekday is None:
            return None
        outside = text[: weekday.start()] + text[weekday.end() :]
        return None if any(char.isalnum() for char in outside) else weekday.group()

    def _is_passthrough(self, text: str) -> bool:
        """Return True for values that should be returned unchanged (non-PHI)."""
        stripped = text.strip()
        # Single character: NER noise (e.g., "w", "-", "/", single digit)
        if len(stripped) <= 1:
            return True
        # Standalone month name (full or abbreviated)
        if stripped.title() in self._month_words:
            return True
        return _NON_DATE_VALUE.fullmatch(stripped) is not None

    def validate(self, params: dict) -> None:
        """Validate operator parameters."""
        try:
            # Always convert to Python int (handles str, numpy.int64, etc.)
            jitter = int(params["jitter"])
            if abs(jitter) < ABSOLUTE_MIN_JITTER_DAYS:
                raise ValueError("Jitter must have absolute value >= 3 for date anonymization.")
        except KeyError as err:
            raise ValueError(
                "Jitter parameter is required and should be greater than 3 for date anonymization."
            ) from err

        entity_type = params.get("entity_type", "DEFAULT")

        if entity_type not in self.supported_entity_types:
            raise ValueError(f"Entity type '{entity_type}' is not supported for DateJitterAnonymizer.")

        if entity_type in _PART_ENTITY_TYPES and "full_date" not in params:
            raise ValueError(f"full_date parameter is required for {entity_type} anonymization.")

    def operator_name(self) -> str:
        """Return the operator name."""
        return "date_jitter"

    def operator_type(self) -> OperatorType:
        """Return the operator type."""
        return OperatorType.Anonymize


@functools.lru_cache(maxsize=_RESULT_CACHE_SIZE)
def _shift_cached(text: str, jitter: int, pivot: int, cap: date | None) -> tuple[str, bool]:
    """Cache ``DateJitterAnonymizer._shift`` for the process (Presidio builds a new operator per span)."""
    return DateJitterAnonymizer()._shift(text, jitter, pivot, cap)


@functools.lru_cache(maxsize=_RESULT_CACHE_SIZE)
def _shift_part_cached(text: str, part: str, full_date: date, jitter: int, cap: date | None) -> tuple[str, bool]:
    """Cache ``DateJitterAnonymizer._shift_part`` for the process."""
    return DateJitterAnonymizer()._shift_part(text, part, full_date, jitter, cap)


def _shifted_date(parsed: date, jitter: int, cap: date | None) -> date:
    """Return ``parsed`` plus ``jitter`` days, limited to ``cap`` plus ``jitter`` days when ``cap`` is set.

    Raises:
        OverflowError: If the result is out of range.
    """
    shifted = parsed + timedelta(days=jitter)
    if cap is not None:
        shifted = min(shifted, cap + timedelta(days=jitter))
    return shifted


def _is_number(token: str, *lengths: int) -> bool:
    """Return True when ``token`` is ASCII digits and its length is one of ``lengths``."""
    return token.isascii() and token.isdigit() and len(token) in lengths


def _is_zero_padded(token: str) -> bool:
    """Return True for a two-digit token with a leading zero ("05")."""
    return len(token) == _PADDED_LEN and token.startswith("0")


def _parse_date_param(value: Any, name: str) -> date | None:
    """Return the operator param ``value`` as a date, or None when it is null or blank.

    Raises:
        ValueError: If ``value`` is not a date, a datetime or an ISO date string.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_iso_date(value, name)
    if is_null(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    # numpy.datetime64 and similar: their str() is ISO.
    return _parse_iso_date(str(value), name)


@functools.lru_cache(maxsize=_RESULT_CACHE_SIZE)
def _parse_iso_date(value: str, name: str) -> date | None:
    """Return the date of an ISO string, or None when it is blank or a null marker."""
    text = value.strip()
    if text in BLANK_NULL_STRINGS:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError as err:
        raise ValueError(f"{name} is not an ISO date: {text!r}") from err


@functools.cache
def _age_cap(birth_date: date, max_age: int) -> date:
    """Return the ``max_age``-th birthday; a 29 February birth has it on 1 March in a non-leap year."""
    year = birth_date.year + max_age
    if birth_date.month == _FEBRUARY and birth_date.day == _LEAP_DAY:
        try:
            return date(year, _FEBRUARY, _LEAP_DAY)
        except ValueError:
            return date(year, 3, 1)
    return birth_date.replace(year=year)


def _age_cap_or_none(birth_date: date, max_age: Any) -> date | None:
    """Return the age cap, or None when ``max_age`` is not a valid age (the cap is then off)."""
    try:
        age = validate_max_age(max_age)
    except (ValueError, TypeError):
        return None
    return _age_cap(birth_date, age)
