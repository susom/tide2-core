"""Privacy-preserving date shifting operator.

Adds a deterministic per-patient jitter (whole days) to every date in a value. The
date tokens are rewritten in place, so the input style survives: numeric or named
month, full or abbreviated name, case, zero padding, ordinal suffix, the word
``of`` and the separators. Everything that is not part of the date (time of day,
the ``T`` separator, timezone, surrounding text) is kept verbatim, except weekday
names, which are rotated by the same jitter so they never reveal the original
weekday. The result depends only on the input and the jitter, never on the clock.
"""

import re
from datetime import date
from datetime import timedelta
from re import Match
from re import Pattern

from presidio_anonymizer.operators import Operator
from presidio_anonymizer.operators import OperatorType

from tide2.anonymizers.guarded import record_fallback
from tide2.cryptographic.date_jitter import ABSOLUTE_MIN_JITTER_DAYS

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
# example, the "30-05" in "10:30-05:30" is never read as dd-mm.
_TIME_PATTERN = re.compile(
    r"[0-9]{1,2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]+)?)?(?:[+-][0-9]{2}:[0-9]{2}|[+-][0-9]{4}(?![0-9]))?"
)

_MONTH_NUMBERS = {
    name: number
    for number, name in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1
    )
}


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
    """

    def __init__(self):
        """Initialize the date jitter anonymizer."""
        super().__init__()

        self.patterns = self._compile_patterns()
        self.default_replacement = "[DATE]"

        self.month_names = {
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

        self.month_abbrev = {
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

        self.days_of_week_full = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        self.days_of_week_abbrev = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        self.supported_entity_types = ["DATE_TIME", "DATE"]

    @staticmethod
    def _get_day_suffix(day: int) -> str:
        """Return the ordinal suffix for a day of the month (st, nd, rd, th)."""
        if _ORDINAL_TEEN_LOW <= day % 100 <= _ORDINAL_TEEN_HIGH:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    @staticmethod
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
            # January 1st, 2010
            f"{_START}{_MONTH} +{_DAY}{_DAY_SUFFIX}[,\\s]+{_YEAR}{_END}",
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
                  the supported types ("DATE_TIME", "DATE"). Any other value
                  (including the absent default) raises ValueError.
                - two_digit_year_pivot (int): Two-digit years below this value are
                  20xx, the rest 19xx. Optional, default 30.

        Raises:
            ValueError: If ``jitter`` is missing, ``entity_type`` is not a
                supported type, or ``two_digit_year_pivot`` is out of range.

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

        if self._is_passthrough(text):
            return text

        matches = self._find_dates(text)
        if not matches:
            weekday = self._standalone_weekday(text)
            if weekday is not None:
                return self._rotate_weekday(weekday, jitter)
            record_fallback(entity_type)
            return self.default_replacement

        try:
            edits = [edit for match in matches for edit in self._date_edits(match, jitter, pivot)]
        except (ValueError, OverflowError):
            record_fallback(entity_type)
            return self.default_replacement

        edits.extend(
            (weekday.start(), weekday.end(), self._rotate_weekday(weekday.group(), jitter))
            for weekday in _WEEKDAY_PATTERN.finditer(text)
        )
        for start, end, replacement in sorted(edits, reverse=True):
            text = text[:start] + replacement + text[end:]
        return text

    def _find_dates(self, text: str) -> list[Match]:
        """Return the non-overlapping date matches in ``text``, highest-priority pattern first."""
        scan = _TIME_PATTERN.sub(lambda match: " " * len(match.group()), text)
        found: list[Match] = []
        while True:
            match = next((m for m in (pattern.search(scan) for pattern in self.patterns) if m), None)
            if match is None:
                return found
            found.append(match)
            scan = scan[: match.start()] + " " * (match.end() - match.start()) + scan[match.end() :]

    def _date_edits(self, match: Match, jitter: int, pivot: int) -> list[tuple[int, int, str]]:
        """Return the (start, end, replacement) edits that shift one matched date.

        Raises:
            ValueError: If the matched tokens are not a real calendar date.
            OverflowError: If the shifted date is out of range.
        """
        tokens = match.groupdict()
        month_text = tokens["month"]
        day_text = tokens.get("day")
        year_text = tokens.get("year")

        month = int(month_text) if month_text.isdigit() else _MONTH_NUMBERS[month_text[:_ABBREV_LEN].lower()]
        day = int(day_text) if day_text is not None else _MONTH_ONLY_DAY
        if year_text is None:
            year = _REFERENCE_LEAP_YEAR
        elif len(year_text) == _TWO_DIGIT_YEAR_LEN:
            year = 2000 + int(year_text) if int(year_text) < pivot else 1900 + int(year_text)
        else:
            year = int(year_text)
        shifted = date(year, month, day) + timedelta(days=jitter)

        numeric_tokens = [token for token in (month_text, day_text) if token is not None and token.isdigit()]
        pad = all(len(token) == _PADDED_LEN for token in numeric_tokens)

        replacements = {}
        if month_text.isdigit():
            replacements["month"] = f"{shifted.month:02d}" if pad else str(shifted.month)
        else:
            replacements["month"] = self._format_month_name(month_text, shifted.month, "sep" in tokens)
        if day_text is not None:
            padded = pad if month_text.isdigit() else day_text.startswith("0")
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
        stripped_title = stripped.title()

        # Standalone month name (full or abbreviated)
        if stripped_title in self.month_names.values() or stripped_title in self.month_abbrev.values():
            return True
        # Standalone year or decade (e.g., "2016", "1990s")
        if re.match(r"^(?:19|20)\d{2}s?$", stripped):
            return True
        # Gestational ages (e.g., "21w 1d", "36w", "4d") — not PHI
        if re.match(r"^\d{1,2}w(?:\s+\d{1,2}d)?$", stripped) or re.match(r"^\d{1,2}d$", stripped):
            return True
        # Single character — NER noise (e.g., "w", "-", "/", single digit)
        return len(stripped) <= 1

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

    def operator_name(self) -> str:
        """Return the operator name."""
        return "date_jitter"

    def operator_type(self) -> OperatorType:
        """Return the operator type."""
        return OperatorType.Anonymize
