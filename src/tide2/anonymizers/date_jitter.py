"""Privacy-preserving date shifting operator.

Adds deterministic per-patient jitter to dates, supporting multiple date formats
and optional per-patient consistency via cryptographic derivation. Standalone
weekdays are rotated by the same per-patient jitter, so the whole operator is a
deterministic function of the jitter.
"""

import contextlib
import inspect
import re
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from re import Pattern

from presidio_anonymizer.operators import Operator
from presidio_anonymizer.operators import OperatorType

# --- Date-parsing constants (named to avoid magic values in comparisons) ---
# Ordinal suffix: 11th-13th (and the 10-20 band) always take "th".
_ORDINAL_TEEN_LOW = 10
_ORDINAL_TEEN_HIGH = 20
# Length thresholds used while normalizing matched month/year groups.
_MONTH_ABBREV_LEN = 3
_TWO_DIGIT_YEAR_LEN = 2
# Minimum absolute jitter required for date anonymization.
_MIN_JITTER = 3

# --- Regular-expression grammar components (module-level constants) ---
_PRECEDING = r"(?:\b)"
_FOLLOWING = r"(?:\b)"
_FOLLOWING_TO_CUTOFF_NUM = r"(?=(?:[^0-9]*\b))"

_MON = r"(?P<month>1|2|3|4|5|6|7|8|9|01|02|03|04|05|06|07|08|09|10|11|12)"
_MONTH = r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
_DAY = r"(?P<day>1|2|3|4|5|6|7|8|9|01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31)"
# Unambiguous day-first value (13-31): too large to be a month, so a numeric
# date starting with it must be dd/mm, not mm/dd. Ambiguous leading values
# (01-12) keep the US mm/dd interpretation handled by the _MON patterns.
_DAY_FIRST = r"(?P<day>13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31)"
_DAY_SUFFIX = r"(?i:st|nd|rd|th)?"
_YEAR = r"(?P<year>(?:19|20)[0-9][0-9])"

_HOUR = r"(?P<hour>1|2|3|4|5|6|7|8|9|01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23)"
_MINUTE = r"(?P<minute>[0-5][0-9])"
_SECOND = r"(?P<second>[0-5][0-9])"
_AMPM = r"(?P<ampm>AM|A|PM|P)"
_TIMEZONE = r"(?i:CT)"
_TIME = f"(?P<time>{_HOUR}:{_MINUTE}(?::{_SECOND})?(?i:\\s*{_AMPM})?(?:\\s*{_TIMEZONE})?(?i:\\s*h)?)"

# Day of week pattern (both full and abbreviated)
_DAY_OF_WEEK = r"(?P<day_of_week>Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)"


class DateJitterAnonymizer(Operator):
    """
    Anonymizer that detects date format and replaces with a data with a jitter of the same format.
    """

    # Pre-computed separator acceptance for format functions (class-level for efficiency)
    _separator_acceptance: dict | None = None

    def __init__(self):
        """Initialize the date jitter anonymizer."""
        super().__init__()

        self.substitutions = self._compile_patterns()
        self.default_replacement = "[DATE]"

        # Pre-compute which format functions accept 'separator' parameter (once per class)
        if DateJitterAnonymizer._separator_acceptance is None:
            DateJitterAnonymizer._separator_acceptance = self._compute_separator_acceptance()

        # Month mappings
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

        # Day of week names (full and abbreviated)
        self.days_of_week_full = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        self.days_of_week_abbrev = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        self.supported_entity_types = ["DATE_TIME", "DATE"]

    @staticmethod
    def _get_day_suffix(day):
        """Generate ordinal suffix for day (st, nd, rd, th)"""
        if _ORDINAL_TEEN_LOW <= day % 100 <= _ORDINAL_TEEN_HIGH:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    @staticmethod
    def _pad_zero(num):
        """Add leading zero if single digit"""
        return f"{num:02d}"

    # Pattern 1: dd/mm/yyyy or dd-mm-yyyy
    def _format_dd_mm_yyyy(self, month, day, year, separator="/"):
        """Format as dd/mm/yyyy or dd-mm-yyyy"""
        return f"{self._pad_zero(day)}{separator}{self._pad_zero(month)}{separator}{year}"

    # Pattern 2: yyyy-mm-dd or yyyy/mm/dd
    def _format_yyyy_mm_dd(self, month, day, year, separator="/"):
        """Format as yyyy-mm-dd or yyyy/mm/dd"""
        return f"{year}{separator}{self._pad_zero(month)}{separator}{self._pad_zero(day)}"

    # Pattern 3: Month/Day/Year (full month name)
    def _format_month_day_year_full(self, month, day, year, separator="/"):
        """Format as Month/Day/Year with full month name"""
        month_name = self.month_names[month]
        return f"{month_name}{separator}{day}{separator}{year}"

    # Pattern 4: mm/dd/yyyy
    def _format_mm_dd_yyyy(self, month, day, year, separator="/"):
        """Format as mm/dd/yyyy"""
        return f"{self._pad_zero(month)}{separator}{self._pad_zero(day)}{separator}{year}"

    # Pattern 5: mm/dd/yy (00-29) - assumes 2000s
    def _format_mm_dd_yy_2000s(self, month, day, year, separator="/"):
        """Format as mm/dd/yy for years 2000-2029"""
        year_suffix = str(year)[-2:]  # Get last 2 digits
        return f"{self._pad_zero(month)}{separator}{self._pad_zero(day)}{separator}{year_suffix}"

    # Pattern 6: mm/dd/yy (30-99) - assumes 1900s
    def _format_mm_dd_yy_1900s(self, month, day, year, separator="/"):
        """Format as mm/dd/yy for years 1930-1999"""
        year_suffix = str(year)[-2:]  # Get last 2 digits
        return f"{self._pad_zero(month)}{separator}{self._pad_zero(day)}{separator}{year_suffix}"

    # Pattern 7: January 1st, 2010
    def _format_month_day_suffix_year(self, month, day, year):
        """Format as 'January 1st, 2010'"""
        month_name = self.month_names[month]
        day_suffix = self._get_day_suffix(day)
        return f"{month_name} {day}{day_suffix}, {year}"

    # Pattern 8: mm/yyyy
    def _format_mm_yyyy(self, month, day, year, separator="/"):
        """Format as mm/yyyy"""
        return f"{self._pad_zero(month)}{separator}{year}"

    # Pattern 9: March of 2010
    def _format_month_of_year(self, month, day, year):
        """Format as 'March of 2010'"""
        month_name = self.month_names[month]
        return f"{month_name} of {year}"

    # Pattern 10: March 10
    def _format_month_day(self, month, day, year, with_suffix=False):
        """Format as 'March 10' or 'March 10th'"""
        month_name = self.month_names[month]
        if with_suffix:
            day_suffix = self._get_day_suffix(day)
            return f"{month_name} {day}{day_suffix}"
        return f"{month_name} {day}"

    # Pattern 13: mm/dd (month/day without year)
    def _format_mm_dd(self, month, day, year, separator="/"):
        """Format as mm/dd"""
        return f"{self._pad_zero(month)}{separator}{self._pad_zero(day)}"

    # Pattern 13b: dd/mm (day/month without year, day-first)
    def _format_dd_mm(self, month, day, year, separator="/"):
        """Format as dd/mm"""
        return f"{self._pad_zero(day)}{separator}{self._pad_zero(month)}"

    # Pattern 14: yyyy/mm (year/month without day)
    def _format_yyyy_mm(self, month, day, year, separator="/"):
        """Format as yyyy/mm"""
        return f"{year}{separator}{self._pad_zero(month)}"

    # Pattern 11: 2008-08-09T15:11:00 (ISO format)
    def _format_iso_date(self, month, day, year, separator="-"):
        """Format as ISO date: yyyy-mm-dd"""
        return f"{year}{separator}{self._pad_zero(month)}{separator}{self._pad_zero(day)}"

    # Pattern 12: 20080809151100 (compact format)
    def _format_compact_date(self, month, day, year):
        """Format as compact date: yyyymmdd"""
        return f"{year}{self._pad_zero(month)}{self._pad_zero(day)}"

    # Pattern 15: Day of week replacement
    def _format_day_of_week(self, original_day, jitter):
        """Replace a day of the week by rotating it ``jitter`` positions.

        Deterministic: the same ``jitter`` always maps a given weekday to the
        same replacement, matching the numeric date path. The weekday can map to
        itself when ``jitter % 7 == 0``; this is intended, unbiased behavior.

        Python's ``%`` returns a non-negative result for negative operands
        (``(0 + -10) % 7 == 4``), so negative jitter is handled with no
        special-casing.
        """
        original_day_normalized = original_day.strip().title()

        if original_day_normalized in self.days_of_week_abbrev:
            table = self.days_of_week_abbrev
        elif original_day_normalized in self.days_of_week_full:
            table = self.days_of_week_full
        else:
            # Unreachable in practice: this method is only invoked after the
            # day_of_week regex group matched an exact day name. If it is ever
            # reached (e.g. a future refactor routes unrecognized input here),
            # return the input unchanged rather than fabricating a weekday.
            return original_day

        idx = table.index(original_day_normalized)
        replacement = table[(idx + jitter) % len(table)]

        # Preserve original casing
        if original_day.isupper():
            return replacement.upper()
        if original_day.islower():
            return replacement.lower()
        return replacement

    def _compile_patterns(self) -> list[tuple[Pattern, Callable]]:
        """
        Compile all date patterns used for matching.
        Returns a list of tuples containing (compiled_pattern, format_function),
        where format_function is a bound formatter method (e.g. _format_mm_dd_yyyy)
        that builds the jittered replacement string for a match.
        """

        patterns = [
            # Day of week (standalone) - should be checked first
            (
                f"{_PRECEDING}{_DAY_OF_WEEK}{_FOLLOWING}",
                self._format_day_of_week,
            ),
            # dd/mm/yyyy or dd-mm-yyyy (full month name)
            (
                f"{_PRECEDING}{_DAY}(/|-){_MONTH}\\2{_YEAR}(?:\\s*{_TIME})?{_FOLLOWING}",
                self._format_dd_mm_yyyy,
            ),
            # dd/mm/yyyy or dd-mm-yyyy (numeric, day-first). Only matches when the
            # leading number is 13-31 so it cannot be confused with mm/dd/yyyy.
            (
                f"{_PRECEDING}{_DAY_FIRST}(/|-){_MON}\\2{_YEAR}(?:\\s*{_TIME})?{_FOLLOWING}",
                self._format_dd_mm_yyyy,
            ),
            # 2008-08-09T15:11:00 (ISO format). Checked before the plain
            # yyyy-mm-dd / yyyy-mm patterns so the 'T'-separated time does not
            # get truncated by an earlier, shorter match.
            (
                f"{_PRECEDING}{_YEAR}(/|-){_MON}\\2{_DAY}(?:T\\s*{_TIME})?{_FOLLOWING}",
                self._format_iso_date,
            ),
            # yyyy-mm-dd or yyyy/mm/dd
            (
                f"{_PRECEDING}{_YEAR}(/|-){_MON}\\2{_DAY}(?:\\s*{_TIME})?{_FOLLOWING}",
                self._format_yyyy_mm_dd,
            ),
            # Month/Day/Year (full month name)
            (
                f"{_PRECEDING}{_MONTH}(/|-){_DAY}\\2{_YEAR}(?:\\s*{_TIME})?{_FOLLOWING_TO_CUTOFF_NUM}",
                self._format_month_day_year_full,
            ),
            # mm/dd/yyyy
            (
                f"{_PRECEDING}{_MON}(/|-){_DAY}\\2{_YEAR}(?:\\s*{_TIME})?{_FOLLOWING_TO_CUTOFF_NUM}",
                self._format_mm_dd_yyyy,
            ),
            # mm/dd/yy (00-29) - assumes 2000s
            (
                f"{_PRECEDING}{_MON}(/|-){_DAY}\\2(?P<year>[0-2][0-9])(?:\\s*{_TIME})?{_FOLLOWING_TO_CUTOFF_NUM}",
                self._format_mm_dd_yy_2000s,
            ),
            # mm/dd/yy (30-99) - assumes 1900s
            (
                f"{_PRECEDING}{_MON}(/|-){_DAY}\\2(?P<year>[3-9][0-9])(?:\\s*{_TIME})?{_FOLLOWING}",
                self._format_mm_dd_yy_1900s,
            ),
            # January 1st, 2010
            (
                f"{_PRECEDING}{_MONTH} +{_DAY}{_DAY_SUFFIX}(,|\\s)+{_YEAR}(?:\\s*{_TIME})?{_FOLLOWING}",
                self._format_month_day_suffix_year,
            ),
            # mm/yyyy
            (f"{_PRECEDING}{_MON}(/|-){_YEAR}{_FOLLOWING}", self._format_mm_yyyy),
            # March of 2010 / March 2010 / March2010
            (f"{_PRECEDING}{_MONTH}\\s*(?:of\\s+)?{_YEAR}", self._format_month_of_year),
            # March 10
            (f"{_PRECEDING}{_MONTH}\\s+{_DAY}{_DAY_SUFFIX}?\\b", self._format_month_day),
            # dd/mm (day/month without year, day-first). Leading 13-31 only, so it
            # cannot collide with the mm/dd interpretation below.
            (f"{_PRECEDING}{_DAY_FIRST}(/|-){_MON}\\b", self._format_dd_mm),
            # mm/dd (month/day without year)
            (f"{_PRECEDING}{_MON}(/|-){_DAY}\\b", self._format_mm_dd),
            # yyyy/mm (year/month without day)
            (f"{_PRECEDING}{_YEAR}(/|-){_MON}\\b", self._format_yyyy_mm),
            # 20080809151100 (compact format)
            (f"{_PRECEDING}{_YEAR}{_MON}{_DAY}(?:\\s*{_TIME})?", self._format_compact_date),
        ]

        return [(re.compile(pattern, re.IGNORECASE), replacement) for pattern, replacement in patterns]

    def _compute_separator_acceptance(self) -> dict:
        """
        Pre-compute which format functions accept 'separator' parameter.

        This avoids calling inspect.signature() in the hot path of operate().

        Keyed by the underlying function (``__func__``) rather than the bound
        method: this cache is class-level and shared across instances, but a
        bound method is unique per instance, so keying by the bound method would
        make every instance after the first miss the cache and silently lose the
        date separator.
        """
        acceptance = {}
        for _pattern, replacement in self.substitutions:
            key = replacement.__func__
            if key not in acceptance:
                sig = inspect.signature(replacement)
                acceptance[key] = "separator" in sig.parameters
        return acceptance

    def _get_jittered_date(self, jitter: int, date_of_service: datetime) -> dict[str, int]:
        """
        Apply jitter to date of service and return date components as dictionary.

        Returns:
            Dict with keys 'day', 'month', 'year' containing integer values
        """
        new_date = date_of_service + timedelta(days=jitter)
        return {"day": new_date.day, "month": new_date.month, "year": new_date.year}

    def operate(self, text: str, params: dict) -> str:
        """Anonymize a date string by shifting it by a deterministic jitter.

        Args:
            text: The original date text in any supported format.
            params: Operator parameters. Required/supported keys:
                - jitter (int): Number of days to shift the date. Required.
                - entity_type (str): Entity type label. Required: must be one of
                  the supported types ("DATE_TIME", "DATE"). Any other value
                  (including the absent default) raises ValueError.

        Raises:
            ValueError: If ``jitter`` is missing or ``entity_type`` is not a
                supported type.

        Returns:
            The date-shifted text preserving the original format, or the
            original text unchanged for non-date values (standalone months,
            years, gestational ages, single characters).
        """

        entity_type = params.get("entity_type", "DEFAULT")

        try:
            # Always convert to Python int (handles str, numpy.int64, etc.)
            jitter = int(params["jitter"])
        except KeyError as err:
            raise ValueError("Jitter parameter is required for date anonymization.") from err

        if entity_type not in self.supported_entity_types:
            raise ValueError(f"Entity type '{entity_type}' is not supported for date anonymization.")

        if self._is_passthrough(text):
            return text

        # Return the result for the first pattern that matches; non-date input
        # that matches nothing falls through to the default replacement.
        for pattern, replacement in self.substitutions:
            match = pattern.search(text)
            if not match:
                continue
            # The day-of-week pattern is a "standalone" fast-path: only take it
            # when the weekday is the only date content in the value. Otherwise a
            # weekday embedded in a longer date (e.g. "Monday, 2023-03-15") would
            # match here and drop the rest of the string, so keep scanning.
            if self._matched_day_of_week(match) and not self._is_standalone_weekday(match, text):
                continue
            return self._replace_match(match, replacement, jitter)

        return self.default_replacement

    @staticmethod
    def _matched_day_of_week(match: re.Match) -> bool:
        """Return True if this match captured a standalone day-of-week group."""
        try:
            return match.group("day_of_week") is not None
        except (IndexError, AttributeError):
            return False

    @staticmethod
    def _is_standalone_weekday(match: re.Match, text: str) -> bool:
        """Return True if the weekday match is the only date content in ``text``.

        The text outside the matched span may contain whitespace or punctuation
        (e.g. "Monday," or " Monday ") and still count as standalone; what must
        NOT appear is any other alphanumeric content, which would indicate the
        weekday is embedded in a longer value (e.g. "Monday, 2023-03-15").
        """
        outside = text[: match.start()] + text[match.end() :]
        return not any(char.isalnum() for char in outside)

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

    def _replace_match(self, match: re.Match, replacement, jitter: int) -> str:
        """Build the jittered replacement string for a single regex match."""
        # Standalone day of week: rotate directly (it carries no numeric date).
        if self._matched_day_of_week(match):
            return self._format_day_of_week(match.group("day_of_week"), jitter)

        month, month_format_str = self._extract_month(match)
        year_str = self._extract_year(match)
        day_str = self._extract_day(match)

        try:
            # Parse date. The result is intentionally naive: it is only used for
            # day/month/year arithmetic, so attaching a timezone would be
            # meaningless (hence the DTZ007 suppression).
            date_str = f"{month}/{day_str}/{year_str}"
            date_of_svc = datetime.strptime(date_str, f"{month_format_str}/%d/%Y")  # noqa: DTZ007
        except ValueError:
            return self.default_replacement

        jittered = self._get_jittered_date(jitter, date_of_svc)
        format_params = {
            "month": jittered["month"],
            "day": jittered["day"],
            "year": jittered["year"],
        }
        # Add separator only if the format function accepts it (pre-computed
        # lookup keyed by the underlying function, see _compute_separator_acceptance).
        if self._separator_acceptance.get(replacement.__func__, False):
            format_params["separator"] = self._extract_separator(match)

        return replacement(**format_params)

    @staticmethod
    def _extract_month(match: re.Match) -> tuple[str, str]:
        """Return (month string, strptime format) for the matched month group."""
        try:
            month = match.group("month")
        except IndexError:
            return "01", "%B"  # Default to January

        # Set month format for datetime parsing
        if len(month) < _MONTH_ABBREV_LEN:
            return month, "%m"
        if len(month) == _MONTH_ABBREV_LEN:
            return month, "%b"
        return month[:_MONTH_ABBREV_LEN], "%b"

    @staticmethod
    def _extract_year(match: re.Match) -> str:
        """Return a 4-digit year string for the matched year group."""
        try:
            year_str = match.group("year")
        except IndexError:
            return str(datetime.now(tz=UTC).year)

        # Expand 2-digit years using a sliding window around the current year.
        if len(year_str) == _TWO_DIGIT_YEAR_LEN:
            current_year = datetime.now(tz=UTC).year % 100
            year = int(year_str)
            return f"20{year_str}" if year <= current_year else f"19{year_str}"
        return year_str

    @staticmethod
    def _extract_day(match: re.Match) -> str:
        """Return the matched day group, defaulting to the middle of the month."""
        try:
            return match.group("day")
        except IndexError:
            return "15"

    @staticmethod
    def _extract_separator(match: re.Match) -> str:
        """Return the date separator captured by the pattern (defaults to '/').

        Only ``/`` and ``-`` are valid separators in the grammar. ``match.group``
        can return ``None`` for a non-participating group, and the group may hold
        a non-separator value; in either case we fall back to ``/`` so the
        separator can never leak a literal ``None`` or a date component into the
        output.
        """
        with contextlib.suppress(IndexError):
            # The separator is captured in group 2 for every separator pattern.
            candidate = match.group(2)
            if candidate in ("/", "-"):
                return candidate
        return "/"

    def validate(self, params: dict) -> None:
        """Validate operator parameters."""
        try:
            # Always convert to Python int (handles str, numpy.int64, etc.)
            jitter = int(params["jitter"])
            if abs(jitter) < _MIN_JITTER:
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
