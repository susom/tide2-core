"""Privacy-preserving date shifting operator.

Adds deterministic per-patient jitter to dates, supporting multiple date formats
and optional per-patient consistency via cryptographic derivation. Standalone
weekdays are rotated by the same per-patient jitter, so the whole operator is a
deterministic function of the jitter.
"""

import inspect
import re
from datetime import datetime
from datetime import timedelta
from re import Pattern

from presidio_anonymizer.operators import Operator
from presidio_anonymizer.operators import OperatorType


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
        if 10 <= day % 100 <= 20:
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

    def _compile_patterns(self) -> list[tuple[Pattern, str]]:
        """
        Compile all date patterns used for matching.
        Returns a list of tuples containing (compiled_pattern, replacement_template)
        """

        # Regular expression components
        PRECEDING = r"(?:\b)"
        FOLLOWING = r"(?:\b)"
        FOLLOWING_TO_CUTOFF_NUM = r"(?=(?:[^0-9]*\b))"

        MON = r"(?P<month>1|2|3|4|5|6|7|8|9|01|02|03|04|05|06|07|08|09|10|11|12)"
        MONTH = r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
        DAY = r"(?P<day>1|2|3|4|5|6|7|8|9|01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31)"
        DAY_SUFFIX = r"(?i:st|nd|rd|th)?"
        YEAR = r"(?P<year>(?:19|20)[0-9][0-9])"

        HOUR = r"(?P<hour>1|2|3|4|5|6|7|8|9|01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23)"
        MINUTE = r"(?P<minute>[0-5][0-9])"
        SECOND = r"(?P<second>[0-5][0-9])"
        AMPM = r"(?P<ampm>AM|A|PM|P)"
        TIMEZONE = r"(?i:CT)"
        TIME = f"(?P<time>{HOUR}:{MINUTE}(?::{SECOND})?(?i:\\s*{AMPM})?(?:\\s*{TIMEZONE})?(?i:\\s*h)?)"

        # Day of week pattern (both full and abbreviated)
        DAY_OF_WEEK = (
            r"(?P<day_of_week>Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
        )

        patterns = [
            # Day of week (standalone) - should be checked first
            (
                f"{PRECEDING}{DAY_OF_WEEK}{FOLLOWING}",
                self._format_day_of_week,
            ),
            # dd/mm/yyyy or dd-mm-yyyy
            (
                f"{PRECEDING}{DAY}(/|-){MONTH}\\2{YEAR}(?:\\s*{TIME})?{FOLLOWING}",
                self._format_dd_mm_yyyy,
            ),
            # yyyy-mm-dd or yyyy/mm/dd
            (
                f"{PRECEDING}{YEAR}(/|-){MON}\\2{DAY}(?:\\s*{TIME})?{FOLLOWING}",
                self._format_yyyy_mm_dd,
            ),
            # Month/Day/Year (full month name)
            (
                f"{PRECEDING}{MONTH}(/|-){DAY}\\2{YEAR}(?:\\s*{TIME})?{FOLLOWING_TO_CUTOFF_NUM}",
                self._format_month_day_year_full,
            ),
            # mm/dd/yyyy
            (
                f"{PRECEDING}{MON}(/|-){DAY}\\2{YEAR}(?:\\s*{TIME})?{FOLLOWING_TO_CUTOFF_NUM}",
                self._format_mm_dd_yyyy,
            ),
            # mm/dd/yy (00-29) - assumes 2000s
            (
                f"{PRECEDING}{MON}(/|-){DAY}\\2(?P<year>[0-2][0-9])(?:\\s*{TIME})?{FOLLOWING_TO_CUTOFF_NUM}",
                self._format_mm_dd_yy_2000s,
            ),
            # mm/dd/yy (30-99) - assumes 1900s
            (
                f"{PRECEDING}{MON}(/|-){DAY}\\2(?P<year>[3-9][0-9])(?:\\s*{TIME})?{FOLLOWING}",
                self._format_mm_dd_yy_1900s,
            ),
            # January 1st, 2010
            (
                f"{PRECEDING}{MONTH} +{DAY}{DAY_SUFFIX}(,|\\s)+{YEAR}(?:\\s*{TIME})?{FOLLOWING}",
                self._format_month_day_suffix_year,
            ),
            # mm/yyyy
            (f"{PRECEDING}{MON}(/|-){YEAR}{FOLLOWING}", self._format_mm_yyyy),
            # March of 2010 / March 2010 / March2010
            (f"{PRECEDING}{MONTH}\\s*(?:of\\s+)?{YEAR}", self._format_month_of_year),
            # March 10
            (f"{PRECEDING}{MONTH}\\s+{DAY}{DAY_SUFFIX}?\\b", self._format_month_day),
            # mm/dd (month/day without year)
            (f"{PRECEDING}{MON}(/|-){DAY}\\b", self._format_mm_dd),
            # yyyy/mm (year/month without day)
            (f"{PRECEDING}{YEAR}(/|-){MON}\\b", self._format_yyyy_mm),
            # 2008-08-09T15:11:00 (ISO format)
            (
                f"{PRECEDING}{YEAR}(/|-){MON}\\2{DAY}(T?:\\s*{TIME})?",
                self._format_iso_date,
            ),
            # 20080809151100 (compact format)
            (f"{PRECEDING}{YEAR}{MON}{DAY}(?:\\s*{TIME})?", self._format_compact_date),
        ]

        return [(re.compile(pattern, re.IGNORECASE), replacement) for pattern, replacement in patterns]

    def _compute_separator_acceptance(self) -> dict:
        """
        Pre-compute which format functions accept 'separator' parameter.

        This avoids calling inspect.signature() in the hot path of operate().
        """
        acceptance = {}
        for _pattern, replacement in self.substitutions:
            if replacement not in acceptance:
                sig = inspect.signature(replacement)
                acceptance[replacement] = "separator" in sig.parameters
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
                - entity_type (str): Entity type label (e.g. "DATE_TIME").
                  Defaults to "DEFAULT".

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

        if entity_type in self.supported_entity_types:
            # Standalone month name (full or abbreviated) — pass through unchanged
            stripped = text.strip()
            stripped_title = stripped.title()
            if stripped_title in self.month_names.values() or stripped_title in self.month_abbrev.values():
                return text

            # Standalone year or decade (e.g., "2016", "1990s") — pass through unchanged
            if re.match(r"^(?:19|20)\d{2}s?$", stripped):
                return text

            # Gestational ages (e.g., "21w 1d", "36w", "4d") — not PHI, pass through
            if re.match(r"^\d{1,2}w(?:\s+\d{1,2}d)?$", stripped) or re.match(r"^\d{1,2}d$", stripped):
                return text

            # Single character — NER noise (e.g., "w", "-", "/", single digit), not PHI
            if len(stripped) <= 1:
                return text

            has_year = False
            has_day = False
            has_month = False

            for pattern, replacement in self.substitutions:
                date_of_svc = None
                month_format_str = "%B"  # Full month name

                match = pattern.search(text)
                if not match:
                    continue  # Skip to next pattern if no match

                # Special handling for day of week pattern
                try:
                    day_of_week = match.group("day_of_week")
                    # This is a standalone day of week. Called directly (not via
                    # the generic replacement(**format_params) dispatch below),
                    # so threading the extra `jitter` arg here is safe; a future
                    # refactor that routes this through generic dispatch must
                    # supply `jitter` another way.
                    return self._format_day_of_week(day_of_week, jitter)
                except (IndexError, AttributeError):
                    # Not a day of week pattern, continue with regular date processing
                    pass

                try:
                    month = match.group("month")
                    has_month = True
                    # Set month format for datetime parsing
                    if len(month) < 3:
                        month_format_str = "%m"
                    elif len(month) == 3:
                        month_format_str = "%b"
                    else:
                        month_format_str = "%b"
                        month = month[:3]
                except IndexError:
                    # If month is not found set a default value
                    month = "01"  # Default to January

                # Get year
                try:
                    year_str = match.group("year")
                    has_year = True
                    # Handle 2-digit years
                    if len(year_str) == 2:
                        current_year = datetime.now().year % 100
                        year = int(year_str)
                        if year <= current_year:
                            year_str = f"20{year_str}"
                        else:
                            year_str = f"19{year_str}"
                except IndexError:
                    year_str = str(datetime.now().year)

                # Get day
                try:
                    day_str = match.group("day")
                    has_day = True
                except IndexError:
                    day_str = "15"  # Default to middle of month

                try:
                    # Parse date
                    date_str = f"{month}/{day_str}/{year_str}"
                    date_of_svc = datetime.strptime(date_str, f"{month_format_str}/%d/%Y")
                    jittered_date_dict = self._get_jittered_date(jitter, date_of_svc)

                    # Extract separator if available
                    separator = "/"
                    try:
                        # The separator is typically in group 2 for most patterns
                        separator = match.group(2)
                    except (IndexError, AttributeError):
                        try:
                            # Fallback to group 1 for some patterns
                            separator = match.group(1)
                        except (IndexError, AttributeError):
                            pass

                    # Use the replacement for the given pattern
                    # Use pre-computed separator acceptance (avoid inspect.signature in hot path)
                    format_params = {
                        "month": jittered_date_dict["month"],
                        "day": jittered_date_dict["day"],
                        "year": jittered_date_dict["year"],
                    }

                    # Add separator if the function accepts it (using pre-computed lookup)
                    if self._separator_acceptance.get(replacement, False):
                        format_params["separator"] = separator

                    new_date = replacement(**format_params)

                    return new_date

                except ValueError:
                    return self.default_replacement

            if not (has_year or has_day or has_month):
                # If all of the components are missing, return default replacement
                return self.default_replacement
            return self.default_replacement
        raise ValueError(f"Entity type '{entity_type}' is not supported for date anonymization.")

    def validate(self, params: dict) -> None:
        """Validate operator parameters."""
        try:
            # Always convert to Python int (handles str, numpy.int64, etc.)
            jitter = int(params["jitter"])
            if abs(jitter) < 3:
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
