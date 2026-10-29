"""
Unit tests for DateJitterAnonymizer.

Tests cover date jittering functionality with various formats including:
- ISO dates (yyyy-mm-dd)
- US dates (mm/dd/yyyy)
- European dates (dd/mm/yyyy)
- Compact dates (yyyymmdd)
- Date with time components
- Month/year only formats
- Jitter validation

The format-fidelity, context, golden and property tests live in
test_date_jitter_format.py.
"""

import random
from datetime import date
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from tide2.anonymizers.date_jitter import DateJitterAnonymizer
from tide2.anonymizers.date_jitter import _shift_cached
from tide2.anonymizers.guarded import end_note
from tide2.anonymizers.guarded import guarded
from tide2.anonymizers.guarded import start_note


class TestDateJitterAnonymizer:
    """Test DateJitterAnonymizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.anonymizer = DateJitterAnonymizer()

    def test_initialization(self):
        """Test DateJitterAnonymizer initialization."""
        assert self.anonymizer.supported_entity_types == ["DATE_TIME", "DATE", "DATE_DAY", "DATE_MONTH", "DATE_YEAR"]
        assert hasattr(self.anonymizer, "month_names")
        assert hasattr(self.anonymizer, "month_abbrev")
        assert hasattr(self.anonymizer, "patterns")
        assert self.anonymizer.default_replacement == "[DATE]"

    def test_operator_name(self):
        """Test operator_name method."""
        assert self.anonymizer.operator_name() == "date_jitter"

    def test_operator_type(self):
        """Test operator_type method."""
        from presidio_anonymizer.operators import OperatorType

        assert self.anonymizer.operator_type() == OperatorType.Anonymize

    # Test validation
    def test_validate_valid_params(self):
        """Test validate method with valid parameters."""
        params = {"entity_type": "DATE", "jitter": 10}
        # Should not raise any exception
        self.anonymizer.validate(params)

    def test_validate_missing_jitter(self):
        """Test validate method with missing jitter parameter."""
        params = {"entity_type": "DATE"}
        with pytest.raises(ValueError, match="Jitter parameter is required"):
            self.anonymizer.validate(params)

    def test_validate_jitter_too_small(self):
        """Test validate method with jitter value too small."""
        params = {
            "entity_type": "DATE",
            "jitter": 2,  # Less than required minimum of 3
        }
        with pytest.raises(ValueError, match="Jitter must have absolute value >= 3"):
            self.anonymizer.validate(params)

    def test_validate_invalid_entity_type(self):
        """Test validate method with invalid entity type."""
        params = {"entity_type": "INVALID", "jitter": 10}
        with pytest.raises(ValueError, match="Entity type 'INVALID' is not supported"):
            self.anonymizer.validate(params)

    def test_validate_negative_jitter(self):
        """Test validate method with negative jitter (should be valid if abs > 3)."""
        params = {"entity_type": "DATE", "jitter": -10}
        # Should not raise exception since abs(-10) > 3
        self.anonymizer.validate(params)

    # Test month mappings
    def test_month_names_mapping(self):
        """Test month names dictionary is properly initialized."""
        assert self.anonymizer.month_names[1] == "January"
        assert self.anonymizer.month_names[12] == "December"
        assert len(self.anonymizer.month_names) == 12

    def test_month_abbrev_mapping(self):
        """Test month abbreviations dictionary is properly initialized."""
        assert self.anonymizer.month_abbrev[1] == "Jan"
        assert self.anonymizer.month_abbrev[12] == "Dec"
        assert len(self.anonymizer.month_abbrev) == 12

    # Test helper methods
    def test_get_day_suffix(self):
        """Test _get_day_suffix static method."""
        test_cases = [
            (1, "st"),
            (21, "st"),
            (31, "st"),
            (2, "nd"),
            (22, "nd"),
            (3, "rd"),
            (23, "rd"),
            (4, "th"),
            (5, "th"),
            (11, "th"),
            (12, "th"),
            (13, "th"),
            (14, "th"),
            (15, "th"),
            (20, "th"),
            (24, "th"),
        ]
        for day, expected in test_cases:
            result = DateJitterAnonymizer._get_day_suffix(day)
            assert result == expected, f"Failed for day {day}"

    # Test operate method with missing jitter
    def test_operate_missing_jitter(self):
        """Test operate method when jitter parameter is missing."""
        params = {"entity_type": "DATE"}

        with pytest.raises(ValueError, match="Jitter parameter is required"):
            self.anonymizer.operate("2023-03-15", params)

    def test_operate_unsupported_entity_type(self):
        """Test operate method with unsupported entity type."""
        params = {"entity_type": "INVALID", "jitter": 10}

        with pytest.raises(ValueError, match="Entity type 'INVALID' is not supported"):
            self.anonymizer.operate("2023-03-15", params)

    # Test operate method with various date patterns
    def test_operate_iso_date_format(self):
        """Test operate method with ISO date format."""
        params = {"entity_type": "DATE", "jitter": 10}

        assert self.anonymizer.operate("2023-03-15", params) == "2023-03-25"

    def test_operate_no_pattern_match(self):
        """Test operate method when no pattern matches."""
        params = {"entity_type": "DATE", "jitter": 10}

        assert self.anonymizer.operate("invalid date text", params) == "[DATE]"

    def test_operate_datetime_parsing_error(self):
        """Test operate method when datetime parsing fails."""
        params = {"entity_type": "DATE", "jitter": 10}

        assert self.anonymizer.operate("2023-13-32", params) == "[DATE]"

    def test_operate_with_default_entity_type(self):
        """Test operate method with default entity type."""
        params = {"jitter": 10}  # No entity_type specified, should default to "DEFAULT"

        # Since "DEFAULT" is not in supported types, should raise error
        with pytest.raises(ValueError, match="Entity type 'DEFAULT' is not supported"):
            self.anonymizer.operate("2023-03-15", params)

    # Test pattern compilation
    def test_compile_patterns_returns_list(self):
        """Test that _compile_patterns returns a list of compiled regexes."""
        patterns = self.anonymizer._compile_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0

        for pattern in patterns:
            assert hasattr(pattern, "search")  # Should be compiled regex

    # Integration tests with real date patterns
    def test_integration_simple_date_formats(self):
        """Integration test asserting full-date formats jitter to an exact value.

        These assert the exact expected output (not just "a non-empty string"),
        so a regression that drops a component or fails to shift the date is
        caught instead of silently passing.
        """
        test_cases = [
            ("03/15/2023", {"entity_type": "DATE", "jitter": 10}, "03/25/2023"),  # US mm/dd/yyyy
            ("2023-03-15", {"entity_type": "DATE", "jitter": -5}, "2023-03-10"),  # ISO yyyy-mm-dd
            ("15/03/2023", {"entity_type": "DATE", "jitter": 7}, "22/03/2023"),  # day-first dd/mm/yyyy
        ]

        for date_str, params, expected in test_cases:
            result = self.anonymizer.operate(date_str, params)
            assert result == expected, f"{date_str} -> {result!r}, expected {expected!r}"

    def test_integration_month_year_formats(self):
        """Month/year only dates are shifted from the 15th of the month."""
        # 15 Mar 2023 + 10 days = 25 Mar 2023; 15 Mar 2023 - 5 days = 10 Mar 2023.
        assert self.anonymizer.operate("03/2023", {"entity_type": "DATE", "jitter": 10}) == "03/2023"
        assert self.anonymizer.operate("2023-03", {"entity_type": "DATE", "jitter": -5}) == "2023-03"
        # 15 Mar 2023 + 112 days = 5 Jul 2023.
        assert self.anonymizer.operate("2023-03", {"entity_type": "DATE", "jitter": 112}) == "2023-07"

    # Test edge cases
    def test_edge_case_leap_year(self):
        """Test jittering around leap year dates."""
        assert self.anonymizer.operate("2024-02-29", {"entity_type": "DATE", "jitter": 1}) == "2024-03-01"

    def test_edge_case_year_boundary(self):
        """Test jittering across year boundary."""
        assert self.anonymizer.operate("2023-12-28", {"entity_type": "DATE", "jitter": 10}) == "2024-01-07"

    def test_edge_case_large_negative_jitter(self):
        """Test with large negative jitter."""
        assert self.anonymizer.operate("2023-06-15", {"entity_type": "DATE", "jitter": -365}) == "2022-06-15"

    def test_edge_case_overflow_falls_back(self):
        """A jitter that pushes the date out of the supported range masks the date."""
        assert self.anonymizer.operate("2023-06-15", {"entity_type": "DATE", "jitter": 10**9}) == "[DATE]"

    # Test day of week replacement
    def test_day_of_week_full_name(self):
        """Test replacement of full day-of-week name."""
        params = {"entity_type": "DATE", "jitter": 10}

        # Test with different full day names
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for day in days:
            result = self.anonymizer.operate(day, params)
            # Result is a deterministic rotation, so it is always a valid day of
            # the week. (It can equal the original when jitter % 7 == 0, which is
            # not the case for jitter=10; see
            # test_day_of_week_jitter_multiple_of_seven_maps_to_self.)
            assert result in days

    def test_day_of_week_abbreviated(self):
        """Test replacement of abbreviated day-of-week name."""
        params = {"entity_type": "DATE", "jitter": 10}

        # Test with different abbreviated day names
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for day in days:
            result = self.anonymizer.operate(day, params)
            # Result is a deterministic rotation within the abbreviated table, so
            # it is always a valid abbreviated day of the week.
            assert result in days

    def test_day_of_week_lowercase(self):
        """Test that lowercase day names are replaced with lowercase."""
        params = {"entity_type": "DATE", "jitter": 10}

        result = self.anonymizer.operate("monday", params)
        assert result.islower()
        assert result.title() in self.anonymizer.days_of_week_full

    def test_day_of_week_uppercase(self):
        """Test that uppercase day names are replaced with uppercase."""
        params = {"entity_type": "DATE", "jitter": 10}

        result = self.anonymizer.operate("MONDAY", params)
        assert result.isupper()
        assert result.title() in self.anonymizer.days_of_week_full

    def test_day_of_week_mixed_case(self):
        """Test that title case day names are replaced with title case."""
        params = {"entity_type": "DATE", "jitter": 10}

        result = self.anonymizer.operate("Monday", params)
        assert result[0].isupper()
        assert result[1:].islower()
        assert result in self.anonymizer.days_of_week_full

    def test_day_of_week_deterministic_reproducible(self):
        """Same jitter maps a weekday to the same replacement on every call."""
        params = {"entity_type": "DATE", "jitter": 10}

        results = [self.anonymizer.operate("Monday", params) for _ in range(5)]

        # Deterministic: every run produces the identical output.
        assert len(set(results)) == 1

    def test_day_of_week_rotation_matches_jitter(self):
        """Replacement equals days_of_week_full[(idx + jitter) % 7]."""
        full = self.anonymizer.days_of_week_full
        for jitter in (3, 10, 47):
            params = {"entity_type": "DATE", "jitter": jitter}
            for idx, day in enumerate(full):
                result = self.anonymizer.operate(day, params)
                assert result == full[(idx + jitter) % 7]

    def test_day_of_week_negative_jitter(self):
        """Negative jitter wraps correctly (Python % is non-negative)."""
        params = {"entity_type": "DATE", "jitter": -10}

        # (0 + -10) % 7 == 4 -> Friday
        result = self.anonymizer.operate("Monday", params)
        assert result == self.anonymizer.days_of_week_full[(0 + -10) % 7]
        assert result == "Friday"

    def test_day_of_week_jitter_multiple_of_seven_maps_to_self(self):
        """jitter % 7 == 0 maps a weekday to itself (intended, unbiased)."""
        full = self.anonymizer.days_of_week_full
        for jitter in (7, 14):
            params = {"entity_type": "DATE", "jitter": jitter}
            for day in full:
                assert self.anonymizer.operate(day, params) == day

    def test_day_of_week_abbrev_stays_abbreviated(self):
        """Abbreviated input maps within the abbreviated table, never a full name."""
        params = {"entity_type": "DATE", "jitter": 10}

        result = self.anonymizer.operate("Mon", params)
        assert result in self.anonymizer.days_of_week_abbrev
        assert result not in self.anonymizer.days_of_week_full

    def test_day_of_week_different_jitter_different_mapping(self):
        """Different jitter values can produce different mappings (sanity)."""
        result_a = self.anonymizer.operate("Monday", {"entity_type": "DATE", "jitter": 3})
        result_b = self.anonymizer.operate("Monday", {"entity_type": "DATE", "jitter": 5})
        assert result_a != result_b

    # Regression tests: numeric day-first dates (dd/mm/yyyy) must jitter the
    # whole date, not silently drop the day and pass month/year through. The
    # earlier integration tests only asserted isinstance(result, str) and
    # len(result) > 0, so they missed that "15-03-2010" collapsed to "03-2010".
    def test_day_first_with_year_jitters_full_date(self):
        """dd-mm-yyyy where the leading number is unambiguously a day (>12)."""
        # 15 March 2010 + 10 days = 25 March 2010, format preserved.
        assert self.anonymizer.operate("15-03-2010", {"entity_type": "DATE", "jitter": 10}) == "25-03-2010"
        assert self.anonymizer.operate("15/03/2010", {"entity_type": "DATE", "jitter": 10}) == "25/03/2010"

    def test_day_first_with_year_rolls_over_month(self):
        """Day-first jitter must roll over month/year boundaries."""
        # 25 Dec 2010 + 10 days = 4 Jan 2011.
        assert self.anonymizer.operate("25/12/2010", {"entity_type": "DATE", "jitter": 10}) == "04/01/2011"

    def test_day_first_does_not_drop_day(self):
        """Guard against the original bug: the day component must not disappear."""
        result = self.anonymizer.operate("15-03-2010", {"entity_type": "DATE", "jitter": 10})
        # The buggy behavior produced "03-2010" (two components). A correct
        # dd-mm-yyyy result always has three dash-separated components.
        assert result.count("-") == 2
        assert result != "03-2010"

    def test_day_first_without_year(self):
        """dd/mm (no year) with an unambiguous leading day must stay day-first."""
        # 15/03 + 7 days = 22/03.
        assert self.anonymizer.operate("15/03", {"entity_type": "DATE", "jitter": 7}) == "22/03"

    def test_ambiguous_leading_value_stays_us_month_first(self):
        """A leading 01-12 is ambiguous and keeps the US mm/dd interpretation."""
        # 03/04/2010 is read as March 4th (mm/dd/yyyy); +10 days = March 14th.
        assert self.anonymizer.operate("03/04/2010", {"entity_type": "DATE", "jitter": 10}) == "03/14/2010"

    @pytest.mark.parametrize(
        ("value", "jitter", "expected"),
        [
            ("31/12/99", 5, "05/01/00"),
            ("28/02/23", 5, "05/03/23"),
            ("31-12-99", 5, "05-01-00"),
            ("13/05/21", 3, "16/05/21"),
            ("13/05/21", -3, "10/05/21"),
        ],
    )
    def test_day_first_two_digit_year_uses_the_year(self, value, jitter, expected):
        """A day-first date with a two-digit year must shift with its year, not as a yearless dd/mm."""
        assert self.anonymizer.operate(value, {"entity_type": "DATE", "jitter": jitter}) == expected

    def test_separator_preserved_across_instances(self):
        """The dash separator must survive on a second instance."""
        first = DateJitterAnonymizer()
        first.operate("15-03-2010", {"entity_type": "DATE", "jitter": 10})

        second = DateJitterAnonymizer()
        assert second.operate("15-03-2010", {"entity_type": "DATE", "jitter": 10}) == "25-03-2010"
        assert second.operate("2023-03-15", {"entity_type": "DATE", "jitter": -5}) == "2023-03-10"

    def test_iso_datetime_with_time_jitters_date_component(self):
        """ISO datetime (yyyy-mm-ddTHH:MM:SS) must jitter the date and keep the time.

        Regression: the ISO pattern used ``(T?:...)`` which required a literal
        ':' after the optional 'T', so a real ``...T15:11:00`` value never
        matched the ISO branch and collapsed to ``yyyy-mm`` instead.
        """
        # 2008-08-09 + 5 days = 2008-08-14; the time of day is kept.
        assert (
            self.anonymizer.operate("2008-08-09T15:11:00", {"entity_type": "DATE_TIME", "jitter": 5})
            == "2008-08-14T15:11:00"
        )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            # A clock after a colon must not consume the day or the last two year digits.
            ("2023-12-31:10:30", "2024-01-05:10:30"),
            ("2023-03-15:10:30:45", "2023-03-20:10:30:45"),
            ("12/31/2023:10:30", "01/05/2024:10:30"),
            ("12/31/99:23:59:59", "01/05/00:23:59:59"),
            ("15-Mar-2023:10:30", "20-Mar-2023:10:30"),
            ("Dec 31, 2023:10:30", "Jan 5, 2024:10:30"),
            # Clocks and offsets are still hidden from date matching.
            ("2023-03-15T10:30-05:30", "2023-03-20T10:30-05:30"),
            ("2023-03-15 10:30", "2023-03-20 10:30"),
            ("10:30-05:30", "[DATE]"),
            ("9-10:30", "[DATE]"),
        ],
    )
    def test_clock_does_not_consume_date_components(self, value, expected):
        assert self.anonymizer.operate(value, {"entity_type": "DATE_TIME", "jitter": 5}) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("Dec 31, 99", "Jan 5, 00"),
            ("Dec 31 99", "Jan 5 00"),
            ("March 10th, 99", "March 15th, 99"),
            ("January 1st, 05", "January 6th, 05"),
            ("Feb 28, 00", "Mar 4, 00"),
            ("March 10 15:30", "March 15 15:30"),
        ],
    )
    def test_month_name_with_two_digit_year(self, value, expected):
        """A two-digit year after a month name takes the pivot and rolls over like mm/dd/yy."""
        assert self.anonymizer.operate(value, {"entity_type": "DATE", "jitter": 5}) == expected

    def test_embedded_weekday_does_not_drop_rest_of_value(self):
        """A weekday embedded in a longer date must not short-circuit the scan.

        Regression: the standalone day-of-week fast-path used ``search`` and
        returned only the rotated weekday, dropping the trailing date (e.g.
        "Monday, 2023-03-15" became just a weekday). The fast-path now only fires
        when the weekday IS the whole stripped value; an embedded weekday is
        rotated in place by the same jitter as the date.
        """
        # 2023-03-15 + 5 days = 2023-03-20; Monday rotated by 5 days is Saturday.
        assert (
            self.anonymizer.operate("Monday, 2023-03-15", {"entity_type": "DATE_TIME", "jitter": 5})
            == "Saturday, 2023-03-20"
        )

    def test_standalone_weekday_with_trailing_punctuation_still_rotates(self):
        """A weekday-only entity with surrounding punctuation/space must rotate.

        Regression: the standalone fast-path checked ``match.group(0) ==
        stripped``, so "Monday," or " Monday " skipped the fast-path and fell
        through to the default replacement. Standalone now means "no other
        alphanumeric content outside the weekday match".
        """
        params = {"entity_type": "DATE_TIME", "jitter": 5}
        # Monday + 5 = Saturday; punctuation/whitespace must not break the path.
        for value in ("Monday,", "  Monday  ", "Monday."):
            assert self.anonymizer.operate(value, params) == "Saturday"


class TestResultCache:
    """The result cache must not change results or fallback reporting."""

    def setup_method(self):
        _shift_cached.cache_clear()

    @pytest.mark.parametrize("value", ["not a date", "13/45/2020", "99/99"])
    def test_fallback_recorded_on_every_call_including_cache_hits(self, value):
        params = {"entity_type": "DATE", "jitter": 10}
        events = start_note()
        try:
            results = [DateJitterAnonymizer().operate(value, params) for _ in range(3)]
        finally:
            end_note()

        assert results == ["[DATE]"] * 3
        assert events == [("fallback", "DATE")] * 3
        assert _shift_cached.cache_info().hits == 2

    def test_no_fallback_recorded_for_cached_valid_date(self):
        params = {"entity_type": "DATE", "jitter": 10}
        events = start_note()
        try:
            for _ in range(3):
                DateJitterAnonymizer().operate("03/04/2010", params)
        finally:
            end_note()

        assert events == []
        assert _shift_cached.cache_info().hits == 2

    def test_cache_is_keyed_on_jitter_and_pivot(self):
        operator = DateJitterAnonymizer()
        assert operator.operate("03/04/2010", {"entity_type": "DATE", "jitter": 10}) == "03/14/2010"
        assert operator.operate("03/04/2010", {"entity_type": "DATE", "jitter": 20}) == "03/24/2010"
        # 2000 is a leap year and 1900 is not, so the pivot changes the result for the same text and jitter.
        leap = {"entity_type": "DATE", "jitter": 1, "two_digit_year_pivot": 30}
        non_leap = {"entity_type": "DATE", "jitter": 1, "two_digit_year_pivot": 0}
        assert operator.operate("02/28/00", leap) == "02/29/00"
        assert operator.operate("02/28/00", non_leap) == "03/01/00"

    def test_cached_result_equals_uncached_result(self):
        operator = DateJitterAnonymizer()
        params = {"entity_type": "DATE_TIME", "jitter": -17}
        values = ["2023-03-15T10:30:00", "Monday, March 4th, 2024", "not a date", "2016", "12/31/99"]
        first = [operator.operate(value, params) for value in values]
        second = [operator.operate(value, params) for value in values]
        assert first == second

    def test_long_values_bypass_the_cache(self):
        value = "x" * 200 + " 03/04/2010"
        DateJitterAnonymizer().operate(value, {"entity_type": "DATE", "jitter": 10})
        assert _shift_cached.cache_info().currsize == 0


def _capped(value, birth_date="1930-06-15", jitter=5, **extra):
    params = {"entity_type": "DATE", "jitter": jitter, "birth_date": birth_date, **extra}
    return DateJitterAnonymizer().operate(value, params)


def _age(on: date, born: date) -> int:
    return on.year - born.year - ((on.month, on.day) < (born.month, born.day))


class TestAgeCap:
    """Dates after the max_age-th birthday move to it before the jitter (DOB 1930-06-15, cap 2020-06-15)."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("2023-03-15", "2020-06-20"),
            ("2019-01-10", "2019-01-15"),
            ("03/15/23", "06/20/20"),
            ("15-MAR-2023", "20-JUN-2020"),
            ("March 10th, 2023", "June 20th, 2020"),
            ("2023-03-15T10:30:00", "2020-06-20T10:30:00"),
            ("March 2023", "June 2020"),
            ("Seen 2021-01-01 and 2019-01-01", "Seen 2020-06-20 and 2019-01-06"),
            # Year-only and yearless values carry no age and are not capped.
            ("2023", "2023"),
            ("03/15", "03/20"),
            # The birth date itself and earlier dates are shifted as usual.
            ("1930-06-15", "1930-06-20"),
            ("1920-01-01", "1920-01-06"),
        ],
    )
    def test_values(self, value, expected):
        assert _capped(value) == expected

    def test_boundary(self):
        assert _capped("2020-06-15") == "2020-06-20"
        assert _capped("2020-06-16") == "2020-06-20"
        assert _capped("2020-06-14") == "2020-06-19"

    def test_max_age(self):
        assert _capped("2023-03-15", max_age=80) == "2010-06-20"
        assert _capped("2023-03-15", max_age=100) == "2023-03-20"

    @pytest.mark.parametrize(
        ("value", "expected"),
        # 2022 is not a leap year, so the 90th birthday of a 29 February 1932 birth is 1 March 2022.
        [("2022-02-28", "2022-03-05"), ("2022-03-01", "2022-03-06"), ("2023-03-01", "2022-03-06")],
    )
    def test_leap_day_birth_non_leap_cap(self, value, expected):
        assert _capped(value, birth_date="1932-02-29") == expected

    def test_leap_day_birth_leap_cap(self):
        assert _capped("2025-01-01", birth_date="1932-02-29", max_age=92, jitter=3) == "2024-03-03"

    def test_shifted_age_never_exceeds_max_age(self):
        rng = random.Random(7)  # noqa: S311
        for _ in range(2000):
            born = date(1903, 1, 1) + timedelta(days=rng.randrange(29000))
            event = born + timedelta(days=rng.randrange(-1000, 42000))
            jitter = rng.choice([-1, 1]) * rng.randrange(3, 181)
            out = date.fromisoformat(_capped(event.isoformat(), birth_date=born.isoformat(), jitter=jitter))
            assert _age(out, born + timedelta(days=jitter)) <= 90
            if _age(event, born) < 90:
                assert out == event + timedelta(days=jitter)

    @pytest.mark.parametrize(
        "birth_date",
        [
            date(1930, 6, 15),
            datetime(1930, 6, 15, 8, 30),
            pd.Timestamp("1930-06-15"),
            np.datetime64("1930-06-15"),
            np.datetime64("1930-06-15T00:00:00.000000000"),
            "1930-06-15",
            "1930-06-15 00:00:00",
        ],
    )
    def test_birth_date_types(self, birth_date):
        assert _capped("2023-03-15", birth_date=birth_date) == "2020-06-20"

    @pytest.mark.parametrize("birth_date", [None, float("nan"), "", "  ", pd.NaT, pd.NA])
    def test_null_birth_date_turns_the_cap_off(self, birth_date):
        assert _capped("2023-03-15", birth_date=birth_date) == "2023-03-20"

    @pytest.mark.parametrize(
        ("birth_date", "max_age", "match"),
        [("not a date", 90, "birth_date"), ("1930-06-15", 0, "max_age"), ("1930-06-15", 151, "max_age")],
    )
    def test_invalid_params(self, birth_date, max_age, match):
        with pytest.raises(ValueError, match=match):
            _capped("2023-03-15", birth_date=birth_date, max_age=max_age)
        events = start_note()
        try:
            params = {"entity_type": "DATE", "jitter": 5, "birth_date": birth_date, "max_age": max_age}
            assert guarded(DateJitterAnonymizer)().operate("2023-03-15", params) == "[DATE]"
        finally:
            end_note()
        assert [kind for kind, _ in events] == ["error"]

    def test_validate_rejects_max_age_out_of_range(self):
        with pytest.raises(ValueError, match="max_age"):
            DateJitterAnonymizer().validate({"entity_type": "DATE", "jitter": 5, "max_age": 0})

    def test_cache_is_keyed_on_the_cap(self):
        _shift_cached.cache_clear()
        assert _capped("2023-03-15", birth_date="1930-06-15") == "2020-06-20"
        assert _capped("2023-03-15", birth_date="1931-06-15") == "2021-06-20"
        assert _capped("2023-03-15", birth_date=None) == "2023-03-20"
        assert _shift_cached.cache_info().currsize == 3

    def test_cached_equals_uncached(self):
        value = "x" * 100 + " 2023-03-15"
        assert _capped(value) == "x" * 100 + " 2020-06-20"
        assert _capped("2023-03-15") == "2020-06-20"


def _part(value, entity_type, full_date="2023-03-30", jitter=5, **extra):
    params = {"entity_type": entity_type, "jitter": jitter, "full_date": full_date, **extra}
    return DateJitterAnonymizer().operate(str(value), params)


class TestDateParts:
    """A day, month or year held on its own is the same part of the shifted full date (2023-03-30 + 5 = 2023-04-04)."""

    @pytest.mark.parametrize(
        ("value", "entity_type", "expected"),
        [
            (30, "DATE_DAY", "4"),
            ("30", "DATE_DAY", "4"),
            (3, "DATE_MONTH", "4"),
            ("3", "DATE_MONTH", "4"),
            ("03", "DATE_MONTH", "04"),
            ("March", "DATE_MONTH", "April"),
            ("Mar", "DATE_MONTH", "Apr"),
            ("MAR", "DATE_MONTH", "APR"),
            ("march", "DATE_MONTH", "april"),
            (2023, "DATE_YEAR", "2023"),
            ("23", "DATE_YEAR", "23"),
            (" 30 ", "DATE_DAY", "4"),
        ],
    )
    def test_values(self, value, entity_type, expected):
        assert _part(value, entity_type) == expected

    def test_zero_padded_day(self):
        assert _part("05", "DATE_DAY", full_date="2023-03-05") == "10"
        assert _part("05", "DATE_DAY", full_date="2023-03-05", jitter=-3) == "02"
        assert _part("02", "DATE_DAY", full_date="2023-03-02", jitter=-3) == "27"
        assert _part("27", "DATE_DAY", full_date="2023-03-27", jitter=7) == "3"
        assert _part("01", "DATE_DAY", full_date="2023-03-01", jitter=3) == "04"

    def test_sept(self):
        assert _part("Sept", "DATE_MONTH", full_date="2023-09-10", jitter=-10) == "Aug"
        assert _part("Aug", "DATE_MONTH", full_date="2023-08-31", jitter=10) == "Sep"
        assert _part("Sept", "DATE_MONTH", full_date="2023-09-10", jitter=30) == "Oct"

    def test_year_rolls_over(self):
        assert _part(2023, "DATE_YEAR", full_date="2023-12-30") == "2024"
        assert _part("23", "DATE_YEAR", full_date="2023-12-30") == "24"
        assert _part(2023, "DATE_YEAR", full_date="2023-01-02", jitter=-5) == "2022"
        assert _part("99", "DATE_YEAR", full_date="1999-12-30") == "00"

    def test_age_cap(self):
        # DOB 1930-06-15: the cap is 2020-06-15, so 2023-03-30 becomes 2020-06-15 + 5 = 2020-06-20.
        extra = {"birth_date": "1930-06-15"}
        assert _part(2023, "DATE_YEAR", **extra) == "2020"
        assert _part(3, "DATE_MONTH", **extra) == "6"
        assert _part(30, "DATE_DAY", **extra) == "20"
        assert _part(2019, "DATE_YEAR", full_date="2019-12-30", **extra) == "2020"

    @pytest.mark.parametrize(
        "full_date",
        [
            date(2023, 3, 30),
            datetime(2023, 3, 30, 14, 0),
            pd.Timestamp("2023-03-30"),
            np.datetime64("2023-03-30"),
            "2023-03-30",
            "2023-03-30T14:00:00",
        ],
    )
    def test_full_date_types(self, full_date):
        assert _part(30, "DATE_DAY", full_date=full_date) == "4"

    def test_parts_agree_with_the_full_date(self):
        rng = random.Random(11)  # noqa: S311
        operator = DateJitterAnonymizer()
        for _ in range(1500):
            born = date(1903, 1, 1) + timedelta(days=rng.randrange(29000))
            event = born + timedelta(days=rng.randrange(-1000, 42000))
            jitter = rng.choice([-1, 1]) * rng.randrange(3, 181)
            base = {"jitter": jitter, "birth_date": born, "full_date": event}
            full = date.fromisoformat(operator.operate(event.isoformat(), {"entity_type": "DATE", **base}))
            assert operator.operate(str(event.day), {"entity_type": "DATE_DAY", **base}) == str(full.day)
            assert operator.operate(str(event.month), {"entity_type": "DATE_MONTH", **base}) == str(full.month)
            assert operator.operate(str(event.year), {"entity_type": "DATE_YEAR", **base}) == str(full.year)

    @pytest.mark.parametrize(
        ("value", "entity_type"),
        [
            (29, "DATE_DAY"),
            (0, "DATE_DAY"),
            (32, "DATE_DAY"),
            ("x", "DATE_DAY"),
            ("²", "DATE_DAY"),
            (2, "DATE_MONTH"),
            (13, "DATE_MONTH"),
            ("April", "DATE_MONTH"),
            ("Marchx", "DATE_MONTH"),
            ("Mon", "DATE_MONTH"),
            (2022, "DATE_YEAR"),
            ("22", "DATE_YEAR"),
            ("223", "DATE_YEAR"),
            ("3", "DATE_YEAR"),
        ],
    )
    def test_mismatch_is_masked(self, value, entity_type):
        events = start_note()
        try:
            assert _part(value, entity_type) == "[DATE]"
        finally:
            end_note()
        assert events == [("fallback", entity_type)]

    @pytest.mark.parametrize("full_date", [None, float("nan"), "", pd.NaT])
    def test_null_full_date_is_masked(self, full_date):
        events = start_note()
        try:
            assert _part(30, "DATE_DAY", full_date=full_date) == "[DATE]"
        finally:
            end_note()
        assert events == [("fallback", "DATE_DAY")]

    def test_shifted_out_of_range_is_masked(self):
        assert _part("9999", "DATE_YEAR", full_date="9999-12-30") == "[DATE]"

    def test_values_that_pass_through_as_dates_are_shifted_as_parts(self):
        assert _part("3", "DATE_MONTH") == "4"
        assert _part("Mar", "DATE_MONTH") == "Apr"
        assert _part("2023", "DATE_YEAR") == "2023"
        assert DateJitterAnonymizer().operate("Mar", {"entity_type": "DATE", "jitter": 5}) == "Mar"

    def test_missing_or_invalid_params(self):
        with pytest.raises(ValueError, match="full_date"):
            DateJitterAnonymizer().operate("30", {"entity_type": "DATE_DAY", "jitter": 5})
        with pytest.raises(ValueError, match="full_date"):
            DateJitterAnonymizer().validate({"entity_type": "DATE_DAY", "jitter": 5})
        with pytest.raises(ValueError, match="full_date"):
            _part(30, "DATE_DAY", full_date="not a date")
        events = start_note()
        try:
            params = {"entity_type": "DATE_DAY", "jitter": 5, "full_date": "not a date"}
            assert guarded(DateJitterAnonymizer)().operate("30", params) == "[DATE_DAY]"
        finally:
            end_note()
        assert [kind for kind, _ in events] == ["error"]

    def test_full_date_is_ignored_for_full_dates(self):
        params = {"entity_type": "DATE", "jitter": 5, "full_date": "1999-01-01"}
        assert DateJitterAnonymizer().operate("2023-03-30", params) == "2023-04-04"

    def test_cache_is_keyed_on_full_date_and_cap(self):
        assert _part(30, "DATE_DAY", full_date="2023-03-30") == "4"
        assert _part(30, "DATE_DAY", full_date="2023-01-30") == "4"
        assert _part(30, "DATE_DAY", full_date="2023-01-30", jitter=10) == "9"
        assert _part(30, "DATE_DAY", full_date="2023-03-30", birth_date="1930-06-15") == "20"
        assert _part(31, "DATE_DAY", full_date="2023-03-30") == "[DATE]"
