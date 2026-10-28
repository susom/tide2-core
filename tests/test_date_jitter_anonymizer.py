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

import pytest

from tide2.anonymizers.date_jitter import DateJitterAnonymizer
from tide2.anonymizers.date_jitter import _shift_cached
from tide2.anonymizers.guarded import end_note
from tide2.anonymizers.guarded import start_note


class TestDateJitterAnonymizer:
    """Test DateJitterAnonymizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.anonymizer = DateJitterAnonymizer()

    def test_initialization(self):
        """Test DateJitterAnonymizer initialization."""
        assert self.anonymizer.supported_entity_types == ["DATE_TIME", "DATE"]
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
