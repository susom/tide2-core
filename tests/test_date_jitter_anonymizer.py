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
"""

from datetime import datetime
from datetime import timedelta

import pytest

from tide2.anonymizers.date_jitter import DateJitterAnonymizer


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
        assert hasattr(self.anonymizer, "substitutions")
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

    def test_pad_zero(self):
        """Test _pad_zero static method."""
        test_cases = [(1, "01"), (9, "09"), (10, "10"), (12, "12")]
        for num, expected in test_cases:
            result = DateJitterAnonymizer._pad_zero(num)
            assert result == expected, f"Failed for number {num}"

    # Test date formatting methods
    def test_format_dd_mm_yyyy(self):
        """Test _format_dd_mm_yyyy method."""
        result = self.anonymizer._format_dd_mm_yyyy(3, 15, 2023, "/")
        assert result == "15/03/2023"

        result = self.anonymizer._format_dd_mm_yyyy(12, 5, 2022, "-")
        assert result == "05-12-2022"

    def test_format_yyyy_mm_dd(self):
        """Test _format_yyyy_mm_dd method."""
        result = self.anonymizer._format_yyyy_mm_dd(3, 15, 2023, "/")
        assert result == "2023/03/15"

        result = self.anonymizer._format_yyyy_mm_dd(12, 5, 2022, "-")
        assert result == "2022-12-05"

    def test_format_month_day_year_full(self):
        """Test _format_month_day_year_full method."""
        result = self.anonymizer._format_month_day_year_full(3, 15, 2023, "/")
        assert result == "March/15/2023"

        result = self.anonymizer._format_month_day_year_full(12, 5, 2022, "-")
        assert result == "December-5-2022"

    def test_format_mm_dd_yyyy(self):
        """Test _format_mm_dd_yyyy method."""
        result = self.anonymizer._format_mm_dd_yyyy(3, 15, 2023, "/")
        assert result == "03/15/2023"

        result = self.anonymizer._format_mm_dd_yyyy(12, 5, 2022, "-")
        assert result == "12-05-2022"

    def test_format_mm_dd_yy_2000s(self):
        """Test _format_mm_dd_yy_2000s method."""
        result = self.anonymizer._format_mm_dd_yy_2000s(3, 15, 2023, "/")
        assert result == "03/15/23"

        result = self.anonymizer._format_mm_dd_yy_2000s(12, 5, 2009, "-")
        assert result == "12-05-09"

    def test_format_mm_dd_yy_1900s(self):
        """Test _format_mm_dd_yy_1900s method."""
        result = self.anonymizer._format_mm_dd_yy_1900s(3, 15, 1985, "/")
        assert result == "03/15/85"

        result = self.anonymizer._format_mm_dd_yy_1900s(12, 5, 1999, "-")
        assert result == "12-05-99"

    def test_format_month_day_suffix_year(self):
        """Test _format_month_day_suffix_year method."""
        result = self.anonymizer._format_month_day_suffix_year(3, 15, 2023)
        assert result == "March 15th, 2023"

        result = self.anonymizer._format_month_day_suffix_year(1, 1, 2022)
        assert result == "January 1st, 2022"

        result = self.anonymizer._format_month_day_suffix_year(5, 22, 2024)
        assert result == "May 22nd, 2024"

    def test_format_mm_yyyy(self):
        """Test _format_mm_yyyy method."""
        result = self.anonymizer._format_mm_yyyy(3, 15, 2023, "/")
        assert result == "03/2023"

        result = self.anonymizer._format_mm_yyyy(12, 5, 2022, "-")
        assert result == "12-2022"

    def test_format_month_of_year(self):
        """Test _format_month_of_year method."""
        result = self.anonymizer._format_month_of_year(3, 15, 2023)
        assert result == "March of 2023"

        result = self.anonymizer._format_month_of_year(12, 5, 2022)
        assert result == "December of 2022"

    def test_format_month_day(self):
        """Test _format_month_day method."""
        result = self.anonymizer._format_month_day(3, 15, 2023, False)
        assert result == "March 15"

        result = self.anonymizer._format_month_day(3, 15, 2023, True)
        assert result == "March 15th"

    def test_format_mm_dd(self):
        """Test _format_mm_dd method."""
        result = self.anonymizer._format_mm_dd(3, 15, 2023, "/")
        assert result == "03/15"

        result = self.anonymizer._format_mm_dd(12, 5, 2022, "-")
        assert result == "12-05"

    def test_format_dd_mm(self):
        """Test _format_dd_mm method (day-first, no year)."""
        result = self.anonymizer._format_dd_mm(3, 15, 2023, "/")
        assert result == "15/03"

        result = self.anonymizer._format_dd_mm(12, 5, 2022, "-")
        assert result == "05-12"

    def test_format_yyyy_mm(self):
        """Test _format_yyyy_mm method."""
        result = self.anonymizer._format_yyyy_mm(3, 15, 2023, "/")
        assert result == "2023/03"

        result = self.anonymizer._format_yyyy_mm(12, 5, 2022, "-")
        assert result == "2022-12"

    def test_format_iso_date(self):
        """Test _format_iso_date method."""
        result = self.anonymizer._format_iso_date(3, 15, 2023, "-")
        assert result == "2023-03-15"

        result = self.anonymizer._format_iso_date(12, 5, 2022, "/")
        assert result == "2022/12/05"

    def test_format_compact_date(self):
        """Test _format_compact_date method."""
        result = self.anonymizer._format_compact_date(3, 15, 2023)
        assert result == "20230315"

        result = self.anonymizer._format_compact_date(12, 5, 2022)
        assert result == "20221205"

    # Test _get_jittered_date method
    def test_get_jittered_date_positive_jitter(self):
        """Test _get_jittered_date with positive jitter."""
        date_of_service = datetime(2023, 3, 15)
        jitter = 10

        result = self.anonymizer._get_jittered_date(jitter, date_of_service)

        expected_date = date_of_service + timedelta(days=10)
        assert result == {"day": expected_date.day, "month": expected_date.month, "year": expected_date.year}

    def test_get_jittered_date_negative_jitter(self):
        """Test _get_jittered_date with negative jitter."""
        date_of_service = datetime(2023, 3, 15)
        jitter = -10

        result = self.anonymizer._get_jittered_date(jitter, date_of_service)

        expected_date = date_of_service + timedelta(days=-10)
        assert result == {"day": expected_date.day, "month": expected_date.month, "year": expected_date.year}

    def test_get_jittered_date_cross_month_boundary(self):
        """Test _get_jittered_date crossing month boundary."""
        date_of_service = datetime(2023, 3, 28)
        jitter = 10

        result = self.anonymizer._get_jittered_date(jitter, date_of_service)

        expected_date = datetime(2023, 4, 7)  # Should cross into April
        assert result == {"day": expected_date.day, "month": expected_date.month, "year": expected_date.year}

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

        # Test with real date jittering logic
        input_date = "2023-03-15"
        result = self.anonymizer.operate(input_date, params)

        # Result should be a date string (possibly jittered)
        assert isinstance(result, str)
        assert len(result) > 0
        # The result might be in the same ISO format or different depending on implementation

    def test_operate_no_pattern_match(self):
        """Test operate method when no pattern matches."""
        params = {"entity_type": "DATE", "jitter": 10}

        # Test with text that doesn't contain a recognizable date pattern
        result = self.anonymizer.operate("invalid date text", params)

        # Should return default replacement or handle gracefully
        assert isinstance(result, str)
        # Might return the default replacement "[DATE]" or the original text

    def test_operate_datetime_parsing_error(self):
        """Test operate method when datetime parsing fails."""
        params = {"entity_type": "DATE", "jitter": 10}

        # Test with an invalid date that might cause parsing errors
        result = self.anonymizer.operate("2023-13-32", params)  # Invalid month and day

        # Should handle the error gracefully
        assert isinstance(result, str)
        # Might return default replacement, original text, or some error handling

    def test_operate_with_default_entity_type(self):
        """Test operate method with default entity type."""
        params = {"jitter": 10}  # No entity_type specified, should default to "DEFAULT"

        # Since "DEFAULT" is not in supported types, should raise error
        with pytest.raises(ValueError, match="Entity type 'DEFAULT' is not supported"):
            self.anonymizer.operate("2023-03-15", params)

    # Test pattern compilation
    def test_compile_patterns_returns_list(self):
        """Test that _compile_patterns returns a list of tuples."""
        patterns = self.anonymizer._compile_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0

        for pattern, replacement in patterns:
            assert hasattr(pattern, "search")  # Should be compiled regex
            assert callable(replacement)  # Should be callable function

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
        """Integration test with month/year only formats."""
        test_cases = [
            ("03/2023", {"entity_type": "DATE", "jitter": 10}),
            ("2023-03", {"entity_type": "DATE", "jitter": -5}),
        ]

        succeeded = 0
        for date_str, params in test_cases:
            try:
                result = self.anonymizer.operate(date_str, params)
                assert isinstance(result, str)
                assert len(result) > 0
                succeeded += 1
            except ValueError:
                # Acceptable if pattern doesn't match
                pass

        # At least one month/year format must be handled without raising.
        assert succeeded > 0, "Expected at least one month/year date format to be handled"

    # Test edge cases
    def test_edge_case_leap_year(self):
        """Test jittering around leap year dates."""
        jittered_dict = self.anonymizer._get_jittered_date(
            1,
            datetime(2024, 2, 29),  # Leap day
        )

        expected = datetime(2024, 3, 1)
        assert jittered_dict == {"day": expected.day, "month": expected.month, "year": expected.year}

    def test_edge_case_year_boundary(self):
        """Test jittering across year boundary."""
        jittered_dict = self.anonymizer._get_jittered_date(10, datetime(2023, 12, 28))

        expected = datetime(2024, 1, 7)  # Should cross into next year
        assert jittered_dict == {"day": expected.day, "month": expected.month, "year": expected.year}

    def test_edge_case_large_negative_jitter(self):
        """Test with large negative jitter."""
        jittered_dict = self.anonymizer._get_jittered_date(-365, datetime(2023, 6, 15))

        expected = datetime(2022, 6, 15)  # Should go back a year
        assert jittered_dict == {"day": expected.day, "month": expected.month, "year": expected.year}

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
        """The dash separator must survive on a second instance.

        Regression for the class-level _separator_acceptance cache being keyed by
        bound methods: a fresh instance used to miss the cache and silently fall
        back to '/', so "15-03-2010" came out as "25/03/2010".
        """
        first = DateJitterAnonymizer()
        first.operate("15-03-2010", {"entity_type": "DATE", "jitter": 10})  # populate class cache

        second = DateJitterAnonymizer()
        assert second.operate("15-03-2010", {"entity_type": "DATE", "jitter": 10}) == "25-03-2010"
        assert second.operate("2023-03-15", {"entity_type": "DATE", "jitter": -5}) == "2023-03-10"
