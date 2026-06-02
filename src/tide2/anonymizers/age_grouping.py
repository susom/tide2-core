"""Age-range anonymization operator.

Groups ages into configurable ranges (e.g., 89+ becomes ">89") while preserving
the original text format (digits, words, ordinals).
"""

import re

from presidio_anonymizer.operators import Operator
from presidio_anonymizer.operators import OperatorType

from tide2.string_parsers.format_detector import FormatDetector
from tide2.string_parsers.format_detector import FormatType


class AgeGroupAnonymizer(Operator):
    """
    Anonymizer that groups ages by applying an upper limit while preserving the original format.
    Ages above the specified limit are set to the limit value, maintaining the input format.
    """

    def __init__(self):
        """Initialize the age grouping anonymizer."""
        super().__init__()
        self.format_detector = FormatDetector()
        self.supported_entity_types = ["AGE"]

    def _contains_written_number(self, text: str) -> bool:
        """Check if text contains written number words that might represent an age."""
        number_words = [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        ]

        text_lower = text.lower()
        age_keywords = ["year", "month", "old"]

        # Check if text contains number words and age-related keywords
        has_number = any(word in text_lower for word in number_words)
        has_age_keyword = any(keyword in text_lower for keyword in age_keywords)

        return has_number and has_age_keyword

    def operate(self, text: str, params: dict) -> str:
        """Anonymize the input text by applying age grouping.

        Args:
            text: The original text containing an age value.
            params: Operator parameters. Supported keys:
                - upper_limit (int): Maximum age value to allow. Ages above this
                  are capped. Defaults to 80.

        Returns:
            The text with the age value capped at upper_limit, formatted to
            match the original style. Returns the original text unchanged if
            no age format is detected.
        """

        upper_limit = params.get("upper_limit", 80)

        # Detect the age format
        format_type, _ = self.format_detector.detect_format(text)

        # If no format detected, try to handle as written numbers manually
        if format_type is None and self._contains_written_number(text):
            format_type = FormatType.AGE_WRITTEN_NUMBERS

        if format_type is None:
            # If still no format detected, return original text
            return text

        # Extract the numeric age from the text
        age_value = self._extract_age_value(text, format_type)

        if age_value is None:
            # If we can't extract age value, return original text
            return text

        # Apply the upper limit
        limited_age = min(age_value, upper_limit)

        # Format the limited age back to the original format
        return self._format_age(text, limited_age, format_type)

    def _extract_age_value(self, text: str, format_type: FormatType) -> int | None:
        """Extract the numeric age value from the text based on format type."""

        if format_type == FormatType.AGE_NUMERIC_ONLY:
            # Extract plain number
            match = re.search(r"\d+", text)
            return int(match.group()) if match else None

        if format_type == FormatType.AGE_WITH_UNITS:
            # Extract number from various unit formats
            # Handle patterns like "18 yo", "32 Y", "12 mo", "75 year old", etc.
            match = re.search(r"(\d+(?:\.\d+)?)", text)
            if match:
                return int(float(match.group(1)))
            return None

        if format_type == FormatType.AGE_GESTATIONAL:
            # For gestational age, extract weeks and convert to approximate age
            # This is a special case - we'll preserve the format but limit the weeks
            weeks_match = re.search(r"(\d+)w", text.lower())
            if weeks_match:
                weeks = int(weeks_match.group(1))
                # Convert gestational weeks to approximate months (rough approximation)
                # Gestational age typically ranges from 20-42 weeks
                # We'll treat this differently and limit weeks instead of converting to years
                return weeks
            return None

        if format_type == FormatType.AGE_WRITTEN_NUMBERS:
            # Handle written numbers like "twenty-seven year old" or "ninety year old"
            number_words = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
                "nine": 9,
                "ten": 10,
                "eleven": 11,
                "twelve": 12,
                "thirteen": 13,
                "fourteen": 14,
                "fifteen": 15,
                "sixteen": 16,
                "seventeen": 17,
                "eighteen": 18,
                "nineteen": 19,
                "twenty": 20,
                "thirty": 30,
                "forty": 40,
                "fifty": 50,
                "sixty": 60,
                "seventy": 70,
                "eighty": 80,
                "ninety": 90,
            }

            text_lower = text.lower()
            total = 0

            # Handle compound numbers like "twenty-seven"
            if "-" in text_lower:
                parts = text_lower.split("-")
                tens_part = parts[0].strip()
                ones_part = parts[1].strip() if len(parts) > 1 else ""

                if tens_part in number_words:
                    total += number_words[tens_part]

                # Extract just the first word from ones_part (e.g., "five year old" -> "five")
                if ones_part:
                    ones_word = ones_part.split()[0]
                    if ones_word in number_words:
                        total += number_words[ones_word]
            else:
                # Handle space-separated numbers like "sixty five" or standalone "ninety"
                words = text_lower.split()
                tens_value = 0
                ones_value = 0

                for word in words:
                    if word in number_words:
                        if word in ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]:
                            tens_value = number_words[word]
                        elif word in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]:
                            ones_value = number_words[word]
                        else:
                            # For teens or other numbers
                            total = number_words[word]
                            break

                # Combine tens and ones if found
                if tens_value > 0:
                    total = tens_value + ones_value

            return total if total > 0 else None

        return None

    def _format_age(self, original_text: str, age_value: int, format_type: FormatType) -> str:
        """Format the age value back to the original format."""

        if format_type == FormatType.AGE_NUMERIC_ONLY:
            # Replace the number while preserving surrounding characters
            return re.sub(r"\d+", str(age_value), original_text)

        if format_type == FormatType.AGE_WITH_UNITS:
            # Replace the number while preserving the unit and format
            return re.sub(r"\d+(?:\.\d+)?", str(age_value), original_text)

        if format_type == FormatType.AGE_GESTATIONAL:
            # For gestational age, we limit the weeks value (different from years)
            # Gestational age upper limit should be different (e.g., 42 weeks max)
            gestational_limit = min(age_value, 42)  # 42 weeks is typical max gestational age
            return re.sub(r"(\d+)w", f"{gestational_limit}w", original_text)

        if format_type == FormatType.AGE_WRITTEN_NUMBERS:
            # Convert numeric age back to written form
            return self._convert_number_to_words(age_value, original_text)

        return original_text

    def _convert_number_to_words(self, number: int, original_text: str) -> str:
        """Convert a number back to written words, preserving the original format."""

        ones = [
            "",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        if number < 20:
            word_number = ones[number]
        elif number < 100:
            tens_digit = number // 10
            ones_digit = number % 10
            if ones_digit == 0:
                word_number = tens[tens_digit]
            else:
                # Preserve hyphenation style from original
                separator = "-" if "-" in original_text else " "
                word_number = tens[tens_digit] + separator + ones[ones_digit]
        else:
            # For numbers >= 100, just use the number (edge case)
            word_number = str(number)

        # Preserve case from original text
        if original_text and original_text[0].isupper():
            word_number = word_number.capitalize()

        # Replace the number words in the original text more carefully
        # Find the number part and replace it while preserving the rest
        text_lower = original_text.lower()

        # Build more comprehensive patterns to match the number part
        # Handle hyphenated numbers like "sixty-five"
        if "-" in text_lower:
            hyphen_pattern = r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-(?:one|two|three|four|five|six|seven|eight|nine)"
            if re.search(hyphen_pattern, text_lower):
                return re.sub(hyphen_pattern, word_number, original_text, flags=re.IGNORECASE)

        # Handle space-separated compound numbers
        compound_pattern = r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(?:one|two|three|four|five|six|seven|eight|nine)"
        if re.search(compound_pattern, text_lower):
            return re.sub(compound_pattern, word_number, original_text, flags=re.IGNORECASE)

        # Handle standalone tens numbers
        tens_pattern = r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
        if re.search(tens_pattern, text_lower):
            return re.sub(tens_pattern, word_number, original_text, flags=re.IGNORECASE)

        # Handle teens and single digits
        teens_pattern = r"(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"
        if re.search(teens_pattern, text_lower):
            return re.sub(teens_pattern, word_number, original_text, flags=re.IGNORECASE)

        ones_pattern = r"(?:one|two|three|four|five|six|seven|eight|nine)"
        if re.search(ones_pattern, text_lower):
            return re.sub(ones_pattern, word_number, original_text, flags=re.IGNORECASE)

        # If no pattern matched, return original text
        return original_text

    def validate(self, params: dict) -> None:
        """Validate operator parameters."""

        entity_type = params.get("entity_type")
        if entity_type not in self.supported_entity_types:
            raise ValueError(f"Entity type '{entity_type}' is not supported for AgeGroupAnonymizer.")

        # Validate upper_limit parameter
        upper_limit = params.get("upper_limit")
        if upper_limit is not None:
            if not isinstance(upper_limit, int) or upper_limit <= 0:
                raise ValueError("Parameter 'upper_limit' must be a positive integer.")

    def operator_name(self) -> str:
        """Return the operator name."""
        return "age_grouping"

    def operator_type(self) -> OperatorType:
        """Return the operator type."""
        return OperatorType.Anonymize
