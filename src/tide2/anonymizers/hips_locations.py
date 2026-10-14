"""Cryptographic location anonymization operator.

Replaces location entities (cities, states, addresses) with deterministic
substitutes selected via HMAC-based secure string selection from census data.
"""

import csv
import re
import string
from pathlib import Path
from typing import ClassVar

from presidio_anonymizer.operators import Operator
from presidio_anonymizer.operators import OperatorType

from tide2.cryptographic.string_selector import secure_string_selector
from tide2.string_parsers.address_parsers import AddressParser
from tide2.utils.resource_utils import CITIES_FILE
from tide2.utils.resource_utils import COUNTRIES_FILE
from tide2.utils.resource_utils import HOSPITALS_FILE
from tide2.utils.resource_utils import STATES_FILE
from tide2.utils.resource_utils import STREET_NAMES_FILE
from tide2.utils.resource_utils import ZIPCODES_FILE
from tide2.utils.resource_utils import get_resource_path

# Hospital length bucket thresholds
TINY_HOSPITAL_MAX_LEN = 5
SHORT_HOSPITAL_MAX_LEN = 15
MEDIUM_HOSPITAL_MAX_LEN = 30


def _read_tsv_column(path: str, col_idx: int = 0) -> list[str]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        return [row[col_idx] for row in reader if row and len(row) > col_idx]


def _read_tsv_columns(path: str, idx1: int, idx2: int) -> tuple[list[str], list[str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        rows = [row for row in reader if row and len(row) > max(idx1, idx2)]
        return [row[idx1] for row in rows], [row[idx2] for row in rows]


def _read_cities(path: str) -> tuple[list[str], list[str], dict[str, str]]:
    cities_dict: dict[str, None] = {}
    states_dict: dict[str, None] = {}
    state2city: dict[str, str] = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            try:
                city, state = row[0], row[1]
            except IndexError:
                continue
            if city:
                cities_dict[city] = None
            if state:
                states_dict[state] = None
                state2city[state] = city
    return list(cities_dict.keys()), list(states_dict.keys()), state2city


class HipsLocationAnonymizer(Operator):
    """
    Anonymizer that replaces addresses with credible addresses but completely fake.
    It will preserve the format, casing, structure, state and country.
    It will not preserve the street name, street number, postal code, city or any other
    specific information, like ZIP code, etc.
    """

    # Class-level cache to avoid reloading CSV files
    _street_names: ClassVar[list | None] = None
    _zipcodes: ClassVar[list | None] = None
    _cities: ClassVar[list | None] = None
    _states: ClassVar[list | None] = None
    _state2city: ClassVar[dict | None] = None
    _states_full: ClassVar[list | None] = None
    _states_abbr: ClassVar[list | None] = None
    _countries: ClassVar[list | None] = None
    _countries_abbr: ClassVar[list | None] = None
    _street_numbers: ClassVar[list | None] = None
    _hospitals: ClassVar[list | None] = None
    _hospitals_by_length: ClassVar[dict | None] = None
    _address_parser: ClassVar[AddressParser | None] = None

    @classmethod
    def _load_location_data(cls):
        """Load location data from CSV files (only once per class)."""
        if cls._street_names is None:
            cls._street_names = _read_tsv_column(get_resource_path(STREET_NAMES_FILE))

        if cls._zipcodes is None:
            cls._zipcodes = _read_tsv_column(get_resource_path(ZIPCODES_FILE))

        if cls._cities is None:
            cls._cities, cls._states, cls._state2city = _read_cities(get_resource_path(CITIES_FILE))

        if cls._states_full is None:
            cls._states_full, cls._states_abbr = _read_tsv_columns(get_resource_path(STATES_FILE), 0, 1)

        if cls._countries is None:
            cls._countries, cls._countries_abbr = _read_tsv_columns(get_resource_path(COUNTRIES_FILE), 1, 0)

        if cls._street_numbers is None:
            # generate street numbers on the fly
            cls._street_numbers = [str(i) for i in range(1, 30000)]

        if cls._hospitals is None:
            # load the list for hospitals
            cls._hospitals = [
                line.strip()
                for line in Path(get_resource_path(HOSPITALS_FILE)).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        if cls._hospitals_by_length is None:
            # Bucket hospitals by length for length-aware selection
            # This preserves document structure by replacing short names with short names
            cls._hospitals_by_length = {
                "tiny": [],  # 1-5 chars (abbreviations: GMC, SMC)
                "short": [],  # 6-15 chars (Genesis Med, Summit Health)
                "medium": [],  # 16-30 chars (Genesis Memorial Center)
                "long": [],  # 31+ chars (Unity Ridge Integrated Health Center)
            }
            for h in cls._hospitals:
                length = len(h)
                if length <= TINY_HOSPITAL_MAX_LEN:
                    cls._hospitals_by_length["tiny"].append(h)
                elif length <= SHORT_HOSPITAL_MAX_LEN:
                    cls._hospitals_by_length["short"].append(h)
                elif length <= MEDIUM_HOSPITAL_MAX_LEN:
                    cls._hospitals_by_length["medium"].append(h)
                else:
                    cls._hospitals_by_length["long"].append(h)

        if cls._address_parser is None:
            # Create AddressParser once at class level
            cls._address_parser = AddressParser()

    def __init__(self):
        """Initialize the HIPS location anonymizer with cached geographic data."""
        super().__init__()
        # Load data once at class level
        self._load_location_data()

        # Use class-level cached AddressParser
        self.address_parser = self._address_parser

        # Use class-level cached data
        self.street_names = self._street_names
        self.zipcodes = self._zipcodes
        self.cities = self._cities
        self.states = self._states
        self.state2city = self._state2city
        self.states_full = self._states_full
        self.states_abbr = self._states_abbr
        self.countries = self._countries
        self.countries_abbr = self._countries_abbr
        self.street_numbers = self._street_numbers
        self.hospitals = self._hospitals
        self.hospitals_by_length = self._hospitals_by_length

        self.supported_entity_types = ["LOCATION", "HOSPITAL", "VENDOR"]

    def _get_hospital_bucket(self, text_length: int) -> list[str]:
        """
        Get appropriate hospital list based on input text length.

        This ensures short abbreviations get replaced with short names,
        preserving document structure.

        Args:
            text_length: Length of the input text

        Returns:
            List of hospitals with similar length, or full list as fallback
        """
        if text_length <= TINY_HOSPITAL_MAX_LEN:
            bucket = self.hospitals_by_length["tiny"]
        elif text_length <= SHORT_HOSPITAL_MAX_LEN:
            bucket = self.hospitals_by_length["short"]
        elif text_length <= MEDIUM_HOSPITAL_MAX_LEN:
            bucket = self.hospitals_by_length["medium"]
        else:
            bucket = self.hospitals_by_length["long"]

        # Fallback to full list if bucket is empty
        return bucket if bucket else self.hospitals

    def _is_spurious_value(self, text: str) -> bool:
        """
        Check if the text is spurious and should not be anonymized.

        A text is considered spurious if it is:
        - Empty or only whitespace
        - Single character (after stripping whitespace)
        - Single letter with punctuation (e.g., "A.", "B,")
        - Only punctuation (with or without whitespace)

        Args:
            text: The text to check

        Returns:
            True if the text is spurious and should be returned unmodified,
            False if it should be anonymized
        """
        if not text or not text.strip():
            return True

        # Check if it's a single character after stripping whitespace
        if len(text.strip()) <= 1:
            return True

        # Check if it's only punctuation and/or whitespace
        stripped_text = text.strip()
        if all(char in string.punctuation + string.whitespace for char in stripped_text):
            return True

        # Strip punctuation and whitespace to check the remaining content
        cleaned_text = stripped_text.strip(string.punctuation + string.whitespace).lower()

        # If nothing is left after stripping, it's spurious
        if not cleaned_text:
            return True

        # Check if it's a single letter with punctuation (e.g., "A.", "B,")
        return bool(len(cleaned_text) == 1 and cleaned_text.isalpha())

    def _component_cleaning(self, components: dict[str, str]) -> dict[str, str]:
        """Clean address components by stripping whitespace, lowercasing, and removing common punctuation."""
        new_dict = {}
        for k, v in components.items():
            if v is not None and k not in ("_component_order", "_format_template"):
                new_dict[k] = re.sub(r'[,.(){}[\]"\'-]', "", v.strip().lower())
            else:
                new_dict[k] = v
        return new_dict

    def operate(self, text: str, params: dict) -> str:
        """Anonymize a location string using deterministic replacement.

        Parses the address into components (street, city, zip, state) and
        replaces each with a cryptographically selected substitute. Falls
        back to format-preserving encryption if parsing fails.

        Args:
            text: The original location/address text.
            params: Operator parameters. Required keys:
                - salt (str): Cryptographic salt for deterministic output.
                - key (str): Encryption key.

        Returns:
            The anonymized location string, or the original text if it is
            spurious (punctuation-only, whitespace, single letter, etc.).
        """

        # Check if the text is spurious (punctuation only, whitespace, single letter, etc.)
        # If so, return it unmodified
        if self._is_spurious_value(text):
            return text

        salt = params["salt"]
        key = params["key"]

        # first try to parse the address. If not possible use FPE to encrypt
        result = self.address_parser.parse_address(text)
        if result is not None:
            # Clean each one of the components stripping whitespace, lowercasing and removing common punctuation
            result_components = self._component_cleaning(result)

            # for each one of the result components use the secure string selector
            # to select a suitable replacement using the appropriate loaded list

            street_number = result_components.get("street_number")
            new_street_number = None
            if street_number:
                new_street_number = secure_string_selector(salt, key, self.street_numbers, street_number)

            street_name = result_components.get("street_name")
            new_street_name = None
            if street_name:
                new_street_name = secure_string_selector(salt, key, self.street_names, street_name)

            city = result_components.get("city")
            new_city = None
            if city:
                new_city = secure_string_selector(salt, key, self.cities, city)

            zipcode = result_components.get("zipcode")
            new_zipcode = None
            if zipcode:
                new_zipcode = secure_string_selector(salt, key, self.zipcodes, zipcode)

            # leave state untouched for now
            new_state = result_components.get("state")
            new_state_abbreviation = result_components.get("state_abbreviation")

            new_text = self.address_parser.format_address(
                street_number=new_street_number,
                street_name=new_street_name,
                city=new_city,
                state=new_state,
                state_abbreviation=new_state_abbreviation,
                zipcode=new_zipcode,
                parsed_result=result,
            )
        else:
            # Use secure string selector for hospitals with length-aware bucket
            cleaned_text = text.strip()
            input_text = cleaned_text.lower()
            hospital_bucket = self._get_hospital_bucket(len(input_text))
            new_text = secure_string_selector(salt, key, hospital_bucket, input_text)
            new_text = self.address_parser._apply_case_pattern(cleaned_text, new_text)

            # Preserve any leading or trailing whitespace from the original span
            leading_ws = text[: len(text) - len(text.lstrip())]
            trailing_ws = text[len(text.rstrip()) :]
            new_text = f"{leading_ws}{new_text}{trailing_ws}"

        return new_text

    def validate(self, params: dict) -> None:
        """Validate operator parameters."""

        entity_type = params.get("entity_type", "DEFAULT")
        if entity_type not in self.supported_entity_types:
            raise ValueError(f"Entity type '{entity_type}' is not supported for HipsLocationAnonimizer.")

        # get the salt and key
        salt = params.get("salt")
        key = params.get("key")
        if not salt or not key:
            raise ValueError("Both 'salt' and 'key' must be provided for HipsLocationAnonimizer.")

    def operator_name(self) -> str:
        """Return the operator name."""
        return "hips_location"

    def operator_type(self) -> OperatorType:
        """Return the operator type."""
        return OperatorType.Anonymize
