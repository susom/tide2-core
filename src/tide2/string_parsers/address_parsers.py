"""US address parsing and formatting utilities.

Wraps the ``usaddress`` library to parse free-text addresses into structured
components (street, city, state, ZIP) and reformat them.
"""

from typing import ClassVar

import pandas as pd
import usaddress

from ..utils.resource_utils import CITIES_FILE
from ..utils.resource_utils import get_resource_path


class AddressParser:
    """
    A class for parsing address strings using the usaddress library.

    The parser extracts components from address strings and maps them to
    standardized keys: street_number, street_name, city, state/state_abbreviation, zipcode.
    """

    # Mapping from usaddress components to our standardized keys
    COMPONENT_MAPPING = {
        "AddressNumber": "street_number",
        "StreetNamePreDirectional": "street_name_parts",
        "StreetName": "street_name_parts",
        "StreetNamePostType": "street_name_parts",
        "LandmarkName": "street_name_parts",  # Sometimes usaddress classifies street names as landmarks
        "Recipient": "city",
        "PlaceName": "city",
        "StateName": "state",
        "ZipCode": "zipcode",
    }

    # Components that indicate a successful parse
    SUCCESS_COMPONENTS = {
        "AddressNumber",
        "StreetNamePreDirectional",
        "StreetName",
        "StreetNamePostType",
        "LandmarkName",
        "PlaceName",
        "StateName",
        "ZipCode",
    }

    # US States and their abbreviations
    STATE_ABBREVIATIONS = {
        "ALABAMA": "AL",
        "ALASKA": "AK",
        "ARIZONA": "AZ",
        "ARKANSAS": "AR",
        "CALIFORNIA": "CA",
        "COLORADO": "CO",
        "CONNECTICUT": "CT",
        "DELAWARE": "DE",
        "FLORIDA": "FL",
        "GEORGIA": "GA",
        "HAWAII": "HI",
        "IDAHO": "ID",
        "ILLINOIS": "IL",
        "INDIANA": "IN",
        "IOWA": "IA",
        "KANSAS": "KS",
        "KENTUCKY": "KY",
        "LOUISIANA": "LA",
        "MAINE": "ME",
        "MARYLAND": "MD",
        "MASSACHUSETTS": "MA",
        "MICHIGAN": "MI",
        "MINNESOTA": "MN",
        "MISSISSIPPI": "MS",
        "MISSOURI": "MO",
        "MONTANA": "MT",
        "NEBRASKA": "NE",
        "NEVADA": "NV",
        "NEW HAMPSHIRE": "NH",
        "NEW JERSEY": "NJ",
        "NEW MEXICO": "NM",
        "NEW YORK": "NY",
        "NORTH CAROLINA": "NC",
        "NORTH DAKOTA": "ND",
        "OHIO": "OH",
        "OKLAHOMA": "OK",
        "OREGON": "OR",
        "PENNSYLVANIA": "PA",
        "RHODE ISLAND": "RI",
        "SOUTH CAROLINA": "SC",
        "SOUTH DAKOTA": "SD",
        "TENNESSEE": "TN",
        "TEXAS": "TX",
        "UTAH": "UT",
        "VERMONT": "VT",
        "VIRGINIA": "VA",
        "WASHINGTON": "WA",
        "WEST VIRGINIA": "WV",
        "WISCONSIN": "WI",
        "WYOMING": "WY",
        "DISTRICT OF COLUMBIA": "DC",
    }

    # Reverse mapping for abbreviations to full names
    ABBREVIATION_TO_STATE = {v: k for k, v in STATE_ABBREVIATIONS.items()}

    # Class-level cache for cities data (loaded once)
    _cities_cache: ClassVar[set | None] = None

    @classmethod
    def _load_cities(cls) -> set:
        """Load cities data from CSV file (only once per class)."""
        if cls._cities_cache is None:
            cities_df = pd.read_csv(get_resource_path(CITIES_FILE), sep="\t", dtype={"city": str, "state": str})
            cls._cities_cache = set(cities_df["city"].dropna().unique().tolist())
        return cls._cities_cache

    def __init__(self) -> None:
        """Initialize the address parser with cached cities data."""
        # Use class-level cached cities data
        self.cities = self._load_cities()

    def _classify_state(self, state_text: str) -> tuple:
        """
        Classify a state text as either full name or abbreviation.

        Args:
            state_text (str): The state text to classify

        Returns:
            tuple: (key, value) where key is either 'state' or 'state_abbreviation'
        """
        if not state_text:
            return "state", state_text

        state_upper = state_text.upper().strip()

        # Check if it's a known abbreviation
        if state_upper in self.ABBREVIATION_TO_STATE:
            return "state_abbreviation", state_text

        # Check if it's a known full state name
        if state_upper in self.STATE_ABBREVIATIONS:
            return "state", state_text

        # If unknown, classify by length (2 chars = abbreviation, longer = full name)
        if len(state_upper) == 2:
            return "state_abbreviation", state_text
        return "state", state_text

    def parse_address(self, address_string: str) -> dict[str, str] | None:
        """
        Parse an address string and return a dictionary of components.
        Preserves the original order of components as they appear in the input.

        Args:
            address_string (str): The address string to parse

        Returns:
            Optional[Dict[str, str]]: Dictionary with available address components
                                   or None if no valid components found
        """
        # Basic input validation
        if not isinstance(address_string, str) or not address_string.strip():
            return None

        address_string = address_string.strip()
        parsed_list = None

        try:
            # Try using usaddress.tag first (preferred method)
            parsed_components, _ = usaddress.tag(address_string)
            # Get the ordered list for maintaining order
            parsed_list = usaddress.parse(address_string)

        except usaddress.RepeatedLabelError:
            # If tag fails, try parse method
            try:
                parsed_list = usaddress.parse(address_string)
                # Convert list of tuples to dictionary
                parsed_components = {}
                for component, label in parsed_list:
                    if label in parsed_components:
                        # Handle repeated labels by concatenating
                        parsed_components[label] += f" {component}"
                    else:
                        parsed_components[label] = component

            except Exception:
                # If both methods fail, return None
                return None

        except Exception:
            # Handle any other parsing errors
            return None

        # Check if we found any of the success components
        found_components = set(parsed_components.keys())
        if not found_components.intersection(self.SUCCESS_COMPONENTS):
            return None

        # Map components to our standardized format while preserving order
        result = {}
        street_name_parts = []
        component_order = []
        format_template = []  # Store the original format structure

        # Process components in the order they appear in the original string
        for component, usaddress_key in parsed_list:
            if usaddress_key in self.COMPONENT_MAPPING:
                mapped_key = self.COMPONENT_MAPPING[usaddress_key]

                if mapped_key == "street_name_parts":
                    # Clean component of trailing commas for street name parts
                    clean_component = component.rstrip(",")
                    street_name_parts.append(clean_component)
                    # Track street name position for ordering
                    if "street_name" not in component_order:
                        component_order.append("street_name")
                        format_template.append(("street_name", component))
                elif mapped_key == "state":
                    # Classify state as full name or abbreviation
                    clean_value = parsed_components[usaddress_key].rstrip(",")
                    state_key, state_value = self._classify_state(clean_value)
                    if state_key not in result:  # Avoid duplicates
                        result[state_key] = state_value
                        component_order.append(state_key)
                        format_template.append((state_key, component))
                elif mapped_key not in result:  # Avoid duplicates
                    if usaddress_key == "Recipient":
                        # Sanitize recipient/place name to match known cities
                        possible_city = parsed_components[usaddress_key].rstrip(",").strip().lower()
                        # if it does not match a city, skip it
                        if possible_city not in self.cities:
                            continue
                    # Clean component of trailing commas
                    clean_value = parsed_components[usaddress_key].rstrip(",")
                    result[mapped_key] = clean_value
                    component_order.append(mapped_key)
                    format_template.append((mapped_key, component))

        # Combine street name parts if any were found
        if street_name_parts:
            result["street_name"] = " ".join(street_name_parts)

        # Store the component order and format template for formatting
        if result:
            result["_component_order"] = component_order
            result["_format_template"] = format_template

        return result if result else None

    def _apply_case_pattern(self, template_text: str, new_text: str) -> str:
        """Apply the capitalization pattern from template to new text."""
        if not template_text or not new_text:
            return new_text

        # Special case: if new_text is a state abbreviation (2 uppercase letters), preserve it
        if len(new_text) == 2 and new_text.upper() in self.ABBREVIATION_TO_STATE:
            return new_text.upper()

        # Analyze the template's case pattern
        if template_text.isupper():
            return new_text.upper()
        if template_text.islower():
            return new_text.lower()
        if template_text.istitle():
            return new_text.title()
        # Mixed case - try to apply word-by-word pattern
        template_words = template_text.split()
        new_words = new_text.split()
        result_words = []

        for i, new_word in enumerate(new_words):
            if i < len(template_words):
                template_word = template_words[i]

                # Special case: if this is a state abbreviation, preserve conventional case
                if (len(new_word) == 2 and new_word.upper() in self.ABBREVIATION_TO_STATE) or template_word.isupper():
                    result_words.append(new_word.upper())
                elif template_word.islower():
                    result_words.append(new_word.lower())
                elif template_word.istitle():
                    result_words.append(new_word.title())
                # For mixed case words, apply character-by-character if same length
                elif len(template_word) == len(new_word):
                    result_word = ""
                    for j, char in enumerate(new_word):
                        if j < len(template_word):
                            if template_word[j].isupper():
                                result_word += char.upper()
                            else:
                                result_word += char.lower()
                        else:
                            result_word += char
                    result_words.append(result_word)
                else:
                    result_words.append(new_word)
            # For extra words, use the pattern from the last template word
            elif template_words:
                last_template = template_words[-1]
                # Special case for state abbreviations
                if (len(new_word) == 2 and new_word.upper() in self.ABBREVIATION_TO_STATE) or last_template.isupper():
                    result_words.append(new_word.upper())
                elif last_template.islower():
                    result_words.append(new_word.lower())
                elif last_template.istitle():
                    result_words.append(new_word.title())
                else:
                    result_words.append(new_word)
            else:
                result_words.append(new_word)

        return " ".join(result_words)

    def _format_components(self, components: dict[str, str], component_order: list | None = None) -> str:
        """
        Internal method to format address components into a standardized address string.
        Preserves the original capitalization and order from the input.

        Args:
            components (Dict[str, str]): Dictionary of address components
            component_order (Optional[list]): Order of components as they appeared in original string

        Returns:
            str: Formatted address string preserving original capitalization and order
        """
        # Filter out None values and metadata
        available_components = {
            k: v
            for k, v in components.items()
            if v and isinstance(v, str) and v.strip() and k not in ["_component_order", "_format_template"]
        }

        if not available_components:
            return ""

        # Use provided order or fall back to standard order
        if component_order:
            return self._format_with_order(available_components, component_order)
        return self._format_standard_order(available_components)

    def _format_with_order(self, components: dict[str, str], component_order: list) -> str:
        """Format components using the specified order."""
        address_parts = []
        street_parts = []
        state_zip_parts = []

        for component_key in component_order:
            value = None

            # Direct match
            if component_key in components:
                value = components[component_key].strip()
            # Handle state/state_abbreviation compatibility
            elif component_key == "state" and "state_abbreviation" in components:
                value = components["state_abbreviation"].strip()
            elif component_key == "state_abbreviation" and "state" in components:
                value = components["state"].strip()

            if value:
                if component_key == "street_number" or component_key == "street_name":
                    street_parts.append(value)
                elif component_key == "city":
                    # Add accumulated street parts first
                    if street_parts:
                        address_parts.append(" ".join(street_parts))
                        street_parts = []
                    address_parts.append(value)
                elif component_key in ["state", "state_abbreviation"] or component_key == "zipcode":
                    state_zip_parts.append(value)

        # Add any remaining street parts
        if street_parts:
            address_parts.insert(0, " ".join(street_parts))

        # Add state/zip parts as a single group
        if state_zip_parts:
            address_parts.append(" ".join(state_zip_parts))

        return ", ".join(address_parts) if len(address_parts) > 1 else (address_parts[0] if address_parts else "")

    def _format_standard_order(self, components: dict[str, str]) -> str:
        """Format components using standard order: street, city, state zipcode."""
        address_parts = []

        # Build street address part
        street_parts = []
        if components.get("street_number"):
            street_parts.append(components["street_number"].strip())
        if components.get("street_name"):
            street_parts.append(components["street_name"].strip())

        if street_parts:
            address_parts.append(" ".join(street_parts))

        # Add city
        if components.get("city"):
            address_parts.append(components["city"].strip())

        # Build state and zipcode part
        state_zip_parts = []
        # Handle both state and state_abbreviation
        if components.get("state"):
            state_zip_parts.append(components["state"].strip())
        elif components.get("state_abbreviation"):
            state_zip_parts.append(components["state_abbreviation"].strip())

        if components.get("zipcode"):
            state_zip_parts.append(components["zipcode"].strip())

        if state_zip_parts:
            address_parts.append(" ".join(state_zip_parts))

        return ", ".join(address_parts) if len(address_parts) > 1 else (address_parts[0] if address_parts else "")

    def format_address(
        self,
        street_number: str | None = None,
        street_name: str | None = None,
        city: str | None = None,
        state: str | None = None,
        state_abbreviation: str | None = None,
        zipcode: str | None = None,
        parsed_result: dict[str, str] | None = None,
    ) -> str:
        """
        Format address components into a standardized address string.

        Args:
            street_number (Optional[str]): Street number
            street_name (Optional[str]): Street name
            city (Optional[str]): City name
            state (Optional[str]): State full name
            state_abbreviation (Optional[str]): State abbreviation
            zipcode (Optional[str]): Zip code
            parsed_result (Optional[Dict[str, str]]): Result from parse_address method.
                                                     If provided, preserves original format and order.

        Returns:
            str: Formatted address string. If parsed_result is provided, preserves
                 original order and capitalization. Otherwise uses standard format.

        Raises:
            ValueError: If no address components are provided
        """
        # Check if any address component is provided
        individual_components = {
            "street_number": street_number,
            "street_name": street_name,
            "city": city,
            "state": state,
            "state_abbreviation": state_abbreviation,
            "zipcode": zipcode,
        }

        # Filter out None values
        provided_components = {k: v for k, v in individual_components.items() if v is not None and str(v).strip()}

        if not provided_components:
            raise ValueError(
                "At least one address component (street_number, street_name, city, state, state_abbreviation, zipcode) must be specified"
            )

        # If parsed_result is provided, use it to preserve format and order
        if parsed_result and isinstance(parsed_result, dict):
            # Apply the format pattern from parsed_result to new components
            return self._apply_format_pattern(parsed_result, provided_components)
        # Use standard formatting
        return self._format_components(provided_components)

    def _apply_format_pattern(self, parsed_result: dict[str, str], new_components: dict[str, str]) -> str:
        """
        Apply the format pattern from a parsed result to new components.
        This includes both component order AND capitalization pattern from the original.

        Args:
            parsed_result (Dict[str, str]): Result from parse_address method
            new_components (Dict[str, str]): New address components to format

        Returns:
            str: New components formatted according to the parsed pattern
        """
        if not parsed_result or "_component_order" not in parsed_result:
            # Fall back to standard formatting if template parsing fails
            return self._format_components(new_components)

        # Get template components and their original case patterns
        template_components = {
            k: v for k, v in parsed_result.items() if k not in ["_component_order", "_format_template"]
        }
        component_order = parsed_result.get("_component_order")

        # Ensure component_order is a list or None
        if component_order is not None and not isinstance(component_order, list):
            component_order = None

        # Apply the template's case patterns to new components
        formatted_components = {}
        for component_key in new_components:
            new_value = str(new_components[component_key]).strip()

            # Apply case pattern from template if available
            template_value = None
            if component_key in template_components:
                template_value = template_components[component_key]
            elif component_key == "state" and "state_abbreviation" in template_components:
                # If input has 'state' but template has 'state_abbreviation', use that pattern
                template_value = template_components["state_abbreviation"]
            elif component_key == "state_abbreviation" and "state" in template_components:
                # If input has 'state_abbreviation' but template has 'state', use that pattern
                template_value = template_components["state"]

            if template_value:
                formatted_components[component_key] = self._apply_case_pattern(template_value, new_value)
            else:
                formatted_components[component_key] = new_value

        return self._format_components(formatted_components, component_order)
