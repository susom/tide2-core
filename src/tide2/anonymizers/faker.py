"""Realistic fake data generation operator.

Uses the Faker library to replace PII entities with plausible synthetic values,
with deterministic seeding for reproducibility.
"""

import random
import re
import string
from typing import ClassVar

from faker import Faker
from presidio_anonymizer.operators import Operator
from presidio_anonymizer.operators import OperatorType

from tide2.string_parsers.format_detector import FormatDetector


class FakerAnonymizer(Operator):
    """
    Anonymizer which replaces the entity value
    with an instance counter per entity.
    """

    # Class-level cache for Faker and FormatDetector (created once)
    _faker: ClassVar[Faker | None] = None
    _format_detector: ClassVar[FormatDetector | None] = None

    @classmethod
    def _get_faker(cls) -> Faker:
        """Get or create the cached Faker instance."""
        if cls._faker is None:
            cls._faker = Faker()
        return cls._faker

    @classmethod
    def _get_format_detector(cls) -> FormatDetector:
        """Get or create the cached FormatDetector instance."""
        if cls._format_detector is None:
            cls._format_detector = FormatDetector()
        return cls._format_detector

    def __init__(self):
        """Initialize the Faker anonymizer with entity-to-generator mappings."""
        # initialize the super class
        super().__init__()
        # Use class-level cached instances
        self.fake = self._get_faker()
        self.format_detector = self._get_format_detector()

        self.fake_dict = {
            "DEFAULT": lambda _: str(self.fake.random_number(digits=8, fix_len=True)),
            "OTHER": lambda _: str(self.fake.random_number(digits=8, fix_len=True)),
            "AGE": lambda _: str(self.fake.random_int(min=18, max=90)),
            "IBAN_CODE": lambda _: self.fake.iban(),
            "CREDIT_CARD": lambda _: self.fake.credit_card_number(),
            "CRYPTO": lambda _: (
                "bc1" + "".join(self.fake.random_choices(string.ascii_lowercase + string.digits, length=26))
            ),
            "IP_ADDRESS": lambda _: self.fake.ipv4_public(),
            "URL": lambda _: self.fake.url(),
            "EMAIL": lambda _: self.fake.email(),
            "EMAIL_ADDRESS": lambda _: self.fake.email(),
            "NRP": lambda _: str(self.fake.random_number(digits=8, fix_len=True)),
            "MEDICAL_LICENSE": lambda _: self.fake.bothify(text="??######").upper(),
            "PHONE": lambda text: self._generate_phone_number(text),
            "PHONE_NUMBER": lambda text: self._generate_phone_number(text),
            # US-specific entities
            "US_BANK_NUMBER": lambda _: self.fake.bban(),
            "US_DRIVER_LICENSE": lambda _: str(self.fake.random_number(digits=9, fix_len=True)),
            "US_ITIN": lambda _: self.fake.bothify(text="9##-7#-####"),
            "US_PASSPORT": lambda _: self.fake.bothify(text="#####??").upper(),
            "US_SSN": lambda _: self.fake.ssn(),
            "ORGANIZATION": lambda _: self.fake.company(),
            "VENDOR": lambda _: self.fake.company(),
            # URL format-specific anonymizers
            "URL_HTTPS": lambda _: self.fake.url(schemes=["https"]),
            "URL_HTTP": lambda _: self.fake.url(schemes=["http"]),
            "URL_FTP": lambda _: f"ftp://{self.fake.domain_name()}/files/{self.fake.file_name()}",
            "URL_WITH_AUTH": lambda _: (
                f"https://{self.fake.user_name()}:{self.fake.password()}@{self.fake.domain_name()}"
            ),
            "URL_WITH_PORT": lambda _: f"http://{self.fake.domain_name()}:{self.fake.random_int(min=1000, max=9999)}",
            "URL_WITH_PARAMS": lambda _: f"{self.fake.url()}?{self.fake.word()}={self.fake.word()}",
            "URL_LOCALHOST": lambda _: f"http://localhost:{self.fake.random_int(min=3000, max=9999)}",
            "URL_IP_BASED": lambda _: f"http://{self.fake.ipv4_private()}:{self.fake.random_int(min=1000, max=9999)}",
            "URL_INCOMPLETE": lambda _: self.fake.domain_name(),
            "URL_DOMAIN_ONLY": lambda _: self.fake.domain_name(),
            "URL_WWW_PREFIX": lambda _: f"www.{self.fake.domain_name()}",
            "URL_LOCALHOST_NO_PROTOCOL": lambda _: f"localhost:{self.fake.random_int(min=3000, max=9999)}",
            "URL_IP_NO_PROTOCOL": lambda _: f"{self.fake.ipv4_private()}:{self.fake.random_int(min=1000, max=9999)}",
            "URL_MALFORMED": lambda _: self.fake.domain_name(),
            "URL_INCOMPLETE_PORT": lambda _: f"{self.fake.domain_name()}:",
            "URL_LEADING_DOT": lambda _: f".{self.fake.domain_name()}",
            "URL_TRAILING_DOT": lambda _: f"{self.fake.domain_name()}.",
            "URL_DOUBLE_DOT": lambda _: f"{self.fake.word()}..{self.fake.domain_name()}",
            "GENETIC_SEQUENCE": lambda text: "".join(
                self.fake.random_element(elements=("A", "T", "G", "C")) for _ in range(len(text))
            ),
            "WEB": "",  # General web entity
        }

        self.entities_supported = set(self.fake_dict.keys())

    def _generate_phone_number(self, text: str) -> str:
        """
        Generate a fake phone number that matches the format of the input.

        Priority:
        1. If digits-only: generate same number of digits
        2. If normalized format (###-###-####): generate with same format
        3. Otherwise: use faker default phone number

        Args:
            text: The original phone number text

        Returns:
            A fake phone number matching the input format
        """

        # Preserve leading and trailing whitespace
        leading_ws = text[: len(text) - len(text.lstrip())]
        trailing_ws = text[len(text.rstrip()) :]
        text_stripped = text.strip()

        # Check if it's digits-only
        if re.match(r"^\d+$", text_stripped):
            # Generate same number of digits
            num_digits = len(text_stripped)
            fake_number = str(self.fake.random_number(digits=num_digits, fix_len=True))
            return leading_ws + fake_number + trailing_ws

        # Check for normalized format with dashes (e.g., 412-456-8708)
        match = re.match(r"^(\d+)-(\d+)-(\d+)$", text_stripped)
        if match:
            # Generate numbers matching each segment length
            segments = match.groups()
            fake_segments = [str(self.fake.random_number(digits=len(seg), fix_len=True)) for seg in segments]
            fake_number = "-".join(fake_segments)
            return leading_ws + fake_number + trailing_ws

        # Fallback to faker default
        return self.fake.phone_number()

    def operate(self, text: str, params: dict) -> str:
        """Anonymize the input text by replacing it with realistic fake data.

        Args:
            text: The original text to anonymize.
            params: Operator parameters. Supported keys:
                - entity_type (str): Entity type label used to select the
                  appropriate faker generator (e.g. "PERSON", "PHONE_NUMBER").
                - faker_seed (int): Seed for reproducible fake data generation.
                  Defaults to a random integer.

        Returns:
            A fake replacement string appropriate for the entity type.
        """

        # type of entity to anonymize
        entity_type = params.get("entity_type")
        # get the faker seed
        faker_seed = params.get("faker_seed", random.randint(0, 100000))

        # If entity type is WEB, try to detect the specific format
        if entity_type in ["WEB", "URL"]:
            detected_format, _ = self.format_detector.detect_format(text)
            if detected_format and detected_format.value.upper() in self.fake_dict:
                entity_type = detected_format.value.upper()
            else:
                # Default to general URL if no specific format detected
                entity_type = "DEFAULT"

        f_map = self.fake_dict[entity_type]
        self.fake.seed_instance(faker_seed)
        return f_map(text)

    def validate(self, params: dict) -> None:
        """Validate operator parameters."""

        entity_type = params.get("entity_type")

        if entity_type not in self.entities_supported:
            raise ValueError(f"Entity type '{entity_type}' is not supported for FakerAnonymizer.")

    def operator_name(self) -> str:
        """Return the operator name."""
        return "faker_anonymizer"

    def operator_type(self) -> OperatorType:
        """Return the operator type."""
        return OperatorType.Anonymize
