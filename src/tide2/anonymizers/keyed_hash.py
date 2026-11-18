"""Anonymizer that replaces an identifier with a keyed hash of its value.

The output is ``UPPER(HEX(HMAC-SHA256(key, salt || "|" || UPPER(TRIM(value)))))``, 64 characters.
"""

import hashlib
import hmac

from presidio_anonymizer.operators import Operator
from presidio_anonymizer.operators import OperatorType

__docformat__ = "google"


class KeyedHashAnonymizer(Operator):
    """Replace a value with the uppercase hex HMAC-SHA256 of the value, keyed by ``salt`` and ``key``."""

    def operate(self, text: str, params: dict) -> str:
        """Return the keyed hash of ``text``.

        Args:
            text: The value to anonymize.
            params: ``salt`` and ``key``, each 32 bytes.

        Returns:
            64 uppercase hexadecimal characters.
        """
        message = params["salt"] + b"|" + text.strip().upper().encode("utf-8")
        return hmac.new(params["key"], message, hashlib.sha256).hexdigest().upper()

    def validate(self, params: dict) -> None:
        """Make sure that ``salt`` and ``key`` are non-empty bytes.

        Raises:
            ValueError: If ``salt`` or ``key`` is missing, empty or not bytes.
        """
        for name in ("salt", "key"):
            value = params.get(name)
            if not isinstance(value, bytes) or not value:
                raise ValueError(f"KeyedHashAnonymizer needs a non-empty bytes '{name}'")

    def operator_name(self) -> str:
        """Return the operator name."""
        return "keyed_hash"

    def operator_type(self) -> OperatorType:
        """Return the operator type."""
        return OperatorType.Anonymize
