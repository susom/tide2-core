"""Entity types, name formats, and shared constants.

Defines enumerations and dataclasses used throughout the package for entity
classification, name ordering, and case formatting.
"""

from dataclasses import dataclass
from enum import StrEnum


class NameFormat(StrEnum):
    """Name ordering formats."""

    FIRST_LAST = "first_last"  # Standard: "John Smith"
    LAST_FIRST = "last_first"  # Comma-separated: "Smith, John"


class CaseFormat(StrEnum):
    """Text casing formats."""

    LOWER = "lower"  # "john smith"
    UPPER = "upper"  # "JOHN SMITH"
    TITLE = "title"  # "John Smith"
    SENTENCE = "sentence"  # "John smith"
    MIXED = "mixed"  # Preserved original mixed casing


@dataclass(frozen=True)
class NameConstants:
    """Immutable constants for name parsing.

    Uses frozen dataclass to ensure constants cannot be modified at runtime.
    """

    salutations: frozenset[str] = frozenset(
        {
            "mr",
            "mrs",
            "ms",
            "miss",
            "dr",
            "prof",
            "professor",
            "rev",
            "reverend",
            "fr",
            "father",
            "sr",
            "sister",
            "hon",
            "honorable",
            "judge",
            "justice",
            "sir",
            "dame",
            "lord",
            "lady",
            "rabbi",
            "imam",
            "pastor",
            "elder",
            "deacon",
            "bishop",
            "archbishop",
            "cardinal",
            "pope",
            "captain",
            "major",
            "colonel",
            "general",
            "admiral",
            "sergeant",
            "lieutenant",
            "commander",
            "chief",
        }
    )

    suffixes: frozenset[str] = frozenset(
        {
            # Generational suffixes
            "jr",
            "sr",
            "ii",
            "iii",
            "iv",
            "v",
            "vi",
            "vii",
            "viii",
            "ix",
            "x",
            "1st",
            "2nd",
            "3rd",
            "4th",
            "5th",
            "6th",
            "7th",
            "8th",
            "9th",
            "10th",
            # Professional/Academic suffixes
            "esq",
            "esquire",
            "phd",
            "md",
            "dds",
            "dmd",
            "jd",
            "od",
            "do",
            "dc",
            "pharmd",
            "rn",
            "lpn",
            "np",
            "pa",
            "pa-c",  # Physician Assistant - Certified
            "dvm",
            "ma",
            "mba",
            "mfa",
            "mpa",
            "mph",
            "ms",
            "msc",
            "msw",
            "ba",
            "bs",
            "bsc",
            "bed",
            "bfa",
            "llb",
            "llm",
            "cpa",
            "cfa",
            "cfp",
            "pe",
            "pmp",
            "cissp",
            "cisa",
            # Healthcare credentials
            "lcsw",  # Licensed Clinical Social Worker
            "lmsw",  # Licensed Master Social Worker
            "rph",  # Registered Pharmacist
            "arnp",  # Advanced Registered Nurse Practitioner
            "aprn",  # Advanced Practice Registered Nurse
            "cnm",  # Certified Nurse Midwife
            "crna",  # Certified Registered Nurse Anesthetist
            "fnp",  # Family Nurse Practitioner
            "fnp-c",  # Family Nurse Practitioner - Certified
            "anp",  # Adult Nurse Practitioner
            "pnp",  # Pediatric Nurse Practitioner
            "cns",  # Clinical Nurse Specialist
            "cna",  # Certified Nursing Assistant
            "lvn",  # Licensed Vocational Nurse
            "msn",  # Master of Science in Nursing
            "bsn",  # Bachelor of Science in Nursing
            "dnp",  # Doctor of Nursing Practice
            "dpm",  # Doctor of Podiatric Medicine
            "dpt",  # Doctor of Physical Therapy
            "ot",  # Occupational Therapist
            "otr",  # Occupational Therapist Registered
            "pt",  # Physical Therapist
            "rt",  # Respiratory Therapist
            "rrt",  # Registered Respiratory Therapist
            "rvt",  # Registered Vascular Technologist
            "rdt",  # Registered Diagnostic Technologist
            "mt",  # Medical Technologist
            "cmt",  # Certified Medical Technician
            "emt",  # Emergency Medical Technician
            "emt-p",  # EMT-Paramedic
            "nrp",  # Nationally Registered Paramedic
            "rd",  # Registered Dietitian
            "rdn",  # Registered Dietitian Nutritionist
            "cde",  # Certified Diabetes Educator
            "ccm",  # Certified Case Manager
            "acnp",  # Acute Care Nurse Practitioner
        }
    )

    # Prefixes that are part of surnames (e.g., "Van Der Berg")
    surname_prefixes: frozenset[str] = frozenset(
        {
            "von",
            "van",
            "der",
            "den",
            "de",
            "del",
            "della",
            "di",
            "da",
            "la",
            "le",
            "los",
            "las",
            "mac",
            "mc",
            "o",
            "o'",
            "san",
            "st",
            "ste",
            "santa",
            "santo",
            "dos",
            "das",
            "ben",
            "bin",
            "al",
        }
    )


# Single instance to be imported
NAME_CONSTANTS = NameConstants()
"""Shared singleton of `NameConstants` for name parsing throughout the package."""
