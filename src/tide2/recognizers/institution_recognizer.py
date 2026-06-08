"""
Institution-specific PHI Recognizer.

Detects institution-specific personally identifiable information (PII) in clinical
text using configurable regex patterns. Ships with Stanford Health Care patterns
covering hospital names, URLs, portals, campus locations, abbreviations, and
common typos.

This recognizer is designed to catch institution-specific PHI that generic NER models
consistently miss — for example, abbreviations like "SHC" or "LPCH", portal names
like "MyHealth", campus addresses like "Pasteur Drive", and institution URLs.

Other institutions can contribute their own pattern sets by adding a class method
(similar to ``stanford_patterns``) or by loading patterns from a JSON config file
via ``from_config``.

Performance: All patterns are pre-compiled at construction time and reused across
calls. Patterns are evaluated individually to preserve per-pattern confidence
scores and category metadata.
"""

import json
import re
from pathlib import Path
from typing import ClassVar

from presidio_analyzer import AnalysisExplanation
from presidio_analyzer import EntityRecognizer
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.recognizer_result import RecognizerResult


class InstitutionRecognizer(EntityRecognizer):
    """
    Institution-specific PHI recognizer.

    Detects institution names, URLs, portals, campus locations, abbreviations,
    system codes, and common typos using configurable regex patterns with
    per-pattern confidence scores.

    Includes built-in Stanford Health Care patterns. Other institutions can
    be added via ``from_config()`` or by subclassing with a new pattern set.

    Example usage::

        # Stanford (built-in)
        recognizer = InstitutionRecognizer()

        # Custom institution from JSON config
        recognizer = InstitutionRecognizer.from_config("path/to/ucsf.json")

        # Use in Presidio AnalyzerEngine
        from presidio_analyzer import AnalyzerEngine
        engine = AnalyzerEngine()
        engine.registry.add_recognizer(recognizer)
        results = engine.analyze(text="Patient seen at Stanford", language="en")
    """

    SUPPORTED_ENTITY: ClassVar[str] = "INSTITUTION"
    DEFAULT_SCORE: ClassVar[float] = 0.85
    _TRAILING_PUNCT = re.compile(r"[.,;:!?)\"']+$")

    # --- Stanford Health Care patterns -----------------------------------------
    # Each tuple: (compiled_regex, score, pattern_name, category)
    # Categories: url, social, facility, location, portal, system_code,
    #             abbreviation, name, typo, zip_code

    @staticmethod
    def stanford_patterns() -> list[tuple[re.Pattern, float, str, str]]:
        """Return the built-in Stanford Health Care pattern set.

        Returns a list of (compiled_regex, confidence_score, pattern_name, category)
        tuples ordered from most specific (longest match) to least specific.
        """
        return [
            # ── URLs & domains (high confidence — almost never false positives) ──
            (re.compile(r"[Ww]ww\.stanford\w+\.(?:com|org|edu)\S*", re.IGNORECASE),
             0.95, "url_full", "url"),
            (re.compile(r"stanfordhealthcare\.[\w./@-]*", re.IGNORECASE),
             0.95, "domain_shc", "url"),
            (re.compile(r"stanfordhospital\.[\w./@-]*", re.IGNORECASE),
             0.95, "domain_sh", "url"),
            (re.compile(r"stanfordmed\.[\w./@-]*", re.IGNORECASE),
             0.95, "domain_sm", "url"),
            (re.compile(r"evercore\.stanfordmed\.org", re.IGNORECASE),
             0.95, "url_evercore", "url"),
            (re.compile(r"partners\.stanfordemedicine\.com", re.IGNORECASE),
             0.95, "url_partners", "url"),
            (re.compile(r"bit\.ly/StanfordPatientEdReg", re.IGNORECASE),
             0.95, "url_bitly", "url"),
            (re.compile(r"StanfordHealthCareLiveDonors\.org", re.IGNORECASE),
             0.95, "url_livedonors", "url"),
            (re.compile(r"StanfordHealthCareLiveDonors", re.IGNORECASE),
             0.90, "bare_livedonors", "url"),
            (re.compile(r"stanfordendo", re.IGNORECASE),
             0.85, "bare_endo", "url"),
            (re.compile(r"stanfordpainmedicine", re.IGNORECASE),
             0.90, "url_pain", "url"),

            # ── Social media handles ──
            (re.compile(r"@[Ss]tanford[\w_]+"),
             0.90, "social_handle", "social"),

            # ── Affiliated facilities (multi-word — high confidence) ──
            (re.compile(r"\bPalo Alto Medical Foundation\b", re.IGNORECASE),
             0.90, "palo_alto_med_fdn", "facility"),
            (re.compile(r"\bSfm Myhealth Clinic Messaging\b", re.IGNORECASE),
             0.90, "sfm_myhealth", "facility"),
            (re.compile(r"\bMyHealth Clinic Messaging\b", re.IGNORECASE),
             0.90, "myhealth_clinic_msg", "facility"),
            (re.compile(r"\bBlake Wilbur Building\b", re.IGNORECASE),
             0.90, "blake_wilbur_bldg", "location"),
            (re.compile(r"\bPackard Children's\b", re.IGNORECASE),
             0.90, "packard_childrens", "facility"),
            (re.compile(r"\bPackard Children\b", re.IGNORECASE),
             0.90, "packard_children", "facility"),
            (re.compile(r"\bLucile\s+Salter\b", re.IGNORECASE),
             0.90, "lucile_salter", "facility"),
            (re.compile(r"\bLucile\s+Community\b", re.IGNORECASE),
             0.85, "lucile_community", "facility"),

            # ── Concatenated location names (no space variants) ──
            (re.compile(r"PaloAlto"),
             0.80, "paloalto_concat", "location"),
            (re.compile(r"BlakeWilbur", re.IGNORECASE),
             0.85, "blakewilbur_concat", "location"),
            (re.compile(r"PasteurDrive", re.IGNORECASE),
             0.90, "pasteurdrive_concat", "location"),
            (re.compile(r"PasteurDr", re.IGNORECASE),
             0.90, "pasteurdr_concat", "location"),

            # ── Spaced location names ──
            (re.compile(r"Palo\s+Alto", re.IGNORECASE),
             0.75, "palo_alto", "location"),
            (re.compile(r"Blake\s+Wilbur", re.IGNORECASE),
             0.85, "blake_wilbur", "location"),
            (re.compile(r"Pasteur\s+Drive", re.IGNORECASE),
             0.90, "pasteur_drive", "location"),
            (re.compile(r"Pasteur\s+Dr\b", re.IGNORECASE),
             0.90, "pasteur_dr", "location"),

            # ── Portal names ──
            (re.compile(r"\bMyHealth\b"), 0.80, "myhealth", "portal"),
            (re.compile(r"\bMyhealth\b"), 0.80, "myhealth_lc", "portal"),
            (re.compile(r"\bmyHealth\b"), 0.80, "myhealth_camel", "portal"),
            (re.compile(r"\bmyhealth\b"), 0.80, "myhealth_lower", "portal"),
            (re.compile(r"\bMYHEALTH\b"), 0.80, "myhealth_upper", "portal"),
            (re.compile(r"\b[Mm][Yy][Hh]ea?th\b"),
             0.75, "myheath_typo", "portal"),

            # ── Abbreviations ──
            (re.compile(r"\bPAMF\b"), 0.70, "pamf", "abbreviation"),
            (re.compile(r"\bPamf\b"), 0.70, "pamf_title", "abbreviation"),

            # ── System codes (very specific — high confidence) ──
            (re.compile(r"SHCECT01"), 0.95, "shcect01", "system_code"),
            (re.compile(r"SHCVCLOGO"), 0.95, "shcvclogo", "system_code"),
            (re.compile(r"SHCTV"), 0.95, "shctv", "system_code"),
            (re.compile(r"\bSHCA\b"), 0.70, "shca", "abbreviation"),
            (re.compile(r"\bSHCE\b"), 0.70, "shce", "abbreviation"),
            (re.compile(r"\bShca\b"), 0.70, "shca_title", "abbreviation"),
            (re.compile(r"\bShce\b"), 0.70, "shce_title", "abbreviation"),
            (re.compile(r"\bSHCe\b"), 0.70, "shce_mixed", "abbreviation"),

            # ── Bare domain/compound names (catch-all for missed URLs) ──
            (re.compile(r"Pasteur", re.IGNORECASE),
             0.60, "pasteur", "location"),
            (re.compile(r"stanfordhealthcare", re.IGNORECASE),
             0.90, "bare_shc", "name"),
            (re.compile(r"stanfordhospital", re.IGNORECASE),
             0.90, "bare_sh", "name"),
            (re.compile(r"stanfordmed", re.IGNORECASE),
             0.90, "bare_sm", "name"),

            # ── Typos ──
            (re.compile(r"\bStandford\b"), 0.85, "standford_typo", "typo"),
            (re.compile(r"\bstandford\b"), 0.85, "standford_typo_lc", "typo"),

            # ── Core institution name (ordered: exact case first, then fallback) ──
            (re.compile(r"STANFORD"), 0.85, "stanford_upper", "name"),
            (re.compile(r"Stanford"), 0.85, "stanford_title", "name"),
            (re.compile(r"stanford"), 0.85, "stanford_lower", "name"),
            (re.compile(r"stanford", re.IGNORECASE),
             0.85, "stanford_mixed", "name"),

            # ── Mail codes & unit codes ──
            (re.compile(r"\bMC\s+\d{4,5}\b", re.IGNORECASE),
             0.75, "mail_code", "location"),
            (re.compile(
                r"\b(?:Floor|ICU|MICU|SICU|NICU|PICU|CCU|HDU)-\d{4,6}\b",
                re.IGNORECASE),
             0.70, "unit_code", "location"),
        ]

    def __init__(
        self,
        patterns: list[tuple[re.Pattern, float, str, str]] | None = None,
        supported_language: str = "en",
        supported_entity: str = "INSTITUTION",
        institution_name: str = "Stanford Health Care",
    ):
        """
        Initialize the institution recognizer.

        Args:
            patterns: List of (compiled_regex, score, name, category) tuples.
                      Defaults to Stanford Health Care patterns.
            supported_language: Language code (default: "en").
            supported_entity: Entity type name (default: "INSTITUTION").
            institution_name: Display name for the institution (used in explanations).
        """
        super().__init__(
            supported_entities=[supported_entity],
            supported_language=supported_language,
            name="InstitutionRecognizer",
        )
        self._supported_entity = supported_entity
        self._institution_name = institution_name
        self._patterns = patterns if patterns is not None else self.stanford_patterns()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        supported_language: str = "en",
        supported_entity: str = "INSTITUTION",
    ) -> "InstitutionRecognizer":
        """
        Create an InstitutionRecognizer from a JSON configuration file.

        The JSON file should contain::

            {
                "institution": "Institution Name",
                "rules": [
                    {
                        "pattern": "regex string",
                        "flags": ["IGNORECASE"],   // optional
                        "score": 0.85,             // optional, default 0.85
                        "label": "pattern_name",
                        "category": "url"          // optional, default "name"
                    },
                    ...
                ]
            }

        Args:
            config_path: Path to the JSON config file.
            supported_language: Language code (default: "en").
            supported_entity: Entity type name (default: "INSTITUTION").

        Returns:
            Configured InstitutionRecognizer instance.
        """
        with open(config_path) as f:
            cfg = json.load(f)

        _VALID_FLAGS = {
            "IGNORECASE", "MULTILINE", "DOTALL", "VERBOSE",
            "ASCII", "LOCALE", "UNICODE",
        }

        patterns: list[tuple[re.Pattern, float, str, str]] = []
        for rule in cfg.get("rules", []):
            flags = 0
            for flag_name in rule.get("flags", []):
                upper = flag_name.upper()
                if upper not in _VALID_FLAGS:
                    raise ValueError(
                        f"Unknown regex flag '{flag_name}' in rule "
                        f"'{rule.get('label', '?')}'. "
                        f"Valid flags: {sorted(_VALID_FLAGS)}"
                    )
                flags |= getattr(re, upper)
            compiled = re.compile(rule["pattern"], flags)
            score = rule.get("score", cls.DEFAULT_SCORE)
            label = rule["label"]
            category = rule.get("category", "name")
            patterns.append((compiled, score, label, category))

        return cls(
            patterns=patterns,
            supported_language=supported_language,
            supported_entity=supported_entity,
            institution_name=cfg.get("institution", "Unknown"),
        )

    def load(self) -> None:
        """No loading required — patterns are compiled at construction time."""

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        """
        Analyze text for institution-specific PHI.

        Applies patterns in priority order (most specific first). Overlapping
        matches are deduplicated, preferring higher-confidence matches.

        Args:
            text: Text to analyze.
            entities: List of entities to detect.
            nlp_artifacts: NLP artifacts (not used).

        Returns:
            List of RecognizerResult objects for detected institution PHI.
        """
        if self._supported_entity not in entities:
            return []

        results: list[RecognizerResult] = []
        seen_spans: set[tuple[int, int]] = set()

        for pattern, score, pattern_name, category in self._patterns:
            for match in pattern.finditer(text):
                start = match.start()
                end = match.end()

                matched_text = text[start:end]
                trailing = self._TRAILING_PUNCT.search(matched_text)
                if trailing:
                    end -= len(trailing.group())
                if end <= start:
                    continue

                span = (start, end)

                if span in seen_spans:
                    continue

                # Skip if this span is fully contained within an existing match
                is_subset = False
                for existing_start, existing_end in seen_spans:
                    if start >= existing_start and end <= existing_end:
                        is_subset = True
                        break
                if is_subset:
                    continue

                # Remove existing spans that are subsets of this longer match
                subsumed = {
                    (es, ee)
                    for es, ee in seen_spans
                    if es >= start and ee <= end
                }
                if subsumed:
                    seen_spans -= subsumed
                    results = [
                        r for r in results
                        if (r.start, r.end) not in subsumed
                    ]

                seen_spans.add(span)

                explanation = AnalysisExplanation(
                    recognizer=self.name,
                    original_score=score,
                    textual_explanation=(
                        f"{self._institution_name} pattern '{pattern_name}' "
                        f"({category}) matched"
                    ),
                    pattern=pattern.pattern,
                )

                results.append(
                    RecognizerResult(
                        entity_type=self._supported_entity,
                        start=start,
                        end=end,
                        score=score,
                        analysis_explanation=explanation,
                        recognition_metadata={
                            "recognizer_name": self.name,
                            "pattern_name": pattern_name,
                            "category": category,
                            "institution": self._institution_name,
                        },
                    )
                )

        return results

    def get_supported_entities(self) -> list[str]:
        """Return supported entities."""
        return [self._supported_entity]
