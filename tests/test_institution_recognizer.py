"""Tests for InstitutionRecognizer.

Covers Stanford Health Care patterns across all categories:
URLs, social handles, facilities, locations, portals, abbreviations,
system codes, typos, and the core institution name.
"""

import json
import tempfile
from pathlib import Path

import pytest

from tide2.recognizers.institution_recognizer import InstitutionRecognizer


@pytest.fixture
def recognizer():
    """Create a default Stanford InstitutionRecognizer."""
    return InstitutionRecognizer()


def _detect(recognizer, text: str, entity: str = "INSTITUTION") -> list:
    """Helper: run analyze and return results."""
    return recognizer.analyze(text=text, entities=[entity])


def _detected_texts(recognizer, text: str) -> list[str]:
    """Helper: return the matched substrings."""
    results = _detect(recognizer, text)
    return [text[r.start : r.end] for r in results]


# ---------------------------------------------------------------------------
# URLs & domains
# ---------------------------------------------------------------------------


class TestUrls:
    def test_full_url(self, recognizer):
        text = "Visit www.stanfordhealthcare.org/appointments for scheduling."
        results = _detect(recognizer, text)
        assert len(results) >= 1
        assert any(r.score >= 0.90 for r in results)

    def test_domain_shc(self, recognizer):
        text = "Portal at stanfordhealthcare.org/patient"
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_domain_hospital(self, recognizer):
        text = "See stanfordhospital.org for directions."
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_domain_med(self, recognizer):
        text = "Login at stanfordmed.org/portal"
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_evercore(self, recognizer):
        text = "Access evercore.stanfordmed.org for records."
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_bitly_link(self, recognizer):
        text = "Register at bit.ly/StanfordPatientEdReg"
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_live_donors(self, recognizer):
        text = "See StanfordHealthCareLiveDonors.org for info."
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_endo(self, recognizer):
        text = "Contact stanfordendo for scheduling."
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_pain_medicine(self, recognizer):
        text = "Refer to stanfordpainmedicine for options."
        results = _detect(recognizer, text)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Social media
# ---------------------------------------------------------------------------


class TestSocial:
    def test_social_handle(self, recognizer):
        text = "Follow us @StanfordHealth on Twitter."
        results = _detect(recognizer, text)
        matched = _detected_texts(recognizer, text)
        assert "@StanfordHealth" in matched

    def test_social_med(self, recognizer):
        text = "See @StanfordMed for updates."
        results = _detect(recognizer, text)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Facilities
# ---------------------------------------------------------------------------


class TestFacilities:
    def test_palo_alto_medical_foundation(self, recognizer):
        text = "Referred from Palo Alto Medical Foundation."
        matched = _detected_texts(recognizer, text)
        assert any("Palo Alto Medical Foundation" in m for m in matched)

    def test_packard_childrens(self, recognizer):
        text = "Admitted to Packard Children's Hospital."
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_lucile_salter(self, recognizer):
        text = "Born at Lucile Salter Packard Children's Hospital."
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_pamf(self, recognizer):
        text = "Records from PAMF on file."
        matched = _detected_texts(recognizer, text)
        assert "PAMF" in matched

    def test_pamf_title_case(self, recognizer):
        text = "Seen at Pamf clinic."
        matched = _detected_texts(recognizer, text)
        assert "Pamf" in matched


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


class TestLocations:
    def test_palo_alto(self, recognizer):
        text = "Patient lives in Palo Alto, CA."
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_blake_wilbur_building(self, recognizer):
        text = "Appointment at Blake Wilbur Building, 2nd floor."
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_pasteur_drive(self, recognizer):
        text = "Located at 300 Pasteur Drive."
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_pasteur_dr(self, recognizer):
        text = "Office at 300 Pasteur Dr, Stanford."
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_mail_code(self, recognizer):
        text = "Send to MC 5985, Stanford."
        matched = _detected_texts(recognizer, text)
        assert any("MC 5985" in m for m in matched)

    def test_unit_code(self, recognizer):
        text = "Patient in ICU-4523 bed 3."
        results = _detect(recognizer, text)
        assert any("ICU-4523" in text[r.start : r.end] for r in results)

    def test_nicu_code(self, recognizer):
        text = "Transferred to NICU-12345."
        results = _detect(recognizer, text)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Portal names
# ---------------------------------------------------------------------------


class TestPortals:
    def test_myhealth_title(self, recognizer):
        text = "Message your doctor via MyHealth."
        matched = _detected_texts(recognizer, text)
        assert "MyHealth" in matched

    def test_myhealth_upper(self, recognizer):
        text = "Access MYHEALTH for lab results."
        matched = _detected_texts(recognizer, text)
        assert "MYHEALTH" in matched

    def test_myhealth_lower(self, recognizer):
        text = "Log into myhealth today."
        matched = _detected_texts(recognizer, text)
        assert "myhealth" in matched

    def test_myhealth_typo(self, recognizer):
        text = "Check MyHeath for results."
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_sfm_myhealth(self, recognizer):
        text = "Sent via Sfm Myhealth Clinic Messaging."
        results = _detect(recognizer, text)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# System codes
# ---------------------------------------------------------------------------


class TestSystemCodes:
    def test_shcect01(self, recognizer):
        text = "Template: SHCECT01"
        matched = _detected_texts(recognizer, text)
        assert "SHCECT01" in matched

    def test_shcvclogo(self, recognizer):
        text = "Image: SHCVCLOGO.png"
        matched = _detected_texts(recognizer, text)
        assert "SHCVCLOGO" in matched

    def test_shctv(self, recognizer):
        text = "Channel SHCTV broadcast."
        matched = _detected_texts(recognizer, text)
        assert "SHCTV" in matched


# ---------------------------------------------------------------------------
# Abbreviations
# ---------------------------------------------------------------------------


class TestAbbreviations:
    def test_shca(self, recognizer):
        text = "Visit SHCA for your appointment."
        matched = _detected_texts(recognizer, text)
        assert "SHCA" in matched

    def test_shce(self, recognizer):
        text = "Referred to SHCE clinic."
        matched = _detected_texts(recognizer, text)
        assert "SHCE" in matched


# ---------------------------------------------------------------------------
# Core institution name
# ---------------------------------------------------------------------------


class TestCoreName:
    def test_stanford_title_case(self, recognizer):
        text = "Patient was seen at Stanford Hospital."
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_stanford_upper(self, recognizer):
        text = "STANFORD HEALTH CARE discharge summary."
        results = _detect(recognizer, text)
        assert len(results) >= 1

    def test_stanford_lower(self, recognizer):
        text = "Records from stanford medical center."
        results = _detect(recognizer, text)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Typos
# ---------------------------------------------------------------------------


class TestTypos:
    def test_standford_typo(self, recognizer):
        text = "Seen at Standford clinic."
        matched = _detected_texts(recognizer, text)
        assert "Standford" in matched

    def test_standford_lower(self, recognizer):
        text = "From standford hospital."
        matched = _detected_texts(recognizer, text)
        assert "standford" in matched


# ---------------------------------------------------------------------------
# Entity filtering
# ---------------------------------------------------------------------------


class TestEntityFiltering:
    def test_ignores_unrequested_entity(self, recognizer):
        text = "Patient at Stanford Hospital."
        results = recognizer.analyze(text=text, entities=["PERSON"])
        assert len(results) == 0

    def test_responds_to_institution_entity(self, recognizer):
        text = "Patient at Stanford Hospital."
        results = recognizer.analyze(text=text, entities=["INSTITUTION"])
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Overlap deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_no_duplicate_spans(self, recognizer):
        text = "Visit stanfordhealthcare.org for info about Stanford."
        results = _detect(recognizer, text)
        spans = [(r.start, r.end) for r in results]
        assert len(spans) == len(set(spans)), "Duplicate spans detected"

    def test_subset_suppressed(self, recognizer):
        text = "Go to www.stanfordhealthcare.org/appointments"
        results = _detect(recognizer, text)
        # The longer URL match should suppress shorter substring matches
        texts = [text[r.start : r.end] for r in results]
        assert any("www.stanfordhealthcare.org" in t for t in texts)


# ---------------------------------------------------------------------------
# from_config (JSON loading)
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_load_from_json(self, tmp_path):
        config = {
            "institution": "Test Hospital",
            "rules": [
                {
                    "pattern": r"\bTestHosp\b",
                    "flags": ["IGNORECASE"],
                    "score": 0.90,
                    "label": "test_hosp",
                    "category": "name",
                },
                {
                    "pattern": r"testhospital\.org",
                    "score": 0.95,
                    "label": "test_url",
                    "category": "url",
                },
            ],
        }
        config_path = tmp_path / "test.json"
        config_path.write_text(json.dumps(config))

        rec = InstitutionRecognizer.from_config(config_path)
        results = _detect(rec, "Visit testhospital.org or call TestHosp.")
        assert len(results) == 2

    def test_config_scores(self, tmp_path):
        config = {
            "institution": "Test",
            "rules": [
                {
                    "pattern": r"\bFooBar\b",
                    "score": 0.42,
                    "label": "foobar",
                },
            ],
        }
        config_path = tmp_path / "test.json"
        config_path.write_text(json.dumps(config))

        rec = InstitutionRecognizer.from_config(config_path)
        results = _detect(rec, "Check FooBar now.")
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Recognition metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_fields(self, recognizer):
        text = "Visit Stanford today."
        results = _detect(recognizer, text)
        assert len(results) >= 1
        meta = results[0].recognition_metadata
        assert meta["recognizer_name"] == "InstitutionRecognizer"
        assert meta["institution"] == "Stanford Health Care"
        assert "pattern_name" in meta
        assert "category" in meta
