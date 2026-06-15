"""Tests for the quantity patterns and the unified RegexExtractor span behaviour."""

import pytest

from bewer import Dataset
from bewer.extractors import RegexExtractor
from bewer.extractors.quantity import (
    CURRENCY_PATTERN,
    DEGREE_PATTERN,
    MEASUREMENT_PATTERN,
    NUMBER_PATTERN,
    PERCENTAGE_PATTERN,
    QUANTITY_CATEGORIES,
)
from bewer.preprocessing.context import set_pipeline


@pytest.fixture
def quantity_context():
    """Activate the pipeline the quantity metrics run under (symbols kept as tokens, case kept)."""
    with set_pipeline(tokenizer="orthographically_complex_term", normalizer="cased"):
        yield


def _extract(text, pattern, *, span=True):
    dataset = Dataset()
    dataset.add(text, "x")
    return RegexExtractor(pattern, span=span)(dataset)


class TestRegexExtractorSpanFlag:
    """The single RegexExtractor switches between per-token and whole-text matching."""

    def test_token_mode_requires_full_token(self, quantity_context):
        # Per-token: NUMBER full-matches "2024" but not the "5" glued inside "5mg".
        assert _extract("year 2024 dose 5mg", NUMBER_PATTERN, span=False) == {"2024"}

    def test_span_mode_crosses_token_boundary(self, quantity_context):
        # Span: a number and its separately-tokenized unit are one term.
        assert _extract("dose 5 mg", MEASUREMENT_PATTERN, span=True) == {"5 mg"}

    def test_repr_marks_span(self):
        assert repr(RegexExtractor("a", span=True)) == "RegexExtractor('a', span=True)"
        assert repr(RegexExtractor("a")) == "RegexExtractor('a')"


class TestQuantityCategoryPatterns:
    """Each category pattern matches its own class of quantity."""

    def test_number_matches_all_numbers(self, quantity_context):
        # Plain numbers and the numeric part of decorated quantities (5 mg, 95%, $100) alike.
        assert _extract("in 2024 at 14:30 give 5 mg and 95% for $100", NUMBER_PATTERN, span=False) == {
            "2024",
            "14:30",
            "5",
            "95",
            "100",
        }

    def test_percentage(self, quantity_context):
        assert _extract("up 95% from 12 %", PERCENTAGE_PATTERN) == {"95%", "12 %"}

    def test_degree(self, quantity_context):
        assert _extract("at 37°C and 451 °F", DEGREE_PATTERN) == {"37°C", "451 °F"}

    def test_currency_either_side(self, quantity_context):
        assert _extract("costs $100 or 80€", CURRENCY_PATTERN) == {"$100", "80€"}

    def test_measurement(self, quantity_context):
        assert _extract("give 5 mg over 120 mmHg", MEASUREMENT_PATTERN) == {"5 mg", "120 mmHg"}

    def test_continental_decimal_measurement(self, quantity_context):
        """Locale-neutral: a comma-decimal measurement is captured like a dot-decimal one."""
        assert _extract("dose 3,5 mg", MEASUREMENT_PATTERN) == {"3,5 mg"}

    def test_measurement_unit_guard(self, quantity_context):
        """'mph' is not a known unit, so it is not matched as a measurement."""
        assert _extract("5 mph wind", MEASUREMENT_PATTERN) == set()


class TestCategoriesMayOverlap:
    """Categories are intentionally not disjoint: number overlaps the decorated categories."""

    def test_number_overlaps_measurement_and_percentage(self, quantity_context):
        text = "give 5 mg and 95%"
        results = {cat: _extract(text, pat, span=span) for cat, (pat, span) in QUANTITY_CATEGORIES.items()}
        assert results["number"] == {"5", "95"}
        assert results["measurement"] == {"5 mg"}
        assert results["percentage"] == {"95%"}
