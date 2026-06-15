"""Tests for register_regex_metric — the public way to define custom regex metrics."""

import pytest

from bewer import Dataset
from bewer.metrics import register_regex_metric
from bewer.metrics.base import METRIC_REGISTRY


@pytest.fixture(scope="module")
def acronym_metrics():
    """Register a custom acronym metric set once for this module."""
    return register_regex_metric("acronym", r"\p{Lu}{2,}", span=False, allow_override=True)


class TestRegisterRegexMetrics:
    def test_registers_recall_precision_fscore(self, acronym_metrics):
        for name in ("acronym_recall", "acronym_precision", "acronym_fscore"):
            assert name in METRIC_REGISTRY.metric_metadata

    def test_returns_the_three_classes(self, acronym_metrics):
        recall, precision, fscore = acronym_metrics
        assert recall.short_name_base == "acronym_recall"
        assert recall.long_name_base == "Acronym Recall"
        assert fscore.long_name_base == "Acronym F-Score"

    def test_metrics_compute_over_extracted_terms(self, acronym_metrics):
        dataset = Dataset()
        dataset.add("the FDA and NATO met", "the FDA and NASA met")  # FDA right, NATO->NASA wrong
        # Recall is 0.5 (1 of 2 reference acronyms transcribed correctly). Precision is 1.0:
        # the vocabulary is reference-derived, so the spurious "NASA" is not a known term and
        # so not a false positive (the same global-vocabulary semantics as the other metrics).
        assert dataset.metrics.acronym_recall().value == 0.5
        assert dataset.metrics.acronym_precision().value == 1.0

    def test_auto_registers_vocabulary(self, acronym_metrics):
        dataset = Dataset()
        dataset.add("the FDA met", "the FDA met")
        assert "acronym_terms" not in dataset._vocabularies
        dataset.metrics.acronym_recall().value
        assert "acronym_terms" in dataset._vocabularies

    def test_custom_vocab_and_label(self):
        recall, _, _ = register_regex_metric(
            "shout", r"\p{Lu}{3,}", vocab="loud_terms", label="Shouting", allow_override=True
        )
        assert recall.long_name_base == "Shouting Recall"
        dataset = Dataset()
        dataset.add("LOUD noise", "LOUD noise")
        dataset.metrics.shout_recall().value
        assert "loud_terms" in dataset._vocabularies
