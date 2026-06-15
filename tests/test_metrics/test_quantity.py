"""Tests for the per-category quantity metrics."""

import pytest

from bewer import Dataset
from bewer.metrics.quantity import MeasurementFscore, MeasurementPrecision, MeasurementRecall


class TestQuantityAutoVocabulary:
    """Each category's vocabulary is registered on first use, out of the box."""

    def test_first_use_registers_vocabulary(self):
        dataset = Dataset()
        dataset.add("give 5 mg", "give 5 mg")
        assert "measurement_terms" not in dataset._vocabularies
        dataset.metrics.measurement_recall().value
        assert "measurement_terms" in dataset._vocabularies

    def test_works_after_dataset_already_frozen(self):
        dataset = Dataset()
        dataset.add("give 5 mg", "give 5 mg")
        dataset.metrics.wer().value  # freezes the dataset
        assert dataset.is_frozen
        assert dataset.metrics.measurement_recall().value == 1.0


class TestPerCategoryMetrics:
    """Categories are scored independently."""

    @pytest.fixture
    def dataset(self):
        d = Dataset()
        d.add("give 5 mg and 95%", "give 5 mg and 96%")  # measurement right, percentage+number wrong
        d.add("costs $100 in 2024", "costs $100 in 2025")  # currency right, number wrong
        return d

    def test_measurement_recall(self, dataset):
        assert dataset.metrics.measurement_recall().value == 1.0

    def test_percentage_recall(self, dataset):
        assert dataset.metrics.percentage_recall().value == 0.0

    def test_currency_recall(self, dataset):
        assert dataset.metrics.currency_recall().value == 1.0

    def test_number_recall_counts_all_numbers(self, dataset):
        # Reference numbers: 5, 95 (ex.0) and 100, 2024 (ex.1) — including the numeric parts of
        # the percentage/currency. Correct: 5 and 100 (95->96 and 2024->2025 are both wrong).
        number = dataset.metrics.number_recall()
        assert number.num_ref_terms == 4
        assert number.num_matches == 2
        assert number.value == 0.5

    def test_fscore_beta_flows_through(self, dataset):
        f1 = dataset.metrics.measurement_fscore().value
        p = dataset.metrics.measurement_precision().value
        r = dataset.metrics.measurement_recall().value
        assert f1 == pytest.approx(2 * p * r / (p + r))
        with pytest.raises(ValueError, match="beta must be positive"):
            dataset.metrics.number_fscore(beta=0).value


class TestQuantityMetricAttributes:
    """Generated metric metadata."""

    def test_each_category_registers_a_flat_trio(self):
        from bewer.metrics.base import METRIC_REGISTRY

        for base in ("number", "percentage", "degree", "currency", "measurement"):
            for kind in ("recall", "precision", "fscore"):
                assert f"{base}_{kind}" in METRIC_REGISTRY.metric_metadata

    def test_names_and_labels(self):
        assert MeasurementRecall.short_name_base == "measurement_recall"
        assert MeasurementRecall.long_name_base == "Measurement Recall"
        assert MeasurementPrecision.long_name_base == "Measurement Precision"
        assert MeasurementFscore.long_name_base == "Measurement F-Score"
        assert all(len(m.description) > 0 for m in (MeasurementRecall, MeasurementPrecision, MeasurementFscore))

    def test_main_value_and_pipeline(self):
        dataset = Dataset()
        dataset.add("give 5 mg", "give 5 mg")
        m = dataset.metrics.measurement_recall()
        assert type(m).metric_values()["main"] == "value"
        assert m.tokenizer == "orthographically_complex_term"
        assert m.normalizer == "cased"

    def test_no_combined_quantity_metric(self):
        """There is intentionally no agglomerated 'quantity' metric."""
        from bewer.metrics.base import METRIC_REGISTRY

        assert "quantity_recall" not in METRIC_REGISTRY.metric_metadata
        assert "qtr" not in METRIC_REGISTRY.metric_metadata
