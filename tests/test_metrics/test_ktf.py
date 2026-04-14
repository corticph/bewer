"""Tests for bewer.metrics.ktf module."""

import pytest

from bewer import Dataset
from bewer.metrics.ktf import KTF, KTF_


class TestKTFExampleMetric:
    """Tests for KTF_ (ExampleMetric) class."""

    @pytest.fixture
    def dataset_perfect(self):
        """Dataset where key term is correctly transcribed (TP=1, FN=0, FP=0)."""
        dataset = Dataset()
        dataset.add(
            ref="the fox jumps",
            hyp="the fox jumps",
            key_terms={"animals": ["fox"]},
        )
        return dataset

    @pytest.fixture
    def dataset_all_fn(self):
        """Dataset where key term is in ref but absent from hyp (TP=0, FN=1, FP=0)."""
        dataset = Dataset()
        dataset.add(
            ref="the fox jumps",
            hyp="the dog jumps",
            key_terms={"animals": ["fox"]},
        )
        return dataset

    @pytest.fixture
    def dataset_partial_recall(self):
        """Dataset with two ref key terms, one correct (TP=1, FN=1, FP=0)."""
        dataset = Dataset()
        dataset.add(
            ref="fox and rabbit",
            hyp="fox and hamster",
            key_terms={"animals": ["fox", "rabbit"]},
        )
        return dataset

    @pytest.fixture
    def dataset_partial_precision(self):
        """Dataset where ref has key term once but hyp has it twice (TP=1, FN=0, FP=1)."""
        dataset = Dataset()
        dataset.add(
            ref="the fox jumps",
            hyp="the fox fox jumps",
            key_terms={"animals": ["fox"]},
        )
        return dataset

    def test_value_perfect(self, dataset_perfect):
        """Test KTF = 1.0 when TP=1, FN=0, FP=0."""
        example = dataset_perfect[0]
        ktf = example.metrics.ktf(vocab="animals")
        assert ktf.value == 1.0

    def test_value_all_fn(self, dataset_all_fn):
        """Test KTF = 0.0 when TP=0, FN=1, FP=0."""
        example = dataset_all_fn[0]
        ktf = example.metrics.ktf(vocab="animals")
        assert ktf.value == 0.0

    def test_value_zero_denominator(self):
        """Test KTF = 0.0 when TP=FN=FP=0 (key term not found in ref or hyp)."""
        dataset = Dataset()
        dataset.add(
            ref="hello world",
            hyp="hello world",
            key_terms={"animals": ["fox"]},
        )
        from bewer.core.key_term import KeyTermNotFoundWarning

        example = dataset[0]
        ktf = example.metrics.ktf(vocab="animals")
        with pytest.warns(KeyTermNotFoundWarning):
            assert ktf.value == 0.0

    def test_partial_recall_full_precision(self, dataset_partial_recall):
        """Test F1 = 2/3 when TP=1, FN=1, FP=0 (full precision, partial recall)."""
        example = dataset_partial_recall[0]
        ktf = example.metrics.ktf(vocab="animals")
        # F1 = 2*1 / (2*1 + 1*1 + 0) = 2/3
        assert ktf.value == pytest.approx(2 / 3)

    def test_full_recall_partial_precision(self, dataset_partial_precision):
        """Test F1 = 2/3 when TP=1, FN=0, FP=1 (full recall, partial precision)."""
        example = dataset_partial_precision[0]
        ktf = example.metrics.ktf(vocab="animals")
        # F1 = 2*1 / (2*1 + 1*0 + 1) = 2/3
        assert ktf.value == pytest.approx(2 / 3)

    def test_beta_greater_than_one_weights_recall(self, dataset_partial_precision):
        """Test that beta=2 gives a higher score than beta=0.5 when FP > 0 and FN = 0."""
        example = dataset_partial_precision[0]
        # TP=1, FN=0, FP=1: beta=2 penalises FP less → higher score
        ktf_beta2 = example.metrics.ktf(vocab="animals", beta=2.0)
        ktf_beta05 = example.metrics.ktf(vocab="animals", beta=0.5)
        assert ktf_beta2.value > ktf_beta05.value

    def test_beta_less_than_one_weights_precision(self, dataset_partial_recall):
        """Test that beta=0.5 gives a higher score than beta=2 when FN > 0 and FP = 0."""
        example = dataset_partial_recall[0]
        # TP=1, FN=1, FP=0: beta=0.5 penalises FN less → higher score
        ktf_beta05 = example.metrics.ktf(vocab="animals", beta=0.5)
        ktf_beta2 = example.metrics.ktf(vocab="animals", beta=2.0)
        assert ktf_beta05.value > ktf_beta2.value

    def test_beta_one_equals_harmonic_mean(self, dataset_partial_precision):
        """Test that KTF(beta=1) equals the harmonic mean of KTP and KTR."""
        example = dataset_partial_precision[0]
        ktf = example.metrics.ktf(vocab="animals")
        ktp = example.metrics.ktp(vocab="animals")
        ktr = example.metrics.ktr(vocab="animals")
        expected = 2 * ktp.value * ktr.value / (ktp.value + ktr.value)
        assert ktf.value == pytest.approx(expected)


class TestKTFDatasetMetric:
    """Tests for KTF (dataset-level Metric) class."""

    @pytest.fixture
    def mixed_dataset(self):
        """Dataset with one correct example and one missed key term."""
        dataset = Dataset()
        dataset.add(
            ref="the fox jumps",
            hyp="the fox jumps",
            key_terms={"animals": ["fox"]},
        )
        dataset.add(
            ref="the rabbit runs",
            hyp="the dog runs",
            key_terms={"animals": ["rabbit"]},
        )
        return dataset

    def test_value_all_correct(self):
        """Test KTF = 1.0 when all key terms are correctly transcribed."""
        dataset = Dataset()
        dataset.add(ref="the fox", hyp="the fox", key_terms={"animals": ["fox"]})
        dataset.add(ref="the rabbit", hyp="the rabbit", key_terms={"animals": ["rabbit"]})
        ktf = dataset.metrics.ktf(vocab="animals")
        assert ktf.value == 1.0

    def test_value_calculation(self, mixed_dataset):
        """Test dataset-level KTF formula: (1+β²)·TP / ((1+β²)·TP + β²·FN + FP)."""
        ktf = mixed_dataset.metrics.ktf(vocab="animals")
        # TP=1, FN=1, FP=0. F1 = 2*1 / (2*1 + 1*1 + 0) = 2/3
        assert ktf.value == pytest.approx(2 / 3)

    def test_empty_dataset(self):
        """Test KTF raises ValueError for unknown vocab."""
        dataset = Dataset()
        with pytest.raises(ValueError, match="not found in dataset key term vocabularies"):
            dataset.metrics.ktf(vocab="animals").value

    def test_beta_affects_value(self, mixed_dataset):
        """Test that changing beta changes the dataset-level score."""
        ktf_f1 = mixed_dataset.metrics.ktf(vocab="animals", beta=1.0)
        ktf_f2 = mixed_dataset.metrics.ktf(vocab="animals", beta=2.0)
        assert ktf_f1.value != ktf_f2.value


class TestKTFMetricAttributes:
    """Tests for KTF metric attributes."""

    def test_short_name_base(self):
        assert KTF.short_name_base == "KTF"

    def test_long_name_base(self):
        assert KTF.long_name_base == "Key Term F-Score"

    def test_description(self):
        assert len(KTF.description) > 0

    def test_example_cls(self):
        assert KTF.example_cls == KTF_

    def test_metric_values_main(self):
        assert KTF.metric_values()["main"] == "value"

    def test_metric_values_other(self):
        """KTF only exposes the main value — no supporting sub-metrics."""
        assert KTF.metric_values()["other"] == []

    def test_beta_in_short_name(self):
        """Test that beta parameter appears in the metric short name."""
        dataset = Dataset()
        dataset.add(ref="the fox", hyp="the fox", key_terms={"animals": ["fox"]})
        ktf = dataset.metrics.ktf(vocab="animals", beta=2.0)
        assert "beta=2.0" in ktf.short_name
