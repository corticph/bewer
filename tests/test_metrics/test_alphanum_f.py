"""Tests for bewer.metrics.alphanum_f module."""

import pytest

from bewer import Dataset
from bewer.metrics.alphanum_f import AlphaNumF, AlphaNumF_


class TestAlphaNumFExampleMetric:
    """Tests for AlphaNumF_ (ExampleMetric)."""

    @pytest.fixture
    def dataset_perfect(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an MRI")
        return dataset

    @pytest.fixture
    def dataset_all_fn(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an exam")
        return dataset

    @pytest.fixture
    def dataset_partial_recall(self):
        """TP=1, FN=1, FP=0."""
        dataset = Dataset()
        dataset.add(ref="MRI and HbA1c", hyp="MRI and haemoglobin")
        return dataset

    @pytest.fixture
    def dataset_partial_precision(self):
        """TP=1, FN=0, FP=1."""
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had MRI MRI")
        return dataset

    def test_value_perfect(self, dataset_perfect):
        f = dataset_perfect[0].metrics.alphanum_f()
        assert f.value == 1.0

    def test_value_all_fn(self, dataset_all_fn):
        f = dataset_all_fn[0].metrics.alphanum_f()
        assert f.value == 0.0

    def test_value_zero_denominator(self):
        """TP=FN=FP=0 → 0.0 (no entities in ref or hyp)."""
        dataset = Dataset()
        dataset.add(ref="patient was fine", hyp="patient was fine")
        f = dataset[0].metrics.alphanum_f()
        assert f.value == 0.0

    def test_partial_recall_full_precision(self, dataset_partial_recall):
        f = dataset_partial_recall[0].metrics.alphanum_f()
        # F1 = 2*1 / (2*1 + 1 + 0) = 2/3
        assert f.value == pytest.approx(2 / 3)

    def test_full_recall_partial_precision(self, dataset_partial_precision):
        f = dataset_partial_precision[0].metrics.alphanum_f()
        # F1 = 2*1 / (2*1 + 0 + 1) = 2/3
        assert f.value == pytest.approx(2 / 3)

    def test_beta_greater_than_one_weights_recall(self, dataset_partial_precision):
        """TP=1, FN=0, FP=1: beta=2 penalises FP less → higher score."""
        f_beta2 = dataset_partial_precision[0].metrics.alphanum_f(beta=2.0)
        f_beta05 = dataset_partial_precision[0].metrics.alphanum_f(beta=0.5)
        assert f_beta2.value > f_beta05.value

    def test_beta_less_than_one_weights_precision(self, dataset_partial_recall):
        """TP=1, FN=1, FP=0: beta=0.5 penalises FN less → higher score."""
        f_beta05 = dataset_partial_recall[0].metrics.alphanum_f(beta=0.5)
        f_beta2 = dataset_partial_recall[0].metrics.alphanum_f(beta=2.0)
        assert f_beta05.value > f_beta2.value

    def test_beta_one_equals_harmonic_mean(self, dataset_partial_precision):
        f = dataset_partial_precision[0].metrics.alphanum_f()
        p = dataset_partial_precision[0].metrics.alphanum_p()
        r = dataset_partial_precision[0].metrics.alphanum_r()
        expected = 2 * p.value * r.value / (p.value + r.value)
        assert f.value == pytest.approx(expected)

    def test_invalid_beta_raises(self):
        dataset = Dataset()
        dataset.add(ref="MRI", hyp="MRI")
        with pytest.raises(ValueError, match="beta must be positive"):
            dataset.metrics.alphanum_f(beta=0.0).value


class TestAlphaNumFDatasetMetric:
    @pytest.fixture
    def mixed_dataset(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an MRI")
        dataset.add(ref="HbA1c elevated", hyp="haemoglobin elevated")
        return dataset

    def test_value_all_correct(self):
        dataset = Dataset()
        dataset.add(ref="MRI", hyp="MRI")
        dataset.add(ref="ECG", hyp="ECG")
        f = dataset.metrics.alphanum_f()
        assert f.value == 1.0

    def test_value_calculation(self, mixed_dataset):
        f = mixed_dataset.metrics.alphanum_f()
        # TP=1, FN=1, FP=0 → 2*1 / (2*1 + 1 + 0) = 2/3
        assert f.value == pytest.approx(2 / 3)

    def test_beta_affects_value(self, mixed_dataset):
        f1 = mixed_dataset.metrics.alphanum_f(beta=1.0)
        f2 = mixed_dataset.metrics.alphanum_f(beta=2.0)
        assert f1.value != f2.value


class TestAlphaNumFMetricAttributes:
    def test_short_name_base(self):
        assert AlphaNumF.short_name_base == "AlphaNumF"

    def test_long_name_base(self):
        assert AlphaNumF.long_name_base == "Alphanumerical Entity F-Score"

    def test_description(self):
        assert len(AlphaNumF.description) > 0

    def test_example_cls(self):
        assert AlphaNumF.example_cls == AlphaNumF_

    def test_metric_values_main(self):
        assert AlphaNumF.metric_values()["main"] == "value"

    def test_beta_in_short_name(self):
        dataset = Dataset()
        dataset.add(ref="MRI", hyp="MRI")
        f = dataset.metrics.alphanum_f(beta=2.0)
        assert "beta=2.0" in f.short_name
