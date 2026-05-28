"""Tests for bewer.metrics.alphanum_p module."""

import pytest

from bewer import Dataset
from bewer.metrics.alphanum_p import AlphaNumP, AlphaNumP_


class TestAlphaNumPExampleMetric:
    """Tests for AlphaNumP_ (ExampleMetric)."""

    @pytest.fixture
    def dataset_correct(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an MRI")
        return dataset

    @pytest.fixture
    def dataset_spurious(self):
        """Hyp invents an MRI not in ref (TP=0, FP=1)."""
        dataset = Dataset()
        dataset.add(ref="patient had an exam", hyp="patient had an MRI")
        return dataset

    @pytest.fixture
    def dataset_partial(self):
        """Ref has MRI once, hyp says MRI twice (TP=1, FP=1)."""
        dataset = Dataset()
        dataset.add(ref="patient had an MRI scan", hyp="patient had an MRI MRI scan")
        return dataset

    def test_value_perfect(self, dataset_correct):
        p = dataset_correct[0].metrics.alphanum_p()
        assert p.value == 1.0

    def test_value_all_fp(self, dataset_spurious):
        p = dataset_spurious[0].metrics.alphanum_p()
        assert p.value == 0.0

    def test_value_partial_precision(self, dataset_partial):
        p = dataset_partial[0].metrics.alphanum_p()
        assert p.value == pytest.approx(0.5)

    def test_num_matches(self, dataset_correct):
        p = dataset_correct[0].metrics.alphanum_p()
        assert p.num_matches == 1

    def test_num_fp(self, dataset_partial):
        p = dataset_partial[0].metrics.alphanum_p()
        assert p.num_fp == 1


class TestAlphaNumPDatasetMetric:
    """Tests for AlphaNumP (dataset-level Metric)."""

    @pytest.fixture
    def mixed_dataset(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an MRI")
        dataset.add(ref="HbA1c elevated", hyp="HbA1c HbA1c elevated")  # FP=1
        return dataset

    def test_num_matches_aggregates(self, mixed_dataset):
        p = mixed_dataset.metrics.alphanum_p()
        expected = sum(ex.metrics.alphanum_p().num_matches for ex in mixed_dataset)
        assert p.num_matches == expected

    def test_num_fp_aggregates(self, mixed_dataset):
        p = mixed_dataset.metrics.alphanum_p()
        expected = sum(ex.metrics.alphanum_p().num_fp for ex in mixed_dataset)
        assert p.num_fp == expected

    def test_value_calculation(self, mixed_dataset):
        p = mixed_dataset.metrics.alphanum_p()
        # TP=2, FP=1 → 2/3
        assert p.num_matches == 2
        assert p.num_fp == 1
        assert p.value == pytest.approx(2 / 3)

    def test_all_correct(self):
        dataset = Dataset()
        dataset.add(ref="MRI", hyp="MRI")
        dataset.add(ref="ECG", hyp="ECG")
        p = dataset.metrics.alphanum_p()
        assert p.value == 1.0

    def test_no_terms_in_hyp(self):
        """When hyp has no alphanumerical entities at all, value is 0.0 (denominator guard)."""
        dataset = Dataset()
        dataset.add(ref="patient was fine", hyp="patient was fine")
        p = dataset.metrics.alphanum_p()
        assert p.value == 0.0


class TestAlphaNumPMetricAttributes:
    def test_short_name_base(self):
        assert AlphaNumP.short_name_base == "AlphaNumP"

    def test_long_name_base(self):
        assert AlphaNumP.long_name_base == "Alphanumerical Entity Precision"

    def test_description(self):
        assert len(AlphaNumP.description) > 0

    def test_example_cls(self):
        assert AlphaNumP.example_cls == AlphaNumP_

    def test_metric_values_main(self):
        assert AlphaNumP.metric_values()["main"] == "value"

    def test_metric_values_other(self):
        values = AlphaNumP.metric_values()
        assert "num_matches" in values["other"]
        assert "num_fp" in values["other"]
