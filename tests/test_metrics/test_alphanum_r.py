"""Tests for bewer.metrics.alphanum_r module."""

import pytest

from bewer import Dataset
from bewer.metrics.alphanum_r import AlphaNumR, AlphaNumR_


class TestAlphaNumRExampleMetric:
    """Tests for AlphaNumR_ (ExampleMetric)."""

    @pytest.fixture
    def dataset_correct(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an MRI")
        return dataset

    @pytest.fixture
    def dataset_missed(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an exam")
        return dataset

    @pytest.fixture
    def dataset_partial(self):
        """Two refs, one matched (TP=1, FN=1)."""
        dataset = Dataset()
        dataset.add(ref="MRI and HbA1c", hyp="MRI and haemoglobin")
        return dataset

    def test_value_perfect(self, dataset_correct):
        r = dataset_correct[0].metrics.alphanum_r()
        assert r.value == 1.0

    def test_value_all_missed(self, dataset_missed):
        r = dataset_missed[0].metrics.alphanum_r()
        assert r.value == 0.0

    def test_value_partial(self, dataset_partial):
        r = dataset_partial[0].metrics.alphanum_r()
        assert r.value == pytest.approx(0.5)

    def test_num_matches(self, dataset_correct):
        r = dataset_correct[0].metrics.alphanum_r()
        assert r.num_matches == 1

    def test_num_ref_terms(self, dataset_partial):
        r = dataset_partial[0].metrics.alphanum_r()
        assert r.num_ref_terms == 2

    def test_lowercased_hyp_drops_recall(self):
        """ref MRI vs hyp mri: case is lost → counts as FN."""
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an mri")
        r = dataset[0].metrics.alphanum_r()
        assert r.value == 0.0
        assert r.num_matches == 0
        assert r.num_ref_terms == 1


class TestAlphaNumRDatasetMetric:
    @pytest.fixture
    def mixed_dataset(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an MRI")
        dataset.add(ref="HbA1c elevated", hyp="haemoglobin elevated")
        return dataset

    def test_num_ref_terms_aggregates(self, mixed_dataset):
        r = mixed_dataset.metrics.alphanum_r()
        expected = sum(ex.metrics.alphanum_r().num_ref_terms for ex in mixed_dataset)
        assert r.num_ref_terms == expected

    def test_num_matches_aggregates(self, mixed_dataset):
        r = mixed_dataset.metrics.alphanum_r()
        expected = sum(ex.metrics.alphanum_r().num_matches for ex in mixed_dataset)
        assert r.num_matches == expected

    def test_value_calculation(self, mixed_dataset):
        r = mixed_dataset.metrics.alphanum_r()
        # TP=1, FN=1 → 0.5
        assert r.num_ref_terms == 2
        assert r.num_matches == 1
        assert r.value == 0.5

    def test_all_correct(self):
        dataset = Dataset()
        dataset.add(ref="MRI", hyp="MRI")
        dataset.add(ref="ECG", hyp="ECG")
        r = dataset.metrics.alphanum_r()
        assert r.value == 1.0


class TestAlphaNumRMetricAttributes:
    def test_short_name_base(self):
        assert AlphaNumR.short_name_base == "AlphaNumR"

    def test_long_name_base(self):
        assert AlphaNumR.long_name_base == "Alphanumerical Entity Recall"

    def test_description(self):
        assert len(AlphaNumR.description) > 0

    def test_example_cls(self):
        assert AlphaNumR.example_cls == AlphaNumR_

    def test_metric_values_main(self):
        assert AlphaNumR.metric_values()["main"] == "value"

    def test_metric_values_other(self):
        values = AlphaNumR.metric_values()
        assert "num_matches" in values["other"]
        assert "num_ref_terms" in values["other"]
