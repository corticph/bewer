"""Tests for bewer.metrics.ktp module."""

import pytest

from bewer import Dataset
from bewer.metrics.ktp import KTP, KTP_


class TestKTPExampleMetric:
    """Tests for KTP_ (ExampleMetric) class."""

    @pytest.fixture
    def dataset_keyword_match(self):
        """Dataset where a key term is correctly transcribed (TP=1, FP=0)."""
        dataset = Dataset()
        dataset.add(
            ref="the fox jumps",
            hyp="the fox jumps",
            key_terms={"animals": ["fox"]},
        )
        return dataset

    @pytest.fixture
    def dataset_keyword_error(self):
        """Dataset where a key term is in ref but absent from hyp (TP=0, FP=0)."""
        dataset = Dataset()
        dataset.add(
            ref="the fox jumps",
            hyp="the dog jumps",
            key_terms={"animals": ["fox"]},
        )
        return dataset

    @pytest.fixture
    def dataset_keyword_fp(self):
        """Dataset where ref has a key term once but hyp has it twice (TP=1, FP=1)."""
        dataset = Dataset()
        dataset.add(
            ref="the fox jumps",
            hyp="the fox fox jumps",
            key_terms={"animals": ["fox"]},
        )
        return dataset

    def test_value_perfect_match(self, dataset_keyword_match):
        """Test KTP = 1.0 when key term is correctly transcribed."""
        example = dataset_keyword_match[0]
        ktp = example.metrics.ktp(vocab="animals")
        assert ktp.value == 1.0

    def test_value_zero_hyp_terms(self, dataset_keyword_error):
        """Test KTP = 0.0 when key term is absent from hypothesis (denominator guard)."""
        example = dataset_keyword_error[0]
        ktp = example.metrics.ktp(vocab="animals")
        assert ktp.value == 0.0

    def test_value_with_false_positive(self, dataset_keyword_fp):
        """Test KTP = 0.5 when hyp has key term twice but ref has it once (TP=1, FP=1)."""
        example = dataset_keyword_fp[0]
        ktp = example.metrics.ktp(vocab="animals")
        assert ktp.value == pytest.approx(0.5)

    def test_num_matches_perfect(self, dataset_keyword_match):
        """Test num_matches = 1 on correct transcription."""
        example = dataset_keyword_match[0]
        ktp = example.metrics.ktp(vocab="animals")
        assert ktp.num_matches == 1

    def test_num_matches_zero(self, dataset_keyword_error):
        """Test num_matches = 0 when key term is absent from hypothesis."""
        example = dataset_keyword_error[0]
        ktp = example.metrics.ktp(vocab="animals")
        assert ktp.num_matches == 0

    def test_num_fp_correct(self, dataset_keyword_match):
        """Test num_fp = 0 on exact match."""
        example = dataset_keyword_match[0]
        ktp = example.metrics.ktp(vocab="animals")
        assert ktp.num_fp == 0

    def test_num_fp_with_fp(self, dataset_keyword_fp):
        """Test num_fp = 1 when key term appears twice in hypothesis (one spurious)."""
        example = dataset_keyword_fp[0]
        ktp = example.metrics.ktp(vocab="animals")
        assert ktp.num_fp == 1

    def test_works_with_any_vocab(self):
        """Test KTP is domain-agnostic and works with any vocabulary name."""
        dataset = Dataset()
        dataset.add(
            ref="patient has diabetes",
            hyp="patient has diabetes",
            key_terms={"medical": ["diabetes"]},
        )
        ktp = dataset[0].metrics.ktp(vocab="medical")
        assert ktp.value == 1.0
        assert ktp.num_matches == 1


class TestKTPDatasetMetric:
    """Tests for KTP (dataset-level Metric) class."""

    @pytest.fixture
    def ktp_dataset(self):
        """Dataset with mixed precision: one exact match, one FP."""
        dataset = Dataset()
        dataset.add(
            ref="the fox jumps",
            hyp="the fox jumps",
            key_terms={"animals": ["fox"]},
        )
        dataset.add(
            ref="the rabbit runs",
            hyp="the rabbit rabbit runs",
            key_terms={"animals": ["rabbit"]},
        )
        return dataset

    def test_num_fp_aggregates(self, ktp_dataset):
        """Test that dataset num_fp sums example-level counts."""
        ktp = ktp_dataset.metrics.ktp(vocab="animals")
        expected = sum(ex.metrics.ktp(vocab="animals").num_fp for ex in ktp_dataset)
        assert ktp.num_fp == expected

    def test_num_matches_aggregates(self, ktp_dataset):
        """Test that dataset num_matches sums example-level counts."""
        ktp = ktp_dataset.metrics.ktp(vocab="animals")
        expected = sum(ex.metrics.ktp(vocab="animals").num_matches for ex in ktp_dataset)
        assert ktp.num_matches == expected

    def test_value_calculation(self, ktp_dataset):
        """Test dataset-level KTP = num_tp / (num_tp + num_fp)."""
        ktp = ktp_dataset.metrics.ktp(vocab="animals")
        # Example 1: TP=1, FP=0. Example 2: TP=1, FP=1.
        assert ktp.num_matches == 2
        assert ktp.num_fp == 1
        assert ktp.value == pytest.approx(2 / 3)

    def test_all_correct(self):
        """Test KTP = 1.0 when all transcriptions are correct."""
        dataset = Dataset()
        dataset.add(ref="the fox", hyp="the fox", key_terms={"animals": ["fox"]})
        dataset.add(ref="the rabbit", hyp="the rabbit", key_terms={"animals": ["rabbit"]})
        ktp = dataset.metrics.ktp(vocab="animals")
        assert ktp.value == 1.0

    def test_empty_dataset(self):
        """Test KTP raises ValueError for unknown vocab."""
        dataset = Dataset()
        with pytest.raises(ValueError, match="not found in dataset key term vocabularies"):
            dataset.metrics.ktp(vocab="animals").value


class TestKTPMetricAttributes:
    """Tests for KTP metric attributes."""

    def test_short_name_base(self):
        assert KTP.short_name_base == "KTP"

    def test_long_name_base(self):
        assert KTP.long_name_base == "Key Term Precision"

    def test_description(self):
        assert len(KTP.description) > 0

    def test_example_cls(self):
        assert KTP.example_cls == KTP_

    def test_metric_values_main(self):
        assert KTP.metric_values()["main"] == "value"

    def test_metric_values_other(self):
        values = KTP.metric_values()
        assert "num_matches" in values["other"]
        assert "num_fp" in values["other"]
