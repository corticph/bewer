"""Tests for bewer.metrics.ktr module."""

import pytest

from bewer import Dataset
from bewer.metrics.ktr import KTR, KTR_


class TestKTRExampleMetric:
    """Tests for KTR_ (ExampleMetric) class."""

    @pytest.fixture
    def dataset_keyword_match(self):
        """Dataset where a key term is correctly transcribed."""
        dataset = Dataset()
        dataset.add(
            ref="the patient has diabetes",
            hyp="the patient has diabetes",
            keywords={"key_terms": ["diabetes"]},
        )
        return dataset

    @pytest.fixture
    def dataset_keyword_error(self):
        """Dataset where a key term is incorrectly transcribed."""
        dataset = Dataset()
        dataset.add(
            ref="the patient has diabetes",
            hyp="the patient has diabetis",
            keywords={"key_terms": ["diabetes"]},
        )
        return dataset

    def test_value_perfect_match(self, dataset_keyword_match):
        """Test KTR value is 1.0 when key term is correctly transcribed."""
        example = dataset_keyword_match[0]
        ktr = example.metrics.ktr(vocab="key_terms")
        assert ktr.value == 1.0

    def test_value_with_error(self, dataset_keyword_error):
        """Test KTR value is 0.0 when key term is incorrectly transcribed."""
        example = dataset_keyword_error[0]
        ktr = example.metrics.ktr(vocab="key_terms")
        assert ktr.value == 0.0

    def test_num_matches_perfect(self, dataset_keyword_match):
        """Test num_matches is 1 when key term matches."""
        example = dataset_keyword_match[0]
        ktr = example.metrics.ktr(vocab="key_terms")
        assert ktr.num_matches == 1

    def test_num_matches_with_error(self, dataset_keyword_error):
        """Test num_matches is 0 when key term has error."""
        example = dataset_keyword_error[0]
        ktr = example.metrics.ktr(vocab="key_terms")
        assert ktr.num_matches == 0

    def test_num_ref_terms(self, dataset_keyword_match):
        """Test num_ref_terms counts correctly."""
        example = dataset_keyword_match[0]
        ktr = example.metrics.ktr(vocab="key_terms")
        assert ktr.num_ref_terms == 1

    def test_ktr_is_complement_of_kter(self, dataset_keyword_error):
        """Test that KTR = 1 - KTER for the same example."""
        example = dataset_keyword_error[0]
        ktr = example.metrics.ktr(vocab="key_terms")
        kter = example.metrics.kter(vocab="key_terms")
        assert ktr.value == pytest.approx(1 - kter.value)

    def test_partial_matches(self):
        """Test KTR with some key terms matching and some not."""
        dataset = Dataset()
        dataset.add(
            ref="patient has diabetes and hypertension",
            hyp="patient has diabetes and hypotension",
            keywords={"key_terms": ["diabetes", "hypertension"]},
        )
        example = dataset[0]
        ktr = example.metrics.ktr(vocab="key_terms")
        assert ktr.num_ref_terms == 2
        assert ktr.num_matches == 1
        assert ktr.value == 0.5

    def test_works_with_any_vocab(self):
        """Test KTR is domain-agnostic and works with any vocabulary name."""
        dataset = Dataset()
        dataset.add(
            ref="the quick brown fox",
            hyp="the quick brown fox",
            keywords={"animals": ["fox"]},
        )
        example = dataset[0]
        ktr = example.metrics.ktr(vocab="animals")
        assert ktr.value == 1.0
        assert ktr.num_matches == 1


class TestKTRDatasetMetric:
    """Tests for KTR (dataset-level Metric) class."""

    @pytest.fixture
    def key_terms_dataset(self):
        """Create a dataset with key terms across multiple examples."""
        dataset = Dataset()
        dataset.add(
            ref="patient has diabetes",
            hyp="patient has diabetes",
            keywords={"key_terms": ["diabetes"]},
        )
        dataset.add(
            ref="patient has asthma",
            hyp="patient has astma",
            keywords={"key_terms": ["asthma"]},
        )
        return dataset

    def test_num_ref_terms_aggregates(self, key_terms_dataset):
        """Test that dataset num_ref_terms sums example-level counts."""
        ktr = key_terms_dataset.metrics.ktr(vocab="key_terms")
        expected = sum(ex.metrics.ktr(vocab="key_terms").num_ref_terms for ex in key_terms_dataset)
        assert ktr.num_ref_terms == expected

    def test_num_matches_aggregates(self, key_terms_dataset):
        """Test that dataset num_matches sums example-level counts."""
        ktr = key_terms_dataset.metrics.ktr(vocab="key_terms")
        expected = sum(ex.metrics.ktr(vocab="key_terms").num_matches for ex in key_terms_dataset)
        assert ktr.num_matches == expected

    def test_value_calculation(self, key_terms_dataset):
        """Test dataset-level KTR value calculation."""
        ktr = key_terms_dataset.metrics.ktr(vocab="key_terms")
        # 1 match out of 2 key terms = 0.5
        assert ktr.num_ref_terms == 2
        assert ktr.num_matches == 1
        assert ktr.value == 0.5

    def test_ktr_is_complement_of_kter(self, key_terms_dataset):
        """Test that dataset-level KTR = 1 - KTER for the same vocab."""
        ktr = key_terms_dataset.metrics.ktr(vocab="key_terms")
        kter = key_terms_dataset.metrics.kter(vocab="key_terms")
        assert ktr.value == pytest.approx(1 - kter.value)

    def test_empty_dataset(self):
        """Test KTR on empty dataset raises when vocab is not registered."""
        dataset = Dataset()
        with pytest.raises(ValueError, match="not found in dataset keyword vocabularies"):
            dataset.metrics.ktr(vocab="key_terms").value

    def test_all_correct(self):
        """Test KTR is 1.0 when all key terms are correct."""
        dataset = Dataset()
        dataset.add(ref="diabetes mellitus", hyp="diabetes mellitus", keywords={"key_terms": ["diabetes"]})
        dataset.add(ref="acute asthma", hyp="acute asthma", keywords={"key_terms": ["asthma"]})
        ktr = dataset.metrics.ktr(vocab="key_terms")
        assert ktr.value == 1.0


class TestKTRMetricAttributes:
    """Tests for KTR metric attributes."""

    def test_short_name_base(self):
        assert KTR.short_name_base == "KTR"

    def test_long_name_base(self):
        assert KTR.long_name_base == "Key Term Recall"

    def test_description(self):
        assert len(KTR.description) > 0

    def test_example_cls(self):
        assert KTR.example_cls == KTR_

    def test_metric_values_main(self):
        values = KTR.metric_values()
        assert values["main"] == "value"

    def test_metric_values_other(self):
        values = KTR.metric_values()
        assert "num_matches" in values["other"]
        assert "num_ref_terms" in values["other"]
