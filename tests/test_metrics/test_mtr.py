"""Tests for bewer.metrics.mtr module."""

import pytest

from bewer import Dataset
from bewer.metrics.mtr import MTR, MTR_


class TestMTRExampleMetric:
    """Tests for MTR_ (ExampleMetric) class."""

    @pytest.fixture
    def dataset_keyword_match(self):
        """Dataset where a medical term is correctly transcribed."""
        dataset = Dataset()
        dataset.add(
            ref="the patient has diabetes",
            hyp="the patient has diabetes",
            keywords={"medical_terms": ["diabetes"]},
        )
        return dataset

    @pytest.fixture
    def dataset_keyword_error(self):
        """Dataset where a medical term is incorrectly transcribed."""
        dataset = Dataset()
        dataset.add(
            ref="the patient has diabetes",
            hyp="the patient has diabetis",
            keywords={"medical_terms": ["diabetes"]},
        )
        return dataset

    def test_value_perfect_match(self, dataset_keyword_match):
        """Test MTR value is 1.0 when medical term is correctly transcribed."""
        example = dataset_keyword_match[0]
        mtr = example.metrics.mtr()
        assert mtr.value == 1.0

    def test_value_with_error(self, dataset_keyword_error):
        """Test MTR value is 0.0 when medical term is incorrectly transcribed."""
        example = dataset_keyword_error[0]
        mtr = example.metrics.mtr()
        assert mtr.value == 0.0

    def test_num_matches_perfect(self, dataset_keyword_match):
        """Test num_matches is 1 when keyword matches."""
        example = dataset_keyword_match[0]
        mtr = example.metrics.mtr()
        assert mtr.num_matches == 1

    def test_num_matches_with_error(self, dataset_keyword_error):
        """Test num_matches is 0 when keyword has error."""
        example = dataset_keyword_error[0]
        mtr = example.metrics.mtr()
        assert mtr.num_matches == 0

    def test_num_keywords(self, dataset_keyword_match):
        """Test num_keywords counts medical terms correctly."""
        example = dataset_keyword_match[0]
        mtr = example.metrics.mtr()
        assert mtr.num_keywords == 1

    def test_mtr_is_complement_of_kwer(self, dataset_keyword_error):
        """Test that MTR = 1 - KWER for the same example."""
        example = dataset_keyword_error[0]
        mtr = example.metrics.mtr()
        kwer = example.metrics.kwer(vocab="medical_terms")
        assert mtr.value == pytest.approx(1 - kwer.value)

    def test_partial_matches(self):
        """Test MTR with some keywords matching and some not."""
        dataset = Dataset()
        dataset.add(
            ref="patient has diabetes and hypertension",
            hyp="patient has diabetes and hypotension",
            keywords={"medical_terms": ["diabetes", "hypertension"]},
        )
        example = dataset[0]
        mtr = example.metrics.mtr()
        assert mtr.num_keywords == 2
        assert mtr.num_matches == 1
        assert mtr.value == 0.5


class TestMTRDatasetMetric:
    """Tests for MTR (dataset-level Metric) class."""

    @pytest.fixture
    def medical_dataset(self):
        """Create a dataset with medical terms across multiple examples."""
        dataset = Dataset()
        dataset.add(
            ref="patient has diabetes",
            hyp="patient has diabetes",
            keywords={"medical_terms": ["diabetes"]},
        )
        dataset.add(
            ref="patient has asthma",
            hyp="patient has astma",
            keywords={"medical_terms": ["asthma"]},
        )
        return dataset

    def test_num_keywords_aggregates(self, medical_dataset):
        """Test that dataset num_keywords sums example-level counts."""
        mtr = medical_dataset.metrics.mtr()
        expected = sum(ex.metrics.mtr().num_keywords for ex in medical_dataset)
        assert mtr.num_keywords == expected

    def test_num_matches_aggregates(self, medical_dataset):
        """Test that dataset num_matches sums example-level counts."""
        mtr = medical_dataset.metrics.mtr()
        expected = sum(ex.metrics.mtr().num_matches for ex in medical_dataset)
        assert mtr.num_matches == expected

    def test_value_calculation(self, medical_dataset):
        """Test dataset-level MTR value calculation."""
        mtr = medical_dataset.metrics.mtr()
        # 1 match out of 2 keywords = 0.5
        assert mtr.num_keywords == 2
        assert mtr.num_matches == 1
        assert mtr.value == 0.5

    def test_mtr_is_complement_of_kwer(self, medical_dataset):
        """Test that dataset-level MTR = 1 - KWER(vocab='medical_terms')."""
        mtr = medical_dataset.metrics.mtr()
        kwer = medical_dataset.metrics.kwer(vocab="medical_terms")
        assert mtr.value == pytest.approx(1 - kwer.value)

    def test_empty_dataset(self):
        """Test MTR on empty dataset raises when medical_terms vocab is not registered."""
        dataset = Dataset()
        with pytest.raises(ValueError, match="not found in dataset keyword vocabularies"):
            dataset.metrics.mtr().value

    def test_all_correct(self):
        """Test MTR is 1.0 when all medical terms are correct."""
        dataset = Dataset()
        dataset.add(ref="diabetes mellitus", hyp="diabetes mellitus", keywords={"medical_terms": ["diabetes"]})
        dataset.add(ref="acute asthma", hyp="acute asthma", keywords={"medical_terms": ["asthma"]})
        mtr = dataset.metrics.mtr()
        assert mtr.value == 1.0


class TestMTRUsesKWERInternally:
    """Tests that MTR correctly delegates to KWER with vocab='medical_terms'."""

    def test_kwer_metric_is_cached(self):
        """Test that _kwer_metric is a cached_property and returns consistent results."""
        dataset = Dataset()
        dataset.add(
            ref="patient has diabetes",
            hyp="patient has diabetes",
            keywords={"medical_terms": ["diabetes"]},
        )
        mtr = dataset.metrics.mtr()
        kwer1 = mtr._kwer_metric
        kwer2 = mtr._kwer_metric
        assert kwer1 is kwer2

    def test_kwer_uses_medical_terms_vocab(self):
        """Test that internal KWER uses the 'medical_terms' vocabulary."""
        dataset = Dataset()
        dataset.add(
            ref="the quick brown fox",
            hyp="the quick brown dog",
            keywords={"medical_terms": ["fox"], "other_terms": ["quick"]},
        )
        mtr = dataset.metrics.mtr()
        # MTR should only care about medical_terms, not other_terms
        assert mtr.num_keywords == 1


class TestMTRMetricAttributes:
    """Tests for MTR metric attributes."""

    def test_short_name_base(self):
        assert MTR.short_name_base == "MTR"

    def test_long_name_base(self):
        assert MTR.long_name_base == "Medical Term Recall"

    def test_description(self):
        assert len(MTR.description) > 0

    def test_example_cls(self):
        assert MTR.example_cls == MTR_

    def test_metric_values_main(self):
        values = MTR.metric_values()
        assert values["main"] == "value"

    def test_metric_values_other(self):
        values = MTR.metric_values()
        assert "num_matches" in values["other"]
        assert "num_keywords" in values["other"]
