"""Tests for the key_term_recall metric."""

import pytest

from bewer import Dataset, Vocabulary, get_config
from bewer.metrics.key_term.key_term_recall import KTR, KTR_


class TestKTRExampleMetric:
    """Tests for KTR_ (ExampleMetric) class."""

    @pytest.fixture
    def dataset_keyword_match(self):
        """Dataset where a key term is correctly transcribed."""
        config = get_config("base").clone().add_vocabulary(Vocabulary(name="key_terms", terms=["diabetes"]))
        dataset = Dataset(config=config)
        dataset.add(
            ref="the patient has diabetes",
            hyp="the patient has diabetes",
        )
        return dataset

    @pytest.fixture
    def dataset_keyword_error(self):
        """Dataset where a key term is incorrectly transcribed."""
        config = get_config("base").clone().add_vocabulary(Vocabulary(name="key_terms", terms=["diabetes"]))
        dataset = Dataset(config=config)
        dataset.add(
            ref="the patient has diabetes",
            hyp="the patient has diabetis",
        )
        return dataset

    def test_value_perfect_match(self, dataset_keyword_match):
        """Test KTR value is 1.0 when key term is correctly transcribed."""
        example = dataset_keyword_match[0]
        key_term_recall = example.metrics.key_term_recall(vocab="key_terms")
        assert key_term_recall.value == 1.0

    def test_value_with_error(self, dataset_keyword_error):
        """Test KTR value is 0.0 when key term is incorrectly transcribed."""
        example = dataset_keyword_error[0]
        key_term_recall = example.metrics.key_term_recall(vocab="key_terms")
        assert key_term_recall.value == 0.0

    def test_num_matches_perfect(self, dataset_keyword_match):
        """Test num_matches is 1 when key term matches."""
        example = dataset_keyword_match[0]
        key_term_recall = example.metrics.key_term_recall(vocab="key_terms")
        assert key_term_recall.num_matches == 1

    def test_num_matches_with_error(self, dataset_keyword_error):
        """Test num_matches is 0 when key term has error."""
        example = dataset_keyword_error[0]
        key_term_recall = example.metrics.key_term_recall(vocab="key_terms")
        assert key_term_recall.num_matches == 0

    def test_num_ref_terms(self, dataset_keyword_match):
        """Test num_ref_terms counts correctly."""
        example = dataset_keyword_match[0]
        key_term_recall = example.metrics.key_term_recall(vocab="key_terms")
        assert key_term_recall.num_ref_terms == 1

    def test_ktr_is_complement_of_kter(self, dataset_keyword_error):
        """Test that KTR = 1 - KTER for the same example."""
        example = dataset_keyword_error[0]
        key_term_recall = example.metrics.key_term_recall(vocab="key_terms")
        key_term_error_rate = example.metrics.key_term_error_rate(vocab="key_terms")
        assert key_term_recall.value == pytest.approx(1 - key_term_error_rate.value)

    def test_partial_matches(self):
        """Test KTR with some key terms matching and some not."""
        config = (
            get_config("base").clone().add_vocabulary(Vocabulary(name="key_terms", terms=["diabetes", "hypertension"]))
        )
        dataset = Dataset(config=config)
        dataset.add(
            ref="patient has diabetes and hypertension",
            hyp="patient has diabetes and hypotension",
        )
        example = dataset[0]
        key_term_recall = example.metrics.key_term_recall(vocab="key_terms")
        assert key_term_recall.num_ref_terms == 2
        assert key_term_recall.num_matches == 1
        assert key_term_recall.value == 0.5

    def test_works_with_any_vocab(self):
        """Test KTR is domain-agnostic and works with any vocabulary name."""
        config = get_config("base").clone().add_vocabulary(Vocabulary(name="animals", terms=["fox"]))
        dataset = Dataset(config=config)
        dataset.add(
            ref="the quick brown fox",
            hyp="the quick brown fox",
        )
        example = dataset[0]
        key_term_recall = example.metrics.key_term_recall(vocab="animals")
        assert key_term_recall.value == 1.0
        assert key_term_recall.num_matches == 1


class TestKTRDatasetMetric:
    """Tests for KTR (dataset-level Metric) class."""

    @pytest.fixture
    def key_terms_dataset(self):
        """Create a dataset with key terms across multiple examples."""
        config = get_config("base").clone().add_vocabulary(Vocabulary(name="key_terms", terms=["diabetes", "asthma"]))
        dataset = Dataset(config=config)
        dataset.add(
            ref="patient has diabetes",
            hyp="patient has diabetes",
        )
        dataset.add(
            ref="patient has asthma",
            hyp="patient has astma",
        )
        return dataset

    def test_num_ref_terms_aggregates(self, key_terms_dataset):
        """Test that dataset num_ref_terms sums example-level counts."""
        key_term_recall = key_terms_dataset.metrics.key_term_recall(vocab="key_terms")
        expected = sum(ex.metrics.key_term_recall(vocab="key_terms").num_ref_terms for ex in key_terms_dataset)
        assert key_term_recall.num_ref_terms == expected

    def test_num_matches_aggregates(self, key_terms_dataset):
        """Test that dataset num_matches sums example-level counts."""
        key_term_recall = key_terms_dataset.metrics.key_term_recall(vocab="key_terms")
        expected = sum(ex.metrics.key_term_recall(vocab="key_terms").num_matches for ex in key_terms_dataset)
        assert key_term_recall.num_matches == expected

    def test_value_calculation(self, key_terms_dataset):
        """Test dataset-level KTR value calculation."""
        key_term_recall = key_terms_dataset.metrics.key_term_recall(vocab="key_terms")
        # 1 match out of 2 key terms = 0.5
        assert key_term_recall.num_ref_terms == 2
        assert key_term_recall.num_matches == 1
        assert key_term_recall.value == 0.5

    def test_ktr_is_complement_of_kter(self, key_terms_dataset):
        """Test that dataset-level KTR = 1 - KTER for the same vocab."""
        key_term_recall = key_terms_dataset.metrics.key_term_recall(vocab="key_terms")
        key_term_error_rate = key_terms_dataset.metrics.key_term_error_rate(vocab="key_terms")
        assert key_term_recall.value == pytest.approx(1 - key_term_error_rate.value)

    def test_empty_dataset(self):
        """Test KTR on empty dataset raises when vocab is not registered."""
        dataset = Dataset()
        with pytest.raises(ValueError, match="not found in dataset config vocabularies"):
            dataset.metrics.key_term_recall(vocab="key_terms").value

    def test_all_correct(self):
        """Test KTR is 1.0 when all key terms are correct."""
        config = get_config("base").clone().add_vocabulary(Vocabulary(name="key_terms", terms=["diabetes", "asthma"]))
        dataset = Dataset(config=config)
        dataset.add(ref="diabetes mellitus", hyp="diabetes mellitus")
        dataset.add(ref="acute asthma", hyp="acute asthma")
        key_term_recall = dataset.metrics.key_term_recall(vocab="key_terms")
        assert key_term_recall.value == 1.0


class TestKTRMetricAttributes:
    """Tests for KTR metric attributes."""

    def test_short_name_base(self):
        assert KTR.short_name_base == "KTR"

    def test_long_name_base(self):
        assert KTR.long_name_base == "Key-Term Recall"

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
