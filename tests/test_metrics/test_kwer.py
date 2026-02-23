"""Tests for bewer.metrics.kwer module."""

import pytest

from bewer import Dataset
from bewer.core.example import KeywordNotFoundWarning
from bewer.metrics.kwer import KWER, KWER_


class TestKWERExampleMetric:
    """Tests for KWER_ (ExampleMetric) class."""

    @pytest.fixture
    def dataset_single_keyword_match(self):
        """Dataset where a single-token keyword is correctly transcribed."""
        dataset = Dataset()
        dataset.add(
            ref="the quick brown fox",
            hyp="the quick brown fox",
            keywords={"animals": ["fox"]},
        )
        return dataset

    @pytest.fixture
    def dataset_single_keyword_error(self):
        """Dataset where a single-token keyword is incorrectly transcribed."""
        dataset = Dataset()
        dataset.add(
            ref="the quick brown fox",
            hyp="the quick brown dog",
            keywords={"animals": ["fox"]},
        )
        return dataset

    @pytest.fixture
    def dataset_multi_token_keyword_match(self):
        """Dataset where a multi-token keyword is correctly transcribed."""
        dataset = Dataset()
        dataset.add(
            ref="the quick brown fox",
            hyp="the quick brown fox",
            keywords={"phrases": ["quick brown"]},
        )
        return dataset

    @pytest.fixture
    def dataset_multi_token_keyword_error(self):
        """Dataset where a multi-token keyword is partially incorrectly transcribed."""
        dataset = Dataset()
        dataset.add(
            ref="the quick brown fox",
            hyp="the slow brown fox",
            keywords={"phrases": ["quick brown"]},
        )
        return dataset

    def test_num_errors_perfect_match(self, dataset_single_keyword_match):
        """Test num_errors is 0 when keyword is correctly transcribed."""
        example = dataset_single_keyword_match[0]
        kwer = example.metrics.kwer(vocab="animals")
        assert kwer.num_errors == 0

    def test_num_errors_single_keyword_error(self, dataset_single_keyword_error):
        """Test num_errors is 1 when keyword is incorrectly transcribed."""
        example = dataset_single_keyword_error[0]
        kwer = example.metrics.kwer(vocab="animals")
        assert kwer.num_errors == 1

    def test_num_keywords_single(self, dataset_single_keyword_match):
        """Test num_keywords counts keyword occurrences correctly."""
        example = dataset_single_keyword_match[0]
        kwer = example.metrics.kwer(vocab="animals")
        assert kwer.num_keywords == 1

    def test_value_perfect_match(self, dataset_single_keyword_match):
        """Test KWER value is 0.0 for perfect keyword match."""
        example = dataset_single_keyword_match[0]
        kwer = example.metrics.kwer(vocab="animals")
        assert kwer.value == 0.0

    def test_value_all_errors(self, dataset_single_keyword_error):
        """Test KWER value is 1.0 when all keywords have errors."""
        example = dataset_single_keyword_error[0]
        kwer = example.metrics.kwer(vocab="animals")
        assert kwer.value == 1.0

    def test_multi_token_keyword_match(self, dataset_multi_token_keyword_match):
        """Test that multi-token keywords are correctly identified as matching."""
        example = dataset_multi_token_keyword_match[0]
        kwer = example.metrics.kwer(vocab="phrases")
        assert kwer.num_errors == 0
        assert kwer.num_keywords == 1
        assert kwer.value == 0.0

    def test_multi_token_keyword_error(self, dataset_multi_token_keyword_error):
        """Test that multi-token keywords with partial errors are counted as errors."""
        example = dataset_multi_token_keyword_error[0]
        kwer = example.metrics.kwer(vocab="phrases")
        assert kwer.num_errors == 1
        assert kwer.num_keywords == 1
        assert kwer.value == 1.0

    def test_multiple_keyword_occurrences(self):
        """Test counting multiple occurrences of the same keyword."""
        dataset = Dataset()
        dataset.add(
            ref="the fox met another fox",
            hyp="the fox met another dog",
            keywords={"animals": ["fox"]},
        )
        example = dataset[0]
        kwer = example.metrics.kwer(vocab="animals")
        assert kwer.num_keywords == 2
        assert kwer.num_errors == 1  # Only the second "fox" is wrong

    def test_multiple_different_keywords(self):
        """Test with multiple different keywords in the same vocabulary."""
        dataset = Dataset()
        dataset.add(
            ref="the quick brown fox",
            hyp="the slow brown dog",
            keywords={"terms": ["quick", "fox"]},
        )
        example = dataset[0]
        kwer = example.metrics.kwer(vocab="terms")
        assert kwer.num_keywords == 2
        assert kwer.num_errors == 2
        assert kwer.value == 1.0

    def test_keyword_not_in_ref(self):
        """Test that keywords not found in reference are not counted."""
        dataset = Dataset()
        with pytest.warns(KeywordNotFoundWarning):
            dataset.add(
                ref="hello world",
                hyp="hello world",
                keywords={"terms": ["missing"]},
            )
        example = dataset[0]
        kwer = example.metrics.kwer(vocab="terms")
        assert kwer.num_keywords == 0
        assert kwer.num_errors == 0

    def test_no_keywords_returns_zero(self):
        """Test value is 0 when there are no keywords to evaluate."""
        dataset = Dataset()
        with pytest.warns(KeywordNotFoundWarning):
            dataset.add(
                ref="hello world",
                hyp="hello world",
                keywords={"terms": ["missing"]},
            )
        example = dataset[0]
        kwer = example.metrics.kwer(vocab="terms")
        assert kwer.value == 0.0


class TestKWERDatasetMetric:
    """Tests for KWER (dataset-level Metric) class."""

    @pytest.fixture
    def keyword_dataset(self):
        """Create a dataset with keywords across multiple examples."""
        dataset = Dataset()
        dataset.add(
            ref="the quick brown fox",
            hyp="the quick brown fox",
            keywords={"animals": ["fox"]},
        )
        dataset.add(
            ref="the lazy brown dog",
            hyp="the lazy brown cat",
            keywords={"animals": ["dog"]},
        )
        return dataset

    def test_num_errors_aggregates(self, keyword_dataset):
        """Test that dataset num_errors sums example-level errors."""
        kwer = keyword_dataset.metrics.kwer(vocab="animals")
        expected = sum(ex.metrics.kwer(vocab="animals").num_errors for ex in keyword_dataset)
        assert kwer.num_errors == expected

    def test_num_keywords_aggregates(self, keyword_dataset):
        """Test that dataset num_keywords sums example-level keyword counts."""
        kwer = keyword_dataset.metrics.kwer(vocab="animals")
        expected = sum(ex.metrics.kwer(vocab="animals").num_keywords for ex in keyword_dataset)
        assert kwer.num_keywords == expected

    def test_value_calculation(self, keyword_dataset):
        """Test dataset-level KWER value calculation."""
        kwer = keyword_dataset.metrics.kwer(vocab="animals")
        # 1 error out of 2 keywords = 0.5
        assert kwer.num_keywords == 2
        assert kwer.num_errors == 1
        assert kwer.value == 0.5

    def test_value_all_correct(self):
        """Test KWER is 0.0 when all keywords match."""
        dataset = Dataset()
        dataset.add(ref="hello world", hyp="hello world", keywords={"terms": ["hello"]})
        dataset.add(ref="foo bar", hyp="foo bar", keywords={"terms": ["foo"]})
        kwer = dataset.metrics.kwer(vocab="terms")
        assert kwer.value == 0.0

    def test_empty_dataset(self):
        """Test KWER on empty dataset raises when vocab is not registered."""
        dataset = Dataset()
        with pytest.raises(ValueError, match="not found in dataset keyword vocabularies"):
            dataset.metrics.kwer(vocab="terms").value


class TestKWERParameterValidation:
    """Tests for KWER parameter validation."""

    def test_missing_vocab_param_raises(self):
        """Test that omitting the required vocab parameter raises ValueError."""
        dataset = Dataset()
        dataset.add(ref="hello world", hyp="hello world", keywords={"terms": ["hello"]})
        with pytest.raises(ValueError, match="Missing required parameters"):
            dataset.metrics.kwer()

    def test_invalid_vocab_raises(self):
        """Test that using a non-existent vocabulary raises ValueError."""
        dataset = Dataset()
        dataset.add(ref="hello world", hyp="hello world", keywords={"terms": ["hello"]})
        with pytest.raises(ValueError, match="not found in dataset keyword vocabularies"):
            dataset.metrics.kwer(vocab="nonexistent").value

    def test_vocab_param_type_validation(self):
        """Test that vocab parameter must be a string."""
        dataset = Dataset()
        dataset.add(ref="hello world", hyp="hello world", keywords={"terms": ["hello"]})
        with pytest.raises(TypeError, match="must be str"):
            dataset.metrics.kwer(vocab=123)


class TestKWERMetricAttributes:
    """Tests for KWER metric attributes."""

    def test_short_name_base(self):
        assert KWER.short_name_base == "KWER"

    def test_long_name_base(self):
        assert KWER.long_name_base == "Keyword Error Rate"

    def test_description(self):
        assert len(KWER.description) > 0

    def test_example_cls(self):
        assert KWER.example_cls == KWER_

    def test_metric_values_main(self):
        values = KWER.metric_values()
        assert values["main"] == "value"

    def test_metric_values_other(self):
        values = KWER.metric_values()
        assert "num_errors" in values["other"]
        assert "num_keywords" in values["other"]

    def test_short_name_includes_params(self):
        """Test that short_name includes the vocab parameter."""
        dataset = Dataset()
        dataset.add(ref="hello", hyp="hello", keywords={"terms": ["hello"]})
        kwer = dataset.metrics.kwer(vocab="terms")
        assert "vocab=terms" in kwer.short_name

    def test_factory_caching(self):
        """Test that same parameters return cached instance."""
        dataset = Dataset()
        dataset.add(ref="hello", hyp="hello", keywords={"terms": ["hello"]})
        kwer1 = dataset.metrics.kwer(vocab="terms")
        kwer2 = dataset.metrics.kwer(vocab="terms")
        assert kwer1 is kwer2
