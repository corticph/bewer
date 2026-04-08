"""Tests for bewer.metrics.kwer module."""

import pytest

from bewer import Dataset
from bewer.core.keyword import KeywordNotFoundWarning
from bewer.metrics.kter import KTER, KTER_


class TestKTERExampleMetric:
    """Tests for KTER_ (ExampleMetric) class."""

    @pytest.fixture
    def dataset_single_keyword_match(self):
        """Dataset where a single-token key term is correctly transcribed."""
        dataset = Dataset()
        dataset.add(
            ref="the quick brown fox",
            hyp="the quick brown fox",
            keywords={"animals": ["fox"]},
        )
        return dataset

    @pytest.fixture
    def dataset_single_keyword_error(self):
        """Dataset where a single-token key term is incorrectly transcribed."""
        dataset = Dataset()
        dataset.add(
            ref="the quick brown fox",
            hyp="the quick brown dog",
            keywords={"animals": ["fox"]},
        )
        return dataset

    @pytest.fixture
    def dataset_multi_token_keyword_match(self):
        """Dataset where a multi-token key term is correctly transcribed."""
        dataset = Dataset()
        dataset.add(
            ref="the quick brown fox",
            hyp="the quick brown fox",
            keywords={"phrases": ["quick brown"]},
        )
        return dataset

    @pytest.fixture
    def dataset_multi_token_keyword_error(self):
        """Dataset where a multi-token key term is partially incorrectly transcribed."""
        dataset = Dataset()
        dataset.add(
            ref="the quick brown fox",
            hyp="the slow brown fox",
            keywords={"phrases": ["quick brown"]},
        )
        return dataset

    def test_num_errors_perfect_match(self, dataset_single_keyword_match):
        """Test num_errors is 0 when key term is correctly transcribed."""
        example = dataset_single_keyword_match[0]
        kter = example.metrics.kter(vocab="animals")
        assert kter.num_errors == 0

    def test_num_errors_single_keyword_error(self, dataset_single_keyword_error):
        """Test num_errors is 1 when key term is incorrectly transcribed."""
        example = dataset_single_keyword_error[0]
        kter = example.metrics.kter(vocab="animals")
        assert kter.num_errors == 1

    def test_num_keywords_single(self, dataset_single_keyword_match):
        """Test num_keywords counts key term occurrences correctly."""
        example = dataset_single_keyword_match[0]
        kter = example.metrics.kter(vocab="animals")
        assert kter.num_keywords == 1

    def test_value_perfect_match(self, dataset_single_keyword_match):
        """Test KTER value is 0.0 for perfect key term match."""
        example = dataset_single_keyword_match[0]
        kter = example.metrics.kter(vocab="animals")
        assert kter.value == 0.0

    def test_value_all_errors(self, dataset_single_keyword_error):
        """Test KTER value is 1.0 when all key terms have errors."""
        example = dataset_single_keyword_error[0]
        kter = example.metrics.kter(vocab="animals")
        assert kter.value == 1.0

    def test_multi_token_keyword_match(self, dataset_multi_token_keyword_match):
        """Test that multi-token key terms are correctly identified as matching."""
        example = dataset_multi_token_keyword_match[0]
        kter = example.metrics.kter(vocab="phrases")
        assert kter.num_errors == 0
        assert kter.num_keywords == 1
        assert kter.value == 0.0

    def test_multi_token_keyword_error(self, dataset_multi_token_keyword_error):
        """Test that multi-token key terms with partial errors are counted as errors."""
        example = dataset_multi_token_keyword_error[0]
        kter = example.metrics.kter(vocab="phrases")
        assert kter.num_errors == 1
        assert kter.num_keywords == 1
        assert kter.value == 1.0

    def test_multiple_keyword_occurrences(self):
        """Test counting multiple occurrences of the same key term."""
        dataset = Dataset()
        dataset.add(
            ref="the fox met another fox",
            hyp="the fox met another dog",
            keywords={"animals": ["fox"]},
        )
        example = dataset[0]
        kter = example.metrics.kter(vocab="animals")
        assert kter.num_keywords == 2
        assert kter.num_errors == 1  # Only the second "fox" is wrong

    def test_multiple_different_keywords(self):
        """Test with multiple different key terms in the same vocabulary."""
        dataset = Dataset()
        dataset.add(
            ref="the quick brown fox",
            hyp="the slow brown dog",
            keywords={"terms": ["quick", "fox"]},
        )
        example = dataset[0]
        kter = example.metrics.kter(vocab="terms")
        assert kter.num_keywords == 2
        assert kter.num_errors == 2
        assert kter.value == 1.0

    def test_keyword_not_in_ref(self):
        """Test that key terms not found in reference are not counted."""
        dataset = Dataset()
        dataset.add(
            ref="hello world",
            hyp="hello world",
            keywords={"terms": ["missing"]},
        )
        example = dataset[0]
        kter = example.metrics.kter(vocab="terms")
        with pytest.warns(KeywordNotFoundWarning):
            assert kter.num_keywords == 0
        assert kter.num_errors == 0

    def test_no_keywords_returns_zero(self):
        """Test value is 0 when there are no key terms to evaluate."""
        dataset = Dataset()
        dataset.add(
            ref="hello world",
            hyp="hello world",
            keywords={"terms": ["missing"]},
        )
        example = dataset[0]
        kter = example.metrics.kter(vocab="terms")
        with pytest.warns(KeywordNotFoundWarning):
            assert kter.value == 0.0


class TestKTERNormalization:
    """Tests for KTER with normalized=False."""

    def test_unnormalized_casing_mismatch(self):
        """Test that differing casing causes an error when normalized=False."""
        dataset = Dataset()
        dataset.add(
            ref="the Fox jumps",
            hyp="the fox jumps",
            keywords={"animals": ["Fox"]},
        )
        kter = dataset[0].metrics.kter(vocab="animals", normalized=False)
        assert kter.num_errors == 1

    def test_unnormalized_casing_match(self):
        """Test that matching casing is fine when normalized=False."""
        dataset = Dataset()
        dataset.add(
            ref="the FOX jumps",
            hyp="the FOX jumps",
            keywords={"animals": ["FOX"]},
        )
        kter = dataset[0].metrics.kter(vocab="animals", normalized=False)
        assert kter.num_errors == 0


class TestKTERDatasetMetric:
    """Tests for KTER (dataset-level Metric) class."""

    @pytest.fixture
    def keyword_dataset(self):
        """Create a dataset with key terms across multiple examples."""
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
        kter = keyword_dataset.metrics.kter(vocab="animals")
        expected = sum(ex.metrics.kter(vocab="animals").num_errors for ex in keyword_dataset)
        assert kter.num_errors == expected

    def test_num_keywords_aggregates(self, keyword_dataset):
        """Test that dataset num_keywords sums example-level keyword counts."""
        kter = keyword_dataset.metrics.kter(vocab="animals")
        expected = sum(ex.metrics.kter(vocab="animals").num_keywords for ex in keyword_dataset)
        assert kter.num_keywords == expected

    def test_value_calculation(self, keyword_dataset):
        """Test dataset-level KTER value calculation."""
        kter = keyword_dataset.metrics.kter(vocab="animals")
        # 1 error out of 2 key terms = 0.5
        assert kter.num_keywords == 2
        assert kter.num_errors == 1
        assert kter.value == 0.5

    def test_value_all_correct(self):
        """Test KTER is 0.0 when all key terms match."""
        dataset = Dataset()
        dataset.add(ref="hello world", hyp="hello world", keywords={"terms": ["hello"]})
        dataset.add(ref="foo bar", hyp="foo bar", keywords={"terms": ["foo"]})
        kter = dataset.metrics.kter(vocab="terms")
        assert kter.value == 0.0

    def test_empty_dataset(self):
        """Test KTER on empty dataset raises when vocab is not registered."""
        dataset = Dataset()
        with pytest.raises(ValueError, match="not found in dataset keyword vocabularies"):
            dataset.metrics.kter(vocab="terms").value


class TestKTERParameterValidation:
    """Tests for KTER parameter validation."""

    def test_missing_vocab_param_raises(self):
        """Test that omitting the required vocab parameter raises ValueError."""
        dataset = Dataset()
        dataset.add(ref="hello world", hyp="hello world", keywords={"terms": ["hello"]})
        with pytest.raises(ValueError, match="Missing required parameters"):
            dataset.metrics.kter()

    def test_invalid_vocab_raises(self):
        """Test that using a non-existent vocabulary raises ValueError."""
        dataset = Dataset()
        dataset.add(ref="hello world", hyp="hello world", keywords={"terms": ["hello"]})
        with pytest.raises(ValueError, match="not found in dataset keyword vocabularies"):
            dataset.metrics.kter(vocab="nonexistent").value

    def test_vocab_param_type_validation(self):
        """Test that vocab parameter must be a string."""
        dataset = Dataset()
        dataset.add(ref="hello world", hyp="hello world", keywords={"terms": ["hello"]})
        with pytest.raises(TypeError, match="must be str"):
            dataset.metrics.kter(vocab=123)


class TestKTERMetricAttributes:
    """Tests for KTER metric attributes."""

    def test_short_name_base(self):
        assert KTER.short_name_base == "KTER"

    def test_long_name_base(self):
        assert KTER.long_name_base == "Key Term Error Rate"

    def test_description(self):
        assert len(KTER.description) > 0

    def test_example_cls(self):
        assert KTER.example_cls == KTER_

    def test_metric_values_main(self):
        values = KTER.metric_values()
        assert values["main"] == "value"

    def test_metric_values_other(self):
        values = KTER.metric_values()
        assert "num_errors" in values["other"]
        assert "num_keywords" in values["other"]

    def test_short_name_includes_params(self):
        """Test that short_name includes the vocab parameter."""
        dataset = Dataset()
        dataset.add(ref="hello", hyp="hello", keywords={"terms": ["hello"]})
        kter = dataset.metrics.kter(vocab="terms")
        assert "vocab=terms" in kter.short_name

    def test_factory_caching(self):
        """Test that same parameters return cached instance."""
        dataset = Dataset()
        dataset.add(ref="hello", hyp="hello", keywords={"terms": ["hello"]})
        kter1 = dataset.metrics.kter(vocab="terms")
        kter2 = dataset.metrics.kter(vocab="terms")
        assert kter1 is kter2


class TestKTERSharesKTStats:
    """Tests that KTER and KTR share the same _KTStats instance."""

    def test_kter_and_ktr_share_kt_stats(self):
        """Test that KTER and KTR with identical params use the same cached _KTStats instance."""
        dataset = Dataset()
        dataset.add(ref="the fox jumps", hyp="the fox jumps", keywords={"animals": ["fox"]})
        kter = dataset.metrics.kter(vocab="animals")
        ktr = dataset.metrics.ktr(vocab="animals")
        assert kter._kt_stats is ktr._kt_stats
