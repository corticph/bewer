"""Tests for parameterized metrics functionality."""

import pytest

from bewer import Dataset
from bewer.metrics.base import Metric


class TestParameterizedMetrics:
    """Test suite for parameterized metrics."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        dataset = Dataset()
        dataset.add(ref="hello world", hyp="hello world")
        dataset.add(ref="foo bar", hyp="foo baz")
        return dataset

    def test_basic_metric_has_empty_params(self, sample_dataset):
        """Test that a basic metric has empty params dict."""
        wer = sample_dataset.metrics.wer
        assert wer.params == {}

    def test_with_params_returns_new_instance(self, sample_dataset):
        """Test that with_params returns a new metric instance."""
        wer_base = sample_dataset.metrics.wer
        wer_params = wer_base.with_params(threshold=0.5)
        assert wer_base is not wer_params
        assert wer_params.params == {"threshold": 0.5}

    def test_with_params_caching(self, sample_dataset):
        """Test that with_params caches instances with same parameters."""
        wer1 = sample_dataset.metrics.wer.with_params(threshold=0.5)
        wer2 = sample_dataset.metrics.wer.with_params(threshold=0.5)
        assert wer1 is wer2

    def test_with_params_different_params_different_instances(self, sample_dataset):
        """Test that different parameters create different instances."""
        wer1 = sample_dataset.metrics.wer.with_params(threshold=0.5)
        wer2 = sample_dataset.metrics.wer.with_params(threshold=0.6)
        assert wer1 is not wer2

    def test_with_params_non_hashable_params(self, sample_dataset):
        """Test that non-hashable parameters raise TypeError with helpful message."""
        with pytest.raises(TypeError) as exc_info:
            sample_dataset.metrics.wer.with_params(config={"a": [1, 2, 3], "b": {"x": 1}})
        error_msg = str(exc_info.value)
        assert "must be hashable" in error_msg or "unhashable type" in error_msg
        assert "config" in error_msg

    def test_with_params_merges_params(self, sample_dataset):
        """Test that with_params merges parameters."""
        wer1 = sample_dataset.metrics.wer.with_params(threshold=0.5)
        wer2 = wer1.with_params(ignore_insertions=True)
        assert wer2.params == {"threshold": 0.5, "ignore_insertions": True}

    def test_with_params_overrides_params(self, sample_dataset):
        """Test that with_params overrides existing parameters."""
        wer1 = sample_dataset.metrics.wer.with_params(threshold=0.5)
        wer2 = wer1.with_params(threshold=0.6)
        assert wer2.params == {"threshold": 0.6}


class TestDynamicNaming:
    """Test suite for dynamic metric naming with parameters."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        dataset = Dataset()
        dataset.add(ref="hello world", hyp="hello world")
        return dataset

    def test_base_metric_short_name_no_params(self, sample_dataset):
        """Test that base metric without params shows base name."""
        wer = sample_dataset.metrics.wer
        assert wer.short_name == "WER"

    def test_base_metric_long_name_no_params(self, sample_dataset):
        """Test that base metric without params shows base long name."""
        wer = sample_dataset.metrics.wer
        assert wer.long_name == "Word Error Rate"

    def test_parameterized_metric_short_name_includes_params(self, sample_dataset):
        """Test that parameterized metric includes params in short name."""
        wer = sample_dataset.metrics.wer.with_params(threshold=0.5)
        assert "threshold=0.5" in wer.short_name
        assert wer.short_name.startswith("WER (")

    def test_parameterized_metric_long_name_includes_params(self, sample_dataset):
        """Test that parameterized metric includes params in long name."""
        wer = sample_dataset.metrics.wer.with_params(threshold=0.5)
        assert "threshold=0.5" in wer.long_name
        assert wer.long_name.startswith("Word Error Rate (")

    def test_multiple_params_in_name(self, sample_dataset):
        """Test that multiple parameters are shown in name."""
        wer = sample_dataset.metrics.wer.with_params(threshold=0.5, ignore_insertions=True)
        assert "threshold=0.5" in wer.short_name
        assert "ignore_insertions=True" in wer.short_name


class TestKeywordAggregatorRefactoring:
    """Test suite for KeywordAggregator refactoring."""

    @pytest.fixture
    def keyword_dataset(self):
        """Create a dataset with keywords for testing."""
        dataset = Dataset()
        dataset.add(
            ref="the patient has diabetes",
            hyp="the patient has diabetis",
            keywords={"medical_terms": ["diabetes"]},
        )
        return dataset

    def test_default_cer_threshold(self, keyword_dataset):
        """Test that default cer_threshold is set correctly."""
        kwa = keyword_dataset.metrics._legacy_kwa
        assert kwa.params["cer_threshold"] == 0.2

    def test_custom_cer_threshold(self, keyword_dataset):
        """Test that custom cer_threshold can be set."""
        kwa = keyword_dataset.metrics._legacy_kwa.with_params(cer_threshold=0.1)
        assert kwa.params["cer_threshold"] == 0.1

    def test_kwa_short_name_includes_threshold(self, keyword_dataset):
        """Test that KWA short name includes threshold parameter."""
        kwa = keyword_dataset.metrics._legacy_kwa
        assert "cer_threshold=0.2" in kwa.short_name

    def test_kwa_computes_with_threshold(self, keyword_dataset):
        """Test that KWA correctly uses threshold in computation."""
        # This is a basic integration test to ensure it doesn't crash
        kwa = keyword_dataset.metrics._legacy_kwa
        _ = kwa.match_count
        _ = kwa.relaxed_match_count


class TestExampleMetricParamsAccess:
    """Test suite for ExampleMetric params property."""

    @pytest.fixture
    def keyword_dataset(self):
        """Create a dataset with keywords for testing."""
        dataset = Dataset()
        dataset.add(
            ref="the patient has diabetes",
            hyp="the patient has diabetis",
            keywords={"medical_terms": ["diabetes"]},
        )
        return dataset

    def test_example_metric_has_params_property(self, keyword_dataset):
        """Test that ExampleMetric has params property."""
        example = keyword_dataset.examples[0]
        kwa_example = example.metrics._legacy_kwa
        assert hasattr(kwa_example, "params")

    def test_example_metric_params_matches_parent(self, keyword_dataset):
        """Test that ExampleMetric params matches parent metric params."""
        kwa = keyword_dataset.metrics._legacy_kwa
        example = keyword_dataset.examples[0]
        kwa_example = example.metrics._legacy_kwa
        assert kwa_example.params == kwa.params

    def test_example_metric_params_with_custom_threshold(self, keyword_dataset):
        """Test that ExampleMetric gets custom params from parent."""
        kwa = keyword_dataset.metrics._legacy_kwa.with_params(cer_threshold=0.1)
        # Force creation of example metric
        _ = kwa.match_count
        example = keyword_dataset.examples[0]
        # Access the example metric for the parameterized parent
        # This requires getting it through the parameterized parent's cache
        kwa_example = kwa._get_example_metric(example)
        assert kwa_example.params["cer_threshold"] == 0.1


class TestCacheKeyGeneration:
    """Test suite for cache key generation."""

    def test_make_cache_key_empty_params(self):
        """Test cache key generation with empty params."""
        key = Metric._make_cache_key()
        assert key == ()

    def test_make_cache_key_single_param(self):
        """Test cache key generation with single parameter."""
        key = Metric._make_cache_key(threshold=0.5)
        assert key == (("threshold", 0.5),)

    def test_make_cache_key_multiple_params(self):
        """Test cache key generation with multiple parameters."""
        key = Metric._make_cache_key(threshold=0.5, ignore_insertions=True)
        # Sorted by key name
        assert key == (("ignore_insertions", True), ("threshold", 0.5))

    def test_make_cache_key_non_hashable_params(self):
        """Test that cache key with non-hashable parameters cannot be used as dict key."""
        # _make_cache_key succeeds, but the result cannot be hashed
        key = Metric._make_cache_key(config={"a": [1, 2, 3]})
        assert isinstance(key, tuple)

        # But trying to use it as a dict key raises TypeError
        with pytest.raises(TypeError):
            test_dict = {}
            test_dict[key] = "value"

    def test_make_cache_key_consistent(self):
        """Test that cache key is consistent for same params."""
        key1 = Metric._make_cache_key(threshold=0.5, value=1)
        key2 = Metric._make_cache_key(threshold=0.5, value=1)
        assert key1 == key2

    def test_make_cache_key_order_independent(self):
        """Test that cache key is order-independent for params."""
        key1 = Metric._make_cache_key(a=1, b=2)
        key2 = Metric._make_cache_key(b=2, a=1)
        assert key1 == key2
