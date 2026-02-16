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
        dataset.add(
            ref="hello world",
            hyp="hello world",
            keywords={"medical_terms": ["hello"]},
        )
        dataset.add(
            ref="foo bar",
            hyp="foo baz",
            keywords={"medical_terms": ["foo"]},
        )
        return dataset

    def test_basic_metric_has_empty_params(self, sample_dataset):
        """Test that a basic metric has empty params dict."""
        wer = sample_dataset.metrics.wer
        assert wer.params == {}

    def test_with_params_returns_new_instance(self, sample_dataset):
        """Test that with_params returns a new metric instance."""
        kwa_base = sample_dataset.metrics._legacy_kwa
        kwa_params = kwa_base.with_params(cer_threshold=0.5)
        assert kwa_base is not kwa_params
        assert kwa_params.params == {"cer_threshold": 0.5}

    def test_with_params_caching(self, sample_dataset):
        """Test that with_params caches instances with same parameters."""
        kwa1 = sample_dataset.metrics._legacy_kwa.with_params(cer_threshold=0.5)
        kwa2 = sample_dataset.metrics._legacy_kwa.with_params(cer_threshold=0.5)
        assert kwa1 is kwa2

    def test_with_params_different_params_different_instances(self, sample_dataset):
        """Test that different parameters create different instances."""
        kwa1 = sample_dataset.metrics._legacy_kwa.with_params(cer_threshold=0.5)
        kwa2 = sample_dataset.metrics._legacy_kwa.with_params(cer_threshold=0.6)
        assert kwa1 is not kwa2

    def test_with_params_non_hashable_params(self, sample_dataset):
        """Test that non-hashable parameters raise TypeError with helpful message."""
        with pytest.raises(TypeError) as exc_info:
            sample_dataset.metrics._legacy_kwa.with_params(config={"a": [1, 2, 3], "b": {"x": 1}})
        error_msg = str(exc_info.value)
        assert "must be hashable" in error_msg or "unhashable type" in error_msg
        assert "config" in error_msg

    def test_with_params_merges_params(self, sample_dataset):
        """Test that with_params merges parameters."""
        # KeywordAggregator has default cer_threshold=0.2
        kwa1 = sample_dataset.metrics._legacy_kwa
        kwa2 = kwa1.with_params(cer_threshold=0.5)
        # Default is merged with new param
        assert kwa2.params == {"cer_threshold": 0.5}

    def test_with_params_overrides_params(self, sample_dataset):
        """Test that with_params overrides existing parameters."""
        kwa1 = sample_dataset.metrics._legacy_kwa.with_params(cer_threshold=0.5)
        kwa2 = kwa1.with_params(cer_threshold=0.6)
        assert kwa2.params == {"cer_threshold": 0.6}


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
        kwa = sample_dataset.metrics._legacy_kwa.with_params(cer_threshold=0.5)
        assert "cer_threshold=0.5" in kwa.short_name
        assert kwa.short_name.startswith("kwa (")

    def test_parameterized_metric_long_name_includes_params(self, sample_dataset):
        """Test that parameterized metric includes params in long name."""
        kwa = sample_dataset.metrics._legacy_kwa.with_params(cer_threshold=0.5)
        assert "cer_threshold=0.5" in kwa.long_name
        assert kwa.long_name.startswith("Keyword Aggregator (")

    def test_multiple_params_in_name(self, sample_dataset):
        """Test that multiple parameters are shown in name."""
        # Use HallucinationAggregator which has threshold param
        # Note: We're just testing the naming mechanism, the actual computation
        # might not work properly without proper data
        hlcn = sample_dataset.metrics._legacy_hlcn.with_params(threshold=5)
        assert "threshold=5" in hlcn.short_name


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


class TestDeclarativeHyperparams:
    """Test suite for declarative hyperparameters."""

    def test_metric_without_hyperparams_rejects_params(self):
        """Test that metrics without _hyperparams reject all parameters."""
        from bewer import Dataset

        dataset = Dataset()
        dataset.add(ref="hello world", hyp="hello world")

        # WER doesn't define _hyperparams, so it should reject params
        with pytest.raises(ValueError) as exc_info:
            dataset.metrics.wer.with_params(threshold=0.5)
        assert "does not accept parameters" in str(exc_info.value)

    def test_metric_with_optional_hyperparam_uses_default(self):
        """Test that optional hyperparams use defaults when not provided."""
        from bewer import Dataset

        dataset = Dataset()
        dataset.add(
            ref="the patient has diabetes",
            hyp="the patient has diabetis",
            keywords={"medical_terms": ["diabetes"]},
        )

        # KeywordAggregator has cer_threshold with default 0.2
        kwa = dataset.metrics._legacy_kwa
        assert kwa.params["cer_threshold"] == 0.2

    def test_metric_with_optional_hyperparam_can_override(self):
        """Test that optional hyperparams can be overridden."""
        from bewer import Dataset

        dataset = Dataset()
        dataset.add(
            ref="the patient has diabetes",
            hyp="the patient has diabetis",
            keywords={"medical_terms": ["diabetes"]},
        )

        # Override cer_threshold
        kwa = dataset.metrics._legacy_kwa.with_params(cer_threshold=0.5)
        assert kwa.params["cer_threshold"] == 0.5

    def test_metric_rejects_unknown_hyperparam(self):
        """Test that unknown hyperparams are rejected with helpful message."""
        from bewer import Dataset

        dataset = Dataset()
        dataset.add(
            ref="the patient has diabetes",
            hyp="the patient has diabetis",
            keywords={"medical_terms": ["diabetes"]},
        )

        # Try to set unknown param
        with pytest.raises(ValueError) as exc_info:
            dataset.metrics._legacy_kwa.with_params(unknown_param=123)
        assert "Unknown parameter 'unknown_param'" in str(exc_info.value)
        assert "Valid parameters:" in str(exc_info.value)
        assert "cer_threshold" in str(exc_info.value)

    def test_metric_validates_hyperparam_type(self):
        """Test that hyperparam types are validated."""
        from bewer import Dataset

        dataset = Dataset()
        dataset.add(
            ref="the patient has diabetes",
            hyp="the patient has diabetis",
            keywords={"medical_terms": ["diabetes"]},
        )

        # Try to set wrong type
        with pytest.raises(TypeError) as exc_info:
            dataset.metrics._legacy_kwa.with_params(cer_threshold="not a float")
        assert "must be float" in str(exc_info.value)


class TestRequiredHyperparams:
    """Test suite for required hyperparameters."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        dataset = Dataset()
        dataset.add(ref="hello world", hyp="hello world")
        dataset.add(ref="foo bar", hyp="foo baz")
        return dataset

    def test_required_hyperparam_raises_on_access(self, sample_dataset):
        """Test that missing required hyperparams raise error on value access."""
        # Create a test metric with required param
        from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_value

        class TestMetricExample(ExampleMetric):
            @metric_value(main=True)
            def value(self) -> float:
                return 0.5

        @METRIC_REGISTRY.register("test_required")
        class TestRequiredMetric(Metric):
            _short_name_base = "Test"
            _long_name_base = "Test Metric"
            description = "Test metric with required param"
            example_cls = TestMetricExample

            _hyperparams = {
                "threshold": float,  # Required (no default)
            }

            @metric_value(main=True)
            def value(self) -> float:
                threshold = self.params["threshold"]
                return sum(
                    ex.metrics.get(self.name).value for ex in self._src if ex.metrics.get(self.name).value >= threshold
                )

        # Try to access without providing required param
        with pytest.raises(ValueError) as exc_info:
            sample_dataset.metrics.test_required.value
        assert "Missing required parameters" in str(exc_info.value)
        assert "threshold" in str(exc_info.value)
        assert ".with_params(threshold=...)" in str(exc_info.value)

    def test_required_hyperparam_works_when_provided(self, sample_dataset):
        """Test that required hyperparams work when provided."""
        from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_value

        class TestMetricExample2(ExampleMetric):
            @metric_value(main=True)
            def value(self) -> float:
                return 0.5

        @METRIC_REGISTRY.register("test_required2")
        class TestRequiredMetric2(Metric):
            _short_name_base = "Test2"
            _long_name_base = "Test Metric 2"
            description = "Test metric with required param"
            example_cls = TestMetricExample2

            _hyperparams = {
                "threshold": float,  # Required
            }

            @metric_value(main=True)
            def value(self) -> float:
                threshold = self.params["threshold"]
                return threshold * 2

        # Provide required param - should work
        result = sample_dataset.metrics.test_required2.with_params(threshold=0.5).value
        assert result == 1.0

    def test_mixed_required_and_optional_hyperparams(self, sample_dataset):
        """Test metrics with both required and optional hyperparams."""
        from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_value

        class TestMetricExample3(ExampleMetric):
            @metric_value(main=True)
            def value(self) -> float:
                return 0.5

        @METRIC_REGISTRY.register("test_mixed")
        class TestMixedMetric(Metric):
            _short_name_base = "TestMixed"
            _long_name_base = "Test Mixed Metric"
            description = "Test metric with mixed params"
            example_cls = TestMetricExample3

            _hyperparams = {
                "threshold": float,  # Required
                "min_length": (int, 1),  # Optional with default
            }

            @metric_value(main=True)
            def value(self) -> float:
                threshold = self.params["threshold"]
                min_length = self.params["min_length"]
                return threshold * min_length

        # Missing required param
        with pytest.raises(ValueError) as exc_info:
            sample_dataset.metrics.test_mixed.value
        assert "threshold" in str(exc_info.value)

        # Provide only required param - should use default for optional
        result = sample_dataset.metrics.test_mixed.with_params(threshold=0.5).value
        assert result == 0.5  # 0.5 * 1 (default min_length)

        # Provide both params
        result = sample_dataset.metrics.test_mixed.with_params(threshold=0.5, min_length=3).value
        assert result == 1.5  # 0.5 * 3
