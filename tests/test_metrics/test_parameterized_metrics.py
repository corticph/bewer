"""Tests for parameterized metrics functionality."""

from dataclasses import dataclass

import pytest

from bewer import Dataset
from bewer.metrics.base import MetricParams


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

    def test_basic_metric_has_no_params(self, sample_dataset):
        """Test that a basic metric has None params."""
        wer = sample_dataset.metrics.wer()
        assert wer.params is None

    def test_factory_returns_new_instance(self, sample_dataset):
        """Test that factory calls with different params return different instances."""
        kwa_base = sample_dataset.metrics._legacy_kwa()
        kwa_params = sample_dataset.metrics._legacy_kwa(cer_threshold=0.5)
        assert kwa_base is not kwa_params
        assert kwa_params.params.cer_threshold == 0.5

    def test_factory_caching(self, sample_dataset):
        """Test that factory caches instances with same parameters."""
        kwa1 = sample_dataset.metrics._legacy_kwa(cer_threshold=0.5)
        kwa2 = sample_dataset.metrics._legacy_kwa(cer_threshold=0.5)
        assert kwa1 is kwa2

    def test_factory_different_params_different_instances(self, sample_dataset):
        """Test that different parameters create different instances."""
        kwa1 = sample_dataset.metrics._legacy_kwa(cer_threshold=0.5)
        kwa2 = sample_dataset.metrics._legacy_kwa(cer_threshold=0.6)
        assert kwa1 is not kwa2

    def test_factory_non_hashable_params(self, sample_dataset):
        """Test that non-hashable parameters raise TypeError with helpful message."""
        with pytest.raises(TypeError) as exc_info:
            sample_dataset.metrics._legacy_kwa(config={"a": [1, 2, 3], "b": {"x": 1}})
        error_msg = str(exc_info.value)
        assert "must be hashable" in error_msg or "unhashable type" in error_msg
        assert "config" in error_msg

    def test_factory_uses_defaults(self, sample_dataset):
        """Test that factory uses default parameters."""
        # KeywordAggregator has default cer_threshold=0.2
        kwa1 = sample_dataset.metrics._legacy_kwa()
        assert kwa1.params.cer_threshold == 0.2

    def test_factory_overrides_defaults(self, sample_dataset):
        """Test that factory can override default parameters."""
        kwa = sample_dataset.metrics._legacy_kwa(cer_threshold=0.5)
        assert kwa.params.cer_threshold == 0.5


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
        wer = sample_dataset.metrics.wer()
        assert wer.short_name == "WER"

    def test_base_metric_long_name_no_params(self, sample_dataset):
        """Test that base metric without params shows base long name."""
        wer = sample_dataset.metrics.wer()
        assert wer.long_name == "Word Error Rate"

    def test_parameterized_metric_short_name_includes_params(self, sample_dataset):
        """Test that parameterized metric includes params in short name."""
        kwa = sample_dataset.metrics._legacy_kwa(cer_threshold=0.5)
        assert "cer_threshold=0.5" in kwa.short_name
        assert kwa.short_name.startswith("kwa (")

    def test_parameterized_metric_long_name_includes_params(self, sample_dataset):
        """Test that parameterized metric includes params in long name."""
        kwa = sample_dataset.metrics._legacy_kwa(cer_threshold=0.5)
        assert "cer_threshold=0.5" in kwa.long_name
        assert kwa.long_name.startswith("Keyword Aggregator (")

    def test_multiple_params_in_name(self, sample_dataset):
        """Test that multiple parameters are shown in name."""
        # Use HallucinationAggregator which has threshold param
        # Note: We're just testing the naming mechanism, the actual computation
        # might not work properly without proper data
        hlcn = sample_dataset.metrics._legacy_hlcn(threshold=5)
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
        kwa = keyword_dataset.metrics._legacy_kwa()
        assert kwa.params.cer_threshold == 0.2

    def test_custom_cer_threshold(self, keyword_dataset):
        """Test that custom cer_threshold can be set."""
        kwa = keyword_dataset.metrics._legacy_kwa(cer_threshold=0.1)
        assert kwa.params.cer_threshold == 0.1

    def test_kwa_short_name_includes_threshold(self, keyword_dataset):
        """Test that KWA short name includes threshold parameter."""
        kwa = keyword_dataset.metrics._legacy_kwa()
        assert "cer_threshold=0.2" in kwa.short_name

    def test_kwa_computes_with_threshold(self, keyword_dataset):
        """Test that KWA correctly uses threshold in computation."""
        # This is a basic integration test to ensure it doesn't crash
        kwa = keyword_dataset.metrics._legacy_kwa()
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
        kwa_example = example.metrics._legacy_kwa()
        assert hasattr(kwa_example, "params")

    def test_example_metric_params_matches_parent(self, keyword_dataset):
        """Test that ExampleMetric params matches parent metric params."""
        kwa = keyword_dataset.metrics._legacy_kwa()
        example = keyword_dataset.examples[0]
        kwa_example = example.metrics._legacy_kwa()
        assert kwa_example.params == kwa.params

    def test_example_metric_params_with_custom_threshold(self, keyword_dataset):
        """Test that ExampleMetric gets custom params from parent."""
        kwa = keyword_dataset.metrics._legacy_kwa(cer_threshold=0.1)
        # Force creation of example metric
        _ = kwa.match_count
        example = keyword_dataset.examples[0]
        # Access the example metric for the parameterized parent
        # This requires getting it through the parameterized parent's cache
        kwa_example = kwa._get_example_metric(example)
        assert kwa_example.params.cer_threshold == 0.1


class TestCacheKeyGeneration:
    """Test suite for cache key generation."""

    def test_make_cache_key_empty_params(self):
        """Test cache key generation with empty params."""
        from bewer.metrics.base import MetricCollection

        key = MetricCollection._make_cache_key()
        assert key == ()

    def test_make_cache_key_single_param(self):
        """Test cache key generation with single parameter."""
        from bewer.metrics.base import MetricCollection

        key = MetricCollection._make_cache_key(threshold=0.5)
        assert key == (("threshold", 0.5),)

    def test_make_cache_key_multiple_params(self):
        """Test cache key generation with multiple parameters."""
        from bewer.metrics.base import MetricCollection

        key = MetricCollection._make_cache_key(threshold=0.5, ignore_insertions=True)
        # Sorted by key name
        assert key == (("ignore_insertions", True), ("threshold", 0.5))

    def test_make_cache_key_non_hashable_params(self):
        """Test that cache key with non-hashable parameters cannot be used as dict key."""
        from bewer.metrics.base import MetricCollection

        # _make_cache_key succeeds, but the result cannot be hashed
        key = MetricCollection._make_cache_key(config={"a": [1, 2, 3]})
        assert isinstance(key, tuple)

        # But trying to use it as a dict key raises TypeError
        with pytest.raises(TypeError):
            test_dict = {}
            test_dict[key] = "value"

    def test_make_cache_key_consistent(self):
        """Test that cache key is consistent for same params."""
        from bewer.metrics.base import MetricCollection

        key1 = MetricCollection._make_cache_key(threshold=0.5, value=1)
        key2 = MetricCollection._make_cache_key(threshold=0.5, value=1)
        assert key1 == key2

    def test_make_cache_key_order_independent(self):
        """Test that cache key is order-independent for params."""
        from bewer.metrics.base import MetricCollection

        key1 = MetricCollection._make_cache_key(a=1, b=2)
        key2 = MetricCollection._make_cache_key(b=2, a=1)
        assert key1 == key2


class TestDeclarativeHyperparams:
    """Test suite for declarative hyperparameters."""

    def test_metric_without_params_rejects_params(self):
        """Test that metrics without param_schema reject all parameters."""
        from bewer import Dataset

        dataset = Dataset()
        dataset.add(ref="hello world", hyp="hello world")

        # WER doesn't define param_schema, so it should reject params
        with pytest.raises(ValueError) as exc_info:
            dataset.metrics.wer(threshold=0.5)
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
        kwa = dataset.metrics._legacy_kwa()
        assert kwa.params.cer_threshold == 0.2

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
        kwa = dataset.metrics._legacy_kwa(cer_threshold=0.5)
        assert kwa.params.cer_threshold == 0.5

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
            dataset.metrics._legacy_kwa(unknown_param=123)
        assert "Unknown parameter" in str(exc_info.value)
        assert "unknown_param" in str(exc_info.value)
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
            dataset.metrics._legacy_kwa(cer_threshold="not a float")
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

    def test_required_hyperparam_raises_on_construction(self, sample_dataset):
        """Test that missing required hyperparams raise error eagerly at construction."""
        # Create a test metric with required param
        from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_value

        class TestMetricExample(ExampleMetric):
            @metric_value(main=True)
            def value(self) -> float:
                return 0.5

        @METRIC_REGISTRY.register("test_required")
        class TestRequiredMetric(Metric):
            short_name_base = "Test"
            long_name_base = "Test Metric"
            description = "Test metric with required param"
            example_cls = TestMetricExample

            @dataclass
            class param_schema(MetricParams):
                threshold: float

            @metric_value(main=True)
            def value(self) -> float:
                threshold = self.params.threshold
                return sum(
                    self._get_example_metric(ex).value
                    for ex in self._src
                    if self._get_example_metric(ex).value >= threshold
                )

        # Try to construct without providing required param — error is eager
        with pytest.raises(ValueError) as exc_info:
            sample_dataset.metrics.test_required()
        assert "Missing required parameters" in str(exc_info.value)
        assert "threshold" in str(exc_info.value)

    def test_required_hyperparam_works_when_provided(self, sample_dataset):
        """Test that required hyperparams work when provided."""
        from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_value

        class TestMetricExample2(ExampleMetric):
            @metric_value(main=True)
            def value(self) -> float:
                return 0.5

        @METRIC_REGISTRY.register("test_required2")
        class TestRequiredMetric2(Metric):
            short_name_base = "Test2"
            long_name_base = "Test Metric 2"
            description = "Test metric with required param"
            example_cls = TestMetricExample2

            @dataclass
            class param_schema(MetricParams):
                threshold: float

            @metric_value(main=True)
            def value(self) -> float:
                threshold = self.params.threshold
                return threshold * 2

        # Provide required param - should work
        result = sample_dataset.metrics.test_required2(threshold=0.5).value
        assert result == 1.0

    def test_mixed_required_and_optional_params(self, sample_dataset):
        """Test metrics with both required and optional hyperparams."""
        from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_value

        class TestMetricExample3(ExampleMetric):
            @metric_value(main=True)
            def value(self) -> float:
                return 0.5

        @METRIC_REGISTRY.register("test_mixed")
        class TestMixedMetric(Metric):
            short_name_base = "TestMixed"
            long_name_base = "Test Mixed Metric"
            description = "Test metric with mixed params"
            example_cls = TestMetricExample3

            @dataclass
            class param_schema(MetricParams):
                threshold: float
                min_length: int = 1

            @metric_value(main=True)
            def value(self) -> float:
                threshold = self.params.threshold
                min_length = self.params.min_length
                return threshold * min_length

        # Missing required param — error is eager at construction
        with pytest.raises(ValueError) as exc_info:
            sample_dataset.metrics.test_mixed()
        assert "threshold" in str(exc_info.value)

        # Provide only required param - should use default for optional
        result = sample_dataset.metrics.test_mixed(threshold=0.5).value
        assert result == 0.5  # 0.5 * 1 (default min_length)

        # Provide both params
        result = sample_dataset.metrics.test_mixed(threshold=0.5, min_length=3).value
        assert result == 1.5  # 0.5 * 3

    def test_params_printed_in_definition_order(self, sample_dataset):
        """Test that parameters are printed in the order they're defined in param_schema."""
        from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_value

        class TestMetricExample4(ExampleMetric):
            @metric_value(main=True)
            def value(self) -> float:
                return 0.5

        @METRIC_REGISTRY.register("test_param_order")
        class TestParamOrderMetric(Metric):
            short_name_base = "TestOrder"
            long_name_base = "Test Parameter Order"
            description = "Test metric with multiple params"
            example_cls = TestMetricExample4

            @dataclass
            class param_schema(MetricParams):
                alpha: str = "a"
                beta: int = 2
                gamma: float = 3.0

            @metric_value(main=True)
            def value(self) -> float:
                return 1.0

        # Set params in different order than definition
        metric = sample_dataset.metrics.test_param_order(gamma=5.0, alpha="z", beta=9)

        # Check that short_name shows params in definition order (alpha, beta, gamma)
        # not in the order they were provided (gamma, alpha, beta)
        assert metric.short_name == "TestOrder (alpha=z, beta=9, gamma=5.0)"
        assert metric.long_name == "Test Parameter Order (alpha=z, beta=9, gamma=5.0)"
