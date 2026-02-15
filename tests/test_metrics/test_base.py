"""Tests for bewer.metrics.base module."""

import pytest

from bewer.metrics.base import (
    METRIC_REGISTRY,
    Metric,
    MetricRegistry,
    list_registered_metrics,
)


class TestMetricValueDecorator:
    """Tests for the metric_value decorator."""

    def test_metric_value_caches_result(self, sample_dataset):
        """Test that metric_value caches computed values."""
        # Access WER metric multiple times
        wer = sample_dataset.metrics.wer
        value1 = wer.value
        value2 = wer.value
        assert value1 == value2

    def test_metric_value_main_flag(self):
        """Test that main flag is correctly set."""
        # The WER class should have value as main
        from bewer.metrics.wer import WER

        metric_values = WER.metric_values()
        assert metric_values["main"] == "value"

    def test_metric_value_other_values(self):
        """Test that other metric values are tracked."""
        from bewer.metrics.wer import WER

        metric_values = WER.metric_values()
        assert "num_edits" in metric_values["other"]
        assert "ref_length" in metric_values["other"]


class TestMetricRegistry:
    """Tests for MetricRegistry class."""

    def test_registry_exists(self):
        """Test that global registry exists."""
        assert METRIC_REGISTRY is not None
        assert isinstance(METRIC_REGISTRY, MetricRegistry)

    def test_registry_has_factories(self):
        """Test that registry has metric factories."""
        assert hasattr(METRIC_REGISTRY, "metric_factories")
        assert isinstance(METRIC_REGISTRY.metric_factories, dict)

    def test_registry_has_classes(self):
        """Test that registry has metric classes."""
        assert hasattr(METRIC_REGISTRY, "metric_classes")
        assert isinstance(METRIC_REGISTRY.metric_classes, dict)


class TestMetricRegistryRegister:
    """Tests for MetricRegistry.register() decorator."""

    def test_register_adds_to_registry(self):
        """Test that register decorator adds metric to registry."""
        # WER and CER should already be registered
        assert "wer" in METRIC_REGISTRY.metric_factories
        assert "cer" in METRIC_REGISTRY.metric_factories

    def test_registered_metric_is_callable(self):
        """Test that registered metric factory is callable."""
        factory = METRIC_REGISTRY.metric_factories["wer"]
        assert callable(factory)


class TestMetricRegistryRegisterMetric:
    """Tests for MetricRegistry.register_metric() method."""

    def test_register_metric_invalid_class_raises(self):
        """Test that registering non-Metric class raises TypeError."""
        registry = MetricRegistry()

        class NotAMetric:
            pass

        with pytest.raises(TypeError, match="must inherit from Metric"):
            registry.register_metric(NotAMetric, name="invalid")

    def test_register_metric_invalid_name_raises(self):
        """Test that registering with non-string name raises TypeError."""
        registry = MetricRegistry()

        # Create a minimal valid Metric subclass for testing
        class DummyMetric(Metric):
            short_name = "TEST"
            long_name = "Test Metric"
            description = "Test"

        with pytest.raises(TypeError, match="name must be a string"):
            registry.register_metric(DummyMetric, name=123)

    def test_register_metric_duplicate_raises(self):
        """Test that registering duplicate name raises ValueError."""
        registry = MetricRegistry()

        class DummyMetric(Metric):
            short_name = "TEST"
            long_name = "Test Metric"
            description = "Test"

        registry.register_metric(DummyMetric, name="test_metric")

        with pytest.raises(ValueError, match="already registered"):
            registry.register_metric(DummyMetric, name="test_metric")

    def test_register_metric_allow_override(self):
        """Test that allow_override permits duplicate registration."""
        registry = MetricRegistry()

        class DummyMetric(Metric):
            short_name = "TEST"
            long_name = "Test Metric"
            description = "Test"

        registry.register_metric(DummyMetric, name="test_metric2")
        registry.register_metric(DummyMetric, name="test_metric2", allow_override=True)
        assert "test_metric2" in registry.metric_factories


class TestMetricCollection:
    """Tests for MetricCollection class."""

    def test_get_registered_metric(self, sample_dataset):
        """Test getting a registered metric."""
        wer = sample_dataset.metrics.get("wer")
        assert wer is not None

    def test_get_unregistered_metric_raises(self, sample_dataset):
        """Test getting unregistered metric raises AttributeError."""
        with pytest.raises(AttributeError, match="not found"):
            sample_dataset.metrics.get("nonexistent_metric")

    def test_getattr_works_like_get(self, sample_dataset):
        """Test that attribute access works like get()."""
        wer = sample_dataset.metrics.wer
        assert wer is not None

    def test_metric_cached(self, sample_dataset):
        """Test that metrics are cached."""
        wer1 = sample_dataset.metrics.get("wer")
        wer2 = sample_dataset.metrics.get("wer")
        assert wer1 is wer2


class TestExampleMetricCollection:
    """Tests for ExampleMetricCollection class."""

    def test_get_example_metric(self, sample_example):
        """Test getting an example-level metric."""
        wer = sample_example.metrics.get("wer")
        assert wer is not None

    def test_get_unregistered_raises(self, sample_example):
        """Test getting unregistered metric raises AttributeError."""
        with pytest.raises(AttributeError, match="not found"):
            sample_example.metrics.get("nonexistent")

    def test_getattr_works_like_get(self, sample_example):
        """Test that attribute access works like get()."""
        wer = sample_example.metrics.wer
        assert wer is not None


class TestListRegisteredMetrics:
    """Tests for list_registered_metrics() function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        metrics = list_registered_metrics()
        assert isinstance(metrics, list)

    def test_includes_wer_cer(self):
        """Test that WER and CER are in the list."""
        metrics = list_registered_metrics()
        assert "wer" in metrics
        assert "cer" in metrics

    def test_show_private_false_excludes_underscore(self):
        """Test that private metrics (starting with _) are excluded by default."""
        metrics = list_registered_metrics(show_private=False)
        for metric in metrics:
            assert not metric.startswith("_")

    def test_show_private_true_includes_all(self):
        """Test that show_private=True includes all metrics."""
        public_metrics = list_registered_metrics(show_private=False)
        all_metrics = list_registered_metrics(show_private=True)
        assert len(all_metrics) >= len(public_metrics)


class TestMetricClass:
    """Tests for Metric base class."""

    def test_metric_has_required_properties(self, sample_dataset):
        """Test that Metric instances have required properties."""
        wer = sample_dataset.metrics.wer
        assert hasattr(wer, "short_name")
        assert hasattr(wer, "long_name")
        assert hasattr(wer, "description")
        assert hasattr(wer, "pipeline")

    def test_metric_pipeline_property(self, sample_dataset):
        """Test that pipeline returns tuple."""
        wer = sample_dataset.metrics.wer
        pipeline = wer.pipeline
        assert isinstance(pipeline, tuple)
        assert len(pipeline) == 3

    def test_set_source(self, sample_dataset):
        """Test that set_source sets the dataset."""
        from bewer.metrics.wer import WER

        metric = WER(name="test_wer")
        metric.set_source(sample_dataset)
        assert metric.src is sample_dataset

    def test_set_standardizer(self, sample_dataset):
        """Test that set_standardizer works."""
        wer = sample_dataset.metrics.wer
        wer.set_standardizer("custom")
        assert wer._standardizer == "custom"

    def test_set_tokenizer(self, sample_dataset):
        """Test that set_tokenizer works."""
        wer = sample_dataset.metrics.wer
        wer.set_tokenizer("custom")
        assert wer._tokenizer == "custom"

    def test_set_normalizer(self, sample_dataset):
        """Test that set_normalizer works."""
        wer = sample_dataset.metrics.wer
        wer.set_normalizer("custom")
        assert wer._normalizer == "custom"
