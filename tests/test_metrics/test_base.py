"""Tests for bewer.metrics.base module."""

import pytest

from bewer.metrics.base import (
    METRIC_REGISTRY,
    ExampleMetric,
    Metric,
    MetricRegistry,
    dependency,
    list_registered_metrics,
    metric_value,
)


class TestMetricValueDecorator:
    """Tests for the metric_value decorator."""

    def test_metric_value_caches_result(self, sample_dataset):
        """Test that metric_value caches computed values."""
        # Access WER metric multiple times
        wer = sample_dataset.metrics.wer()
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

    def test_underscore_name_excluded_by_default(self):
        """Underscore-named metric_value is excluded from metric_values()['other'] by default."""

        class DummyMetric(ExampleMetric):
            @metric_value
            def public_val(self):
                return 1

            @metric_value
            def _private_val(self):
                return 2

        mv = DummyMetric.metric_values()
        assert "public_val" in mv["other"]
        assert "_private_val" not in mv["other"]
        assert "_private_val" not in (mv.get("private") or [])

    def test_underscore_name_appears_with_include_private(self):
        """Underscore-named metric_value is included when include_private=True."""

        class DummyMetric(ExampleMetric):
            @metric_value
            def public_val(self):
                return 1

            @metric_value
            def _private_val(self):
                return 2

        mv = DummyMetric.metric_values(include_private=True)
        assert "_private_val" in mv["private"]
        assert "public_val" in mv["other"]

    def test_explicit_private_false_overrides_underscore(self):
        """@metric_value(private=False) on an underscore name forces it into 'other'."""

        class DummyMetric(ExampleMetric):
            @metric_value(private=False)
            def _not_actually_private(self):
                return 1

        mv = DummyMetric.metric_values()
        assert "_not_actually_private" in mv["other"]

    def test_explicit_private_true_on_public_name(self):
        """@metric_value(private=True) on a public name routes it to 'private'."""

        class DummyMetric(ExampleMetric):
            @metric_value(private=True)
            def hidden_val(self):
                return 1

        mv = DummyMetric.metric_values()
        assert "hidden_val" not in mv["other"]
        mv_full = DummyMetric.metric_values(include_private=True)
        assert "hidden_val" in mv_full["private"]

    def test_private_values_deduplicated_across_mro(self):
        """Private metric values are deduplicated when inherited across multiple bases."""

        class Base(ExampleMetric):
            @metric_value
            def _shared(self):
                return 1

        class Child(Base):
            pass

        mv = Child.metric_values(include_private=True)
        assert mv["private"].count("_shared") == 1


class TestDependencyDecorator:
    """Tests for the dependency decorator."""

    def test_dependency_registered_on_class(self):
        """Test that @dependency registers the property name in _dependencies."""

        class DummyMetric(Metric):
            short_name_base = "TEST"
            long_name_base = "Test Metric"
            description = "Test"

            @dependency
            def _dep_a(self):
                pass

            @dependency
            def _dep_b(self):
                pass

        assert DummyMetric.dependencies() == ["_dep_a", "_dep_b"]

    def test_dependency_order_preserved(self):
        """Test that definition order is preserved (no set shuffling)."""

        class DummyMetric(Metric):
            short_name_base = "TEST"
            long_name_base = "Test Metric"
            description = "Test"

            @dependency
            def _z(self):
                pass

            @dependency
            def _a(self):
                pass

            @dependency
            def _m(self):
                pass

        assert DummyMetric.dependencies() == ["_z", "_a", "_m"]

    def test_dependency_inherited(self):
        """Test that dependencies are collected from base classes via MRO."""

        class BaseMetric(Metric):
            short_name_base = "BASE"
            long_name_base = "Base Metric"
            description = "Base"

            @dependency
            def _base_dep(self):
                pass

        class ChildMetric(BaseMetric):
            short_name_base = "CHILD"
            long_name_base = "Child Metric"
            description = "Child"

            @dependency
            def _child_dep(self):
                pass

        assert "_base_dep" in ChildMetric.dependencies()
        assert "_child_dep" in ChildMetric.dependencies()

    def test_dependency_no_duplicates(self):
        """Test that a dependency name appearing in multiple bases is deduplicated."""

        class BaseMixin(Metric):
            short_name_base = "MIX"
            long_name_base = "Mixin"
            description = "Mixin"

            @dependency
            def _shared(self):
                pass

        class ChildMetric(BaseMixin):
            short_name_base = "CHILD"
            long_name_base = "Child"
            description = "Child"

            @dependency
            def _shared(self):
                pass

        deps = ChildMetric.dependencies()
        assert deps.count("_shared") == 1

    def test_dependency_caches_result(self):
        """Test that @dependency caches like cached_property."""
        call_count = 0

        class DummyMetric(Metric):
            short_name_base = "TEST"
            long_name_base = "Test Metric"
            description = "Test"

            @dependency
            def _dep(self):
                nonlocal call_count
                call_count += 1
                return object()

        m = DummyMetric.__new__(DummyMetric)
        m.__dict__.clear()
        result1 = DummyMetric._dep.__get__(m, DummyMetric)
        result2 = DummyMetric._dep.__get__(m, DummyMetric)
        assert result1 is result2
        assert call_count == 1

    def test_ktp_has_kt_stats_dependency(self):
        """Test that KTP registers _kt_stats as a dependency."""
        from bewer.metrics.ktp import KTP

        assert "_kt_stats" in KTP.dependencies()


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
        """Test that registered metric metadata exists."""
        metadata = METRIC_REGISTRY.metric_metadata["wer"]
        assert isinstance(metadata, dict)
        assert "metric_cls" in metadata


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
        """Test getting a registered metric factory."""
        wer_factory = sample_dataset.metrics.get("wer")
        assert wer_factory is not None
        assert callable(wer_factory)

    def test_get_unregistered_metric_raises(self, sample_dataset):
        """Test getting unregistered metric raises AttributeError."""
        with pytest.raises(AttributeError, match="not found"):
            sample_dataset.metrics.get("nonexistent_metric")

    def test_getattr_works_like_get(self, sample_dataset):
        """Test that attribute access works like get()."""
        wer_factory = sample_dataset.metrics.wer
        assert wer_factory is not None
        assert callable(wer_factory)

    def test_metric_cached(self, sample_dataset):
        """Test that metric instances are cached when called with same params."""
        wer1 = sample_dataset.metrics.get("wer")()
        wer2 = sample_dataset.metrics.get("wer")()
        assert wer1 is wer2


class TestExampleMetricCollection:
    """Tests for ExampleMetricCollection class."""

    def test_get_example_metric(self, sample_example):
        """Test getting an example-level metric factory."""
        wer_factory = sample_example.metrics.get("wer")
        assert wer_factory is not None
        assert callable(wer_factory)

    def test_get_unregistered_raises(self, sample_example):
        """Test getting unregistered metric raises AttributeError."""
        with pytest.raises(AttributeError, match="not found"):
            sample_example.metrics.get("nonexistent")

    def test_getattr_works_like_get(self, sample_example):
        """Test that attribute access works like get()."""
        wer_factory = sample_example.metrics.wer
        assert wer_factory is not None
        assert callable(wer_factory)


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
        wer = sample_dataset.metrics.wer()
        assert hasattr(wer, "short_name")
        assert hasattr(wer, "long_name")
        assert hasattr(wer, "description")
        assert hasattr(wer, "pipeline")

    def test_metric_pipeline_property(self, sample_dataset):
        """Test that pipeline returns tuple."""
        wer = sample_dataset.metrics.wer()
        pipeline = wer.pipeline
        assert isinstance(pipeline, tuple)
        assert len(pipeline) == 3

    def test_set_source(self, sample_dataset):
        """Test that set_source sets the dataset."""
        from bewer.metrics.wer import WER

        metric = WER(name="test_wer")
        metric.set_source(sample_dataset)
        assert metric.src is sample_dataset

    def test_init_standardizer(self):
        """Test that standardizer can be set via __init__."""
        from bewer.metrics.wer import WER

        wer = WER(name="test", standardizer="custom")
        assert wer._standardizer == "custom"

    def test_init_tokenizer(self):
        """Test that tokenizer can be set via __init__."""
        from bewer.metrics.wer import WER

        wer = WER(name="test", tokenizer="custom")
        assert wer._tokenizer == "custom"

    def test_init_normalizer(self):
        """Test that normalizer can be set via __init__."""
        from bewer.metrics.wer import WER

        wer = WER(name="test", normalizer="custom")
        assert wer._normalizer == "custom"
