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

    def test_get_metric_factory(self, sample_example):
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

    def test_src_set_at_construction(self, sample_dataset):
        """Test that the src passed at construction is stored."""
        from bewer.metrics.wer import WER

        metric = WER(name="test_wer", src=sample_dataset)
        assert metric.src is sample_dataset

    def test_init_standardizer(self, sample_dataset):
        """Test that standardizer can be set via __init__."""
        from bewer.metrics.wer import WER

        wer = WER(name="test", src=sample_dataset, standardizer="custom")
        assert wer._standardizer == "custom"

    def test_init_tokenizer(self, sample_dataset):
        """Test that tokenizer can be set via __init__."""
        from bewer.metrics.wer import WER

        wer = WER(name="test", src=sample_dataset, tokenizer="custom")
        assert wer._tokenizer == "custom"

    def test_init_normalizer(self, sample_dataset):
        """Test that normalizer can be set via __init__."""
        from bewer.metrics.wer import WER

        wer = WER(name="test", src=sample_dataset, normalizer="custom")
        assert wer._normalizer == "custom"


class TestMetricSequenceProtocol:
    """Tests for the Metric __len__/__iter__/__getitem__ sequence protocol."""

    def test_len_matches_dataset(self, sample_dataset):
        """len(metric) equals the number of examples in the dataset."""
        wer = sample_dataset.metrics.wer()
        assert len(wer) == len(sample_dataset)

    def test_getitem_returns_example_metric(self, sample_dataset):
        """Indexing returns the ExampleMetric for that positional example."""
        from bewer.metrics.wer import WER_

        wer = sample_dataset.metrics.wer()
        em = wer[0]
        assert isinstance(em, WER_)
        assert em.example is sample_dataset[0]

    def test_getitem_is_cached(self, sample_dataset):
        """Repeated access returns the identical cached object."""
        wer = sample_dataset.metrics.wer()
        assert wer[0] is wer[0]

    def test_iter_matches_getitem(self, sample_dataset):
        """Iteration yields the same objects (and order) as positional indexing."""
        wer = sample_dataset.metrics.wer()
        iterated = list(wer)
        assert len(iterated) == len(wer)
        for i, em in enumerate(iterated):
            assert em is wer[i]

    def test_negative_index(self, sample_dataset):
        """Negative indices behave like list indexing."""
        wer = sample_dataset.metrics.wer()
        assert wer[-1] is wer[len(wer) - 1]

    def test_slice_returns_list(self, sample_dataset):
        """Slicing returns a list of ExampleMetric objects."""
        wer = sample_dataset.metrics.wer()
        sliced = wer[0:2]
        assert isinstance(sliced, list)
        assert len(sliced) == 2
        assert sliced[0] is wer[0]
        assert sliced[1] is wer[1]

    def test_out_of_range_raises_index_error(self, sample_dataset):
        """Out-of-range integer indices raise IndexError."""
        wer = sample_dataset.metrics.wer()
        with pytest.raises(IndexError):
            _ = wer[len(wer)]

    def test_no_example_cls_raises_type_error(self, sample_dataset):
        """Metrics without an example_cls are not iterable/indexable."""
        from bewer.metrics.base import Metric

        class NoExampleMetric(Metric):
            short_name_base = "NEM"
            long_name_base = "No Example Metric"
            description = "A metric with no example-level metric."
            example_cls = None

        metric = NoExampleMetric(src=sample_dataset)
        with pytest.raises(TypeError):
            len(metric)
        with pytest.raises(TypeError):
            _ = metric[0]
        with pytest.raises(TypeError):
            iter(metric)


class TestFormatRegisteredParams:
    """Tests for the parameter summary shown by list_metrics()."""

    def test_defaulted_params_show_their_default(self):
        """A parameter with a default renders as name=default."""
        from bewer.metrics.base import _format_registered_params

        assert _format_registered_params("wer") == "normalized=True"

    def test_required_params_are_marked(self):
        """A parameter without a default is marked with a star, not a value."""
        from bewer.metrics.base import _format_registered_params

        summary = _format_registered_params("ktr")

        assert summary.startswith("vocab*")
        assert "normalized=True" in summary

    def test_reads_the_registry_not_an_instance(self):
        """Metrics with required params cannot be constructed, but are still describable."""
        from bewer.metrics.base import _format_registered_params

        # ktr requires `vocab`, so no instance exists to introspect.
        assert "vocab*" in _format_registered_params("ktr")

    def test_specialization_shows_its_own_default(self):
        """A subclass that gives a required param a default shows the value, unmarked."""
        from bewer.metrics.base import _format_registered_params

        summary = _format_registered_params("orthographically_complex_term_recall")

        assert "vocab='orthographically_complex_terms'" in summary
        assert "*" not in summary

    def test_registration_default_overrides_the_schema_default(self):
        """A default supplied at registration wins over the one on param_schema."""
        from bewer.metrics.base import METRIC_REGISTRY, _format_registered_params

        metadata = METRIC_REGISTRY.metric_metadata["wer"]
        original = metadata["param_defaults"]
        metadata["param_defaults"] = {"normalized": False}
        try:
            assert _format_registered_params("wer") == "normalized=False"
        finally:
            metadata["param_defaults"] = original

    def test_metric_without_params_renders_a_placeholder(self):
        """A metric with no parameters renders as '-' rather than an empty cell."""
        from bewer.metrics.base import _format_registered_params

        assert _format_registered_params("summary") == "-"


class TestListMetricsIncludesParams:
    """list_metrics() surfaces the parameter summary."""

    def test_params_appear_in_the_table(self, capsys, sample_dataset):
        """The rendered table contains a Params column and the summaries."""
        sample_dataset.metrics.list_metrics()
        out = capsys.readouterr().out

        assert "Params" in out
        assert "required parameter" in out  # the caption explaining '*'

    def test_params_shown_once_per_metric(self, capsys):
        """The summary is attached to the dataset row, not repeated on the example row.

        Renders one two-level row directly with a short value, so the assertion cannot be
        defeated by the cell wrapping at the console's width.
        """
        from bewer.reporting.python.tables import print_metric_table

        print_metric_table([("m", "n=1", (("value", "-"), ("value", "-")))])
        out = capsys.readouterr().out

        assert out.count("n=1") == 1


class TestParamFormattingEdgeCases:
    """Values that the registry can hold but the shipped metrics do not exercise."""

    def test_default_factory_is_resolved(self):
        """A dataclass default_factory is called, not printed as the factory itself."""
        from dataclasses import dataclass, field

        from bewer.metrics.base import (
            METRIC_REGISTRY,
            ExampleMetric,
            Metric,
            MetricParams,
            _format_registered_params,
            metric_value,
        )

        class _FactoryProbe_(ExampleMetric):
            @metric_value(main=True)
            def value(self) -> float:
                return 0.0

        @METRIC_REGISTRY.register("_factory_probe", allow_override=True)
        class _FactoryProbe(Metric):
            short_name_base = "_FactoryProbe"
            long_name_base = "Factory Probe"
            description = "Test metric with a default_factory parameter."
            example_cls = _FactoryProbe_

            @dataclass
            class param_schema(MetricParams):
                tags: tuple = field(default_factory=tuple)

        try:
            assert _format_registered_params("_factory_probe") == "tags=()"
        finally:
            del METRIC_REGISTRY.metric_metadata["_factory_probe"]
            del METRIC_REGISTRY.metric_classes["_factory_probe"]

    def test_markup_like_defaults_render_literally(self, capsys):
        """A default that looks like Rich markup must not be parsed as markup.

        Rich treats "[" followed by a lowercase letter, "#", "/" or "@" as a tag, which
        would silently swallow a value such as a regex default (and raise on "[/]").
        """
        from bewer.reporting.python.tables import print_metric_table

        for default in ("pattern='[a-z]+'", "pattern='[bold]'", "pattern='[/]'"):
            print_metric_table([("m", default, (("value", "-"), None))])
            out = capsys.readouterr().out
            # The bracketed fragment survives; only wrapping may break the full string.
            assert "[" in out and "]" in out, f"markup consumed for {default!r}"
