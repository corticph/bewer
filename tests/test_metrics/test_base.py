"""Tests for bewer.metrics.base module."""

from dataclasses import dataclass
from typing import Literal

import pytest

from bewer import Dataset, Vocabulary
from bewer.metrics.base import (
    METRIC_REGISTRY,
    DependencyEdge,
    ExampleMetric,
    Metric,
    MetricParams,
    MetricPropertyType,
    MetricRegistry,
    _check_dependency_graph,
    dependency,
    list_registered_metrics,
    metric_property,
)


class TestMetricPropertyDecorator:
    """Tests for the metric_property decorator."""

    def test_metric_property_caches_result(self, sample_dataset):
        """Test that metric_property caches computed values."""
        # Access WER metric multiple times
        wer = sample_dataset.metrics.wer()
        value1 = wer.value
        value2 = wer.value
        assert value1 == value2

    def test_metric_property_main_flag(self):
        """Test that main flag is correctly set."""
        # The WER class should have value as main
        from bewer.metrics.general.wer import WER

        metric_values = WER.metric_values()
        assert metric_values["main"] == "value"

    def test_metric_property_other_values(self):
        """Test that other metric values are tracked."""
        from bewer.metrics.general.wer import WER

        metric_values = WER.metric_values()
        assert "num_edits" in metric_values["other"]
        assert "ref_length" in metric_values["other"]

    def test_underscore_name_excluded_by_default(self):
        """Underscore-named metric_property is excluded from metric_values()['other'] by default."""

        class DummyMetric(ExampleMetric):
            @metric_property(type="value")
            def public_val(self):
                return 1

            @metric_property(type="value")
            def _private_val(self):
                return 2

        mv = DummyMetric.metric_values()
        assert "public_val" in mv["other"]
        assert "_private_val" not in mv["other"]
        assert "_private_val" not in (mv.get("private") or [])

    def test_underscore_name_appears_with_include_private(self):
        """Underscore-named metric_property is included when include_private=True."""

        class DummyMetric(ExampleMetric):
            @metric_property(type="value")
            def public_val(self):
                return 1

            @metric_property(type="value")
            def _private_val(self):
                return 2

        mv = DummyMetric.metric_values(include_private=True)
        assert "_private_val" in mv["private"]
        assert "public_val" in mv["other"]

    def test_explicit_private_false_overrides_underscore(self):
        """@metric_property(type="value", private=False) on an underscore name forces it into 'other'."""

        class DummyMetric(ExampleMetric):
            @metric_property(type="value", private=False)
            def _not_actually_private(self):
                return 1

        mv = DummyMetric.metric_values()
        assert "_not_actually_private" in mv["other"]

    def test_explicit_private_true_on_public_name(self):
        """@metric_property(type="value", private=True) on a public name routes it to 'private'."""

        class DummyMetric(ExampleMetric):
            @metric_property(type="value", private=True)
            def hidden_val(self):
                return 1

        mv = DummyMetric.metric_values()
        assert "hidden_val" not in mv["other"]
        mv_full = DummyMetric.metric_values(include_private=True)
        assert "hidden_val" in mv_full["private"]

    def test_private_values_deduplicated_across_mro(self):
        """Private metric values are deduplicated when inherited across multiple bases."""

        class Base(ExampleMetric):
            @metric_property(type="value")
            def _shared(self):
                return 1

        class Child(Base):
            pass

        mv = Child.metric_values(include_private=True)
        assert mv["private"].count("_shared") == 1


class TestDependencyDecorator:
    """Tests for function-form dependencies (metric_property(type="dependency"))."""

    def test_dependency_registered_on_class(self):
        """Test that type="dependency" registers the property name in _dependencies."""

        class DummyMetric(Metric):
            short_name_base = "TEST"
            long_name_base = "Test Metric"
            description = "Test"

            @metric_property(type="dependency")
            def _dep_a(self):
                pass

            @metric_property(type="dependency")
            def _dep_b(self):
                pass

        assert DummyMetric.dependencies() == ["_dep_a", "_dep_b"]

    def test_dependency_order_preserved(self):
        """Test that definition order is preserved (no set shuffling)."""

        class DummyMetric(Metric):
            short_name_base = "TEST"
            long_name_base = "Test Metric"
            description = "Test"

            @metric_property(type="dependency")
            def _z(self):
                pass

            @metric_property(type="dependency")
            def _a(self):
                pass

            @metric_property(type="dependency")
            def _m(self):
                pass

        assert DummyMetric.dependencies() == ["_z", "_a", "_m"]

    def test_dependency_inherited(self):
        """Test that dependencies are collected from base classes via MRO."""

        class BaseMetric(Metric):
            short_name_base = "BASE"
            long_name_base = "Base Metric"
            description = "Base"

            @metric_property(type="dependency")
            def _base_dep(self):
                pass

        class ChildMetric(BaseMetric):
            short_name_base = "CHILD"
            long_name_base = "Child Metric"
            description = "Child"

            @metric_property(type="dependency")
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

            @metric_property(type="dependency")
            def _shared(self):
                pass

        class ChildMetric(BaseMixin):
            short_name_base = "CHILD"
            long_name_base = "Child"
            description = "Child"

            @metric_property(type="dependency")
            def _shared(self):
                pass

        deps = ChildMetric.dependencies()
        assert deps.count("_shared") == 1

    def test_dependency_caches_result(self):
        """Test that @metric_property(type="dependency") caches like cached_property."""
        call_count = 0

        class DummyMetric(Metric):
            short_name_base = "TEST"
            long_name_base = "Test Metric"
            description = "Test"

            @metric_property(type="dependency")
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

    def test_ktp_has_kt_outcomes_dependency(self):
        """Test that KTP registers _kt_outcomes as a dependency."""
        from bewer.metrics.key_term.key_term_precision import KTP

        assert "_kt_outcomes" in KTP.dependencies()


class TestMetricPropertyType:
    """Tests for the type argument of metric_property."""

    def test_unknown_type_raises(self):
        """An unknown type is rejected when the decorator is built."""
        with pytest.raises(ValueError, match="Unknown metric_property type 'values'"):
            metric_property(type="values")

    def test_targets_only_for_dependency_types(self):
        """targets is rejected on a non-dependency property."""
        with pytest.raises(ValueError, match="`targets` applies only to dependency types"):
            metric_property(type="value", targets=("wer",))

    def test_member_and_string_are_interchangeable(self):
        """A member and its string value give the same type, which equals the plain string."""
        assert metric_property(type=MetricPropertyType.MAIN).type is MetricPropertyType.MAIN
        assert metric_property(type="main").type is MetricPropertyType.MAIN
        assert MetricPropertyType.MAIN == "main"

    @pytest.mark.parametrize(
        ("member", "is_value", "is_dependency"),
        [
            (MetricPropertyType.VALUE, True, False),
            (MetricPropertyType.MAIN, True, False),
            (MetricPropertyType.DEPENDENCY, False, True),
            (MetricPropertyType.ALIGNMENT, False, True),
        ],
    )
    def test_member_roles(self, member, is_value, is_dependency):
        """Main is also a value; alignment is also a dependency."""
        assert member.is_value is is_value
        assert member.is_dependency is is_dependency


@pytest.fixture
def register_probe():
    """Register throwaway metric classes under private names; unregister them afterwards."""
    registered = []

    def register(name, metric_cls):
        METRIC_REGISTRY.register_metric(metric_cls, name)
        registered.append(name)
        return metric_cls

    yield register
    for name in registered:
        del METRIC_REGISTRY.metric_metadata[name]
        del METRIC_REGISTRY.metric_classes[name]


class TestDependencyDeclaration:
    """Tests for dependency(): declaration, static edges, building and graph checks."""

    def test_target_and_selected_by_are_exclusive(self):
        """Exactly one of target and selected_by is required."""
        with pytest.raises(ValueError, match="exactly one of `target` or `selected_by`"):
            dependency()
        with pytest.raises(ValueError, match="exactly one of `target` or `selected_by`"):
            dependency("wip", selected_by="alignment")

    def test_type_must_be_a_dependency_type(self):
        """A dependency cannot be typed as a value."""
        with pytest.raises(ValueError, match="A dependency's type must be one of"):
            dependency("wip", type="value")

    def test_edges_for_target(self):
        """A target dependency has one static target."""
        from bewer.metrics.general.wil import WIL

        assert WIL.dependencies() == ["_wip"]
        assert WIL.dependency_edges() == {"_wip": DependencyEdge(type="dependency", targets=("wip",))}

    def test_edges_mark_alignment(self):
        """An alignment dependency carries type 'alignment'."""
        from bewer.metrics.general.mer import MER

        assert MER.dependency_edges()["_alignment"] == DependencyEdge(type="alignment", targets=("levenshtein",))
        assert MER.dependency_edges()["_alignment"].type is MetricPropertyType.ALIGNMENT

    def test_edges_for_selected_by_come_from_literal(self):
        """A selected_by dependency's targets are the Literal values of its param."""
        from bewer.metrics.hallucination.insertion_rate import InsertionRate

        edge = InsertionRate.dependency_edges()["_alignment"]
        assert edge == DependencyEdge(type="alignment", targets=("levenshtein", "error_align"))

    def test_selected_by_needs_literal_param(self):
        """selected_by on a param that is not Literal-typed fails when edges are read."""

        @dataclass
        class Params(MetricParams):
            alignment: str = "levenshtein"

        class Probe(Metric):
            short_name_base = long_name_base = description = "Probe"
            param_schema = Params
            _alignment = dependency(selected_by="alignment", type="alignment")

        with pytest.raises(TypeError, match=r"selected_by='alignment' needs a param typed Literal\[\.\.\.\]"):
            Probe.dependency_edges()

    @pytest.mark.parametrize("value", [True, 1.0])
    def test_literal_param_rejects_equal_value_of_other_type(self, value):
        """A Literal param checks the value's type, so True and 1.0 do not pass as Literal[1]."""

        @dataclass
        class Params(MetricParams):
            level: Literal[1, 2] = 1

        with pytest.raises(ValueError, match=r"Parameter 'level' must be one of \[1, 2\]"):
            Params(level=value)

    def test_function_form_edges(self):
        """A function-form dependency reports its declared targets, or None without them."""

        class Probe(Metric):
            short_name_base = long_name_base = description = "Probe"

            @metric_property(type="dependency", targets=("wip",))
            def _declared(self):
                pass

            @metric_property(type="dependency")
            def _undeclared(self):
                pass

        assert Probe.dependency_edges() == {
            "_declared": DependencyEdge(type="dependency", targets=("wip",)),
            "_undeclared": DependencyEdge(type="dependency", targets=None),
        }

    def test_plain_override_of_inherited_dependency(self, sample_dataset, register_probe):
        """A plain property that overrides an inherited dependency has unknown targets and still builds."""
        from bewer.metrics.general.wil import WIL

        class Probe(WIL):
            @property
            def _wip(self):
                return self.dataset.metrics.wip()

        assert Probe.dependency_edges() == {"_wip": DependencyEdge(type="dependency", targets=None)}
        register_probe("_probe_plain_override", Probe)
        assert sample_dataset.metrics.get("_probe_plain_override")()._wip is sample_dataset.metrics.wip()

    def test_builds_target_with_owner_pipeline_and_forwarded_params(self, sample_dataset):
        """The built dependency is the shared instance for the owner's pipeline and params."""
        config = sample_dataset.config.clone().add_vocabulary(Vocabulary(name="terms", terms=["quick"]))
        dataset = Dataset(config=config)
        dataset.add("the quick brown fox", "the quick brown dog")
        ktr = dataset.metrics.key_term_recall(vocab="terms", partial_credit=True)
        expected = dataset.metrics._kt_outcomes(
            vocab="terms", allow_subset_matches=False, partial_credit=True, tokenizer="key_term"
        )
        assert ktr._kt_outcomes is expected

    def test_builds_selected_target(self, sample_dataset):
        """selected_by builds the metric named by the param value."""
        levenshtein_rate = sample_dataset.metrics.insertion_rate()
        error_align_rate = sample_dataset.metrics.insertion_rate(alignment="error_align")
        assert levenshtein_rate._alignment is sample_dataset.metrics.levenshtein()
        assert error_align_rate._alignment is sample_dataset.metrics.error_align()

    def test_unregistered_target_raises_on_construction(self, sample_dataset, register_probe):
        """A target missing from the registry fails when the metric is built, not on first access."""

        class Probe(Metric):
            short_name_base = long_name_base = description = "Probe"
            _missing = dependency("_no_such_metric")

        register_probe("_probe_missing", Probe)
        with pytest.raises(ValueError, match="Probe._missing depends on unregistered metric '_no_such_metric'"):
            sample_dataset.metrics.get("_probe_missing")()

    def test_forward_unknown_to_owner_raises(self, sample_dataset, register_probe):
        """Forwarding a param the owner does not have is an error."""

        class Probe(Metric):
            short_name_base = long_name_base = description = "Probe"
            _wip = dependency("wip", forward=("vocab",))

        register_probe("_probe_forward_owner", Probe)
        with pytest.raises(ValueError, match=r"Probe._wip forwards \['vocab'\], which Probe does not accept"):
            sample_dataset.metrics.get("_probe_forward_owner")()

    def test_forward_unknown_to_target_raises(self, sample_dataset, register_probe):
        """Forwarding a param the target does not accept is an error."""

        @dataclass
        class Params(MetricParams):
            vocab: str = "terms"

        class Probe(Metric):
            short_name_base = long_name_base = description = "Probe"
            param_schema = Params
            _wip = dependency("wip", forward=("vocab",))

        register_probe("_probe_forward_target", Probe)
        with pytest.raises(ValueError, match=r"Probe._wip forwards \['vocab'\], which target WIP does not accept"):
            sample_dataset.metrics.get("_probe_forward_target")()

    def test_cycle_raises(self, sample_dataset, register_probe):
        """A dependency cycle is reported with its chain."""

        class ProbeA(Metric):
            short_name_base = long_name_base = description = "ProbeA"
            _b = dependency("_probe_cycle_b")

        class ProbeB(Metric):
            short_name_base = long_name_base = description = "ProbeB"
            _a = dependency("_probe_cycle_a")

        register_probe("_probe_cycle_a", ProbeA)
        register_probe("_probe_cycle_b", ProbeB)
        with pytest.raises(ValueError, match="Dependency cycle: ProbeA -> ProbeB -> ProbeA"):
            sample_dataset.metrics.get("_probe_cycle_a")()

    def test_all_registered_graphs_are_valid(self):
        """Every built-in metric's static dependency graph checks cleanly."""
        for metric_cls in METRIC_REGISTRY.metric_classes.values():
            _check_dependency_graph(metric_cls)


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

    def test_invalid_pipeline_variant_raises_before_freeze(self, sample_dataset):
        """A typo in a pipeline variant name raises ValueError and does not freeze."""
        with pytest.raises(ValueError, match="Standardizer 'nonexistent' not found"):
            sample_dataset.metrics.wer(standardizer="nonexistent")
        assert sample_dataset.is_frozen is False

        with pytest.raises(ValueError, match="Tokenizer 'nonexistent' not found"):
            sample_dataset.metrics.wer(tokenizer="nonexistent")
        assert sample_dataset.is_frozen is False

        with pytest.raises(ValueError, match="Normalizer 'nonexistent' not found"):
            sample_dataset.metrics.wer(normalizer="nonexistent")
        assert sample_dataset.is_frozen is False

    def test_invalid_pipeline_variant_lists_available(self, sample_dataset):
        """The error message lists available variant names."""
        with pytest.raises(ValueError, match="Available:"):
            sample_dataset.metrics.wer(normalizer="defualt")


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
        from bewer.metrics.general.wer import WER

        metric = WER(name="test_wer", src=sample_dataset)
        assert metric.src is sample_dataset

    def test_init_standardizer(self, sample_dataset):
        """Test that standardizer can be set via __init__."""
        from bewer.metrics.general.wer import WER

        wer = WER(name="test", src=sample_dataset, standardizer="custom")
        assert wer._standardizer == "custom"

    def test_init_tokenizer(self, sample_dataset):
        """Test that tokenizer can be set via __init__."""
        from bewer.metrics.general.wer import WER

        wer = WER(name="test", src=sample_dataset, tokenizer="custom")
        assert wer._tokenizer == "custom"

    def test_init_normalizer(self, sample_dataset):
        """Test that normalizer can be set via __init__."""
        from bewer.metrics.general.wer import WER

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
        from bewer.metrics.general.wer import WER_

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

        assert _format_registered_params("insertion_rate") == "min_run_length=1, alignment='levenshtein'"

    def test_required_params_are_marked(self):
        """A parameter without a default is marked with a star, not a value."""
        from bewer.metrics.base import _format_registered_params

        summary = _format_registered_params("key_term_recall")

        assert summary.startswith("vocab*")
        assert "allow_subset_matches=False" in summary

    def test_reads_the_registry_not_an_instance(self):
        """Metrics with required params cannot be constructed, but are still describable."""
        from bewer.metrics.base import _format_registered_params

        # key_term_recall requires `vocab`, so no instance exists to introspect.
        assert "vocab*" in _format_registered_params("key_term_recall")

    def test_specialization_shows_its_own_default(self):
        """A registration default for a required param shows the value, unmarked."""
        from bewer.metrics import KTR
        from bewer.metrics.base import METRIC_REGISTRY, _format_registered_params

        METRIC_REGISTRY.register_metric(KTR, "_default_probe", vocab="terms", allow_override=True)
        try:
            summary = _format_registered_params("_default_probe")
        finally:
            del METRIC_REGISTRY.metric_metadata["_default_probe"]
            del METRIC_REGISTRY.metric_classes["_default_probe"]

        assert "vocab='terms'" in summary
        assert "*" not in summary

    def test_registration_default_overrides_the_schema_default(self):
        """A default supplied at registration wins over the one on param_schema."""
        from bewer.metrics.base import METRIC_REGISTRY, _format_registered_params

        metadata = METRIC_REGISTRY.metric_metadata["insertion_rate"]
        original = metadata["param_defaults"]
        metadata["param_defaults"] = {"min_run_length": 2}
        try:
            assert _format_registered_params("insertion_rate") == "min_run_length=2, alignment='levenshtein'"
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
            metric_property,
        )

        class _FactoryProbe_(ExampleMetric):
            @metric_property(type="main")
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
