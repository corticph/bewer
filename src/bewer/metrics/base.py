from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, fields
from enum import StrEnum
from functools import cached_property, update_wrapper
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Literal,
    NamedTuple,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from typeguard import check_type

from bewer.flags import DEFAULT, NONE, NORMALIZERS, STANDARDIZERS, TOKENIZERS
from bewer.preprocessing.context import set_pipeline
from bewer.reporting.python.tables import print_metric_table

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset
    from bewer.core.example import Example

__all__ = [
    "Metric",
    "ExampleMetric",
    "MetricParams",
    "metric_property",
    "MetricPropertyType",
    "dependency",
    "Dependency",
    "DependencyEdge",
    "METRIC_REGISTRY",
    "MetricRegistry",
    "list_registered_metrics",
    "MetricCollection",
    "ExampleMetricCollection",
    "ConfidenceInterval",
]


class MetricPropertyType(StrEnum):
    """The kinds of ``metric_property``.

    Members compare equal to their values, so ``type="main"`` and
    ``type=MetricPropertyType.MAIN`` are interchangeable.
    """

    VALUE = "value"
    """A metric value, listed for introspection."""
    MAIN = "main"
    """The metric's main value (at most one per class); also a value."""
    DEPENDENCY = "dependency"
    """A borrowed metric this metric is computed from."""
    ALIGNMENT = "alignment"
    """A dependency whose target is an alignment metric."""

    @property
    def is_value(self) -> bool:
        """Whether properties of this type are registered as metric values."""
        return self in (MetricPropertyType.VALUE, MetricPropertyType.MAIN)

    @property
    def is_dependency(self) -> bool:
        """Whether properties of this type are registered as dependencies."""
        return self in (MetricPropertyType.DEPENDENCY, MetricPropertyType.ALIGNMENT)

    @classmethod
    def parse(cls, value: "MetricPropertyType | str") -> "MetricPropertyType":
        """Convert a member or its string value to a member.

        Raises:
            ValueError: If ``value`` names no member.
        """
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Unknown metric_property type {value!r}. Valid types: {[m.value for m in cls]}."
            ) from None


class metric_property(cached_property):
    """Unified decorator for metric properties.

    Combines caching, pipeline context, value classification, and dependency
    registration into a single decorator.

    Common usage patterns::

        @metric_property(type="main")           # the metric's main value
        @metric_property(type="value")          # a regular metric value
        @metric_property                        # internal computed property (caching + pipeline only)

    Declare dependencies with :func:`dependency` instead of a decorated function. A
    dependency that needs custom logic can still use the function form, and should
    name the registry metrics it can return in ``targets``::

        @metric_property(type="dependency", targets=("wip",))
        def _wip(self): ...

    Args:
        type: The kind of property: a ``MetricPropertyType`` member or its string value.
            ``None`` makes an internal computed property that is cached but not registered.
        private: Explicitly control privacy of a value. If None, inferred from a
            leading underscore.
        set_pipeline: Wrap computation in ``set_pipeline(*obj.pipeline)``. Default True.
            Dependencies never set the pipeline: the borrowed metric sets its own.
        targets: Function-form dependencies only. The registry names the function can
            return, so the dependency graph is known without building metrics.
    """

    def __init__(
        self,
        func=None,
        *,
        type: MetricPropertyType | str | None = None,
        private: bool | None = None,
        set_pipeline: bool = True,
        targets: tuple[str, ...] | None = None,
    ):
        self._type = MetricPropertyType.parse(type) if type is not None else None
        is_dependency_type = self._type is not None and self._type.is_dependency
        if targets is not None and not is_dependency_type:
            raise ValueError("`targets` applies only to dependency types ('dependency', 'alignment').")
        self._private = private
        self._set_pipeline = set_pipeline and not is_dependency_type
        self._targets = tuple(targets) if targets is not None else None
        if func is not None:
            super().__init__(func)
            update_wrapper(self, func)

    def __call__(self, func):
        super().__init__(func)
        update_wrapper(self, func)
        return self

    @property
    def type(self) -> MetricPropertyType | None:
        """The kind of property, or None for an internal computed property."""
        return self._type

    def dependency_targets(self, owner: type) -> tuple[str, ...] | None:
        """Get the registry names this dependency can resolve to, without building it.

        Args:
            owner: The metric class the dependency is read through.

        Returns:
            The registry names, or None when a function-form dependency declares no targets.
        """
        return self._targets

    def __set_name__(self, owner: type, name: str):
        super().__set_name__(owner, name)

        if self._type is None:
            return  # Internal computed property: cached, not registered

        if self._type.is_dependency:
            deps = owner.__dict__.get("_dependencies")
            if deps is None:
                deps = []
                owner._dependencies = deps
            deps.append(name)
            return  # Dependencies are not values

        if self._type.is_value:
            # Determine privacy: explicit flag takes precedence, otherwise infer from leading underscore.
            private = self._private if self._private is not None else name.startswith("_")

            metric_values = owner.__dict__.get("_metric_values")
            if metric_values is None:
                metric_values = {"other": [], "main": None, "private": []}
                owner._metric_values = metric_values

            if self._type is MetricPropertyType.MAIN:
                if metric_values["main"] is not None:
                    raise ValueError(f"Multiple main metric values defined in {owner.__name__}.")
                metric_values["main"] = name
            elif private:
                metric_values["private"].append(name)
            else:
                metric_values["other"].append(name)

    def __get__(self, obj: Optional[Any], objtype: Optional[type] = None) -> Any:
        if obj is None:
            return self

        name = self.attrname
        if name in obj.__dict__:
            return obj.__dict__[name]

        if self._set_pipeline:
            with set_pipeline(*obj.pipeline):
                value = self.func(obj)
        else:
            value = self.func(obj)

        obj.__dict__[name] = value
        return value


class Dependency(metric_property):
    """A metric borrowed by registry name, built from the owner's pipeline and params.

    Create it with :func:`dependency`. On first access it builds the target through
    ``dataset.metrics`` with the owner's standardizer, tokenizer and normalizer, plus
    the forwarded params, and caches the instance like any ``metric_property``.
    """

    def __init__(
        self,
        target: str | None,
        *,
        selected_by: str | None,
        forward: tuple[str, ...],
        type: MetricPropertyType | str,
    ):
        if (target is None) == (selected_by is None):
            raise ValueError("Pass exactly one of `target` or `selected_by`.")
        type = MetricPropertyType.parse(type)
        if not type.is_dependency:
            valid = [m.value for m in MetricPropertyType if m.is_dependency]
            raise ValueError(f"A dependency's type must be one of {valid}, got {type.value!r}.")
        self.target = target
        self.selected_by = selected_by
        self.forward = tuple(forward)
        super().__init__(self._build, type=type)
        source = repr(target) if target is not None else f"the value of params.{selected_by}"
        self.__doc__ = f"Borrowed metric {source}, built with this metric's pipeline."

    def dependency_targets(self, owner: type) -> tuple[str, ...]:
        """Get the registry names this dependency can resolve to, without building it.

        A ``target`` dependency has exactly one. A ``selected_by`` dependency has the
        values allowed by its param's ``Literal`` type in ``owner.param_schema``.

        Args:
            owner: The metric class the dependency is read through.

        Raises:
            TypeError: If the ``selected_by`` param is missing or not ``Literal``-typed.
        """
        if self.target is not None:
            return (self.target,)
        schema = getattr(owner, "param_schema", None)
        hint = get_type_hints(schema).get(self.selected_by) if schema is not None else None
        if get_origin(hint) is not Literal:
            raise TypeError(
                f"{owner.__name__}.{self.attrname}: selected_by={self.selected_by!r} needs a param "
                f"typed Literal[...] in param_schema, listing the registry names it can select."
            )
        return get_args(hint)

    def _build(self, metric: "Metric") -> "Metric":
        """Build the borrowed metric for one owner instance."""
        name = self.target if self.target is not None else getattr(metric.params, self.selected_by)
        forwarded = {param: getattr(metric.params, param) for param in self.forward}
        return metric.dataset.metrics.get(name)(
            **forwarded,
            standardizer=metric.standardizer,
            tokenizer=metric.tokenizer,
            normalizer=metric.normalizer,
        )


def dependency(
    target: str | None = None,
    *,
    selected_by: str | None = None,
    forward: tuple[str, ...] = (),
    type: MetricPropertyType | str = MetricPropertyType.DEPENDENCY,
) -> Dependency:
    """Declare a metric this metric is computed from.

    The borrowed metric always runs with the owner's pipeline (standardizer, tokenizer
    and normalizer). Name it in one of two ways:

    - ``target``: the registry name of the one metric to borrow.
    - ``selected_by``: the name of an owner param whose value is the registry name.
      The param must be typed ``Literal[...]``; its allowed values are the possible targets.

    Examples::

        _wip = dependency("wip")
        _kt_outcomes = dependency("_kt_outcomes", forward=("vocab", "allow_subset_matches"))
        _alignment = dependency("levenshtein", type="alignment")
        _alignment = dependency(selected_by="alignment", type="alignment")

    Args:
        target: Registry name of the borrowed metric.
        selected_by: Name of the owner param that holds the registry name.
        forward: Owner params passed to the borrowed metric under the same names.
        type: ``"dependency"``, or ``"alignment"`` when the target is an alignment metric.

    Returns:
        The dependency descriptor to assign as a class attribute.
    """
    return Dependency(target, selected_by=selected_by, forward=forward, type=type)


def _get_dependencies(cls) -> list[str]:
    """Get dependency metric names defined in the class and its bases."""
    deps = []
    for base in reversed(cls.__mro__):
        _deps = base.__dict__.get("_dependencies")
        if _deps:
            deps.extend(_deps)
    return list(dict.fromkeys(deps))


class DependencyEdge(NamedTuple):
    """A dependency of a metric class, known without building any metric.

    Attributes:
        type: ``MetricPropertyType.DEPENDENCY``, or ``MetricPropertyType.ALIGNMENT`` when
            the target is an alignment metric.
        targets: Registry names the dependency can resolve to, or None when a
            function-form dependency declares none.
    """

    type: MetricPropertyType
    targets: tuple[str, ...] | None


_CHECKED_DEPENDENCY_GRAPHS: set[type] = set()


def _check_dependency_graph(cls: type["Metric"]) -> None:
    """Check the static dependency graph reachable from a metric class.

    Runs once per class, and again after any new registration. Edges without declared
    targets (function-form dependencies) are skipped.

    Raises:
        ValueError: If a dependency names an unregistered metric, forwards a param that
            the owner or the target lacks, or the graph has a cycle.
    """
    if cls in _CHECKED_DEPENDENCY_GRAPHS:
        return

    def visit(metric_cls: type["Metric"], path: tuple[type["Metric"], ...]) -> None:
        if metric_cls in path:
            chain = " -> ".join(c.__name__ for c in (*path[path.index(metric_cls) :], metric_cls))
            raise ValueError(f"Dependency cycle: {chain}.")
        if metric_cls in _CHECKED_DEPENDENCY_GRAPHS:
            return
        path = (*path, metric_cls)
        for attr, edge in metric_cls.dependency_edges().items():
            descriptor = getattr(metric_cls, attr)
            forward = getattr(descriptor, "forward", ())
            _check_forwarded_params(metric_cls, attr, forward)
            for target in edge.targets or ():
                if target not in METRIC_REGISTRY.metric_classes:
                    raise ValueError(f"{metric_cls.__name__}.{attr} depends on unregistered metric {target!r}.")
                target_cls = METRIC_REGISTRY.metric_classes[target]
                _check_forwarded_params(target_cls, attr, forward, owner=metric_cls)
                visit(target_cls, path)
        _CHECKED_DEPENDENCY_GRAPHS.add(metric_cls)

    visit(cls, ())


def _check_forwarded_params(
    metric_cls: type["Metric"], attr: str, forward: tuple[str, ...], owner: type["Metric"] | None = None
) -> None:
    """Check that every forwarded param is a field of ``metric_cls.param_schema``.

    Args:
        metric_cls: The class whose params must include the forwarded names.
        attr: The dependency attribute, for the error message.
        forward: The forwarded param names.
        owner: The class declaring the dependency, when ``metric_cls`` is its target.
    """
    if not forward:
        return
    schema = metric_cls.param_schema
    known = {f.name for f in fields(schema)} if schema is not None else set()
    missing = [param for param in forward if param not in known]
    if missing:
        declared_on = (owner or metric_cls).__name__
        role = "target " + metric_cls.__name__ if owner is not None else metric_cls.__name__
        raise ValueError(f"{declared_on}.{attr} forwards {missing}, which {role} does not accept.")


def _get_metric_values(cls, include_private: bool = False) -> dict[str, Union[str, list[str]]]:
    """Get the metric values defined in the class and its bases."""
    main_value = None
    other_values = []
    private_values = []
    for base in reversed(cls.__mro__):
        _metric_values = base.__dict__.get("_metric_values")
        if _metric_values:
            main_value = _metric_values["main"] or main_value
            other_values.extend(_metric_values["other"])
            private_values.extend(_metric_values.get("private", []))
    result: dict[str, Any] = {"main": main_value, "other": list(dict.fromkeys(other_values))}
    if include_private:
        result["private"] = list(dict.fromkeys(private_values))
    return result


def _format_registered_params(name: str) -> str:
    """Render a registered metric's parameters for the metrics table.

    Reads the registry rather than an instance, so metrics with required parameters
    (which cannot be constructed without them) are still describable. Parameters with
    a default are shown as ``name=default``; required ones are marked with ``*``.

    ``param_schema`` stores a dataclass ``default_factory`` as the factory itself, so it is
    called here to show the value a user actually receives, matching ``resolve_params()``.

    Args:
        name: The registered metric name.

    Returns:
        str: Comma-separated parameters, or "-" if the metric takes none.
    """
    metadata = METRIC_REGISTRY.metric_metadata[name]
    schema = metadata["param_schema"]
    if not schema:
        return "-"
    registered_defaults = metadata["param_defaults"]
    parts = []
    for param_name, param_spec in schema.items():
        if param_name in registered_defaults:
            # A default supplied at registration overrides the one on param_schema.
            parts.append(f"{param_name}={registered_defaults[param_name]!r}")
        elif isinstance(param_spec, tuple):
            default = param_spec[1]
            parts.append(f"{param_name}={(default() if callable(default) else default)!r}")
        else:
            parts.append(f"{param_name}*")
    return ", ".join(parts)


def _get_metric_table_row_values(metric: "Metric") -> tuple[str, str]:
    metric_values = metric.metric_values()
    main_value = "-" if metric_values["main"] is None else metric_values["main"]
    other_values = "-" if len(metric_values["other"]) == 0 else ", ".join(metric_values["other"])
    return (main_value, other_values)


@dataclass
class MetricParams:
    """Base class for metric parameter dataclasses.

    Subclass this with @dataclass to define metric parameters:

        @dataclass
        class param_schema(MetricParams):
            threshold: float = 0.5

    The framework sets `_metric` after construction so that
    `self.metric` / `self.src` are available in `validate()`.
    """

    def __post_init__(self):
        hints = get_type_hints(type(self))
        for f in fields(self):
            value = getattr(self, f.name)
            expected_type = hints[f.name]
            if get_origin(expected_type) is Literal:
                allowed = get_args(expected_type)
                # Match the type as well, so that True and 1.0 do not pass as Literal[1].
                if not any(type(value) is type(option) and value == option for option in allowed):
                    raise ValueError(f"Parameter '{f.name}' must be one of {list(allowed)}, got {value!r}.")
                continue
            try:
                check_type(value, expected_type)
            except Exception:
                type_name = getattr(expected_type, "__name__", str(expected_type))
                raise TypeError(f"Parameter '{f.name}' must be {type_name}, got {type(value).__name__}")

    @property
    def metric(self) -> "Metric":
        """Alias for the src property."""
        return self._metric

    @property
    def src(self) -> "Metric":
        """The parent Metric instance."""
        return self._metric

    def validate(self) -> None:
        """Override to validate params against the source metric/dataset.

        Called from Metric.__init__ after src is set, so self.metric.src is the Dataset.
        """
        pass


class Metric(ABC):
    example_cls: type["ExampleMetric"] | None = None
    param_schema: type["MetricParams"] | None = None
    short_name_base: str
    long_name_base: str

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        src: "Dataset",
        standardizer: str = DEFAULT,
        tokenizer: str = DEFAULT,
        normalizer: str = DEFAULT,
        **params,
    ):
        """Initialize the Metric object.

        Args:
            name: Metric name. Defaults to the lowercase class name.
            src: Dataset object for computing the metric (required).
            standardizer: Standardizer pipeline name.
            tokenizer: Tokenizer pipeline name.
            normalizer: Normalizer pipeline name.
            **params: Optional parameters for metric configuration.
        """
        _check_dependency_graph(type(self))
        self.name = name or type(self).__name__.lower()
        self._examples = {}

        self._standardizer = standardizer
        self._tokenizer = tokenizer
        self._normalizer = normalizer
        self._pipeline = (standardizer, tokenizer, normalizer)

        if "normalized" in params:
            raise ValueError(
                "The 'normalized' metric parameter was removed. Pass normalizer='none' to skip token normalization."
            )

        # Construct params dataclass or reject unexpected params
        if self.param_schema is not None:
            try:
                self.params = self.param_schema(**params)
            except TypeError as e:
                dc_fields = {f.name for f in fields(self.param_schema)}
                unknown = set(params.keys()) - dc_fields
                if unknown:
                    raise ValueError(
                        f"Unknown parameter(s) {sorted(unknown)} for {self.short_name_base}. "
                        f"Valid parameters: {sorted(dc_fields)}"
                    ) from e
                missing = {
                    f.name for f in fields(self.param_schema) if f.default is MISSING and f.default_factory is MISSING
                } - set(params.keys())
                if missing:
                    param_hints = ", ".join(f"{p}=..." for p in sorted(missing))
                    raise ValueError(
                        f"Missing required parameters for {self.short_name_base}: {sorted(missing)}. "
                        f"Pass them as keyword arguments: metric({param_hints})"
                    ) from e
                raise
            self.params._metric = self
        else:
            if params:
                raise ValueError(f"Metric {self.short_name_base} does not accept parameters")
            self.params = None

        self._src = src
        if self.params is not None:
            self.params.validate()

    @property
    def short_name(self) -> str:
        """Get the short name, including parameters if present."""
        if self.params is None:
            return self.short_name_base
        param_strs = [f"{f.name}={getattr(self.params, f.name)}" for f in fields(self.params)]
        return f"{self.short_name_base} ({', '.join(param_strs)})"

    @property
    def long_name(self) -> str:
        """Get the long name, including parameters if present."""
        if self.params is None:
            return self.long_name_base
        param_strs = [f"{f.name}={getattr(self.params, f.name)}" for f in fields(self.params)]
        return f"{self.long_name_base} ({', '.join(param_strs)})"

    @property
    @abstractmethod
    def description(self) -> str:
        """Get a description of the metric."""
        pass

    @property
    def src(self) -> "Dataset":
        """Get the parent Dataset object."""
        return self._src

    @property
    def dataset(self) -> "Dataset":
        """Alias for src property."""
        return self._src

    @property
    def pipeline(self) -> tuple[str, str, str]:
        """Get the preprocessing pipeline for the metric. Cached property to ensure immutability."""
        return self._pipeline

    @property
    def standardizer(self) -> str:
        """Get the standardizer for the metric."""
        return self._standardizer

    @property
    def tokenizer(self) -> str:
        """Get the tokenizer for the metric."""
        return self._tokenizer

    @property
    def normalizer(self) -> str:
        """Get the normalizer for the metric."""
        return self._normalizer

    @classmethod
    def metric_values(cls, include_private: bool = False) -> dict[str, Union[str, list[str]]]:
        """Get the metric values defined in the class and its bases."""
        return _get_metric_values(cls, include_private=include_private)

    @classmethod
    def dependencies(cls) -> list[str]:
        """Get the dependency metric names defined in the class and its bases."""
        return _get_dependencies(cls)

    @classmethod
    def dependency_edges(cls) -> dict[str, DependencyEdge]:
        """Get each dependency's type and possible targets, without building any metric.

        A subclass can override an inherited dependency with a plain ``property`` or an
        untyped ``metric_property``. That override has no dependency metadata, so its edge
        has type ``MetricPropertyType.DEPENDENCY`` and unknown targets.

        Returns:
            A mapping from dependency attribute name to its ``DependencyEdge``.
        """
        edges = {}
        for name in cls.dependencies():
            descriptor = getattr(cls, name)
            prop_type = descriptor.type if isinstance(descriptor, metric_property) else None
            if prop_type is not None and prop_type.is_dependency:
                edges[name] = DependencyEdge(type=descriptor.type, targets=descriptor.dependency_targets(cls))
            else:
                edges[name] = DependencyEdge(type=MetricPropertyType.DEPENDENCY, targets=None)
        return edges

    @classmethod
    def _get_row_values(cls) -> tuple[str, str, str] | None:
        """Get the table row values for the main and example metric."""
        metric_row_values = _get_metric_table_row_values(cls)
        if cls.example_cls is not None:
            example_metric_row_values = _get_metric_table_row_values(cls.example_cls)
        else:
            example_metric_row_values = None
        return (metric_row_values, example_metric_row_values)

    def _example_metric(self, example: "Example") -> "ExampleMetric":
        """Get (creating and caching if needed) the ExampleMetric object for a
        given example. Internal helper backing the sequence protocol below."""
        if example.index in self._examples:
            return self._examples[example.index]
        example_metric = self.example_cls(parent_metric=self, src=example)
        self._examples[example.index] = example_metric
        return example_metric

    def __len__(self) -> int:
        if self.example_cls is None:
            raise TypeError(f"{type(self).__name__} has no example-level metric")
        return len(self._src)

    def __iter__(self) -> "Iterator[ExampleMetric]":
        if self.example_cls is None:
            raise TypeError(f"{type(self).__name__} has no example-level metric")
        return (self._example_metric(example) for example in self._src)

    def __getitem__(self, index):
        if self.example_cls is None:
            raise TypeError(f"{type(self).__name__} has no example-level metric")
        if isinstance(index, slice):
            return [self._example_metric(ex) for ex in self._src[index]]
        return self._example_metric(self._src[index])

    def compute_confidence_interval(
        self,
        level: float = 0.95,
        *,
        n_resamples: int = 1000,
        seed: Optional[int] = None,
        method: str = "bootstrap",
    ) -> "ConfidenceInterval":
        """Compute a confidence interval for this metric's main value via
        bootstrap resampling of examples (with replacement).

        Args:
            level: Confidence level in (0, 1). Default 0.95.
            n_resamples: Number of bootstrap resamples. Default 1000.
            seed: Optional seed for reproducibility.
            method: Currently only "bootstrap" is supported.

        Returns:
            A ConfidenceInterval. Iterable as (low, high) for tuple unpacking.

        Raises:
            ValueError: if level is not in (0, 1), n_resamples < 1, the metric
                has no main value, or the dataset is empty.
            TypeError: if the metric's main value is not numeric.
            NotImplementedError: if method is not "bootstrap".
        """
        if method != "bootstrap":
            raise NotImplementedError(f"Unknown CI method: {method!r}")
        from bewer.metrics.confidence import bootstrap_confidence_interval

        return bootstrap_confidence_interval(
            self,
            level=level,
            n_resamples=n_resamples,
            seed=seed,
        )


class ExampleMetric(ABC):
    def __init__(
        self,
        parent_metric: "Metric",
        *,
        src: "Example",
    ):
        """Initialize the ExampleMetric object.

        Args:
            parent_metric: The parent Metric object.
            src: Parent Example object (required).
        """
        self.parent_metric = parent_metric
        self._src = src

    @property
    def params(self) -> Optional["MetricParams"]:
        """Access parent metric's parameters."""
        return self.parent_metric.params

    @property
    def src(self) -> "Example":
        """Get the parent Example object."""
        return self._src

    @property
    def example(self) -> "Example":
        """Alias for src property."""
        return self._src

    @property
    def pipeline(self) -> tuple[str, str, str]:
        """Get the preprocessing pipeline for the metric."""
        return self.parent_metric.pipeline

    @property
    def standardizer(self) -> str:
        """Get the standardizer for the metric."""
        return self.parent_metric.standardizer

    @property
    def tokenizer(self) -> str:
        """Get the tokenizer for the metric."""
        return self.parent_metric.tokenizer

    @property
    def normalizer(self) -> str:
        """Get the normalizer for the metric."""
        return self.parent_metric.normalizer

    @classmethod
    def metric_values(cls, include_private: bool = False) -> dict[str, Union[str, list[str]]]:
        """Get the metric values defined in the class and its bases."""
        return _get_metric_values(cls, include_private=include_private)

    @classmethod
    def dependencies(cls) -> list[str]:
        """Get the dependency metric names defined in the class and its bases."""
        return _get_dependencies(cls)


_PIPELINE_STAGES = (
    ("standardizer", STANDARDIZERS),
    ("tokenizer", TOKENIZERS),
    ("normalizer", NORMALIZERS),
)


def _validate_pipeline_variants(resolved: dict, pipelines) -> None:
    """Check that resolved pipeline variant names exist in the dataset's pipelines.

    Raises ValueError with the available variant names if a name is not found.
    Called before metric creation so a typo does not freeze the dataset.
    """
    for stage_name, stage_attr in _PIPELINE_STAGES:
        variant = resolved.get(stage_name, DEFAULT)
        available = getattr(pipelines, stage_attr)
        if stage_name == "tokenizer" and variant == NONE:
            raise ValueError(f"Tokenizer {NONE!r} is not supported: tokenization cannot be skipped.")
        if variant not in available:
            raise ValueError(f"{stage_name.capitalize()} '{variant}' not found. Available: {sorted(available)}")


class MetricCollection(object):
    """Collection of metrics for a dataset or an example.

    Attributes:
        src (Union[Dataset, Example]): The source object (dataset or example) to compute metrics for.
        metrics (list[Metric]): A list of valid Metric objects depending on the source object.
    """

    def __init__(self, src: "Dataset"):
        """Initialize the MetricCollection object.

        Args:
            src (Union[Dataset, Example]): The source object (dataset or example) to compute metrics for.
        """
        self._src = src
        self._metric_cache = {}  # (name, kwargs_key) -> Metric instance

    @staticmethod
    def _make_cache_key(**kwargs) -> tuple:
        """Create a hashable cache key from kwargs.

        Args:
            **kwargs: Keyword arguments to hash. All values must be hashable.

        Returns:
            Tuple of sorted (key, value) pairs.
        """
        if not kwargs:
            return ()
        return tuple(sorted(kwargs.items()))

    def list_metrics(self, show_private: bool = False) -> None:
        """Print all registered metrics with their parameters and values."""
        metric_rows = []
        for metric_name, metric_cls in METRIC_REGISTRY.metric_classes.items():
            if not show_private and metric_name.startswith("_"):
                continue
            metric_rows.append((metric_name, _format_registered_params(metric_name), metric_cls._get_row_values()))
        print_metric_table(metric_rows)

    def get(self, name: str):
        """Get a metric factory function by name.

        Returns a callable that creates/caches metric instances with specified parameters.

        Args:
            name: The registered metric name.

        Returns:
            A factory function that accepts **kwargs and returns a Metric instance.

        Example:
            >>> wer_metric = dataset.metrics.get("wer")()
            >>> wer_value = wer_metric.value
        """
        if name not in METRIC_REGISTRY.metric_metadata:
            raise AttributeError(f"Metric '{name}' not found.")

        # Return factory function
        def metric_factory(**kwargs):
            # Resolve kwargs against all defaults for a canonical cache key
            resolved = METRIC_REGISTRY.resolve_params(name, **kwargs)
            try:
                cache_key = (name, self._make_cache_key(**resolved))
                if cache_key in self._metric_cache:
                    # A cached metric only exists because a prior successful request already
                    # froze the dataset, so the "cached implies frozen" invariant holds here.
                    return self._metric_cache[cache_key]
            except TypeError as e:
                # Find non-hashable parameters for better error message
                non_hashable = []
                for key, value in kwargs.items():
                    try:
                        hash(value)
                    except TypeError:
                        non_hashable.append(f"{key} ({type(value).__name__})")
                raise TypeError(
                    f"All metric parameters must be hashable. Non-hashable parameters: {', '.join(non_hashable)}. "
                    f"Use hashable alternatives: tuple instead of list, frozenset instead of set, etc."
                ) from e

            # Create new metric instance. This may raise on invalid/missing params; in that
            # case no metric was obtained, so the dataset must be left modifiable.
            # Validate pipeline variant names before creation so a typo doesn't freeze.
            _validate_pipeline_variants(resolved, self._src.config)
            metric_instance = METRIC_REGISTRY.create_metric(name, src=self._src, **kwargs)

            # A metric was successfully requested: freeze the dataset so its contents (and the
            # cached metric results derived from them) can no longer change underneath us.
            self._src.freeze()

            # Cache and return
            self._metric_cache[cache_key] = metric_instance
            return metric_instance

        return metric_factory

    def __getattr__(self, name: str):
        # NOTE: This method is only called if the attribute is not found the usual ways.
        return self.get(name)

    def __repr__(self):
        # TODO: Improve representation to list available metrics and their values.
        return "MetricCollection()"


class ExampleMetricCollection(object):
    """Collection of metrics for an example."""

    def __init__(self, src: "Example"):
        """Initialize the ExampleMetricCollection object."""
        self._src_example = src
        self._cache = {}

    @property
    def _src_collection(self):
        """The parent Dataset's MetricCollection.

        Resolved lazily so an Example built standalone (``src=None``) can still be
        constructed; only fails if a parent-dependent example metric is actually requested.
        """
        dataset = self._src_example.src
        if dataset is None:
            raise ValueError(
                "Dataset-backed metrics are unavailable for an Example constructed without a "
                "parent Dataset (src=None). Add the Example to a Dataset to compute metrics."
            )
        return dataset.metrics

    def get(self, name: str):
        """Get a metric factory function by name.

        Returns a callable that creates/caches example metric instances.

        Args:
            name: The registered metric name.

        Returns:
            A factory function that accepts **kwargs and returns an ExampleMetric instance.
        """
        if name not in METRIC_REGISTRY.metric_metadata:
            raise AttributeError(f"Metric '{name}' not found.")

        # Return factory function
        def example_metric_factory(**kwargs):
            # Resolve kwargs against all defaults for a canonical cache key
            resolved = METRIC_REGISTRY.resolve_params(name, **kwargs)
            try:
                cache_key = (name, MetricCollection._make_cache_key(**resolved))
                if cache_key in self._cache:
                    return self._cache[cache_key]
            except TypeError as e:
                non_hashable = []
                for key, value in kwargs.items():
                    try:
                        hash(value)
                    except TypeError:
                        non_hashable.append(f"{key} ({type(value).__name__})")
                raise TypeError(
                    f"All metric parameters must be hashable. Non-hashable parameters: {', '.join(non_hashable)}. "
                    f"Use hashable alternatives: tuple instead of list, frozenset instead of set, etc."
                ) from e

            # Get the parent metric (with same params)
            parent_metric_factory = self._src_collection.__getattr__(name)
            parent_metric = parent_metric_factory(**kwargs)

            # Get example metric from parent
            example_metric_instance = parent_metric[self._src_example.index]

            # Cache and return
            self._cache[cache_key] = example_metric_instance
            return example_metric_instance

        return example_metric_factory

    def __getattr__(self, name: str):
        # NOTE: This method is only called if the attribute is not found the usual ways.
        return self.get(name)

    def __repr__(self):
        # TODO: Improve representation to list available metrics and their values.
        return "ExampleMetricCollection()"


class MetricRegistry:
    def __init__(self) -> None:
        self.metric_metadata = {}  # name -> metadata dict
        self.metric_classes = {}  # name -> class (kept for compatibility)

    def __contains__(self, name: str) -> bool:
        return name in self.metric_metadata

    @property
    def metric_factories(self):
        """Backwards compatibility - returns dict of names."""
        return self.metric_metadata

    def register_metric(
        self,
        metric_cls: type["Metric"],
        name: str,
        allow_override: bool = False,
        standardizer: str = DEFAULT,
        tokenizer: str = DEFAULT,
        normalizer: str = DEFAULT,
        **kwargs,
    ):
        """
        Register a metric.

        Args:
            metric_cls (type): The metric class to register.
            name (str): The registered name for the metric.
            allow_override (bool): Whether to allow overriding an existing metric with the same name.
            standardizer (str): Default standardizer for the metric.
            tokenizer (str): Default tokenizer for the metric.
            normalizer (str): Default normalizer for the metric.
            **kwargs: Default parameters to pass to the metric class upon instantiation.
        """

        # Validate metric class.
        if not issubclass(metric_cls, Metric):
            raise TypeError(f"Metric class {metric_cls.__name__} must inherit from Metric.")

        # Validate metric name.
        if not isinstance(name, str):
            raise TypeError("Metric name must be a string.")

        # Register metric based on its type.
        if name in self.metric_metadata and not allow_override:
            raise ValueError(f"Metric '{name}' already registered.")

        # Extract parameter schema from dataclass fields
        if metric_cls.param_schema is not None:
            if not hasattr(metric_cls.param_schema, "__dataclass_fields__"):
                raise TypeError(f"param_schema on {metric_cls.__name__} must be a @dataclass inheriting MetricParams.")
            hints = get_type_hints(metric_cls.param_schema)
            param_schema = {}
            for f in fields(metric_cls.param_schema):
                field_type = hints[f.name]
                if f.default is not MISSING:
                    param_schema[f.name] = (field_type, f.default)
                elif f.default_factory is not MISSING:
                    param_schema[f.name] = (field_type, f.default_factory)
                else:
                    param_schema[f.name] = field_type
        else:
            param_schema = {}

        # Determine required parameters (those without defaults)
        required_params = {
            param_name
            for param_name, param_spec in param_schema.items()
            if not isinstance(param_spec, tuple)  # No tuple = no default
        }

        # Store metadata for factory creation
        metadata = {
            "metric_cls": metric_cls,
            "pipeline_defaults": {
                "standardizer": standardizer,
                "tokenizer": tokenizer,
                "normalizer": normalizer,
            },
            "param_defaults": kwargs,  # Params passed at registration
            "param_schema": param_schema,
            "required_params": required_params,
        }

        self.metric_metadata[name] = metadata
        self.metric_classes[name] = metric_cls
        # A new or replaced class can change any checked graph that reaches its name.
        _CHECKED_DEPENDENCY_GRAPHS.clear()

    def resolve_params(self, name: str, **kwargs) -> dict:
        """Resolve kwargs against all defaults to get canonical form.

        Merges pipeline defaults, registry param defaults, and class-level param_schema
        defaults with provided kwargs to produce a fully-resolved parameter dict.

        Args:
            name: The registered metric name.
            **kwargs: Raw kwargs (pipeline overrides + metric params).

        Returns:
            Dict with all resolved key-value pairs (pipeline + metric params).
        """
        metadata = self.metric_metadata[name]

        pipeline_args = {}
        metric_params = {}
        for key, value in kwargs.items():
            if key in ("standardizer", "tokenizer", "normalizer"):
                pipeline_args[key] = value
            else:
                metric_params[key] = value

        # Merge with registry-level defaults
        final_pipeline = {**metadata["pipeline_defaults"], **pipeline_args}
        final_params = {**metadata["param_defaults"], **metric_params}

        # Apply class-level param_schema defaults for any still-missing params
        for param_name, param_spec in metadata["param_schema"].items():
            if isinstance(param_spec, tuple) and param_name not in final_params:
                default = param_spec[1]
                final_params[param_name] = default() if callable(default) else default

        return {**final_pipeline, **final_params}

    def create_metric(self, name: str, *, src: "Dataset", **kwargs) -> "Metric":
        """Create a metric instance with merged defaults and overrides.

        Args:
            name: The registered metric name.
            src: The Dataset the metric is computed for (required).
            **kwargs: Parameters and pipeline overrides. Can include:
                - standardizer, tokenizer, normalizer (pipeline overrides)
                - Any metric-specific parameters

        Returns:
            Configured Metric instance.
        """
        if name not in self.metric_metadata:
            raise ValueError(f"Metric '{name}' not registered.")

        resolved = self.resolve_params(name, **kwargs)

        # Separate pipeline args from metric params
        pipeline_keys = ("standardizer", "tokenizer", "normalizer")
        pipeline_args = {k: resolved[k] for k in pipeline_keys}
        metric_params = {k: v for k, v in resolved.items() if k not in pipeline_keys}

        return self.metric_metadata[name]["metric_cls"](name=name, src=src, **pipeline_args, **metric_params)

    def register(
        self,
        name: str | None = None,
        allow_override: bool = False,
        standardizer: str = DEFAULT,
        tokenizer: str = DEFAULT,
        normalizer: str = DEFAULT,
        **kwargs,
    ):
        """
        Decorator version of 'register_metric'.

        Usage:
            @registry.register("foo")
            class Foo(...): ...

            @registry.register(name="foo", bar=3)
            class Foo(...): ...
        """

        def decorator(metric_cls):
            # Delegate to the existing register() method
            self.register_metric(
                metric_cls=metric_cls,
                name=name,
                allow_override=allow_override,
                standardizer=standardizer,
                tokenizer=tokenizer,
                normalizer=normalizer,
                **kwargs,
            )
            return metric_cls

        return decorator


METRIC_REGISTRY = MetricRegistry()


def list_registered_metrics(show_private: bool = False) -> list[str]:
    """List all registered metric names.

    Args:
        show_private (bool): Whether to include private metrics (those starting with an underscore).

    Returns:
        list[str]: List of registered metric names.
    """
    if show_private:
        return list(METRIC_REGISTRY.metric_factories.keys())
    else:
        return [name for name in METRIC_REGISTRY.metric_factories.keys() if not name.startswith("_")]


from bewer.metrics.confidence import ConfidenceInterval  # noqa: E402
