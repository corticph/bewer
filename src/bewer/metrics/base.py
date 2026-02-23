from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, fields
from functools import cached_property, update_wrapper
from typing import TYPE_CHECKING, Any, Optional, Union, get_type_hints

from typeguard import check_type

from bewer.flags import DEFAULT
from bewer.preprocessing.context import set_pipeline
from bewer.reporting.python.tables import print_metric_table

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset
    from bewer.core.example import Example


class metric_value(cached_property):
    def __init__(self, func=None, *, main: bool = False):
        self.main = main
        if func is not None:
            super().__init__(func)
            update_wrapper(self, func)

    def __call__(self, func):
        super().__init__(func)
        update_wrapper(self, func)
        return self

    def __set_name__(self, owner: type, name: str):
        super().__set_name__(owner, name)

        # Create or get the _metric_values dict on the owner class
        metric_values = owner.__dict__.get("_metric_values")
        if metric_values is None:
            metric_values = {"other": [], "main": None}
            owner._metric_values = metric_values

        # Update the metric values
        if self.main:
            if metric_values["main"] is not None:
                raise ValueError(f"Multiple main metric values defined in {owner.__name__}.")
            metric_values["main"] = name
        else:
            metric_values["other"].append(name)

    def __get__(self, obj: Optional[Any], objtype: Optional[type] = None) -> Any:
        if obj is None:
            return self

        # TODO: Figure out if this is necessary given cached_property behavior
        name = self.attrname
        if name in obj.__dict__:
            return obj.__dict__[name]

        with set_pipeline(*obj.pipeline):
            value = self.func(obj)

        obj.__dict__[name] = value
        return value


def _get_metric_values(cls) -> dict[str, Union[str, list[str]]]:
    """Get the metric values defined in the class and its bases."""
    main_value = None
    other_values = []
    for base in reversed(cls.__mro__):
        _metric_values = base.__dict__.get("_metric_values")
        if _metric_values:
            main_value = _metric_values["main"] or main_value
            other_values.extend(_metric_values["other"])
    metric_values = {"main": main_value, "other": list(set(other_values))}
    return metric_values


def _get_metric_table_row_values(metric: "Metric") -> tuple[str, str, str]:
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

        Called after set_source(), so self.metric.src is the Dataset.
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
        src: Optional["Dataset"] = None,
        *,
        standardizer: str = DEFAULT,
        tokenizer: str = DEFAULT,
        normalizer: str = DEFAULT,
        **params,
    ):
        """Initialize the Metric object.

        Args:
            name: Metric name. Defaults to the lowercase class name.
            src: Dataset object for computing the metric. Can be set later via set_source().
            standardizer: Standardizer pipeline name.
            tokenizer: Tokenizer pipeline name.
            normalizer: Normalizer pipeline name.
            **params: Optional parameters for metric configuration.
        """
        self.name = name or type(self).__name__.lower()
        self._examples = {}

        self._standardizer = standardizer
        self._tokenizer = tokenizer
        self._normalizer = normalizer
        self._pipeline = (standardizer, tokenizer, normalizer)

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

        self._src = None
        if src is not None:
            self.set_source(src)

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
    def src(self) -> Optional["Dataset"]:
        """Get the parent Dataset object."""
        return self._src

    @property
    def dataset(self) -> Optional["Dataset"]:
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
    def metric_values(cls) -> dict[str, Union[str, list[str]]]:
        """Get the metric values defined in the class and its bases."""
        return _get_metric_values(cls)

    @classmethod
    def _get_row_values(cls) -> tuple[str, str, str] | None:
        """Get the table row values for the main and example metric."""
        metric_row_values = _get_metric_table_row_values(cls)
        if cls.example_cls is not None:
            example_metric_row_values = _get_metric_table_row_values(cls.example_cls)
        else:
            example_metric_row_values = None
        return (metric_row_values, example_metric_row_values)

    def set_source(self, src: "Dataset") -> None:
        """Set the parent Dataset object and validate parameters if needed.

        Args:
            src: The parent Dataset object.

        Raises:
            ValueError: If source is already set or if parameters are invalid for this dataset.
        """
        if self._src is not None:
            raise ValueError("Source already set for Metric")
        self._src = src

        if self.params is not None:
            self.params.validate()

    def _get_example_metric(self, example: "Example") -> "ExampleMetric":
        """Get the ExampleMetric object for a given example index."""
        if example._index in self._examples:
            return self._examples[example._index]
        if self.example_cls is None:
            return None
        example_metric = self.example_cls(parent_metric=self)
        example_metric.set_source(example)
        self._examples[example._index] = example_metric
        return example_metric


class ExampleMetric(ABC):
    def __init__(
        self,
        parent_metric: "Metric",
        src: Optional["Example"] = None,
    ):
        """Initialize the ExampleMetric object.

        Args:
            parent_metric: The parent Metric object.
            src: Parent Example object. Can be set later via set_source().
        """
        self.parent_metric = parent_metric

        self._src = None
        if src is not None:
            self.set_source(src)

    @property
    def params(self) -> Optional["MetricParams"]:
        """Access parent metric's parameters."""
        return self.parent_metric.params

    @property
    def src(self) -> Optional["Example"]:
        """Get the parent Example object."""
        return self._src

    @property
    def example(self) -> Optional["Example"]:
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

    def set_source(self, src: "Example") -> None:
        """Set the parent Example object.

        Args:
            src: The parent Example object.

        Raises:
            ValueError: If source is already set.
        """
        if self._src is not None:
            raise ValueError("Source already set for ExampleMetric")
        self._src = src

    @classmethod
    def metric_values(cls) -> dict[str, Union[str, list[str]]]:
        """Get the metric values defined in the class and its bases."""
        return _get_metric_values(cls)


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
        """Print all registered example metric and their values."""
        metric_rows = []
        for metric_name, metric_cls in METRIC_REGISTRY.metric_classes.items():
            if not show_private and metric_name.startswith("_"):
                continue
            metric_rows.append((metric_name, metric_cls._get_row_values()))
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

            # Create new metric instance
            metric_instance = METRIC_REGISTRY.create_metric(name, **kwargs)
            metric_instance.set_source(self._src)

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
        self._src_collection = src.src.metrics if src.src is not None else None
        self._cache = {}

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
            example_metric_instance = parent_metric._get_example_metric(self._src_example)

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

    def create_metric(self, name: str, **kwargs) -> "Metric":
        """Create a metric instance with merged defaults and overrides.

        Args:
            name: The registered metric name.
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

        return self.metric_metadata[name]["metric_cls"](name=name, **pipeline_args, **metric_params)

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
