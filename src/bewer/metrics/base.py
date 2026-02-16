from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from functools import cached_property, update_wrapper
from typing import TYPE_CHECKING, Any, Optional, Union

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

        # Validate required params before computing (only for Metric, not ExampleMetric)
        if hasattr(obj, "_validate_required_params"):
            obj._validate_required_params()

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


class Metric(ABC):
    example_cls: type["ExampleMetric"] | None = None
    _param_schema: dict[str, type] | None = None  # Deprecated: use _hyperparams
    _hyperparams: dict[str, type | tuple[type, Any]] | None = None
    _short_name_base: str
    _long_name_base: str

    def __init__(
        self,
        name: str,
        src: Optional["Dataset"] = None,
        **params,
    ):
        """Initialize the Metric object.

        Args:
            name: Metric name.
            src: Parent Dataset object. Can be set later via set_source().
            **params: Optional parameters for metric configuration.
        """
        self.name = name

        # Apply defaults from _hyperparams
        if self._hyperparams is not None:
            for param_name, param_spec in self._hyperparams.items():
                # Parse spec: type or (type, default)
                if isinstance(param_spec, tuple):
                    param_type, default_value = param_spec
                    # Apply default if not provided
                    if param_name not in params:
                        params[param_name] = default_value
                # else: just type, means required (no default applied)

        self.params = params
        self._examples = {}
        self._param_cache = {}

        self._standardizer = DEFAULT
        self._tokenizer = DEFAULT
        self._normalizer = DEFAULT

        # Validate parameters
        if params:
            self._validate_params(params)

        self._src = None
        if src is not None:
            self.set_source(src)

    @property
    def short_name(self) -> str:
        """Get the short name, including parameters if present."""
        if not self.params:
            return self._short_name_base

        # Format parameters for display
        param_strs = [f"{k}={v}" for k, v in self.params.items()]
        return f"{self._short_name_base} ({', '.join(param_strs)})"

    @property
    def long_name(self) -> str:
        """Get the long name, including parameters if present."""
        if not self.params:
            return self._long_name_base

        param_strs = [f"{k}={v}" for k, v in self.params.items()]
        return f"{self._long_name_base} ({', '.join(param_strs)})"

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
        """Alias for src property for backward compatibility."""
        return self._src

    @property
    def pipeline(self) -> tuple[str, str, str]:
        """Get the preprocessing pipeline for the metric. Cached property to ensure immutability."""
        return (self._standardizer, self._tokenizer, self._normalizer)

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

        # Validate parameters against dataset if present
        if self.params:
            self._validate_params_with_dataset(self.params, src)

    def set_standardizer(self, standardizer: str):
        """Set the standardizer for the metric."""
        # TODO: Validate standardizer
        self._standardizer = standardizer

    def set_tokenizer(self, tokenizer: str):
        """Set the tokenizer for the metric."""
        # TODO: Validate tokenizer
        self._tokenizer = tokenizer

    def set_normalizer(self, normalizer: str):
        """Set the normalizer for the metric."""
        # TODO: Validate normalizer
        self._normalizer = normalizer

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

    @staticmethod
    def _make_cache_key(**params) -> tuple:
        """Convert parameters to canonical hashable cache key.

        All parameter values must be hashable (int, float, str, bool, tuple, etc.).
        Non-hashable types like lists or dicts will raise TypeError.

        Args:
            **params: Parameters to convert to cache key. All values must be hashable.

        Returns:
            tuple: Tuple of sorted (key, value) pairs for use as cache key.
        """
        if not params:
            return ()
        # Sort keys for canonical representation and return as tuple
        return tuple(sorted(params.items()))

    def with_params(self, **new_params) -> "Metric":
        """Create or retrieve cached instance with specified parameters.

        Args:
            **new_params: Parameters for the metric variant. All values must be hashable.

        Returns:
            Metric instance configured with the specified parameters.

        Raises:
            TypeError: If any parameter value is not hashable (e.g., list, dict, set).

        Example:
            >>> wer_threshold = dataset.metrics.wer.with_params(threshold=0.5)
            >>> wer_threshold.value
        """
        # Merge params: current params + new params (new overrides)
        merged_params = {**self.params, **new_params}

        # Create cache key and check cache
        # This will raise TypeError if params are not hashable
        try:
            cache_key = self._make_cache_key(**merged_params)
            # Check cache - this is where TypeError actually happens for non-hashable values
            if cache_key in self._param_cache:
                return self._param_cache[cache_key]
        except TypeError as e:
            # Find which parameter is non-hashable for better error message
            non_hashable = []
            for key, value in merged_params.items():
                try:
                    hash(value)
                except TypeError:
                    non_hashable.append(f"{key} ({type(value).__name__})")
            raise TypeError(
                f"All metric parameters must be hashable. Non-hashable parameters: {', '.join(non_hashable)}. "
                f"Use hashable alternatives: tuple instead of list, frozenset instead of set, etc."
            ) from e

        # Validate parameters if schema defined
        if self._param_schema is not None:
            self._validate_params(merged_params)

        # Create new instance with merged params
        new_instance = self.__class__(name=self.name, src=self._src, **merged_params)

        # Copy preprocessing pipeline settings
        new_instance._standardizer = self._standardizer
        new_instance._tokenizer = self._tokenizer
        new_instance._normalizer = self._normalizer

        # Cache and return
        self._param_cache[cache_key] = new_instance
        return new_instance

    def _validate_params(self, params: dict) -> None:
        """Validate parameters against hyperparam definition.

        Args:
            params: The parameters to validate.

        Raises:
            ValueError: If parameter name is unknown or no params accepted.
            TypeError: If parameter type doesn't match schema.
        """
        # Use _hyperparams if defined, otherwise fall back to _param_schema
        param_def = self._hyperparams if self._hyperparams is not None else self._param_schema

        # If no param definition, reject all params
        if param_def is None:
            if params:
                raise ValueError(f"Metric {self._short_name_base} does not accept parameters")
            return

        # Check for unknown params
        for param_name in params:
            if param_name not in param_def:
                valid_params = list(param_def.keys())
                raise ValueError(
                    f"Unknown parameter '{param_name}' for {self._short_name_base}. Valid parameters: {valid_params}"
                )

        # Type validation
        for param_name, param_value in params.items():
            param_spec = param_def[param_name]
            # Extract type from spec (could be just type or (type, default))
            param_type = param_spec[0] if isinstance(param_spec, tuple) else param_spec

            if not isinstance(param_value, param_type):
                raise TypeError(
                    f"Parameter '{param_name}' must be {param_type.__name__}, got {type(param_value).__name__}"
                )

    def _validate_params_with_dataset(self, params: dict, dataset: "Dataset") -> None:
        """Optional hook for dataset-aware validation.

        Override in subclasses that need to validate parameters against dataset state.

        Args:
            params: The parameters to validate.
            dataset: The source dataset.

        Raises:
            ValueError: If parameters are invalid for this dataset.
        """
        pass  # Default: no dataset-specific validation

    def _validate_required_params(self) -> None:
        """Ensure all required parameters are present.

        Required parameters are those defined in _hyperparams without defaults.

        Raises:
            ValueError: If required parameters are missing.
        """
        # Use _hyperparams if defined, otherwise fall back to _param_schema
        param_def = self._hyperparams if self._hyperparams is not None else self._param_schema

        if param_def is None:
            return

        # Find required params (those without defaults)
        required = {
            name
            for name, spec in param_def.items()
            if not isinstance(spec, tuple)  # No tuple = no default = required
        }

        missing = required - set(self.params.keys())
        if missing:
            missing_list = sorted(missing)
            param_hints = ", ".join(f"{p}=..." for p in missing_list)
            raise ValueError(
                f"Missing required parameters for {self._short_name_base}: {missing_list}. "
                f"Use .with_params({param_hints}) to set them."
            )


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
    def params(self) -> dict:
        """Access parent metric's parameters."""
        return self.parent_metric.params

    @property
    def src(self) -> Optional["Example"]:
        """Get the parent Example object."""
        return self._src

    @property
    def example(self) -> Optional["Example"]:
        """Alias for src property for backward compatibility."""
        return self._src

    @property
    def pipeline(self) -> tuple[str, str, str]:
        """Get the preprocessing pipeline for the metric."""
        return self.parent_metric.pipeline

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
        self._cache = {}

    def list_metrics(self, show_private: bool = False) -> None:
        """Print all registered example metric and their values."""
        metric_rows = []
        for metric_name, metric_cls in METRIC_REGISTRY.metric_classes.items():
            if not show_private and metric_name.startswith("_"):
                continue
            metric_rows.append((metric_name, metric_cls._get_row_values()))
        print_metric_table(metric_rows)

    def get(self, name: str) -> Metric:
        """Get a metric by name.

        If the metric is not already computed, it will be created and set as an attribute of the MetricCollection.
        """
        if name in self._cache:
            return self._cache[name]
        elif name in METRIC_REGISTRY.metric_factories:
            metric_factory = METRIC_REGISTRY.metric_factories[name]
            metric_instance = metric_factory()
            metric_instance.set_source(self._src)
            self._cache[name] = metric_instance
            return metric_instance
        else:
            raise AttributeError(f"Metric '{name}' not found.")

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

    def get(self, name: str) -> ExampleMetric:
        """Get a metric by name.

        If the metric is not already computed, it will be created and cached.

        Metrics without an ExampleMetric counterpart will return None.
        """
        if name in self._cache:
            return self._cache[name]
        elif name in METRIC_REGISTRY.metric_factories:
            metric = self._src_collection.__getattr__(name)
            example_metric_instance = metric._get_example_metric(self._src_example)
            self._cache[name] = example_metric_instance
            return example_metric_instance
        else:
            raise AttributeError(f"Metric '{name}' not found.")

    def __getattr__(self, name: str):
        # NOTE: This method is only called if the attribute is not found the usual ways.
        return self.get(name)

    def __repr__(self):
        # TODO: Improve representation to list available metrics and their values.
        return "ExampleMetricCollection()"


class MetricRegistry:
    def __init__(self) -> None:
        self.metric_factories = {}
        self.metric_classes = {}

    def register_metric(
        self,
        metric_cls: type["Metric"],
        name: str | None = None,
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
            name (str | None): The name of the metric. If None, the attr_name of the metric class will be used.
            allow_override (bool): Whether to allow overriding an existing metric with the same name.
            **kwargs: Additional keyword arguments to pass to the metric class upon instantiation.
        """

        # Validate metric class.
        if not issubclass(metric_cls, Metric):
            raise TypeError(f"Metric class {metric_cls.__name__} must inherit from Metric.")

        # Validate metric name.
        if not isinstance(name, str):
            raise TypeError("Metric name must be a string or None.")

        # Register metric based on its type.
        if name in self.metric_factories and not allow_override:
            raise ValueError(f"Metric '{name}' already registered.")

        def metric_factory():
            metric = metric_cls(name=name, **kwargs)
            metric.set_standardizer(standardizer)
            metric.set_tokenizer(tokenizer)
            metric.set_normalizer(normalizer)
            return metric

        # Validate that all parameters (except the first) have default values.
        sig = inspect.signature(metric_factory)
        params = list(sig.parameters.values())  # NOTE: `self` not included when wrapped with partial.
        for param in params:
            if param.default is inspect.Parameter.empty:
                raise ValueError(f"Parameter '{param.name}' for metric '{name}' must have a default value.")

        self.metric_factories[name] = metric_factory
        self.metric_classes[name] = metric_cls

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
