from __future__ import annotations

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
    _params: dict[str, type | tuple[type, Any]] | None = None
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

        # Apply defaults from _params
        if self._params is not None:
            for param_name, param_spec in self._params.items():
                # Parse spec: type or (type, default)
                if isinstance(param_spec, tuple):
                    param_type, default_value = param_spec
                    # Apply default if not provided
                    if param_name not in params:
                        params[param_name] = default_value
                # else: just type, means required (no default applied)

        self.params = params
        self._examples = {}

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

        # Format parameters in the order they're defined in _params
        if self._params is not None:
            param_strs = [f"{k}={self.params[k]}" for k in self._params if k in self.params]
        else:
            param_strs = [f"{k}={v}" for k, v in self.params.items()]
        return f"{self._short_name_base} ({', '.join(param_strs)})"

    @property
    def long_name(self) -> str:
        """Get the long name, including parameters if present."""
        if not self.params:
            return self._long_name_base

        # Format parameters in the order they're defined in _params
        if self._params is not None:
            param_strs = [f"{k}={self.params[k]}" for k in self._params if k in self.params]
        else:
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

    def _validate_params(self, params: dict) -> None:
        """Validate parameters against hyperparam definition.

        Args:
            params: The parameters to validate.

        Raises:
            ValueError: If parameter name is unknown or no params accepted.
            TypeError: If parameter type doesn't match schema.
        """
        # If no param definition, reject all params
        if self._params is None:
            if params:
                raise ValueError(f"Metric {self._short_name_base} does not accept parameters")
            return

        # Check for unknown params
        for param_name in params:
            if param_name not in self._params:
                valid_params = list(self._params.keys())
                raise ValueError(
                    f"Unknown parameter '{param_name}' for {self._short_name_base}. Valid parameters: {valid_params}"
                )

        # Type validation
        for param_name, param_value in params.items():
            param_spec = self._params[param_name]
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

        Required parameters are those defined in _params without defaults.

        Raises:
            ValueError: If required parameters are missing.
        """
        if self._params is None:
            return

        # Find required params (those without defaults)
        required = {
            name
            for name, spec in self._params.items()
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
            >>> wer_metric = dataset.metrics.get("wer")(threshold=0.5)
            >>> wer_value = wer_metric.value
        """
        if name not in METRIC_REGISTRY.metric_metadata:
            raise AttributeError(f"Metric '{name}' not found.")

        # Return factory function
        def metric_factory(**kwargs):
            # Create cache key from kwargs
            try:
                cache_key = (name, self._make_cache_key(**kwargs))
                # Check cache - this is where TypeError happens for non-hashable values
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
            # Create cache key from kwargs
            cache_key = (name, MetricCollection._make_cache_key(**kwargs))

            # Check cache
            if cache_key in self._cache:
                return self._cache[cache_key]

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
            raise TypeError("Metric name must be a string or None.")

        # Register metric based on its type.
        if name in self.metric_metadata and not allow_override:
            raise ValueError(f"Metric '{name}' already registered.")

        # Extract parameter schema from class
        param_schema = metric_cls._params or {}

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

        metadata = self.metric_metadata[name]

        # Separate pipeline args from metric params
        pipeline_args = {}
        metric_params = {}

        for key, value in kwargs.items():
            if key in ("standardizer", "tokenizer", "normalizer"):
                pipeline_args[key] = value
            else:
                metric_params[key] = value

        # Merge with defaults
        final_pipeline = {**metadata["pipeline_defaults"], **pipeline_args}
        final_params = {**metadata["param_defaults"], **metric_params}

        # Create metric instance
        metric = metadata["metric_cls"](name=name, **final_params)
        metric.set_standardizer(final_pipeline["standardizer"])
        metric.set_tokenizer(final_pipeline["tokenizer"])
        metric.set_normalizer(final_pipeline["normalizer"])

        return metric

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
