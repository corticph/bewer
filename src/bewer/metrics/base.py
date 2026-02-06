from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Hashable
from functools import cached_property, update_wrapper
from typing import TYPE_CHECKING, Any, Optional, Union

from bewer.flags import DEFAULT
from bewer.preprocessing.context import set_pipeline
from bewer.style.python.tables import print_metric_table

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


class Metric(ABC):
    example_cls: type["ExampleMetric"] | None = None

    def __init__(
        self,
        name: str,
        key: Optional[Hashable] = None,
    ):
        """Initialize the Metric object.

        Args:
            src (Dataset): The dataset to compute the metric for.
        """
        self.name = name
        self.key = key
        self._src_dataset = None
        self._examples = {}
        self._members = {}

        self._standardizer = DEFAULT
        self._tokenizer = DEFAULT
        self._normalizer = DEFAULT

    @property
    @abstractmethod
    def long_name(self) -> str:
        """Get the long/full name of the metric."""
        pass

    @property
    @abstractmethod
    def short_name(self) -> str:
        """Get the short name (e.g., an abbreviation) of the metric."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get a description of the metric."""
        pass

    @property
    def dataset(self) -> "Dataset":
        """Get the dataset for the metric."""
        return self._src_dataset

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

    def set_source(self, src: "Dataset"):
        """Set the source dataset for the metric."""
        self._src_dataset = src

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
        example_metric = self.example_cls(self)
        example_metric.set_source(example)
        self._examples[example._index] = example_metric
        return example_metric

    def is_valid_key(self, key: Hashable) -> bool:
        """Check if the filter is valid."""
        raise NotImplementedError(f"Metric '{self.short_name}' does not support filtering.")

    def __getitem__(self, key: Hashable) -> "Metric":
        """Get a metric by a hashable key."""
        if key in self._members:
            return self._members[key]
        elif self.is_valid_key(key):
            self._members[key] = self.__class__(self._src_dataset, key)
            return self._members[key]
        raise AttributeError(f"'{key}' is not a valid key for this metric.")


class ExampleMetric(ABC):
    def __init__(
        self,
        src_metric: "Metric",
        key: Optional[Hashable] = None,
    ):
        """Initialize the Metric object.

        Args:
            src (Example): The Example to compute the metric for.
        """

        self.src_metric = src_metric
        self.key = key
        self._src_example = None
        self._members = {}

    @property
    def example(self) -> "Example":
        """Get the example for the metric."""
        return self._src_example

    @property
    def pipeline(self) -> tuple[str, str, str]:
        """Get the preprocessing pipeline for the metric."""
        return self.src_metric.pipeline

    def set_source(self, src: "Example"):
        """Set the source example for the metric."""
        self._src_example = src

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
        self._src_collection = src._src_dataset.metrics
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
