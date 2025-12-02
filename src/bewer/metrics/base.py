from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Hashable
from functools import partial
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset
    from bewer.core.example import Example


class Metric(ABC):
    def __init__(self, src: "Dataset", name: str, key: Optional[Hashable] = None):
        """Initialize the Metric object.

        Args:
            src (Dataset): The dataset to compute the metric for.
        """
        self.name = name
        self.key = key
        self._src_dataset = src
        self._examples = {}
        self._members = {}

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
    def example_cls(self) -> type["ExampleMetric"] | None:
        """Get the ExampleMetric class associated with this Metric."""
        return None

    def _get_example_metric(self, example: "Example") -> "ExampleMetric":
        """Get the ExampleMetric object for a given example index."""
        if example._index in self._examples:
            return self._examples[example._index]
        if self.example_cls is None:
            return None
        example_metric = self.example_cls(example, self)
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
        example: "Example",
        src_metric: "Metric",
        key: Optional[Hashable] = None,
    ):
        """Initialize the Metric object.

        Args:
            src (Example): The Example to compute the metric for.
        """
        self.example = example
        self.src_metric = src_metric
        self.key = key
        self._members = {}


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

    def get(self, name: str) -> Metric:
        """Get a metric by name.

        If the metric is not already computed, it will be created and set as an attribute of the MetricCollection.
        """
        if name in self._cache:
            return self._cache[name]
        elif name in METRIC_REGISTRY.metrics:
            metric_factory = METRIC_REGISTRY.metrics[name]
            metric_instance = metric_factory(self._src)
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
        elif name in METRIC_REGISTRY.metrics:
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
    def __init__(self):
        self.metrics = {}

    def register_metric(
        self,
        metric_cls: type["Metric"],
        name: str | None = None,
        allow_override: bool = False,
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
        if name in self.metrics and not allow_override:
            raise ValueError(f"Metric '{name}' already registered.")
        metric_factory = partial(metric_cls, name=name, **kwargs) if kwargs else partial(metric_cls, name=name)
        self.metrics[name] = metric_factory

        # Validate that all parameters (except the first) have default values.
        sig = inspect.signature(metric_factory)
        params = list(sig.parameters.values())  # NOTE: `self` not included when wrapped with partial.
        if len(params) == 0 or params[0].default is not inspect.Parameter.empty:
            raise ValueError("Metric '__init__' must have 'src' as the first parameter without a default value.")
        for param in params[1:]:
            if param.default is inspect.Parameter.empty:
                raise ValueError(f"Parameter '{param.name}' for metric '{name}' must have a default value.")

    def register(self, name: str | None = None, allow_override: bool = False, **kwargs):
        """
        Decorator version of 'register_metric'.

        Usage:
            @registry.register()
            class Foo(...): ...

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
                **kwargs,
            )
            return metric_cls

        return decorator


METRIC_REGISTRY = MetricRegistry()


def list_registered_metrics() -> list[str]:
    """List all registered metric names."""
    return [metric for metric in METRIC_REGISTRY.metrics.keys() if not metric.startswith("_")]
