"""Unified registry for BeWER pluggable components.

This module provides a single ``REGISTRY`` singleton with typed namespaces
for configs.  Metrics continue to use the existing ``MetricRegistry`` class,
exposed as ``REGISTRY.metrics``.

Registration API
----------------

    REGISTRY.configs.register("da", da_config)

Preprocessing functions (transforms, tokenizers) and vocabularies are NOT
registered — they are passed directly in ``BewerConfig``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


class ComponentRegistry:
    """A name -> component registry."""

    def __init__(self, kind: str):
        self._kind = kind
        self._components: dict[str, Any] = {}

    def register(self, name: str, component: Any) -> Any:
        """Register a component under ``name``.

        Raises ValueError if the name is already registered.
        """
        if name in self._components:
            raise ValueError(f"{self._kind} '{name}' is already registered")
        self._components[name] = component
        return component

    def get(self, name: str) -> Any:
        """Retrieve a component by name. Raises KeyError if missing."""
        if name not in self._components:
            available = sorted(self._components)
            raise KeyError(f"{self._kind} '{name}' not found. Available: {available}")
        return self._components[name]

    def list(self) -> list[str]:
        """Return sorted names of all registered components."""
        return sorted(self._components)

    def __contains__(self, name: str) -> bool:
        return name in self._components

    def __repr__(self) -> str:
        return f"ComponentRegistry({self._kind!r}, {list(self._components)})"


class Registry:
    """Top-level registry holding all typed namespaces."""

    def __init__(self):
        self.configs = ComponentRegistry("config")
        self._metrics = None

    @property
    def metrics(self):
        if self._metrics is None:
            from bewer.metrics.base import METRIC_REGISTRY

            self._metrics = METRIC_REGISTRY
        return self._metrics

    @contextmanager
    def isolated(self):
        """Snapshot and restore all non-metric namespaces.

        Useful for tests that register temporary components.
        """
        snapshots = {ns: dict(getattr(self, ns)._components) for ns in ("configs",)}
        try:
            yield
        finally:
            for ns, snap in snapshots.items():
                getattr(self, ns)._components.clear()
                getattr(self, ns)._components.update(snap)


REGISTRY = Registry()
