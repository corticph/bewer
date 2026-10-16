"""Config registry for BeWER.

Provides ``CONFIG_REGISTRY``, a singleton ``ComponentRegistry`` for config
presets. Metrics use the separate ``MetricRegistry`` class in
``bewer.metrics.base``, exposed as ``METRIC_REGISTRY``.

Registration API
----------------

    CONFIG_REGISTRY.register("da", da_config)

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

        If the component has a ``freeze`` method, it is called before registration.
        Raises ValueError if the name is already registered.
        """
        if name in self._components:
            raise ValueError(f"{self._kind} '{name}' is already registered")
        if hasattr(component, "freeze"):
            component.freeze()
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

    @contextmanager
    def isolated(self):
        """Snapshot and restore all registered components.

        Useful for tests that register temporary components.
        """
        snapshot = dict(self._components)
        try:
            yield self
        finally:
            self._components.clear()
            self._components.update(snapshot)


CONFIG_REGISTRY = ComponentRegistry("config")
