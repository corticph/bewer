"""Unified registry for BeWER pluggable components.

This module provides a single ``REGISTRY`` singleton with typed namespaces
for extractors, vocabularies, and configs.  Metrics continue to use the
existing ``MetricRegistry`` class, exposed as ``REGISTRY.metrics``.

Registration API
----------------

All non-metric namespaces use ``ComponentRegistry.register``::

    REGISTRY.extractors.register("orthographically_complex", MyExtractor())
    REGISTRY.vocabularies.register("medical", vocab_instance)

    @REGISTRY.configs.register("da", extends="base")
    def danish_delta() -> BewerConfig:
        ...

Preprocessing functions (transforms, tokenizers) are NOT registered —
they are passed directly as callables in ``Transform``.
"""

from __future__ import annotations

import difflib
from contextlib import contextmanager
from typing import Any, Callable, Iterator


class ComponentNotFoundError(LookupError):
    """Raised when a component is not found in the registry.

    Includes a "did you mean?" hint when a close match exists.
    """

    def __init__(self, kind: str, name: str, available: list[str]):
        self.kind = kind
        self.name = name
        self.available = available
        matches = difflib.get_close_matches(name, available, n=1, cutoff=0.6)
        hint = f" Did you mean '{matches[0]}'?" if matches else ""
        super().__init__(f"{kind.capitalize()} '{name}' not found. Available: {sorted(available)}{hint}")


class ComponentRegistry:
    """A name -> component registry with decorator-friendly registration.

    Parameters
    ----------
    kind:
        Human-readable label for error messages (e.g. "transform").
    validate_fn:
        Optional callable invoked on the component at registration time.
        If it raises, registration is aborted.
    """

    def __init__(self, kind: str, *, validate_fn: Callable[[Any], None] | None = None):
        self._kind = kind
        self._validate_fn = validate_fn
        self._components: dict[str, Any] = {}

    def register(
        self,
        name: str | Callable | None = None,
        component: Any = None,
        *,
        allow_override: bool = False,
        **attrs: Any,
    ) -> Any:
        """Register a component, optionally as a decorator.

        Direct call::

            registry.register("my_func", my_func)
            registry.register("my_extractor", extractor_instance)

        Decorator (name defaults to ``fn.__name__``)::

            @registry.register
            def my_func(...): ...

            @registry.register("explicit_name")
            def my_func(...): ...

            @registry.register(custom_attr=True)
            def my_func(...): ...

        Extra ``**attrs`` are set as attributes on the component (useful for
        metadata like ``length_preserving`` or ``token_only``).

        Returns the component itself so decorator stacking works.
        """
        # Decorator form: @registry.register or @registry.register("name")
        if component is None and callable(name):
            component = name
            name = component.__name__

        def _do_register(n: str, c: Any) -> Any:
            if not allow_override and n in self._components:
                raise ValueError(f"{self._kind} '{n}' is already registered")
            if self._validate_fn is not None:
                self._validate_fn(c)
            for k, v in attrs.items():
                setattr(c, k, v)
            self._components[n] = c
            return c

        if component is not None:
            return _do_register(name, component)

        # @registry.register("name", ...) — returns decorator
        def decorator(fn: Callable) -> Callable:
            return _do_register(name or fn.__name__, fn)

        return decorator

    def get(self, name: str) -> Any:
        """Retrieve a component by name. Raises ComponentNotFoundError if missing."""
        if name not in self._components:
            raise ComponentNotFoundError(self._kind, name, list(self._components))
        return self._components[name]

    def list(self) -> list[str]:
        """Return sorted names of all registered components."""
        return sorted(self._components)

    def __contains__(self, name: str) -> bool:
        return name in self._components

    def __iter__(self) -> Iterator[str]:
        return iter(self._components)

    def __len__(self) -> int:
        return len(self._components)

    def __repr__(self) -> str:
        return f"ComponentRegistry({self._kind!r}, {list(self._components)})"


class Registry:
    """Top-level registry holding all typed namespaces.

    Attributes
    ----------
    extractors:
        Extractor instances (``dataset -> Iterable[str]``).
    vocabularies:
        ``Vocabulary`` instances (frozen on registration).
    configs:
        Config functions returning ``BewerConfig`` deltas.
    metrics:
        The existing ``MetricRegistry`` (not a ``ComponentRegistry``).
    """

    def __init__(self):
        self.extractors = ComponentRegistry("extractor")
        self.vocabularies = ComponentRegistry("vocabulary")
        self.configs = ComponentRegistry("config")
        # metrics is set lazily to avoid a circular import: MetricRegistry
        # lives in metrics/base.py which imports from preprocessing modules
        # that import REGISTRY from this module.
        self._metrics = None

    @property
    def metrics(self):
        if self._metrics is None:
            from bewer.metrics.base import METRIC_REGISTRY

            self._metrics = METRIC_REGISTRY
        return self._metrics

    @contextmanager
    def isolated(self):
        """Context manager that snapshots and restores all non-metric namespaces.

        Useful for tests that register temporary components::

            with REGISTRY.isolated():
                REGISTRY.extractors.register("temp", my_extractor)
                # ... run tests ...
            # "temp" is gone after the context exits
        """
        snapshots = {ns: dict(getattr(self, ns)._components) for ns in ("extractors", "vocabularies", "configs")}
        try:
            yield
        finally:
            for ns, snap in snapshots.items():
                getattr(self, ns)._components.clear()
                getattr(self, ns)._components.update(snap)


REGISTRY = Registry()
