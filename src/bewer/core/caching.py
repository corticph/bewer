"""Context-sensitive caching descriptor for pipeline-derived properties."""

from contextvars import ContextVar
from typing import Any, Callable

from bewer.preprocessing.context import PIPELINE_STAGES

# Build lookup: context_var -> (context_vars_up_to_and_including, pipeline_attr)
_STAGE_INDEX = {}
for i, (cv, attr) in enumerate(PIPELINE_STAGES):
    _STAGE_INDEX[cv] = (tuple(cv for cv, _ in PIPELINE_STAGES[: i + 1]), attr)


class PipelineCachedProperty:
    """Descriptor that implements context-sensitive lazy caching for pipeline properties.

    Reads one or more ContextVars to form a cache key, looks up the pipeline function
    from ``instance._pipelines``, calls it on cache miss, and stores the result in a
    per-instance dict whose name is auto-derived as ``_cache_{property_name}``.

    Use via the :func:`pipeline_cached_property` decorator factory.
    """

    def __init__(
        self,
        context_vars: tuple[ContextVar, ...],
        pipeline: str,
        compute: Callable[[Any, Any], Any],
        doc: str | None = None,
    ):
        self._context_vars = context_vars
        self._pipeline = pipeline
        self._compute = compute
        self.__doc__ = doc

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name
        self._cache_attr = f"_cache_{name}"

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self

        # Build cache key from active context variable values
        key_parts = tuple(cv.get() for cv in self._context_vars)
        cache_key = key_parts[0] if len(key_parts) == 1 else key_parts

        # Check per-instance cache
        cache = getattr(instance, self._cache_attr)
        if cache_key in cache:
            return cache[cache_key]

        # Resolve pipeline function (last context var names the pipeline entry)
        pipelines = instance._pipelines
        if pipelines is None:
            raise ValueError(f"No {self._pipeline} found in pipelines.")

        pipeline_name = key_parts[-1]
        registry = getattr(pipelines, self._pipeline)
        func = registry.get(pipeline_name, None)
        if func is None:
            raise ValueError(f"'{pipeline_name}' not found in {self._pipeline}.")

        # Compute, cache, return
        result = self._compute(instance, func)
        cache[cache_key] = result
        return result


def pipeline_cached_property(stage: ContextVar):
    """Decorator for context-sensitive cached properties backed by pipeline functions.

    The pipeline stage ordering is defined in ``PIPELINE_STAGES``. Passing a stage
    automatically includes all preceding stages in the cache key and resolves the
    pipeline attribute name.

    Args:
        stage: The ContextVar identifying the pipeline stage (e.g. ``NORMALIZER_NAME``).

    Example::

        @pipeline_cached_property(STANDARDIZER_NAME)
        def standardized(self, func):
            \"\"\"The standardized text.\"\"\"
            return func(self.raw)
    """
    context_vars, pipeline = _STAGE_INDEX[stage]

    def decorator(method: Callable) -> PipelineCachedProperty:
        return PipelineCachedProperty(
            context_vars=context_vars,
            pipeline=pipeline,
            compute=method,
            doc=method.__doc__,
        )

    return decorator
