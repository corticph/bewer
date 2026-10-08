from __future__ import annotations

import inspect
from collections import namedtuple
from dataclasses import dataclass, field, replace
from importlib import import_module
from os import PathLike
from typing import Any, Callable

import yaml

from bewer.flags import NORMALIZERS, STANDARDIZERS, TOKENIZERS
from bewer.preprocessing.normalization import Normalizer
from bewer.preprocessing.tokenization import Tokenizer
from bewer.registry import REGISTRY

__all__ = [
    "BewerConfig",
    "Transform",
    "Pipelines",
    "merge_configs",
    "resolve_config",
    "get_config",
    "from_yaml",
    "to_yaml",
    "SerializationError",
]


# ============================================================
# Pipelines namedtuple
# ============================================================

_PipelinesBase = namedtuple("_PipelinesBase", [STANDARDIZERS, TOKENIZERS, NORMALIZERS])


class Pipelines(_PipelinesBase):
    """The preprocessing variants resolved from a configuration, grouped by stage.

    A plain namedtuple of three ``{name: pipeline}`` dicts, with a repr that lists the
    available variant names per stage instead of dumping every resolved object.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        stages = ((STANDARDIZERS, self.standardizers), (TOKENIZERS, self.tokenizers), (NORMALIZERS, self.normalizers))
        width = max(len(name) for name, _ in stages) + 1
        rows = "\n".join(f"    {name + ':':<{width}} {', '.join(entries) or '-'}" for name, entries in stages)
        return f"Pipelines(\n{rows}\n)"


# ============================================================
# Transform — a callable with bound params
# ============================================================


class Transform:
    """A transform function bound with parameters.

    Calls ``component(text, **params)`` when invoked.
    """

    __slots__ = ("component", "params")

    def __init__(self, component: Callable[..., Any], **params: Any):
        self.component = component
        self.params = params

    def __call__(self, text: str) -> str:
        return self.component(text, **self.params)

    def __repr__(self) -> str:
        if self.params:
            param_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
            return f"Transform({self.component.__name__}, {param_str})"
        return f"Transform({self.component.__name__})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transform):
            return NotImplemented
        return self.component is other.component and self.params == other.params

    def __hash__(self) -> int:
        return hash((self.component, tuple(sorted(self.params.items()))))


# ============================================================
# BewerConfig
# ============================================================


@dataclass(frozen=True)
class BewerConfig:
    """A frozen preprocessing configuration.

    Each field maps variant names to pipeline definitions:

    - ``standardizers`` / ``normalizers``: variant name -> tuple of Transforms
    - ``tokenizers``: variant name -> compiled regex Pattern
    - ``vocabularies``: tuple of vocabulary names (for auto-resolution)
    """

    standardizers: dict[str, tuple[Transform, ...]] = field(default_factory=dict)
    tokenizers: dict[str, Any] = field(default_factory=dict)
    normalizers: dict[str, tuple[Transform, ...]] = field(default_factory=dict)
    vocabularies: tuple[str, ...] = ()

    def replace(self, **kwargs) -> BewerConfig:
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)


# ============================================================
# Merge
# ============================================================


def merge_configs(base: BewerConfig, delta: BewerConfig) -> BewerConfig:
    """Merge ``delta`` onto ``base`` at variant granularity.

    A variant in ``delta`` replaces the same-named variant in ``base`` entirely.
    Variants not mentioned in ``delta`` are inherited from ``base``.
    """
    return BewerConfig(
        standardizers={**base.standardizers, **delta.standardizers},
        tokenizers={**base.tokenizers, **delta.tokenizers},
        normalizers={**base.normalizers, **delta.normalizers},
        vocabularies=base.vocabularies + delta.vocabularies,
    )


# ============================================================
# Resolution
# ============================================================


def resolve_config(config: BewerConfig) -> Pipelines:
    """Resolve a BewerConfig to Pipelines namedtuple."""
    standardizers = {name: _resolve_normalizer(name, steps) for name, steps in config.standardizers.items()}
    tokenizers = {name: _resolve_tokenizer(name, pattern) for name, pattern in config.tokenizers.items()}
    normalizers = {name: _resolve_normalizer(name, steps) for name, steps in config.normalizers.items()}
    return Pipelines(
        standardizers=standardizers,
        tokenizers=tokenizers,
        normalizers=normalizers,
    )


def _resolve_normalizer(name: str, steps: tuple[Transform, ...]) -> Normalizer:
    """Resolve a tuple of Transforms to a Normalizer."""
    pipeline = []
    for step in steps:
        fn = _resolve_component(step.component)
        _validate_params(fn, step.params, skip_first=True)
        pipeline.append((fn, step.params))
    return Normalizer(pipeline, name)


def _resolve_tokenizer(name: str, pattern: Any) -> Tokenizer:
    """Wrap a pattern (from a factory call) in a Tokenizer."""
    return Tokenizer(pattern, name)


def _resolve_component(component: str | Callable[..., Any]) -> Callable[..., Any]:
    """Resolve a component reference to a callable.

    Strings with dots are treated as dotted-path imports.
    Callables are returned as-is.
    """
    if isinstance(component, str):
        if "." in component:
            module_name, func_name = component.rsplit(".", 1)
            module = import_module(module_name)
            return getattr(module, func_name)
        raise ValueError(
            f"Component name '{component}' is not a dotted path. Use 'module.function' or pass a callable directly."
        )
    elif callable(component):
        return component
    else:
        raise TypeError(f"Expected str or callable, got {type(component)}")


def _validate_params(fn: Callable, params: dict[str, Any], *, skip_first: bool = False) -> None:
    """Validate config params against a function's signature."""
    sig = inspect.signature(fn)
    func_params = sig.parameters
    param_iter = iter(func_params.items())

    if skip_first:
        first_param = next(param_iter)[0]
        if first_param in params:
            raise ValueError(f"First positional argument '{first_param}' should not be passed in params")

    for param, value in param_iter:
        if value.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if value.default is inspect.Parameter.empty and param not in params:
            raise ValueError(f"Parameter '{param}' not found in function '{fn.__name__}'")

    for param in params:
        if param not in func_params:
            raise ValueError(f"Unexpected parameter '{param}' for function '{fn.__name__}'")


# ============================================================
# Config name resolution
# ============================================================


def get_config(name: str) -> BewerConfig:
    """Resolve a registered config by name, following the extends chain.

    A config without ``extends`` returns its delta directly.
    A config with ``extends`` is merged onto its parent.
    """
    fn = REGISTRY.configs.get(name)
    extends = getattr(fn, "extends", None)
    delta = fn()
    if extends:
        base = get_config(extends)
        return merge_configs(base, delta)
    return delta


# ============================================================
# YAML I/O
# ============================================================


class SerializationError(ValueError):
    """Raised when a BewerConfig cannot be serialized to YAML."""


def to_yaml(config: BewerConfig) -> str:
    """Serialize a BewerConfig to YAML using dotted-path names."""
    data = {
        "standardizers": _pipelines_to_yaml(config.standardizers),
        "tokenizers": {name: _pattern_to_yaml_dict(pattern) for name, pattern in config.tokenizers.items()},
        "normalizers": _pipelines_to_yaml(config.normalizers),
        "vocabularies": list(config.vocabularies) if config.vocabularies else None,
    }
    data = {k: v for k, v in data.items() if v}
    return yaml.dump(data, sort_keys=False, allow_unicode=True)


def _pipelines_to_yaml(pipelines: dict[str, tuple[Transform, ...]]) -> dict[str, dict[str, dict[str, Any]]] | None:
    if not pipelines:
        return None
    result = {}
    for name, steps in pipelines.items():
        result[name] = {}
        seen = set()
        for step in steps:
            key = _transform_to_yaml_key(step)
            if key in seen:
                raise SerializationError(f"Duplicate component '{key}' in pipeline '{name}'")
            seen.add(key)
            result[name][key] = step.params or None
    return result


def _transform_to_yaml_key(step: Transform) -> str:
    module = getattr(step.component, "__module__", None)
    qualname = getattr(step.component, "__qualname__", None) or getattr(step.component, "__name__", None)
    if not module or not qualname:
        raise SerializationError(f"Cannot serialize callable {step.component!r}: no module/name metadata.")
    if "<locals>" in qualname:
        raise SerializationError(
            f"Cannot serialize callable {step.component!r}: it is a local function "
            f"that cannot be imported by dotted path."
        )
    return f"{module}.{qualname}"


def _pattern_to_yaml_dict(pattern: Any) -> dict[str, Any]:
    """Serialize a tokenizer pattern to YAML.

    Tokenizer patterns are compiled regex Patterns produced by factory calls.
    We store the factory's dotted path and the params used to create the pattern.
    """
    # Patterns from factory calls carry their factory's metadata via the pattern
    # itself — but we can't reconstruct the factory from the pattern.
    # For YAML round-trip, we store the pattern string.
    import regex as re

    if isinstance(pattern, (re.Pattern, type(__import__("re").compile("")))):
        return {pattern.pattern: None}
    raise SerializationError(f"Cannot serialize tokenizer pattern of type {type(pattern)!r}")


def from_yaml(source: str | PathLike) -> BewerConfig:
    """Parse YAML into a BewerConfig.

    Accepts dotted paths (e.g. ``bewer.preprocessing.normalization.lowercase``).
    Any name containing ``.`` is treated as a dotted path and imported.

    ``source`` may be a YAML string, a file path (``str`` or ``PathLike``).
    """
    from pathlib import Path

    if isinstance(source, PathLike):
        text = Path(source).read_text(encoding="utf-8")
    elif isinstance(source, str):
        try:
            path = Path(source)
            if path.is_file():
                text = path.read_text(encoding="utf-8")
            else:
                text = source
        except OSError:
            text = source
    else:
        raise TypeError(f"from_yaml expects str or PathLike, got {type(source)}")

    raw = yaml.safe_load(text) or {}
    return _raw_to_config(raw)


def _raw_to_config(raw: dict) -> BewerConfig:
    standardizers = _parse_pipelines(raw.get("standardizers", {}))
    tokenizers = _parse_tokenizers(raw.get("tokenizers", {}))
    normalizers = _parse_pipelines(raw.get("normalizers", {}))
    vocabularies = raw.get("vocabularies", ())
    if isinstance(vocabularies, list):
        vocabularies = tuple(vocabularies)
    elif isinstance(vocabularies, dict):
        vocabularies = tuple(vocabularies.keys())
    else:
        vocabularies = ()
    return BewerConfig(
        standardizers=standardizers,
        tokenizers=tokenizers,
        normalizers=normalizers,
        vocabularies=vocabularies,
    )


def _parse_pipelines(raw: dict) -> dict[str, tuple[Transform, ...]]:
    result = {}
    for name, steps in raw.items():
        if steps is None:
            result[name] = ()
            continue
        step_list = []
        for component_path, params in steps.items():
            fn = _resolve_component(component_path)
            step_list.append(Transform(fn, **(params or {})))
        result[name] = tuple(step_list)
    return result


def _parse_tokenizers(raw: dict) -> dict[str, Any]:
    result = {}
    for name, steps in raw.items():
        if steps is None:
            continue
        if len(steps) != 1:
            raise ValueError(f"Tokenizer config for '{name}' must contain exactly one definition, got {len(steps)}")
        key, params = next(iter(steps.items()))
        if _looks_like_dotted_path(key):
            fn = _resolve_component(key)
            result[name] = fn(**(params or {}))
        else:
            import regex as re

            result[name] = re.compile(key, re.V1)
    return result


def _looks_like_dotted_path(s: str) -> bool:
    """Heuristic: a dotted path has at least one dot and each segment is a valid identifier."""
    if "." not in s:
        return False
    parts = s.rsplit(".", 1)
    return all(p.isidentifier() for p in parts)
