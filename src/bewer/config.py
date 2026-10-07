from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from importlib import import_module
from os import PathLike
from typing import Any, Callable

import yaml

from bewer.configs.resolve import Pipelines
from bewer.preprocessing.normalization import Normalizer
from bewer.preprocessing.tokenization import Tokenizer
from bewer.registry import REGISTRY

__all__ = [
    "BewerConfig",
    "PipelineStep",
    "merge_configs",
    "resolve_config",
    "resolve_profile",
    "from_yaml",
    "to_yaml",
    "SerializationError",
]


# ============================================================
# Data classes
# ============================================================


@dataclass(frozen=True)
class PipelineStep:
    """A single step in a preprocessing pipeline.

    ``component`` is a registered name (str) or a direct callable.
    Direct callables make the config non-serializable.
    """

    component: str | Callable[..., Any]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BewerConfig:
    """A frozen preprocessing configuration.

    Each field maps variant names to pipeline definitions:

    - ``standardizers`` / ``normalizers``: variant name -> tuple of PipelineSteps
    - ``tokenizers``: variant name -> single PipelineStep
    - ``vocabularies``: alias name -> vocabulary name (for auto-resolution)
    """

    standardizers: dict[str, tuple[PipelineStep, ...]] = field(default_factory=dict)
    tokenizers: dict[str, PipelineStep] = field(default_factory=dict)
    normalizers: dict[str, tuple[PipelineStep, ...]] = field(default_factory=dict)
    vocabularies: dict[str, str] = field(default_factory=dict)

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
        vocabularies={**base.vocabularies, **delta.vocabularies},
    )


# ============================================================
# Resolution
# ============================================================


def resolve_config(config: BewerConfig) -> Pipelines:
    """Resolve a BewerConfig to the existing Pipelines namedtuple."""
    standardizers = {name: _resolve_transform_pipeline(name, steps) for name, steps in config.standardizers.items()}
    tokenizers = {name: _resolve_tokenizer(name, step) for name, step in config.tokenizers.items()}
    normalizers = {name: _resolve_transform_pipeline(name, steps) for name, steps in config.normalizers.items()}
    return Pipelines(
        standardizers=standardizers,
        tokenizers=tokenizers,
        normalizers=normalizers,
    )


def _resolve_transform_pipeline(name: str, steps: tuple[PipelineStep, ...]) -> Normalizer:
    """Resolve a list of PipelineSteps to a Normalizer."""
    pipeline = []
    for step in steps:
        fn = _resolve_component(step.component, "transforms")
        _validate_params(fn, step.params, skip_first=True)
        pipeline.append((fn, step.params))
    return Normalizer(pipeline, name)


def _resolve_tokenizer(name: str, step: PipelineStep) -> Tokenizer:
    """Resolve a PipelineStep to a Tokenizer."""
    fn = _resolve_component(step.component, "tokenizers")
    _validate_params(fn, step.params, skip_first=False)
    pattern = fn(**step.params)
    return Tokenizer(pattern, name)


def _resolve_component(component: str | Callable[..., Any], namespace: str) -> Callable[..., Any]:
    if isinstance(component, str):
        if "." in component:
            module_name, func_name = component.rsplit(".", 1)
            module = import_module(module_name)
            return getattr(module, func_name)
        return getattr(REGISTRY, namespace).get(component)
    elif callable(component):
        return component
    else:
        raise TypeError(f"Expected str or callable, got {type(component)}")


def _validate_params(fn: Callable, params: dict[str, Any], *, skip_first: bool = False) -> None:
    """Validate config params against a function's signature.

    Preserves the checks from the original resolve.py:
    - first positional arg not in params (only when ``skip_first`` — transforms)
    - required params supplied
    - no unknown params
    """
    sig = inspect.signature(fn)
    func_params = sig.parameters
    param_iter = iter(func_params.items())

    if skip_first:
        first_param = next(param_iter)[0]
        if first_param in params:
            raise ValueError(f"First positional argument '{first_param}' should not be passed in params")

    for param, value in param_iter:
        if value.default is inspect.Parameter.empty and param not in params:
            raise ValueError(f"Parameter '{param}' not found in function '{fn.__name__}'")

    for param in params:
        if param not in func_params:
            raise ValueError(f"Unexpected parameter '{param}' for function '{fn.__name__}'")


# ============================================================
# Profile resolution
# ============================================================


def resolve_profile(name: str) -> BewerConfig:
    """Resolve a profile by name, following the extends chain.

    A profile without ``extends`` returns its delta directly.
    A profile with ``extends`` is merged onto its parent.
    """
    fn = REGISTRY.profiles.get(name)
    extends = getattr(fn, "extends", None)
    delta = fn()
    if extends:
        base = resolve_profile(extends)
        return merge_configs(base, delta)
    return delta


# ============================================================
# YAML I/O
# ============================================================


class SerializationError(ValueError):
    """Raised when a BewerConfig cannot be serialized to YAML."""


def to_yaml(config: BewerConfig) -> str:
    """Serialize a BewerConfig to YAML using registry names.

    Direct callables (not registered) raise SerializationError.
    """
    data = {
        "standardizers": _pipelines_to_yaml(config.standardizers),
        "tokenizers": {name: _step_to_yaml_dict(step) for name, step in config.tokenizers.items()},
        "normalizers": _pipelines_to_yaml(config.normalizers),
        "vocabularies": dict(config.vocabularies) if config.vocabularies else None,
    }
    data = {k: v for k, v in data.items() if v}
    return yaml.dump(data, sort_keys=False, allow_unicode=True)


def _pipelines_to_yaml(pipelines: dict[str, tuple[PipelineStep, ...]]) -> dict[str, dict[str, dict[str, Any]]] | None:
    if not pipelines:
        return None
    result = {}
    for name, steps in pipelines.items():
        result[name] = {}
        for step in steps:
            result[name][_step_to_yaml_key(step)] = step.params or None
    return result


def _step_to_yaml_key(step: PipelineStep) -> str:
    if isinstance(step.component, str):
        return step.component
    elif callable(step.component):
        name = getattr(step.component, "__name__", None)
        if name and name in REGISTRY.transforms:
            return name
        raise SerializationError(
            f"Cannot serialize callable {step.component!r}: not registered in REGISTRY.transforms. "
            f"Register it with @REGISTRY.transforms.register to make it serializable."
        )
    else:
        raise SerializationError(f"Cannot serialize component of type {type(step.component)!r}")


def _step_to_yaml_dict(step: PipelineStep) -> dict[str, Any]:
    key = _step_to_yaml_key(step)
    return {key: step.params or None}


def from_yaml(source: str | PathLike) -> BewerConfig:
    """Parse YAML into a BewerConfig.

    Accepts registry names (e.g. ``lowercase``) and, as a permanent fallback,
    dotted paths (e.g. ``bewer.preprocessing.normalization.lowercase``).
    Any name containing ``.`` is treated as a dotted path and imported.

    ``source`` may be a YAML string, a file path (``str`` or ``PathLike``).
    """
    from pathlib import Path

    if isinstance(source, PathLike):
        text = Path(source).read_text(encoding="utf-8")
    elif isinstance(source, str):
        path = Path(source)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
        else:
            text = source
    else:
        raise TypeError(f"from_yaml expects str or PathLike, got {type(source)}")

    raw = yaml.safe_load(text) or {}
    return _raw_to_config(raw)


def _raw_to_config(raw: dict) -> BewerConfig:
    standardizers = _parse_pipelines(raw.get("standardizers", {}))
    tokenizers = _parse_tokenizers(raw.get("tokenizers", {}))
    normalizers = _parse_pipelines(raw.get("normalizers", {}))
    vocabularies = raw.get("vocabularies", {})
    if vocabularies and isinstance(vocabularies, dict):
        vocabularies = {k: v for k, v in vocabularies.items()}
    else:
        vocabularies = {}
    return BewerConfig(
        standardizers=standardizers,
        tokenizers=tokenizers,
        normalizers=normalizers,
        vocabularies=vocabularies,
    )


def _parse_pipelines(raw: dict) -> dict[str, tuple[PipelineStep, ...]]:
    result = {}
    for name, steps in raw.items():
        if steps is None:
            result[name] = ()
            continue
        step_list = []
        for component, params in steps.items():
            step_list.append(PipelineStep(component=component, params=params or {}))
        result[name] = tuple(step_list)
    return result


def _parse_tokenizers(raw: dict) -> dict[str, PipelineStep]:
    result = {}
    for name, steps in raw.items():
        if steps is None:
            continue
        if len(steps) != 1:
            raise ValueError(f"Tokenizer config for '{name}' must contain exactly one definition, got {len(steps)}")
        component, params = next(iter(steps.items()))
        result[name] = PipelineStep(component=component, params=params or {})
    return result
