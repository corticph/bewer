from __future__ import annotations

import inspect
from collections import namedtuple
from dataclasses import dataclass, field, replace
from typing import Any, Callable

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
        _validate_params(component, params, skip_first=True)

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
    pipeline = [(step.component, step.params) for step in steps]
    return Normalizer(pipeline, name)


def _resolve_tokenizer(name: str, pattern: Any) -> Tokenizer:
    """Wrap a pattern (from a factory call) in a Tokenizer."""
    return Tokenizer(pattern, name)


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
