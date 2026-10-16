from __future__ import annotations

import inspect
from collections.abc import Sequence
from contextlib import contextmanager
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable

from bewer.preprocessing.normalization import Normalizer, Standardizer
from bewer.preprocessing.tokenization import Tokenizer

if TYPE_CHECKING:
    from bewer.core.vocabulary import Vocabulary


class ConfigFrozenError(RuntimeError):
    """Raised when attempting to modify a frozen BewerConfig."""


__all__ = [
    "BewerConfig",
    "ConfigFrozenError",
    "ConfigRegistry",
    "CONFIG_REGISTRY",
    "Transform",
    "get_config",
]


# ============================================================
# ConfigRegistry
# ============================================================


class ConfigRegistry:
    """A name -> BewerConfig registry.

    Auto-freezes configs on registration. Provides collision detection,
    nice error messages, and an ``isolated()`` context manager for tests.
    """

    def __init__(self):
        self._configs: dict[str, BewerConfig] = {}

    def register(self, name: str, config: BewerConfig) -> BewerConfig:
        """Register a config under ``name``, freezing it first.

        Raises ValueError if the name is already registered.
        """
        if name in self._configs:
            raise ValueError(f"Config '{name}' is already registered")
        config.freeze()
        self._configs[name] = config
        return config

    def get(self, name: str) -> BewerConfig:
        """Retrieve a config by name. Raises KeyError if missing."""
        if name not in self._configs:
            available = sorted(self._configs)
            raise KeyError(f"Config '{name}' not found. Available: {available}")
        return self._configs[name]

    def list(self) -> list[str]:
        """Return sorted names of all registered configs."""
        return sorted(self._configs)

    def __contains__(self, name: str) -> bool:
        return name in self._configs

    def __repr__(self) -> str:
        return f"ConfigRegistry({list(self._configs)})"

    @contextmanager
    def isolated(self):
        """Snapshot and restore all registered configs.

        Useful for tests that register temporary configs.
        """
        snapshot = dict(self._configs)
        try:
            yield self
        finally:
            self._configs.clear()
            self._configs.update(snapshot)


CONFIG_REGISTRY = ConfigRegistry()


# ============================================================
# Transform — a callable with bound params
# ============================================================


class Transform:
    """A transform function bound with parameters.

    Calls ``component(text, **params)`` when invoked.
    """

    __slots__ = ("fn", "params")

    def __init__(self, fn: Callable[..., Any], **params: Any):
        self.fn = fn
        self.params = params
        _validate_params(fn, params, skip_first=True)

    def __call__(self, text: str) -> str:
        return self.fn(text, **self.params)

    def __repr__(self) -> str:
        if self.params:
            param_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
            return f"Transform({self.fn.__name__}, {param_str})"
        return f"Transform({self.fn.__name__})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transform):
            return NotImplemented
        return self.fn is other.fn and self.params == other.params

    def __hash__(self) -> int:
        return hash((self.fn, tuple(sorted(self.params.items()))))


def _to_dict(items) -> dict:
    """Convert a sequence of named objects into a dict keyed by ``.name``.

    Raises ``ValueError`` on duplicate names.
    """
    if items is None:
        return {}
    result = {}
    for item in items:
        if item.name in result:
            raise ValueError(f"Duplicate name: {item.name!r}")
        result[item.name] = item
    return result


# ============================================================
# BewerConfig
# ============================================================


class BewerConfig:
    """A preprocessing configuration.

    Each field accepts a sequence of pipeline objects (each carrying its own
    ``.name``). Internally stored as ``{name: object}`` dicts:

    - ``standardizers``: sequence of Standardizer
    - ``tokenizers``: sequence of Tokenizer
    - ``normalizers``: sequence of Normalizer
    - ``vocabularies``: sequence of Vocabulary

    Configs are mutable until frozen. Freezing happens automatically when a
    config is registered or handed to a Dataset. Use ``clone()`` to get a
    mutable copy of a frozen config.
    """

    _FIELDS = ("standardizers", "tokenizers", "normalizers", "vocabularies")

    def __init__(
        self,
        standardizers: Sequence[Standardizer] | None = None,
        tokenizers: Sequence[Tokenizer] | None = None,
        normalizers: Sequence[Normalizer] | None = None,
        vocabularies: Sequence[Vocabulary] | None = None,
        *,
        extends: BewerConfig | str | None = None,
    ):
        if extends is not None:
            base = get_config(extends) if isinstance(extends, str) else extends
            self.standardizers = {**base.standardizers, **_to_dict(standardizers)}
            self.tokenizers = {**base.tokenizers, **_to_dict(tokenizers)}
            self.normalizers = {**base.normalizers, **_to_dict(normalizers)}
            self.vocabularies = {**base.vocabularies, **_to_dict(vocabularies)}
        else:
            self.standardizers = _to_dict(standardizers)
            self.tokenizers = _to_dict(tokenizers)
            self.normalizers = _to_dict(normalizers)
            self.vocabularies = _to_dict(vocabularies)
        self._frozen = False

    # -- freezing ----------------------------------------------------------

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """Freeze the config — all dict fields become read-only ``MappingProxyType`` views."""
        if self._frozen:
            return
        for name in self._FIELDS:
            setattr(self, name, MappingProxyType(dict(self.__dict__[name])))
        self._frozen = True

    def clone(self) -> BewerConfig:
        """Return a mutable copy of this config."""
        clone = BewerConfig()
        for name in self._FIELDS:
            setattr(clone, name, dict(self.__dict__[name]))
        return clone

    # -- add methods -------------------------------------------------------

    def _add(self, field: str, item, allow_override: bool = False) -> BewerConfig:
        if self._frozen:
            raise ConfigFrozenError(f"Cannot modify frozen BewerConfig (adding to {field})")
        items = getattr(self, field)
        if not allow_override and item.name in items:
            raise ValueError(f"{item.name!r} already exists in {field}")
        items[item.name] = item
        return self

    def add_standardizer(self, standardizer: Standardizer, *, allow_override: bool = False) -> BewerConfig:
        return self._add("standardizers", standardizer, allow_override=allow_override)

    def add_tokenizer(self, tokenizer: Tokenizer, *, allow_override: bool = False) -> BewerConfig:
        return self._add("tokenizers", tokenizer, allow_override=allow_override)

    def add_normalizer(self, normalizer: Normalizer, *, allow_override: bool = False) -> BewerConfig:
        return self._add("normalizers", normalizer, allow_override=allow_override)

    def add_vocabulary(self, vocab: "Vocabulary", *, allow_override: bool = False) -> BewerConfig:
        self._add("vocabularies", vocab, allow_override=allow_override)
        vocab.freeze()
        return self

    def __repr__(self) -> str:
        parts = []
        for name, value in (
            ("standardizers", self.standardizers),
            ("tokenizers", self.tokenizers),
            ("normalizers", self.normalizers),
            ("vocabularies", self.vocabularies),
        ):
            if value:
                parts.append(f"{name}={value!r}")
        return f"BewerConfig({', '.join(parts)})"


# ============================================================
# Resolution
# ============================================================


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
    """Retrieve a registered config by name."""
    return CONFIG_REGISTRY.get(name)
