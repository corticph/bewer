from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from types import MappingProxyType
from typing import TYPE_CHECKING

from bewer.preprocessing.normalization import Normalizer, Standardizer
from bewer.preprocessing.tokenization import Tokenizer

if TYPE_CHECKING:
    from bewer.key_term.vocabulary import Vocabulary

__all__ = [
    "BewerConfig",
    "ConfigFrozenError",
    "ConfigRegistry",
    "CONFIG_REGISTRY",
    "get_config",
]


class ConfigFrozenError(RuntimeError):
    """Raised when attempting to modify a frozen BewerConfig."""


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
# BewerConfig
# ============================================================


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
        stages = (
            ("standardizers", self.standardizers),
            ("tokenizers", self.tokenizers),
            ("normalizers", self.normalizers),
            ("vocabularies", self.vocabularies),
        )
        width = max(len(name) for name, _ in stages) + 1
        rows = "\n".join(
            f"    {name + ':':<{width}} {', '.join(entries) or '-'}"
            for name, entries in stages
        )
        return f"BewerConfig(\n{rows}\n)"


# ============================================================
# Config name resolution
# ============================================================


def get_config(name: str) -> BewerConfig:
    """Retrieve a registered config by name."""
    return CONFIG_REGISTRY.get(name)
