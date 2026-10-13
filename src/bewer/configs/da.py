"""Danish config."""

from __future__ import annotations

from bewer.configs.config import BewerConfig, Transform
from bewer.preprocessing.normalization import Normalizer, lowercase, transliterate_latin_letters, transliterate_symbols
from bewer.registry import REGISTRY

__all__: list[str] = []


REGISTRY.configs.register(
    "da",
    BewerConfig(
        extends="base",
        normalizers={
            "default": Normalizer(
                "default",
                Transform(lowercase),
                Transform(transliterate_latin_letters, preserve="æøå"),
                Transform(transliterate_symbols),
            ),
        },
    ),
)
