"""German config."""

from __future__ import annotations

from bewer.configs.config import BewerConfig, Transform
from bewer.preprocessing.normalization import lowercase, transliterate_latin_letters, transliterate_symbols
from bewer.registry import REGISTRY

__all__: list[str] = []


REGISTRY.configs.register(
    "de",
    BewerConfig(
        extends="base",
        normalizers={
            "default": (
                Transform(lowercase),
                Transform(transliterate_latin_letters, preserve="äöüß"),
                Transform(transliterate_symbols),
            ),
        },
    ),
)
