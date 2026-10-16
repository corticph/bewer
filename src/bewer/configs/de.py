"""German config."""

from __future__ import annotations

from bewer.configs.base import BASE_CONFIG
from bewer.configs.config import CONFIG_REGISTRY, BewerConfig
from bewer.preprocessing.normalization import (
    Normalizer,
    Transform,
    lowercase,
    transliterate_latin_letters,
    transliterate_symbols,
)

__all__: list[str] = []


DE_CONFIG = BewerConfig(
    extends=BASE_CONFIG,
    normalizers=(
        Normalizer(
            "default",
            Transform(lowercase),
            Transform(transliterate_latin_letters, preserve="äöüß"),
            Transform(transliterate_symbols),
        ),
    ),
)

CONFIG_REGISTRY.register("de", DE_CONFIG)
