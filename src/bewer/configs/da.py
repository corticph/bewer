"""Danish config."""

from __future__ import annotations

from bewer.configs.base import BASE_CONFIG
from bewer.configs.config import CONFIG_REGISTRY, BewerConfig, Transform
from bewer.preprocessing.normalization import Normalizer, lowercase, transliterate_latin_letters, transliterate_symbols

__all__: list[str] = []


DA_CONFIG = BewerConfig(
    extends=BASE_CONFIG,
    normalizers=(
        Normalizer(
            "default",
            Transform(lowercase),
            Transform(transliterate_latin_letters, preserve="æøå"),
            Transform(transliterate_symbols),
        ),
    ),
)

CONFIG_REGISTRY.register("da", DA_CONFIG)
