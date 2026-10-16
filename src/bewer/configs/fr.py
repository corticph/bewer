"""French config."""

from __future__ import annotations

from bewer.configs.base import BASE_CONFIG
from bewer.configs.config import CONFIG_REGISTRY, BewerConfig, Transform
from bewer.preprocessing.normalization import Normalizer, lowercase, transliterate_latin_letters, transliterate_symbols
from bewer.preprocessing.tokenization import Tokenizer, strip_punctuation_keep_symbols_pattern

__all__: list[str] = []

DEFAULT_PATTERN = strip_punctuation_keep_symbols_pattern(split_on_escaped="-/'")

FR_CONFIG = BewerConfig(
    extends=BASE_CONFIG,
    tokenizers=(Tokenizer("default", DEFAULT_PATTERN),),
    normalizers=(
        Normalizer(
            "default",
            Transform(lowercase),
            Transform(transliterate_latin_letters, preserve="àâäçéèêëîïôöùûüÿœæ"),
            Transform(transliterate_symbols),
        ),
    ),
)

CONFIG_REGISTRY.register("fr", FR_CONFIG)
