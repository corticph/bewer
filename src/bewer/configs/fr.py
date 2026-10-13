"""French config."""

from __future__ import annotations

from bewer.configs.config import BewerConfig, Transform
from bewer.preprocessing.normalization import Normalizer, lowercase, transliterate_latin_letters, transliterate_symbols
from bewer.preprocessing.tokenization import Tokenizer, strip_punctuation_keep_symbols_pattern
from bewer.registry import REGISTRY

__all__: list[str] = []


REGISTRY.configs.register(
    "fr",
    BewerConfig(
        extends="base",
        tokenizers={
            "default": Tokenizer("default", strip_punctuation_keep_symbols_pattern(split_on_escaped="-/'")),
        },
        normalizers={
            "default": Normalizer(
                "default",
                Transform(lowercase),
                Transform(transliterate_latin_letters, preserve="àâäçéèêëîïôöùûüÿœæ"),
                Transform(transliterate_symbols),
            ),
        },
    ),
)
