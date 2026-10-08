"""Built-in config presets.

Each config returns a BewerConfig delta — only the variants it changes.
The registry builds the full chain from the ``extends`` attribute.
"""

from __future__ import annotations

from bewer.config import BewerConfig, Transform
from bewer.preprocessing.normalization import (
    lowercase,
    nfc,
    normalize_apostrophe_variants,
    normalize_hyphen_variants,
    normalize_slash_variants,
    transliterate_latin_letters,
    transliterate_symbols,
)
from bewer.preprocessing.tokenization import (
    keep_symbols_and_punctuation_pattern,
    strip_punctuation_keep_symbols_pattern,
)
from bewer.registry import REGISTRY

__all__: list[str] = []


# ============================================================
# Base config
# ============================================================


@REGISTRY.configs.register("base")
def base_config() -> BewerConfig:
    return BewerConfig(
        standardizers={
            "default": (
                Transform(nfc),
                Transform(normalize_apostrophe_variants),
                Transform(normalize_hyphen_variants),
                Transform(normalize_slash_variants),
            ),
        },
        tokenizers={
            "default": strip_punctuation_keep_symbols_pattern(split_on_escaped="-/"),
            "with_punctuation": keep_symbols_and_punctuation_pattern(
                punct_chars='.,!?:;"-/()“”«»„¡¿',
                keep_newlines=True,
            ),
            "key_term": strip_punctuation_keep_symbols_pattern(split_on_escaped="-/'"),
            "orthographically_complex_term": strip_punctuation_keep_symbols_pattern(split_on_escaped="/'"),
        },
        normalizers={
            "default": (
                Transform(lowercase),
                Transform(transliterate_latin_letters),
                Transform(transliterate_symbols),
            ),
            "cased": (
                Transform(transliterate_latin_letters),
                Transform(transliterate_symbols),
            ),
        },
    )


# ============================================================
# English — no changes from base
# ============================================================


@REGISTRY.configs.register("en", extends="base")
def english_delta() -> BewerConfig:
    return BewerConfig()


# ============================================================
# Danish
# ============================================================


@REGISTRY.configs.register("da", extends="base")
def danish_delta() -> BewerConfig:
    return BewerConfig(
        normalizers={
            "default": (
                Transform(lowercase),
                Transform(transliterate_latin_letters, preserve="æøå"),
                Transform(transliterate_symbols),
            ),
        },
    )


# ============================================================
# German
# ============================================================


@REGISTRY.configs.register("de", extends="base")
def german_delta() -> BewerConfig:
    return BewerConfig(
        normalizers={
            "default": (
                Transform(lowercase),
                Transform(transliterate_latin_letters, preserve="äöüß"),
                Transform(transliterate_symbols),
            ),
        },
    )


# ============================================================
# French
# ============================================================


@REGISTRY.configs.register("fr", extends="base")
def french_delta() -> BewerConfig:
    return BewerConfig(
        tokenizers={
            "default": strip_punctuation_keep_symbols_pattern(split_on_escaped="-/'"),
        },
        normalizers={
            "default": (
                Transform(lowercase),
                Transform(transliterate_latin_letters, preserve="àâäçéèêëîïôöùûüÿœæ"),
                Transform(transliterate_symbols),
            ),
        },
    )
