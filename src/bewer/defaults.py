"""Built-in config presets.

Each config is a BewerConfig instance registered directly. Language configs
extend the base config via the ``extends`` keyword argument.
"""

from __future__ import annotations

from bewer.config import BewerConfig, Transform, get_config
from bewer.core.vocabulary import Vocabulary
from bewer.extractors.orthographically_complex_term import OrthographicallyComplexTermExtractor
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

REGISTRY.configs.register(
    "base",
    BewerConfig(
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
        vocabularies={
            "orthographically_complex_terms": Vocabulary("orthographically_complex_terms").add_extractor(
                OrthographicallyComplexTermExtractor()
            ),
        },
    ),
)


# ============================================================
# English — no changes from base
# ============================================================

REGISTRY.configs.register("en", get_config("base"))


# ============================================================
# German
# ============================================================

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


# ============================================================
# French
# ============================================================

REGISTRY.configs.register(
    "fr",
    BewerConfig(
        extends="base",
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
    ),
)


# ============================================================
# Danish
# ============================================================

REGISTRY.configs.register(
    "da",
    BewerConfig(
        extends="base",
        normalizers={
            "default": (
                Transform(lowercase),
                Transform(transliterate_latin_letters, preserve="æøå"),
                Transform(transliterate_symbols),
            ),
        },
    ),
)
