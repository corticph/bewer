"""Base config — shared by all language configs."""

from __future__ import annotations

from bewer.configs.config import CONFIG_REGISTRY, BewerConfig
from bewer.core.vocabulary import Vocabulary
from bewer.extractors.orthographically_complex_term import OrthographicallyComplexTermExtractor
from bewer.preprocessing.normalization import (
    Normalizer,
    Standardizer,
    Transform,
    lowercase,
    nfc,
    normalize_apostrophe_variants,
    normalize_hyphen_variants,
    normalize_slash_variants,
    transliterate_latin_letters,
    transliterate_symbols,
)
from bewer.preprocessing.tokenization import (
    Tokenizer,
    keep_symbols_and_punctuation_pattern,
    strip_punctuation_keep_symbols_pattern,
)

__all__: list[str] = []

DEFAULT_PATTERN = strip_punctuation_keep_symbols_pattern(split_on_escaped="-/")
WITH_PUNCTUATION_PATTERN = keep_symbols_and_punctuation_pattern(punct_chars='.,!?:;"-/()“”«»„¡¿', keep_newlines=True)
KEY_TERM_PATTERN = strip_punctuation_keep_symbols_pattern(split_on_escaped="-/'")
ORTHOGRAPHICALLY_COMPLEX_TERM_PATTERN = strip_punctuation_keep_symbols_pattern(split_on_escaped="/'")

BASE_CONFIG = BewerConfig(
    standardizers=(
        Standardizer(
            "default",
            Transform(nfc),
            Transform(normalize_apostrophe_variants),
            Transform(normalize_hyphen_variants),
            Transform(normalize_slash_variants),
        ),
    ),
    tokenizers=(
        Tokenizer("default", DEFAULT_PATTERN),
        Tokenizer("with_punctuation", WITH_PUNCTUATION_PATTERN),
        Tokenizer("key_term", KEY_TERM_PATTERN),
        Tokenizer("orthographically_complex_term", ORTHOGRAPHICALLY_COMPLEX_TERM_PATTERN),
    ),
    normalizers=(
        Normalizer(
            "default",
            Transform(lowercase),
            Transform(transliterate_latin_letters),
            Transform(transliterate_symbols),
        ),
        Normalizer(
            "cased",
            Transform(transliterate_latin_letters),
            Transform(transliterate_symbols),
        ),
    ),
    vocabularies=(Vocabulary("orthographically_complex_terms").add_extractor(OrthographicallyComplexTermExtractor()),),
)

CONFIG_REGISTRY.register("base", BASE_CONFIG)
