"""Base config — shared by all language configs."""

from __future__ import annotations

from bewer.configs.config import BewerConfig, Transform
from bewer.core.vocabulary import Vocabulary
from bewer.extractors.orthographically_complex_term import OrthographicallyComplexTermExtractor
from bewer.preprocessing.normalization import (
    Normalizer,
    Standardizer,
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
from bewer.registry import REGISTRY

__all__: list[str] = []


REGISTRY.configs.register(
    "base",
    BewerConfig(
        standardizers={
            "default": Standardizer(
                "default",
                Transform(nfc),
                Transform(normalize_apostrophe_variants),
                Transform(normalize_hyphen_variants),
                Transform(normalize_slash_variants),
            ),
        },
        tokenizers={
            "default": Tokenizer("default", strip_punctuation_keep_symbols_pattern(split_on_escaped="-/")),
            "with_punctuation": Tokenizer(
                "with_punctuation",
                keep_symbols_and_punctuation_pattern(
                    punct_chars='.,!?:;"-/()“”«»„¡¿',
                    keep_newlines=True,
                ),
            ),
            "key_term": Tokenizer("key_term", strip_punctuation_keep_symbols_pattern(split_on_escaped="-/'")),
            "orthographically_complex_term": Tokenizer(
                "orthographically_complex_term",
                strip_punctuation_keep_symbols_pattern(split_on_escaped="/'"),
            ),
        },
        normalizers={
            "default": Normalizer(
                "default",
                Transform(lowercase),
                Transform(transliterate_latin_letters),
                Transform(transliterate_symbols),
            ),
            "cased": Normalizer(
                "cased",
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
