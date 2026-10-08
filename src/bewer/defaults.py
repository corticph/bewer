"""Built-in config presets.

Each config returns a BewerConfig delta — only the variants it changes.
The registry builds the full chain from the ``extends`` attribute.
"""

from __future__ import annotations

from bewer.config import BewerConfig, PipelineStep
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
                PipelineStep("nfc"),
                PipelineStep("normalize_apostrophe_variants"),
                PipelineStep("normalize_hyphen_variants"),
                PipelineStep("normalize_slash_variants"),
            ),
        },
        tokenizers={
            "default": PipelineStep(
                "strip_punctuation_keep_symbols_pattern",
                params={"split_on_escaped": "-/"},
            ),
            "with_punctuation": PipelineStep(
                "keep_symbols_and_punctuation_pattern",
                params={"punct_chars": '.,!?:;"-/()\u201c\u201d\u00ab\u00bb\u201e\u00a1\u00bf', "keep_newlines": True},
            ),
            "key_term": PipelineStep(
                "strip_punctuation_keep_symbols_pattern",
                params={"split_on_escaped": "-/'"},
            ),
            "orthographically_complex_term": PipelineStep(
                "strip_punctuation_keep_symbols_pattern",
                params={"split_on_escaped": "/'"},
            ),
        },
        normalizers={
            "default": (
                PipelineStep("lowercase"),
                PipelineStep("transliterate_latin_letters"),
                PipelineStep("transliterate_symbols"),
            ),
            "cased": (
                PipelineStep("transliterate_latin_letters"),
                PipelineStep("transliterate_symbols"),
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
                PipelineStep("lowercase"),
                PipelineStep("transliterate_latin_letters", params={"preserve": "æøå"}),
                PipelineStep("transliterate_symbols"),
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
                PipelineStep("lowercase"),
                PipelineStep("transliterate_latin_letters", params={"preserve": "äöüß"}),
                PipelineStep("transliterate_symbols"),
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
            "default": PipelineStep(
                "strip_punctuation_keep_symbols_pattern",
                params={"split_on_escaped": "-/'"},
            ),
        },
        normalizers={
            "default": (
                PipelineStep("lowercase"),
                PipelineStep(
                    "transliterate_latin_letters",
                    params={"preserve": "àâäçéèêëîïôöùûüÿœæ"},
                ),
                PipelineStep("transliterate_symbols"),
            ),
        },
    )
