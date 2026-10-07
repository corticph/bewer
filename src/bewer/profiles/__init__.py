"""Built-in language profiles.

Each profile returns a BewerConfig delta — only the variants it changes.
The registry builds the full chain from the ``extends`` attribute.
"""

from __future__ import annotations

from bewer.config import BewerConfig, PipelineStep
from bewer.registry import REGISTRY

__all__: list[str] = []


# ============================================================
# Base profile (mirrors configs/base.yml byte-for-byte)
# ============================================================


@REGISTRY.register_profile("base")
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
                params={"punct_chars": '.,!?:;"-/()“”«»„¡¿', "keep_newlines": True},
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
# English — no changes from base (mirrors en.yml)
# ============================================================


@REGISTRY.register_profile("en", extends="base")
def english_delta() -> BewerConfig:
    return BewerConfig()


# ============================================================
# Danish (mirrors da.yml)
# ============================================================


@REGISTRY.register_profile("da", extends="base")
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
# German (mirrors de.yml)
# ============================================================


@REGISTRY.register_profile("de", extends="base")
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
# French (mirrors fr.yml)
# ============================================================


@REGISTRY.register_profile("fr", extends="base")
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
