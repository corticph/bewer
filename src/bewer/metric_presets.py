"""Pre-built keyword-argument presets for metric calls.

A preset bundles the call-time arguments that run a metric under a specific
preprocessing pipeline and vocabulary. Splat it into any metric factory call:

    from bewer.metric_presets import ORTHOGRAPHICALLY_COMPLEX_TERM_KWARGS

    recall = dataset.metrics.key_term_recall(**ORTHOGRAPHICALLY_COMPLEX_TERM_KWARGS)

Override individual arguments by merging a dict first:

    loose = dataset.metrics.key_term_recall(
        **{**ORTHOGRAPHICALLY_COMPLEX_TERM_KWARGS, "normalizer": "default"}
    )

The orthographically-complex term preset runs any key-term metric on abbreviations,
acronyms, alphanumerics and hyphen compounds (e.g. ``MRI``, ``HbA1c``, ``CT-scan``),
auto-extracted into the ``orthographically_complex_terms`` vocabulary of the base
config. The ``orthographically_complex_term`` tokenizer keeps hyphen compounds whole
and the ``cased`` normalizer skips lowercasing, so a term's surface form is scored
strictly.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = ["ORTHOGRAPHICALLY_COMPLEX_TERM_KWARGS"]

ORTHOGRAPHICALLY_COMPLEX_TERM_KWARGS: Mapping[str, str] = MappingProxyType(
    {
        "vocab": "orthographically_complex_terms",
        "tokenizer": "orthographically_complex_term",
        "normalizer": "cased",
    }
)
