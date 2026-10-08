"""Orthographically-complex term metrics: recall, precision and F-score.

These mirror the key-term metrics (KTR / KTP / KTF) but draw their terms from an
*orthographically-complex term* vocabulary — abbreviations, acronyms, alphanumerics and
hyphen compounds auto-extracted from the references. The vocabulary is declared in the
base config and attached at Dataset init.

They run under the ``orthographically_complex_term`` tokenizer (no hyphen split, so
``CT-scan`` is one token) and the ``cased`` normalizer (no lowercasing), so a term's
surface form is scored strictly.
"""

from __future__ import annotations

from bewer.metrics.base import METRIC_REGISTRY
from bewer.metrics.ktf import KTF
from bewer.metrics.ktp import KTP
from bewer.metrics.ktr import KTR

DEFAULT_ORTHOGRAPHICALLY_COMPLEX_TERM_VOCAB = "orthographically_complex_terms"

METRIC_REGISTRY.register_metric(
    KTR,
    "orthographically_complex_term_recall",
    tokenizer="orthographically_complex_term",
    normalizer="cased",
    vocab=DEFAULT_ORTHOGRAPHICALLY_COMPLEX_TERM_VOCAB,
)

METRIC_REGISTRY.register_metric(
    KTP,
    "orthographically_complex_term_precision",
    tokenizer="orthographically_complex_term",
    normalizer="cased",
    vocab=DEFAULT_ORTHOGRAPHICALLY_COMPLEX_TERM_VOCAB,
)

METRIC_REGISTRY.register_metric(
    KTF,
    "orthographically_complex_term_fscore",
    tokenizer="orthographically_complex_term",
    normalizer="cased",
    vocab=DEFAULT_ORTHOGRAPHICALLY_COMPLEX_TERM_VOCAB,
)
