"""Orthographically-complex term metric — a predefined regex metric.

``orthographically_complex_term_recall`` / ``_precision`` / ``_fscore`` are registered via
:func:`~bewer.metrics.regex_metrics.register_regex_metric` over the
``orthographically_complex_terms`` vocabulary auto-extracted by
:data:`~bewer.extractors.orthographically_complex_term.ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN`
(abbreviations, acronyms, alphanumerics, hyphen compounds, Greek-bearing tokens). They are flat
siblings of the quantity metrics and of any metric a user registers — none is special; they
merely share being defined by a regular expression.

The name marks the *orthographic* dimension of term complexity (cf. a future
``phonetically_complex_term``). They run under the ``orthographically_complex_term`` tokenizer (no hyphen split,
so ``CT-scan`` is one token) and the ``cased`` normalizer (no lowercasing), so a term's surface
form is scored strictly::

    >>> dataset.metrics.orthographically_complex_term_recall().value
"""

from __future__ import annotations

from bewer.extractors.orthographically_complex_term import ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN
from bewer.metrics.regex_metrics import register_regex_metric

(
    OrthographicallyComplexTermRecall,
    OrthographicallyComplexTermPrecision,
    OrthographicallyComplexTermFscore,
) = register_regex_metric(
    "orthographically_complex_term",
    ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN,
    vocab="orthographically_complex_terms",
)

__all__ = [
    "OrthographicallyComplexTermRecall",
    "OrthographicallyComplexTermPrecision",
    "OrthographicallyComplexTermFscore",
]
