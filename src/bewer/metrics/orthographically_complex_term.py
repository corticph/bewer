"""Orthographically-complex term metrics: recall, precision and F-score.

These mirror the key-term metrics (KTR / KTP / KTF) but draw their terms from an
*orthographically-complex term* vocabulary — abbreviations, acronyms, alphanumerics and
hyphen compounds auto-extracted from the references by
:class:`~bewer.extractors.OrthographicallyComplexTermExtractor`. They answer: of those terms
in the references, how many did the system transcribe correctly (recall), how precisely
(precision), and their F-score.

The terms are selected by their *orthography* (capitalization, digit/letter mixing, symbols,
Greek), so this is one dimension of "term complexity" (cf. a future ``phonetically_complex_term``).
The vocabulary (named ``orthographically_complex_terms`` by default) is registered on the
dataset on first use if not already present, so the metrics work out of the box::

    >>> dataset.metrics.orthographically_complex_term_recall().value

They run under the ``complex_term`` tokenizer (which does not split on hyphens, so ``CT-scan``
is one token) and the ``cased`` normalizer (which does not lowercase), so a term's surface form
is scored strictly: neither ``CT scan`` nor ``ct-scan`` matches a ``CT-scan`` term.
"""

from __future__ import annotations

from dataclasses import dataclass

from bewer.core.vocabulary import Vocabulary
from bewer.extractors.orthographically_complex_term import OrthographicallyComplexTermExtractor
from bewer.metrics.base import METRIC_REGISTRY
from bewer.metrics.ktf import KTF
from bewer.metrics.ktp import KTP
from bewer.metrics.ktr import KTR

__all__ = [
    "OrthographicallyComplexTermRecall",
    "OrthographicallyComplexTermPrecision",
    "OrthographicallyComplexTermFscore",
    "DEFAULT_ORTHOGRAPHICALLY_COMPLEX_TERM_VOCAB",
]

#: Name of the auto-registered orthographically-complex term vocabulary.
DEFAULT_ORTHOGRAPHICALLY_COMPLEX_TERM_VOCAB = "orthographically_complex_terms"


def _ensure_orthographically_complex_term_vocabulary(dataset, name: str) -> None:
    """Register a vocabulary backed by :class:`OrthographicallyComplexTermExtractor` if absent.

    Uses :meth:`Dataset._register_derived_vocabulary` so the lookup succeeds even when the
    dataset has already frozen on an earlier metric request.
    """
    if name not in dataset._vocabularies:
        dataset._register_derived_vocabulary(Vocabulary(name).add_extractor(OrthographicallyComplexTermExtractor()))


@dataclass
class OrthographicallyComplexTermMetricParams(KTR.param_schema):
    """Parameters for the orthographically-complex term recall/precision metrics.

    Identical to the key-term metric parameters, except that ``vocab`` defaults to the
    auto-registered orthographically-complex term vocabulary.

    Attributes:
        vocab: Name of the vocabulary. Registered with a default
            :class:`OrthographicallyComplexTermExtractor` on first use if not already present.
    """

    vocab: str = DEFAULT_ORTHOGRAPHICALLY_COMPLEX_TERM_VOCAB

    def validate(self) -> None:
        """Ensure the vocabulary exists, then validate as a key-term metric."""
        _ensure_orthographically_complex_term_vocabulary(self.metric.dataset, self.vocab)
        super().validate()


@dataclass
class OrthographicallyComplexTermFScoreParams(KTF.param_schema):
    """Parameters for the orthographically-complex term F-score metric.

    Identical to the key-term F-score parameters, except that ``vocab`` defaults to the
    auto-registered orthographically-complex term vocabulary.

    Attributes:
        vocab: Name of the vocabulary. Registered with a default
            :class:`OrthographicallyComplexTermExtractor` on first use if not already present.
        beta: F-score beta parameter. beta=1 gives F1 (equal weight to precision and recall);
            beta>1 weights recall more heavily; beta<1 weights precision more heavily.
    """

    vocab: str = DEFAULT_ORTHOGRAPHICALLY_COMPLEX_TERM_VOCAB

    def validate(self) -> None:
        """Ensure the vocabulary exists, then validate as a key-term F-score metric."""
        _ensure_orthographically_complex_term_vocabulary(self.metric.dataset, self.vocab)
        super().validate()


@METRIC_REGISTRY.register("orthographically_complex_term_recall", tokenizer="complex_term", normalizer="cased")
class OrthographicallyComplexTermRecall(KTR):
    short_name_base = "orthographically_complex_term_recall"
    long_name_base = "Orthographically Complex Term Recall"
    description = (
        "Orthographically-complex term recall is key term recall (TP / (TP + FN)) computed over terms "
        "selected by their orthography — abbreviations, acronyms, alphanumerics and hyphen compounds "
        "(e.g. MRI, HbA1c, CO2, CT-scan). It measures the fraction of such reference terms the system "
        "transcribed correctly. Scoring is case-sensitive and does not split hyphen compounds."
    )
    param_schema = OrthographicallyComplexTermMetricParams


@METRIC_REGISTRY.register("orthographically_complex_term_precision", tokenizer="complex_term", normalizer="cased")
class OrthographicallyComplexTermPrecision(KTP):
    short_name_base = "orthographically_complex_term_precision"
    long_name_base = "Orthographically Complex Term Precision"
    description = (
        "Orthographically-complex term precision is key term precision (TP / (TP + FP)) computed over terms "
        "selected by their orthography. It measures how precisely the system transcribed the such terms it "
        "produced. Scoring is case-sensitive and does not split hyphen compounds."
    )
    param_schema = OrthographicallyComplexTermMetricParams


@METRIC_REGISTRY.register("orthographically_complex_term_fscore", tokenizer="complex_term", normalizer="cased")
class OrthographicallyComplexTermFscore(KTF):
    short_name_base = "orthographically_complex_term_fscore"
    long_name_base = "Orthographically Complex Term F-Score"
    description = (
        "Orthographically-complex term F-score is the weighted harmonic mean of the precision and recall "
        "computed over terms selected by their orthography. The beta parameter controls the trade-off "
        "(beta=1 gives F1). At the dataset level it is a micro F-score: TP, FN and FP are summed across "
        "examples before applying the formula."
    )
    param_schema = OrthographicallyComplexTermFScoreParams
