"""Complex-term metrics: recall (CTR), precision (CTP) and F-score (CTF).

These mirror the key-term metrics (KTR / KTP / KTF) but draw their terms from a
*complex-term* vocabulary — abbreviations, acronyms, alphanumerics and hyphen compounds
auto-extracted from the references by :class:`~bewer.extractors.ComplexTermExtractor`.
They answer: of the complex terms in the references, how many did the system transcribe
correctly (recall), how precisely (precision), and their F-score.

The complex-term vocabulary (named ``complex_terms`` by default) is registered on the
dataset on first use if it is not already present, so the metrics work out of the box::

    >>> dataset.metrics.ctr().value

The metrics run under the ``complex_term`` tokenizer (which does not split on hyphens, so
``CT-scan`` is one token) and the ``cased`` normalizer (which does not lowercase), so a
term's surface form is scored strictly: neither ``CT scan`` nor ``ct-scan`` matches a
``CT-scan`` term.
"""

from __future__ import annotations

from dataclasses import dataclass

from bewer.core.vocabulary import Vocabulary
from bewer.extractors.complex_term import ComplexTermExtractor
from bewer.metrics.base import METRIC_REGISTRY
from bewer.metrics.ktf import KTF
from bewer.metrics.ktp import KTP
from bewer.metrics.ktr import KTR

__all__ = ["CTR", "CTP", "CTF", "DEFAULT_COMPLEX_TERM_VOCAB"]

#: Name of the auto-registered complex-term vocabulary.
DEFAULT_COMPLEX_TERM_VOCAB = "complex_terms"


def _ensure_complex_term_vocabulary(dataset, name: str) -> None:
    """Register a complex-term vocabulary backed by :class:`ComplexTermExtractor` if absent.

    Uses :meth:`Dataset._register_derived_vocabulary` so the lookup succeeds even when the
    dataset has already frozen on an earlier metric request.
    """
    if name not in dataset._vocabularies:
        dataset._register_derived_vocabulary(Vocabulary(name).add_extractor(ComplexTermExtractor()))


@dataclass
class ComplexTermMetricParams(KTR.param_schema):
    """Parameters for the complex-term recall/precision metrics (CTR / CTP).

    Identical to the key-term metric parameters, except that ``vocab`` defaults to the
    auto-registered complex-term vocabulary.

    Attributes:
        vocab: Name of the complex-term vocabulary. Registered with a default
            :class:`ComplexTermExtractor` on first use if not already present.
    """

    vocab: str = DEFAULT_COMPLEX_TERM_VOCAB

    def validate(self) -> None:
        """Ensure the complex-term vocabulary exists, then validate as a key-term metric."""
        _ensure_complex_term_vocabulary(self.metric.dataset, self.vocab)
        super().validate()


@dataclass
class ComplexTermFScoreParams(KTF.param_schema):
    """Parameters for the complex-term F-score metric (CTF).

    Identical to the key-term F-score parameters, except that ``vocab`` defaults to the
    auto-registered complex-term vocabulary.

    Attributes:
        vocab: Name of the complex-term vocabulary. Registered with a default
            :class:`ComplexTermExtractor` on first use if not already present.
        beta: F-score beta parameter. beta=1 gives F1 (equal weight to precision and recall);
            beta>1 weights recall more heavily; beta<1 weights precision more heavily.
    """

    vocab: str = DEFAULT_COMPLEX_TERM_VOCAB

    def validate(self) -> None:
        """Ensure the complex-term vocabulary exists, then validate as a key-term F-score metric."""
        _ensure_complex_term_vocabulary(self.metric.dataset, self.vocab)
        super().validate()


@METRIC_REGISTRY.register("ctr", tokenizer="complex_term", normalizer="cased")
class CTR(KTR):
    short_name_base = "CTR"
    long_name_base = "Complex Term Recall"
    description = (
        "Complex term recall (CTR) is key term recall (TP / (TP + FN)) computed over complex terms — "
        "abbreviations, acronyms, alphanumerics and hyphen compounds (e.g. MRI, HbA1c, CO2, CT-scan) "
        "auto-extracted from the references. It measures the fraction of reference complex terms the "
        "system transcribed correctly. Scoring is case-sensitive and does not split hyphen compounds."
    )
    param_schema = ComplexTermMetricParams


@METRIC_REGISTRY.register("ctp", tokenizer="complex_term", normalizer="cased")
class CTP(KTP):
    short_name_base = "CTP"
    long_name_base = "Complex Term Precision"
    description = (
        "Complex term precision (CTP) is key term precision (TP / (TP + FP)) computed over complex terms "
        "auto-extracted from the references. It measures how precisely the system transcribed the complex "
        "terms it produced. Scoring is case-sensitive and does not split hyphen compounds."
    )
    param_schema = ComplexTermMetricParams


@METRIC_REGISTRY.register("ctf", tokenizer="complex_term", normalizer="cased")
class CTF(KTF):
    short_name_base = "CTF"
    long_name_base = "Complex Term F-Score"
    description = (
        "Complex term F-score (CTF) is the weighted harmonic mean of complex term precision (CTP) and "
        "recall (CTR), computed over complex terms auto-extracted from the references. The beta parameter "
        "controls the trade-off (beta=1 gives F1). At the dataset level it is a micro F-score: TP, FN and FP "
        "are summed across examples before applying the formula."
    )
    param_schema = ComplexTermFScoreParams
