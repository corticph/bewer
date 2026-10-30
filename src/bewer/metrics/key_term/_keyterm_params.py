from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import MetricParams

__all__ = ["KeyTermParams", "KeyTermSubsetParams"]


@dataclass
class KeyTermParams(MetricParams):
    """Base class for key-term metric parameter dataclasses.

    Subclass this with @dataclass to define key-term metric parameters.
    Provides the common ``vocab`` and ``normalized`` fields and validates
    that the vocabulary exists on the dataset.

    Attributes:
        vocab: The vocabulary name to use for key term identification.
        normalized: Whether to use normalized tokens for alignment and key term matching.
    """

    vocab: str
    normalized: bool = True

    def validate(self) -> None:
        """Validate that the vocabulary exists on the dataset."""
        if self.vocab not in self.metric.dataset.config.vocabularies:
            raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset config vocabularies.")


@dataclass
class KeyTermSubsetParams(KeyTermParams):
    """Base class for key-term metric parameters that expose subset matching.

    Adds ``allow_subset_matches`` on top of :class:`KeyTermParams` for metrics
    that consume the matched-term list (KTR, KTP, KTF, KTFPR, KTCER, RKTR, KTER,
    and the hidden stats metrics). Position-based consumers such as
    B-WER/U-WER derive from :class:`KeyTermParams` directly: the union of
    covered token positions is invariant under subset removal, so subset
    matching cannot affect their counts.

    Attributes:
        allow_subset_matches: Whether to allow subset matches.
    """

    allow_subset_matches: bool = False
