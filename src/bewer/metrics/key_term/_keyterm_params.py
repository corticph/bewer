from __future__ import annotations

from dataclasses import dataclass, field

from bewer.metrics.base import AlignmentParams

__all__ = ["KeyTermParams", "KeyTermSubsetParams", "RelaxedKeyTermParams"]


@dataclass
class KeyTermParams(AlignmentParams):
    """Base class for key-term metric parameter dataclasses.

    Subclass this with @dataclass to define key-term metric parameters.
    Provides the common ``vocab`` field and validates that the vocabulary exists on the dataset.
    Inherits the keyword-only ``alignment`` field from :class:`AlignmentParams`.

    Attributes:
        vocab: The vocabulary name to use for key term identification.
    """

    vocab: str

    def validate(self) -> None:
        """Validate the alignment and that the vocabulary exists on the dataset."""
        super().validate()
        if self.vocab not in self.metric.dataset.config.vocabularies:
            raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset config vocabularies.")


@dataclass
class KeyTermSubsetParams(KeyTermParams):
    """Base class for key-term metric parameters that expose subset matching.

    Adds ``allow_subset_matches`` on top of :class:`KeyTermParams` for metrics
    that consume the matched-term list (KTR, KTP, KTF, KTFPR, KTCER, RKTR, KTER,
    and the hidden outcomes metrics). Position-based consumers such as
    B-WER/U-WER derive from :class:`KeyTermParams` directly: the union of
    covered token positions is invariant under subset removal, so subset
    matching cannot affect their counts.

    Attributes:
        allow_subset_matches: Whether to allow subset matches.
    """

    allow_subset_matches: bool = False


@dataclass
class RelaxedKeyTermParams(KeyTermSubsetParams):
    """Base class for relaxed key-term metric parameters (RKTR, KTCER and their shared outcomes).

    Same fields as :class:`KeyTermSubsetParams`, but ``alignment`` defaults to ``"error_align"``:
    its partial-token ops let a split or merged compound score as a near miss.

    Attributes:
        alignment: The alignment algorithm the metric is computed from.
    """

    alignment: str = field(default="error_align", kw_only=True)
