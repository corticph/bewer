from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import MetricParams

__all__ = ["KeyTermParams"]


@dataclass
class KeyTermParams(MetricParams):
    """Base class for key-term metric parameter dataclasses.

    Subclass this with @dataclass to define key-term metric parameters.
    Provides the common ``vocab``, ``normalized``, and ``allow_subset_matches``
    fields and validates that the vocabulary exists on the dataset.

    Attributes:
        vocab: The vocabulary name to use for key term identification.
        normalized: Whether to use normalized tokens for alignment and key term matching.
        allow_subset_matches: Whether to allow subset matches.
    """

    vocab: str
    normalized: bool = True
    allow_subset_matches: bool = False

    def validate(self) -> None:
        """Validate that the vocabulary exists on the dataset."""
        if self.vocab not in self.metric.dataset._vocabularies:
            raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset key term vocabularies.")
