from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import MetricParams

__all__ = ["KeyTermMetricParams"]


@dataclass
class KeyTermMetricParams(MetricParams):
    """Shared parameters for key term metrics.

    Attributes:
        vocab: The vocabulary name to use for key term identification.
        normalized: Whether to use normalized tokens for alignment and key term matching.
        allow_subset_matches: Whether to allow subset matches.
        only_local_matches: Whether to scope matches to the terms each example regards
            (examples with no regarded terms then contribute no matches).
    """

    vocab: str
    normalized: bool = True
    allow_subset_matches: bool = False
    only_local_matches: bool = False

    def validate(self) -> None:
        """Validate that the referenced vocabulary exists on the dataset."""
        self.metric.dataset.get_vocabulary(self.vocab)
