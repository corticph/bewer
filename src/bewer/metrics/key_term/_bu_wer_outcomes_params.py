from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import MetricParams

__all__ = ["BUWEROutcomesParams"]


@dataclass
class BUWEROutcomesParams(MetricParams):
    """Base class for B-WER/U-WER metric parameter dataclasses.

    Subclass this with @dataclass (or use directly) to define B-WER/U-WER metric
    parameters. Provides the common ``vocab`` and ``normalized`` fields, validates
    them, and checks that the vocabulary exists on the dataset.

    Attributes:
        vocab: The vocabulary name to use for the biased/unbiased word split.
        normalized: Whether to use normalized tokens for alignment and vocabulary matching.
    """

    vocab: str
    normalized: bool = True

    def validate(self) -> None:
        """Validate the parameter values against the metric's dataset."""
        if self.vocab not in self.metric.dataset.config.vocabularies:
            raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset config vocabularies.")
