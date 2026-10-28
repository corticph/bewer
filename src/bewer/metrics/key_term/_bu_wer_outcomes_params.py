from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import MetricParams

__all__ = ["BUWEROutcomesParams"]


@dataclass
class BUWEROutcomesParams(MetricParams):
    """Base class for B-WER/U-WER metric parameter dataclasses.

    Subclass this with @dataclass (or use directly) to define B-WER/U-WER metric
    parameters. Provides the common ``vocab``, ``normalized``, and ``multi_token``
    fields, validates them, and checks that the vocabulary exists on the dataset.

    Attributes:
        vocab: The vocabulary name to use for the biased/unbiased word split.
        normalized: Whether to use normalized tokens for alignment and vocabulary matching.
        multi_token: How multi-token vocabulary terms attribute bias — 'occurrence',
            'flag', or 'split'. 'occurrence' (default) counts a token as biased only
            when it is part of a matched term occurrence. 'flag' behaves identically but
            warns when the vocabulary contains terms that tokenize to multiple tokens.
            'split' treats every term token as an individually in-vocabulary word
            (word-list semantics).
    """

    vocab: str
    normalized: bool = True
    multi_token: str = "occurrence"

    def validate(self) -> None:
        """Validate the parameter values against the metric's dataset."""
        if self.multi_token not in ("occurrence", "flag", "split"):
            raise ValueError(f"multi_token must be one of 'occurrence', 'flag', or 'split', got '{self.multi_token}'.")
        if self.vocab not in self.metric.dataset.config.vocabularies:
            raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset config vocabularies.")
