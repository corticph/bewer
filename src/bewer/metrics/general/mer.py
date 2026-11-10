"""Match Error Rate (MER) metric.

Defined by Morris, Maier and Green (2004), "From WER and RIL to MER and WIL: improved
evaluation measures for connected speech recognition", ICSLP. Matches the jiwer
reference implementation, which computes MER from a word-level Levenshtein alignment.
"""

from __future__ import annotations

from bewer.alignment import Alignment
from bewer.metrics.base import (
    ALIGNMENT_ALGORITHMS,
    METRIC_REGISTRY,
    AlignmentParams,
    ExampleMetric,
    Metric,
    metric_property,
)

__all__ = ["MER"]


class MER_(ExampleMetric):
    @metric_property
    def _alignment(self) -> Alignment:
        return self.parent_metric._alignment[self.example.index].alignment

    @metric_property(is_value=True)
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference text."""
        return self._alignment.num_edits

    @metric_property(is_value=True)
    def num_matches(self) -> int:
        """Get the number of matched tokens between the hypothesis and reference text."""
        return self._alignment.num_matches

    @metric_property(is_main=True)
    def value(self) -> float:
        """Get the example-level match error rate."""
        denom = self.num_edits + self.num_matches
        if denom == 0:
            return 0.0
        return self.num_edits / denom


@METRIC_REGISTRY.register("mer")
class MER(Metric):
    short_name_base = "MER"
    long_name_base = "Match Error Rate"
    description = (
        "Match error rate (MER) is computed as the number of edits between the reference and "
        "hypothesis texts, divided by the total number of matched and edited words. Unlike WER, "
        "its denominator includes inserted words, so it never exceeds 1. It was defined by "
        "Morris, Maier and Green (2004, From WER and RIL to MER and WIL)."
    )
    example_cls = MER_
    param_schema = AlignmentParams
    supported_alignments = ALIGNMENT_ALGORITHMS

    @metric_property(is_dependency=True)
    def _alignment(self):
        """The shared alignment metric instance selected by ``params.alignment``."""
        return self._get_alignment_metric()

    @metric_property(is_value=True)
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference texts."""
        return sum(em.num_edits for em in self)

    @metric_property(is_value=True)
    def num_matches(self) -> int:
        """Get the number of matched tokens between the hypothesis and reference texts."""
        return sum(em.num_matches for em in self)

    @metric_property(is_main=True)
    def value(self) -> float:
        """Get the match error rate."""
        denom = self.num_edits + self.num_matches
        if denom == 0:
            return 0.0
        return self.num_edits / denom
