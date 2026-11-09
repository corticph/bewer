"""Word Information Preserved (WIP) metric.

Defined by Morris, Maier and Green (2004), "From WER and RIL to MER and WIL: improved
evaluation measures for connected speech recognition", ICSLP. Matches the jiwer
reference implementation, which computes WIP from a word-level Levenshtein alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

from bewer.alignment import Alignment
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, metric_property

__all__ = ["WIP"]


class WIP_(ExampleMetric):
    @metric_property
    def _alignment(self) -> Alignment:
        return self.parent_metric._alignment[self.example.index].alignment

    @metric_property(is_value=True)
    def num_matches(self) -> int:
        """Get the number of matched tokens between the hypothesis and reference text."""
        return self._alignment.num_matches

    @metric_property(is_value=True)
    def ref_length(self) -> int:
        """Get the number of tokens in the reference text."""
        if self.params.normalized:
            return len(self.example.ref.tokens.normalized)
        return len(self.example.ref.tokens.standardized)

    @metric_property(is_value=True)
    def hyp_length(self) -> int:
        """Get the number of tokens in the hypothesis text."""
        if self.params.normalized:
            return len(self.example.hyp.tokens.normalized)
        return len(self.example.hyp.tokens.standardized)

    @metric_property(is_main=True)
    def value(self) -> float:
        """Get the example-level word information preserved."""
        if self.ref_length == 0 and self.hyp_length == 0:
            return 1.0
        if self.ref_length == 0 or self.hyp_length == 0:
            return 0.0
        return (self.num_matches / self.ref_length) * (self.num_matches / self.hyp_length)


@METRIC_REGISTRY.register("wip")
class WIP(Metric):
    short_name_base = "WIP"
    long_name_base = "Word Information Preserved"
    description = (
        "Word information preserved (WIP) is the product of the share of reference words that "
        "were matched and the share of hypothesis words that were matched: "
        "(num_matches / ref_length) * (num_matches / hyp_length). It approximates the proportion "
        "of word information communicated and never exceeds 1. It was defined by Morris, Maier "
        "and Green (2004, From WER and RIL to MER and WIL)."
    )
    example_cls = WIP_

    @dataclass
    class param_schema(MetricParams):
        normalized: bool = True

    @metric_property(is_dependency=True)
    def _alignment(self):
        return self.dataset.metrics.levenshtein(
            normalized=self.params.normalized,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_property(is_value=True)
    def num_matches(self) -> int:
        """Get the number of matched tokens between the hypothesis and reference texts."""
        return sum(em.num_matches for em in self)

    @metric_property(is_value=True)
    def ref_length(self) -> int:
        """Get the number of tokens in the reference texts."""
        return sum(em.ref_length for em in self)

    @metric_property(is_value=True)
    def hyp_length(self) -> int:
        """Get the number of tokens in the hypothesis texts."""
        return sum(em.hyp_length for em in self)

    @metric_property(is_main=True)
    def value(self) -> float:
        """Get the word information preserved."""
        if self.ref_length == 0 and self.hyp_length == 0:
            return 1.0
        if self.ref_length == 0 or self.hyp_length == 0:
            return 0.0
        return (self.num_matches / self.ref_length) * (self.num_matches / self.hyp_length)
