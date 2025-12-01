from __future__ import annotations

from functools import cached_property

from rapidfuzz.distance import Levenshtein

from bewer.core.context import set_pipeline
from bewer.metrics.base import ExampleMetric


class WER(ExampleMetric):
    @cached_property
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference text."""
        with set_pipeline(*self.src_metric.pipeline):
            # if self.example._index == 0:
            #     import IPython; IPython.embed(using=False, banner1="WER ExampleMetric num_edits debugging")
            return Levenshtein.distance(
                self.example.ref.tokens.normalized,
                self.example.hyp.tokens.normalized,
            )

    @cached_property
    def ref_length(self) -> int:
        """Get the number of tokens in the reference text."""
        with set_pipeline(*self.src_metric.pipeline):
            return len(self.example.ref.tokens.normalized)

    @cached_property
    def value(self) -> float:
        """Get the example-level word error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length


class CER(ExampleMetric):
    @cached_property
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference text."""
        with set_pipeline(*self.src_metric.pipeline):
            return Levenshtein.distance(
                self.example.hyp.joined(normalized=True),
                self.example.ref.joined(normalized=True),
            )

    @cached_property
    def ref_length(self) -> int:
        """Get the number of characters in the reference text."""
        with set_pipeline(*self.src_metric.pipeline):
            return len(self.example.ref.joined(normalized=True))

    @cached_property
    def value(self) -> float:
        """Get the example-level character error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length
