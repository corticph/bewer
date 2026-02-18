from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, metric_value


class WER_(ExampleMetric):
    @metric_value
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference text."""
        if self.params.normalized:
            return Levenshtein.distance(
                self.example.ref.tokens.normalized,
                self.example.hyp.tokens.normalized,
            )
        return Levenshtein.distance(
            self.example.ref.tokens.raw,
            self.example.hyp.tokens.raw,
        )

    @metric_value
    def ref_length(self) -> int:
        """Get the number of tokens in the reference text."""
        if self.params.normalized:
            return len(self.example.ref.tokens.normalized)
        return len(self.example.ref.tokens.raw)

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level word error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length


@METRIC_REGISTRY.register("wer")
class WER(Metric):
    short_name_base = "WER"
    long_name_base = "Word Error Rate"
    description = (
        "Word Error Rate (WER) is computed as the token-level (i.e., word-level) edit distance between the reference "
        "and hypothesis texts, divided by the total number of tokens in the reference texts."
    )
    example_cls = WER_

    @dataclass
    class param_schema(MetricParams):
        normalized: bool = True

    @metric_value
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference texts."""
        return sum([self._get_example_metric(example).num_edits for example in self._src])

    @metric_value
    def ref_length(self) -> int:
        """Get the number of tokens in the reference texts."""
        return sum([self._get_example_metric(example).ref_length for example in self._src])

    @metric_value(main=True)
    def value(self) -> float:
        """Get the word error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length
