from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, metric_value

__all__ = ["CER", "CER_"]


class CER_(ExampleMetric):
    @metric_value
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference text."""
        return Levenshtein.distance(
            self.example.hyp.joined(normalized=self.params.normalized),
            self.example.ref.joined(normalized=self.params.normalized),
        )

    @metric_value
    def ref_length(self) -> int:
        """Get the number of characters in the reference text."""
        return len(self.example.ref.joined(normalized=self.params.normalized))

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level character error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length


@METRIC_REGISTRY.register("cer")
class CER(Metric):
    short_name_base = "CER"
    long_name_base = "Character Error Rate"
    description = (
        "Character error rate (CER) is computed as the character-level edit distance between the reference "
        "and hypothesis texts, divided by the total number of characters in the reference texts."
    )
    example_cls = CER_

    @dataclass
    class param_schema(MetricParams):
        normalized: bool = True

    @metric_value
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference texts."""
        return sum([self.get_example_metric(example).num_edits for example in self._src])

    @metric_value
    def ref_length(self) -> int:
        """Get the number of characters in the reference texts."""
        return sum([self.get_example_metric(example).ref_length for example in self._src])

    @metric_value(main=True)
    def value(self) -> float:
        """Get the character error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length
