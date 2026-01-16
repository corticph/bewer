from __future__ import annotations

from rapidfuzz.distance import Levenshtein

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_value


class WER_(ExampleMetric):
    @metric_value
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference text."""
        return Levenshtein.distance(
            self.example.ref.tokens.normalized,
            self.example.hyp.tokens.normalized,
        )

    @metric_value
    def ref_length(self) -> int:
        """Get the number of tokens in the reference text."""
        return len(self.example.ref.tokens.normalized)

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level word error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length


@METRIC_REGISTRY.register("wer")
class WER(Metric):
    short_name = "WER"
    long_name = "Word Error Rate"
    description = (
        "Word Error Rate (WER) is computed as the token-level (i.e., word-level) edit distance between the reference "
        "and hypothesis texts, divided by the total number of tokens in the reference texts."
    )
    example_cls = WER_

    @metric_value
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference texts."""
        return sum([example.metrics.get(self.name).num_edits for example in self._src_dataset])

    @metric_value
    def ref_length(self) -> int:
        """Get the number of tokens in the reference texts."""
        return sum([example.metrics.get(self.name).ref_length for example in self._src_dataset])

    @metric_value(main=True)
    def value(self) -> float:
        """Get the word error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length
