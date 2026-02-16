from __future__ import annotations

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_value


class DatasetSummary_(ExampleMetric):
    @metric_value
    def num_ref_words(self) -> int:
        """The number of tokens in the reference text."""
        return len(self.example.ref.tokens)

    @metric_value
    def num_ref_chars(self) -> int:
        """The number of characters in the reference text."""
        return len(self.example.ref.standardized)

    @metric_value
    def num_hyp_words(self) -> int:
        """The number of tokens in the hypothesis text."""
        return len(self.example.hyp.tokens)

    @metric_value
    def num_hyp_chars(self) -> int:
        """The number of characters in the hypothesis text."""
        return len(self.example.hyp.standardized)


@METRIC_REGISTRY.register("summary")
class DatasetSummary(Metric):
    short_name = "Summary"
    long_name = "Dataset Summary"
    description = (
        "Dataset Summary provides basic statistics about the dataset, including the number of words and characters "
        "in both the reference and hypothesis texts."
    )
    example_cls = DatasetSummary_

    @metric_value
    def num_examples(self) -> int:
        """The total number of examples in the dataset."""
        return len(self._src)

    @metric_value
    def num_ref_words(self) -> int:
        """The total number of tokens in the reference texts."""
        return sum(example.metrics.get(self.name).num_ref_words for example in self._src)

    @metric_value
    def num_ref_chars(self) -> int:
        """The total number of characters in the reference texts."""
        return sum(example.metrics.get(self.name).num_ref_chars for example in self._src)

    @metric_value
    def num_hyp_words(self) -> int:
        """The total number of tokens in the hypothesis texts."""
        return sum(example.metrics.get(self.name).num_hyp_words for example in self._src)

    @metric_value
    def num_hyp_chars(self) -> int:
        """The total number of characters in the hypothesis texts."""
        return sum(example.metrics.get(self.name).num_hyp_chars for example in self._src)
