from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, metric_value

__all__ = ["MTR"]


class MTR_(ExampleMetric):
    @metric_value
    def num_matches(self) -> int:
        """Get the number of medical keywords correctly transcribed in the hypothesis text."""
        return self.num_keywords - self.parent_metric._kwer_metric.get_example_metric(self.example).num_errors

    @metric_value
    def num_keywords(self) -> int:
        """Get the number of medical keywords in the reference text."""
        return self.parent_metric._kwer_metric.get_example_metric(self.example).num_keywords

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level medical term recall."""
        return 1 - self.parent_metric._kwer_metric.get_example_metric(self.example).value


@METRIC_REGISTRY.register("mtr")
class MTR(Metric):
    short_name_base = "MTR"
    long_name_base = "Medical Term Recall"
    description = (
        "Medical term recall (MTR) is computed as the number of medical keywords (or key terms) correctly identified "
        "in the hypothesis texts, divided by the total number of medical keywords identified in the reference texts. "
        "A keyword may consist of one or more tokens, but is treated as a single unit for the purpose of MTR "
        "calculation. MTR is the complement of KWER (Keyword Error Rate) with the keyword vocabulary set to "
        "'medical_terms'. Specifically, MTR can be calculated as 1 - KWER."
    )
    example_cls = MTR_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the MTR metric.

        Attributes:
            normalized: Whether to use normalized tokens for alignment and keyword matching.
        """

        normalized: bool = True

        def validate(self) -> None:
            """Validate that the metric can be computed with the given parameters and source data."""
            is_dynamic_vocab = "medical_terms" in self.metric.dataset._dynamic_keyword_vocabs
            is_static_vocab = "medical_terms" in self.metric.dataset._static_keyword_vocabs
            if not is_dynamic_vocab and not is_static_vocab:
                raise ValueError("Vocabulary 'medical_terms' not found in dataset keyword vocabularies.")

    @cached_property
    def _kwer_metric(self):
        """Get the corresponding KWER metric instance for calculating MTR."""
        return self.dataset.metrics.kwer(
            vocab="medical_terms",
            normalized=self.params.normalized,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_value
    def num_matches(self) -> int:
        """Get the number of medical keywords correctly transcribed in the hypothesis texts."""
        return self._kwer_metric.num_keywords - self._kwer_metric.num_errors

    @metric_value
    def num_keywords(self) -> int:
        """Get the number of medical keywords in the reference texts."""
        return self._kwer_metric.num_keywords

    @metric_value(main=True)
    def value(self) -> float:
        """Get the medical term recall."""
        return 1 - self._kwer_metric.value
