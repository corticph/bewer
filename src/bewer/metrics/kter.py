from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, metric_value

__all__ = ["KTER"]


class KTER_(ExampleMetric):
    @metric_value
    def num_errors(self) -> int:
        """Get the number of key terms incorrectly transcribed in the hypothesis text."""
        return self.parent_metric._kt_stats.get_example_metric(self.example).num_fn

    @metric_value
    def num_key_terms(self) -> int:
        """Get the number of key terms in the reference text."""
        return self.parent_metric._kt_stats.get_example_metric(self.example).num_ref_terms

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level key term error rate."""
        stats = self.parent_metric._kt_stats.get_example_metric(self.example)
        if stats.num_ref_terms == 0:
            return float(stats.num_fn)
        return stats.num_fn / stats.num_ref_terms


@METRIC_REGISTRY.register("kter", tokenizer="key_term")
class KTER(Metric):
    short_name_base = "KTER"
    long_name_base = "Key Term Error Rate"
    description = (
        "Key term error rate (KTER) is computed as the number of key terms incorrectly transcribed in "
        "the hypothesis texts, divided by the total number of key terms identified in the reference texts. "
        "A key term may consist of one or more tokens, but is treated as a single unit for the purpose of "
        "KTER calculation."
    )
    example_cls = KTER_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the KTER metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            normalized: Whether to use normalized tokens for alignment and key term matching.
            allow_subsets: Whether to allow subset matches. If False, overlapping key term matches
                are deduplicated, keeping only the longest match.
        """

        vocab: str
        normalized: bool = True
        allow_subsets: bool = True

        def validate(self) -> None:
            """Validate that the metric can be computed with the given parameters and source data."""
            is_dynamic_vocab = self.vocab in self.metric.dataset._dynamic_key_term_vocabs
            is_static_vocab = self.vocab in self.metric.dataset._static_key_term_vocabs
            if not is_dynamic_vocab and not is_static_vocab:
                raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset key term vocabularies.")

    @cached_property
    def _kt_stats(self):
        """Get the shared _KTStats metric instance."""
        return self.dataset.metrics._kt_stats(
            vocab=self.params.vocab,
            normalized=self.params.normalized,
            allow_subsets=self.params.allow_subsets,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_value
    def num_errors(self) -> int:
        """Get the number of key terms incorrectly transcribed in the hypothesis texts."""
        return self._kt_stats.num_fn

    @metric_value
    def num_key_terms(self) -> int:
        """Get the number of key terms in the reference texts."""
        return self._kt_stats.num_ref_terms

    @metric_value(main=True)
    def value(self) -> float:
        """Get the key term error rate."""
        if self._kt_stats.num_ref_terms == 0:
            return float(self._kt_stats.num_fn)
        return self._kt_stats.num_fn / self._kt_stats.num_ref_terms
