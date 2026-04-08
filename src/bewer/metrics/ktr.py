from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, metric_value

__all__ = ["KTR"]


class KTR_(ExampleMetric):
    @metric_value
    def num_matches(self) -> int:
        """Get the number of key terms correctly transcribed in the hypothesis text."""
        return self.parent_metric._kt_stats.get_example_metric(self.example).num_tp

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the number of key terms in the reference text."""
        return self.parent_metric._kt_stats.get_example_metric(self.example).num_ref_terms

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level key term recall."""
        stats = self.parent_metric._kt_stats.get_example_metric(self.example)
        if stats.num_ref_terms == 0:
            return 0.0
        return stats.num_tp / stats.num_ref_terms


@METRIC_REGISTRY.register("ktr")
class KTR(Metric):
    short_name_base = "KTR"
    long_name_base = "Key Term Recall"
    description = (
        "Key term recall (KTR) is computed as the number of key terms correctly transcribed divided by the total "
        "number of key terms in the reference texts. A key term may consist of one or more tokens, but is treated as "
        "a single unit for the purpose of KTR calculation. KTR is the complement of KTER (Key Term Error Rate) for a "
        "given vocabulary. Specifically, KTR = 1 - KTER."
    )
    example_cls = KTR_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the KTR metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            normalized: Whether to use normalized tokens for alignment and key term matching.
            allow_subsets: Whether to allow subset matches.
        """

        vocab: str
        normalized: bool = True
        allow_subsets: bool = True

        def validate(self) -> None:
            """Validate that the metric can be computed with the given parameters and source data."""
            is_dynamic_vocab = self.vocab in self.metric.dataset._dynamic_keyword_vocabs
            is_static_vocab = self.vocab in self.metric.dataset._static_keyword_vocabs
            if not is_dynamic_vocab and not is_static_vocab:
                raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset keyword vocabularies.")

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
    def num_matches(self) -> int:
        """Get the number of key terms correctly transcribed in the hypothesis texts."""
        return self._kt_stats.num_tp

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the number of key terms in the reference texts."""
        return self._kt_stats.num_ref_terms

    @metric_value(main=True)
    def value(self) -> float:
        """Get the key term recall."""
        if self._kt_stats.num_ref_terms == 0:
            return 0.0
        return self._kt_stats.num_tp / self._kt_stats.num_ref_terms
