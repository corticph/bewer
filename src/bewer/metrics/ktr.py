from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, dependency, metric_value

__all__ = ["KTR"]


class KTR_(ExampleMetric):
    @metric_value
    def num_matches(self) -> int:
        """Get the number of key terms correctly transcribed in the hypothesis text."""
        return self.parent_metric._kt_stats[self.example.index].num_tp

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the number of key terms in the reference text."""
        return self.parent_metric._kt_stats[self.example.index].num_ref_terms

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level key term recall."""
        stats = self.parent_metric._kt_stats[self.example.index]
        if (stats.num_tp + stats.num_fn) == 0:
            return 0.0
        return stats.num_tp / (stats.num_tp + stats.num_fn)


@METRIC_REGISTRY.register("ktr", tokenizer="key_term")
class KTR(Metric):
    short_name_base = "KTR"
    long_name_base = "Key Term Recall"
    description = (
        "Key term recall (KTR) is computed as TP / (TP + FN). "
        "When partial_credit=False (default, exact-match view), each key term occurrence is treated as a single unit: "
        "TP if every constituent token is correctly transcribed, FN otherwise. "
        "When partial_credit=True (partial-credit view), TP and FN are counted at the token-position level within "
        "key term spans, giving proportional credit for partially correct multi-token terms. "
        "KTR is the complement of KTER (Key Term Error Rate): KTR = 1 - KTER."
    )
    example_cls = KTR_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the KTR metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            normalized: Whether to use normalized tokens for alignment and key term matching.
            allow_subset_matches: Whether to allow subset matches.
            partial_credit: When False (default, exact-match view), each key term occurrence is a single
                TP or FN unit. When True (partial-credit view), TP and FN are counted at the
                token-position level within key term spans.
        """

        vocab: str
        normalized: bool = True
        allow_subset_matches: bool = False
        partial_credit: bool = False

        def validate(self) -> None:
            """Validate that the metric can be computed with the given parameters and source data."""
            if self.vocab not in self.metric.dataset._vocabularies:
                raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset key term vocabularies.")

    @dependency
    def _kt_stats(self):
        """Get the shared _KTStats metric instance."""
        return self.dataset.metrics._kt_stats(
            vocab=self.params.vocab,
            normalized=self.params.normalized,
            allow_subset_matches=self.params.allow_subset_matches,
            partial_credit=self.params.partial_credit,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_value
    def num_matches(self) -> int:
        """Get the number of correctly transcribed key term units (occurrences or token positions)."""
        return self._kt_stats.num_tp

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the number of key terms in the reference texts."""
        return self._kt_stats.num_ref_terms

    @metric_value(main=True)
    def value(self) -> float:
        """Get the key term recall."""
        if (self._kt_stats.num_tp + self._kt_stats.num_fn) == 0:
            return 0.0
        return self._kt_stats.num_tp / (self._kt_stats.num_tp + self._kt_stats.num_fn)
