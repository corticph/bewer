from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, dependency, metric_value

__all__ = ["KTER"]


class KTER_(ExampleMetric):
    @metric_value
    def num_errors(self) -> int:
        """Get the number of key terms incorrectly transcribed in the hypothesis text."""
        return self.parent_metric._kt_stats[self.example.index].num_fn

    @metric_value
    def num_key_terms(self) -> int:
        """Get the number of key terms in the reference text."""
        return self.parent_metric._kt_stats[self.example.index].num_ref_terms

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level key term error rate."""
        stats = self.parent_metric._kt_stats[self.example.index]
        denom = stats.num_tp + stats.num_fn
        if denom == 0:
            return 0.0
        return stats.num_fn / denom


@METRIC_REGISTRY.register("kter", tokenizer="key_term")
class KTER(Metric):
    short_name_base = "KTER"
    long_name_base = "Key Term Error Rate"
    description = (
        "Key term error rate (KTER) is computed as FN / (TP + FN). "
        "When partial_credit=False (default, exact-match view), each key term occurrence is treated as a single unit: "
        "FN if any constituent token is incorrectly transcribed, TP otherwise. The denominator equals the number of "
        "reference key term occurrences. "
        "When partial_credit=True (partial-credit view), FN and TP are counted at the token-position level within "
        "key term spans; the denominator equals the total number of reference term token positions. "
        "KTER is the complement of KTR: KTER = 1 - KTR."
    )
    example_cls = KTER_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the KTER metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            normalized: Whether to use normalized tokens for alignment and key term matching.
            allow_subset_matches: Whether to allow subset matches. If False, overlapping key term matches
                are deduplicated, keeping only the longest match.
            partial_credit: When False (default, exact-match view), each key term occurrence is a single
                FN or TP unit. When True (partial-credit view), FN and TP are counted at the
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
        denom = self._kt_stats.num_tp + self._kt_stats.num_fn
        if denom == 0:
            return 0.0
        return self._kt_stats.num_fn / denom
