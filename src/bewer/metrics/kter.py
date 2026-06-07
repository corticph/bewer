from __future__ import annotations

from bewer.metrics._kt_params import KeyTermMetricParams
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, dependency, metric_value

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
    param_schema = KeyTermMetricParams

    @dependency
    def _kt_stats(self):
        """Get the shared _KTStats metric instance."""
        return self.dataset.metrics._kt_stats(
            vocab=self.params.vocab,
            normalized=self.params.normalized,
            allow_subset_matches=self.params.allow_subset_matches,
            only_local_matches=self.params.only_local_matches,
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
