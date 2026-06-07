from __future__ import annotations

from bewer.metrics._kt_params import KeyTermMetricParams
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, dependency, metric_value

__all__ = ["KTP"]


class KTP_(ExampleMetric):
    @metric_value
    def num_matches(self) -> int:
        """Get the number of key terms correctly transcribed in the hypothesis text."""
        return self.parent_metric._kt_stats[self.example.index].num_tp

    @metric_value
    def num_fp(self) -> int:
        """Get the number of spurious key term occurrences in the hypothesis text."""
        return self.parent_metric._kt_stats[self.example.index].num_fp

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level key term precision."""
        stats = self.parent_metric._kt_stats[self.example.index]
        if (stats.num_tp + stats.num_fp) == 0:
            return 0.0
        return stats.num_tp / (stats.num_tp + stats.num_fp)


@METRIC_REGISTRY.register("ktp", tokenizer="key_term")
class KTP(Metric):
    short_name_base = "KTP"
    long_name_base = "Key Term Precision"
    description = (
        "Key term precision (KTP) is computed as TP / (TP + FP), where TP is the number of key terms correctly "
        "transcribed and FP is the number of spurious key term occurrences in the hypothesis. A key term may "
        "consist of one or more tokens, but is treated as a single unit for the purpose of KTP calculation."
    )
    example_cls = KTP_
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
    def num_matches(self) -> int:
        """Get the number of key terms correctly transcribed in the hypothesis texts."""
        return self._kt_stats.num_tp

    @metric_value
    def num_fp(self) -> int:
        """Get the number of spurious key term occurrences in the hypothesis texts."""
        return self._kt_stats.num_fp

    @metric_value(main=True)
    def value(self) -> float:
        """Get the key term precision."""
        if (self._kt_stats.num_tp + self._kt_stats.num_fp) == 0:
            return 0.0
        return self._kt_stats.num_tp / (self._kt_stats.num_tp + self._kt_stats.num_fp)
