from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, dependency, metric_value

__all__ = ["KTF"]


class KTF_(ExampleMetric):
    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level key term F-score."""
        stats = self.parent_metric._kt_stats.get_example_metric(self.example)
        beta_sq = self.params.beta**2
        denominator = (1 + beta_sq) * stats.num_tp + beta_sq * stats.num_fn + stats.num_fp
        if denominator == 0:
            return 0.0
        return (1 + beta_sq) * stats.num_tp / denominator


@METRIC_REGISTRY.register("ktf", tokenizer="key_term")
class KTF(Metric):
    short_name_base = "KTF"
    long_name_base = "Key Term F-Score"
    description = (
        "Key term F-score (KTF) is the weighted harmonic mean of key term precision (KTP) and key term recall (KTR). "
        "The beta parameter controls the trade-off: beta > 1 weights recall more heavily, beta < 1 weights precision "
        "more heavily, and beta = 1 (default) gives the standard F1 score. "
        "At the dataset level, KTF is computed as a micro F-score: TP, FN, and FP counts are summed across all "
        "examples before applying the formula, so examples with more key terms contribute more to the final score. "
        "Computed as: (1 + beta²) × TP / ((1 + beta²) × TP + beta² × FN + FP)."
    )
    example_cls = KTF_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the KTF metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            normalized: Whether to use normalized tokens for alignment and key term matching.
            allow_subset_matches: Whether to allow subset matches.
            beta: F-score beta parameter. beta=1 gives F1 (equal weight to precision and recall).
                beta>1 weights recall more heavily; beta<1 weights precision more heavily.
            only_local_matches: If True, restrict matching to per-example local key terms only.
        """

        vocab: str
        normalized: bool = True
        allow_subset_matches: bool = False
        beta: float = 1.0
        only_local_matches: bool = False

        def validate(self) -> None:
            """Validate that the metric can be computed with the given parameters and source data."""
            if self.beta <= 0:
                raise ValueError(f"beta must be positive, got {self.beta}.")
            is_global_vocab = self.vocab in self.metric.dataset._global_key_term_vocabs
            is_local_vocab = self.vocab in self.metric.dataset._local_key_term_vocabs
            if not is_global_vocab and not is_local_vocab:
                raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset key term vocabularies.")

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

    @metric_value(main=True)
    def value(self) -> float:
        """Get the key term F-score."""
        beta_sq = self.params.beta**2
        denominator = (1 + beta_sq) * self._kt_stats.num_tp + beta_sq * self._kt_stats.num_fn + self._kt_stats.num_fp
        if denominator == 0:
            return 0.0
        return (1 + beta_sq) * self._kt_stats.num_tp / denominator
