from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import ALIGNMENT_ALGORITHMS, METRIC_REGISTRY, ExampleMetric, Metric, metric_property

from ._keyterm_params import KeyTermSubsetParams

__all__ = ["KTF"]


class KTF_(ExampleMetric):
    @metric_property(is_main=True)
    def value(self) -> float:
        """Get the example-level key term F-score."""
        outcomes = self.parent_metric._kt_outcomes[self.example.index]
        beta_sq = self.params.beta**2
        denominator = (1 + beta_sq) * outcomes.num_tp + beta_sq * outcomes.num_fn + outcomes.num_fp
        if denominator == 0:
            return 0.0
        return (1 + beta_sq) * outcomes.num_tp / denominator


@METRIC_REGISTRY.register("ktf", tokenizer="key_term")
class KTF(Metric):
    short_name_base = "KTF"
    long_name_base = "Key-Term F-Score"
    description = (
        "Key term F-score (KTF) is the weighted harmonic mean of key term precision (KTP) and key term recall (KTR). "
        "The beta parameter controls the trade-off: beta > 1 weights recall more heavily, beta < 1 weights precision "
        "more heavily, and beta = 1 (default) gives the standard F1 score. "
        "At the dataset level, KTF is computed as a micro F-score: TP, FN, and FP counts are summed across all "
        "examples before applying the formula, so examples with more key terms contribute more to the final score. "
        "Computed as: (1 + beta²) × TP / ((1 + beta²) × TP + beta² × FN + FP)."
    )
    example_cls = KTF_
    supported_alignments = ALIGNMENT_ALGORITHMS

    @dataclass
    class param_schema(KeyTermSubsetParams):
        """Parameters for the KTF metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            allow_subset_matches: Whether to allow subset matches.
            partial_credit: When False (default, exact-match view), each key term occurrence is a single
                TP, FN, or FP unit. When True (partial-credit view), TP, FN, and FP are counted at the
                token-position level within key term spans.
            beta: F-score beta parameter. beta=1 gives F1 (equal weight to precision and recall).
                beta>1 weights recall more heavily; beta<1 weights precision more heavily.
        """

        partial_credit: bool = False
        beta: float = 1.0

        def validate(self) -> None:
            """Validate that the metric can be computed with the given parameters and source data."""
            if self.beta <= 0:
                raise ValueError(f"beta must be positive, got {self.beta}.")
            super().validate()

    @metric_property(is_dependency=True)
    def _kt_outcomes(self):
        """Get the shared _KTOutcomes metric instance."""
        return self.dataset.metrics._kt_outcomes(
            vocab=self.params.vocab,
            alignment=self.params.alignment,
            allow_subset_matches=self.params.allow_subset_matches,
            partial_credit=self.params.partial_credit,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_property(is_main=True)
    def value(self) -> float:
        """Get the key term F-score."""
        beta_sq = self.params.beta**2
        denominator = (
            (1 + beta_sq) * self._kt_outcomes.num_tp + beta_sq * self._kt_outcomes.num_fn + self._kt_outcomes.num_fp
        )
        if denominator == 0:
            return 0.0
        return (1 + beta_sq) * self._kt_outcomes.num_tp / denominator
