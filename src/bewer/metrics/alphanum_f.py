from __future__ import annotations

from dataclasses import dataclass

import regex as re

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, dependency, metric_value
from bewer.preprocessing.regex_match import ALPHANUM_DEFAULT_PATTERN

__all__ = ["AlphaNumF"]


class AlphaNumF_(ExampleMetric):
    @metric_value(main=True)
    def value(self) -> float:
        stats = self.parent_metric._alphanum_stats.get_example_metric(self.example)
        beta_sq = self.params.beta**2
        denominator = (1 + beta_sq) * stats.num_tp + beta_sq * stats.num_fn + stats.num_fp
        if denominator == 0:
            return 0.0
        return (1 + beta_sq) * stats.num_tp / denominator


@METRIC_REGISTRY.register("alphanum_f")
class AlphaNumF(Metric):
    short_name_base = "AlphaNumF"
    long_name_base = "Alphanumerical Entity F-Score"
    description = (
        "Alphanumerical entity F-score (AlphaNumF) is the weighted harmonic mean of AlphaNumP and AlphaNumR. "
        "The beta parameter controls the trade-off: beta > 1 weights recall more heavily, beta < 1 weights "
        "precision more heavily, and beta = 1 (default) gives the standard F1 score. At the dataset level, "
        "AlphaNumF is computed as a micro F-score: TP, FN, and FP counts are summed across all examples "
        "before applying the formula. Computed as: (1 + beta²) × TP / ((1 + beta²) × TP + beta² × FN + FP). "
        "See AlphaNumP / AlphaNumR for the detection regex and case-sensitivity behavior."
    )
    example_cls = AlphaNumF_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the AlphaNumF metric.

        Attributes:
            pattern: Regex pattern matched against each token's case-preserving form via `fullmatch`.
            beta: F-score beta parameter. beta=1 gives F1; beta>1 weights recall; beta<1 weights precision.
        """

        pattern: str = ALPHANUM_DEFAULT_PATTERN
        beta: float = 1.0

        def validate(self) -> None:
            if self.beta <= 0:
                raise ValueError(f"beta must be positive, got {self.beta}.")
            try:
                self._compiled = re.compile(self.pattern)
            except re.error as e:
                raise ValueError(f"Invalid alphanumerical regex pattern: {e}") from e

    @dependency
    def _alphanum_stats(self):
        return self.dataset.metrics._alphanum_stats(
            pattern=self.params.pattern,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_value(main=True)
    def value(self) -> float:
        beta_sq = self.params.beta**2
        stats = self._alphanum_stats
        denominator = (1 + beta_sq) * stats.num_tp + beta_sq * stats.num_fn + stats.num_fp
        if denominator == 0:
            return 0.0
        return (1 + beta_sq) * stats.num_tp / denominator
