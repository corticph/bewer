from __future__ import annotations

from dataclasses import dataclass

import regex as re

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, dependency, metric_value
from bewer.preprocessing.regex_match import ALPHANUM_DEFAULT_PATTERN

__all__ = ["AlphaNumP"]


class AlphaNumP_(ExampleMetric):
    @metric_value
    def num_matches(self) -> int:
        """Number of alphanumerical entities correctly transcribed in the hypothesis."""
        return self.parent_metric._alphanum_stats.get_example_metric(self.example).num_tp

    @metric_value
    def num_fp(self) -> int:
        """Number of spurious alphanumerical entities in the hypothesis."""
        return self.parent_metric._alphanum_stats.get_example_metric(self.example).num_fp

    @metric_value(main=True)
    def value(self) -> float:
        stats = self.parent_metric._alphanum_stats.get_example_metric(self.example)
        if (stats.num_tp + stats.num_fp) == 0:
            return 0.0
        return stats.num_tp / (stats.num_tp + stats.num_fp)


@METRIC_REGISTRY.register("alphanum_p")
class AlphaNumP(Metric):
    short_name_base = "AlphaNumP"
    long_name_base = "Alphanumerical Entity Precision"
    description = (
        "Alphanumerical entity precision (AlphaNumP) is computed as TP / (TP + FP), where TP is the number of "
        "alphanumerical entities correctly transcribed in the hypothesis and FP is the number of spurious "
        "alphanumerical entities. Entities are detected via a regex predicate over each token's "
        "case-preserving form, covering initialisms (MRI, FBI), acronyms (NATO), chemical/unit notation "
        "(CH3, mmHg, CO2), mixed-case medical/brand terms (HbA1c, mRNA, iPhone), and digit-prefixed "
        "entities with an uppercase tail (3D, 5G). The default pattern is Unicode-aware: Greek letters "
        "integrate naturally (ΔG, μM, β2). Each match is a single token. Note: this metric is inherently "
        "case-sensitive — ASR systems that emit lowercase-only output will show low recall on case-only "
        "entities (e.g. mri vs MRI counts as a miss), and period-style abbreviations (Dr., e.g.) are not "
        "detected because the default tokenizer splits on periods."
    )
    example_cls = AlphaNumP_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the AlphaNumP metric.

        Attributes:
            pattern: Regex pattern matched against each token's case-preserving form via `fullmatch`.
        """

        pattern: str = ALPHANUM_DEFAULT_PATTERN

        def validate(self) -> None:
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

    @metric_value
    def num_matches(self) -> int:
        return self._alphanum_stats.num_tp

    @metric_value
    def num_fp(self) -> int:
        return self._alphanum_stats.num_fp

    @metric_value(main=True)
    def value(self) -> float:
        if (self._alphanum_stats.num_tp + self._alphanum_stats.num_fp) == 0:
            return 0.0
        return self._alphanum_stats.num_tp / (self._alphanum_stats.num_tp + self._alphanum_stats.num_fp)
