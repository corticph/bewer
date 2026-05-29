from __future__ import annotations

from dataclasses import dataclass

import regex as re

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, dependency, metric_value
from bewer.preprocessing.regex_match import ALPHANUM_DEFAULT_PATTERN

__all__ = ["AlphaNumR"]


class AlphaNumR_(ExampleMetric):
    @metric_value
    def num_matches(self) -> int:
        """Number of alphanumerical entities correctly transcribed in the hypothesis."""
        return self.parent_metric._alphanum_stats.get_example_metric(self.example).num_tp

    @metric_value
    def num_ref_terms(self) -> int:
        """Number of alphanumerical entities in the reference text."""
        return self.parent_metric._alphanum_stats.get_example_metric(self.example).num_ref_terms

    @metric_value(main=True)
    def value(self) -> float:
        stats = self.parent_metric._alphanum_stats.get_example_metric(self.example)
        if (stats.num_tp + stats.num_fn) == 0:
            return 0.0
        return stats.num_tp / (stats.num_tp + stats.num_fn)


@METRIC_REGISTRY.register("alphanum_r")
class AlphaNumR(Metric):
    short_name_base = "AlphaNumR"
    long_name_base = "Alphanumerical Entity Recall"
    description = (
        "Alphanumerical entity recall (AlphaNumR) is computed as TP / (TP + FN), where TP is the number of "
        "alphanumerical entities correctly transcribed and FN is the number missed. Entities are detected "
        "via a regex predicate applied to each token (and to hyphen-joined runs of consecutive tokens) "
        "using the token's case-preserving form. The default pattern matches a candidate string when (a) "
        "it contains at least one uppercase letter and is not an ordinary capitalised compound (one or "
        "more parts that are each either init-cap or all lowercase, joined by hyphens — so 'Patient', "
        "'Hello', 'Hello-World', 'up-to-date' are excluded while 'MRI', 'mmHg', 'CT-scan', 'X-ray', "
        "'pre-MRI', 'MRI-CT' are kept), (b) it consists of one or more letters followed by at least one "
        "digit, or (c) it contains at least one Greek letter (so μg, α-helix, β-blocker are entities "
        "regardless of case). Hyphen structure is enforced strictly: a multi-token compound entity in "
        "ref counts as TP only when hyp preserves the hyphens — otherwise the entity is FN. The pattern "
        "uses Unicode letter properties (\\p{Lu}, \\p{Ll}, \\p{L}, \\p{Greek}), so Greek letters and "
        "other scripts integrate naturally. This metric is inherently case-sensitive — ASR systems that "
        "emit lowercase-only output will show low recall on case-only entities (e.g. 'mri' vs 'MRI' "
        "counts as a miss)."
    )
    example_cls = AlphaNumR_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the AlphaNumR metric.

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
    def num_ref_terms(self) -> int:
        return self._alphanum_stats.num_ref_terms

    @metric_value(main=True)
    def value(self) -> float:
        if (self._alphanum_stats.num_tp + self._alphanum_stats.num_fn) == 0:
            return 0.0
        return self._alphanum_stats.num_tp / (self._alphanum_stats.num_tp + self._alphanum_stats.num_fn)
