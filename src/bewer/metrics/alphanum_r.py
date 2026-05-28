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
        "via a regex predicate applied to each token's case-preserving form (Token.raw). The default "
        "pattern matches a token when either: (a) it contains at least one uppercase letter and is not "
        "the shape <single uppercase letter><lowercase letters> (so ordinary capitalised words such as "
        "'Patient', 'Hello', 'The' are excluded, while tokens with an internal uppercase letter, multiple "
        "uppercase letters, or non-letter characters alongside an uppercase letter are kept), or (b) it "
        "consists of one or more letters followed by at least one digit. The pattern uses Unicode letter "
        "properties (\\p{Lu}, \\p{Ll}, \\p{L}), so Greek letters and other scripts are classified by case "
        "and integrate naturally. Each match is a single token. Note: this metric is inherently "
        "case-sensitive — ASR systems that emit lowercase-only output will show low recall on case-only "
        "entities (e.g. 'mri' vs 'MRI' counts as a miss)."
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
