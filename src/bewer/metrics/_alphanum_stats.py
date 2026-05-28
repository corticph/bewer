from __future__ import annotations

from dataclasses import dataclass

import regex as re

from bewer.alignment import Alignment
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, metric_value
from bewer.preprocessing.regex_match import ALPHANUM_DEFAULT_PATTERN, match_token_regex

__all__: list[str] = []


class _AlphaNumStats_(ExampleMetric):
    def _get_alignment(self):
        return self.example.metrics.levenshtein(
            normalized=False,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        ).alignment

    def _get_ref_matches(self) -> list[slice]:
        return match_token_regex(self.example.ref.tokens, self.params._compiled)

    def _get_hyp_matches(self) -> list[slice]:
        return match_token_regex(self.example.hyp.tokens, self.params._compiled)

    @metric_value
    def _ref_match_classification(self) -> dict[str, list[Alignment]]:
        """Partition ref matches into tp and fn alignment segments."""
        matches = self._get_ref_matches()
        if not matches:
            return {"tp": [], "fn": []}
        alignment = self._get_alignment()
        tp: list[Alignment] = []
        fn: list[Alignment] = []
        for m in matches:
            op_start = alignment.ref_index_mapping.get(m.start)
            op_stop = alignment.ref_index_mapping.get(m.stop - 1) + 1
            segment: Alignment = alignment[op_start:op_stop]
            if segment.num_edits == 0:
                tp.append(segment)
            else:
                fn.append(segment)
        return {"tp": tp, "fn": fn}

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the number of alphanumerical entities in the reference text."""
        return len(self._get_ref_matches())

    @metric_value
    def num_hyp_terms(self) -> int:
        """Get the number of alphanumerical entities in the hypothesis text."""
        return len(self._get_hyp_matches())

    @metric_value
    def tp_alignments(self) -> list[Alignment]:
        """Alignment segments for each correctly transcribed alphanumerical entity (TP)."""
        return self._ref_match_classification["tp"]

    @metric_value
    def fn_alignments(self) -> list[Alignment]:
        """Alignment segments for each missed alphanumerical entity (FN)."""
        return self._ref_match_classification["fn"]

    @metric_value
    def fp_alignments(self) -> list[Alignment]:
        """Alignment segments for each spurious alphanumerical entity in the hypothesis (FP).

        A hyp match is an FP when its alignment span contains at least one edit op.
        Hyp matches whose span is entirely MATCH are excluded — mirroring _KTStats.
        """
        hyp_matches = self._get_hyp_matches()
        if not hyp_matches:
            return []
        alignment = self._get_alignment()
        result: list[Alignment] = []
        for m in hyp_matches:
            op_start = alignment.hyp_index_mapping[m.start]
            op_stop = alignment.hyp_index_mapping[m.stop - 1] + 1
            segment: Alignment = alignment[op_start:op_stop]
            if segment.num_edits > 0:
                result.append(segment)
        return result

    @metric_value
    def num_tp(self) -> int:
        return len(self.tp_alignments)

    @metric_value
    def num_fn(self) -> int:
        return len(self.fn_alignments)

    @metric_value
    def num_fp(self) -> int:
        return len(self.fp_alignments)


@METRIC_REGISTRY.register("_alphanum_stats")
class _AlphaNumStats(Metric):
    short_name_base = "_AlphaNumStats"
    long_name_base = "Alphanumerical Entity Statistics"
    description = (
        "Private metric that computes shared alphanumerical-entity statistics (num_ref_terms, num_hyp_terms, "
        "TP, FN, FP) used by AlphaNumP, AlphaNumR, and AlphaNumF. Not intended for direct use. "
        "Entities are detected by applying a regex predicate to each token's case-preserving form."
    )
    example_cls = _AlphaNumStats_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the _AlphaNumStats metric.

        Attributes:
            pattern: Regex pattern matched against each token's case-preserving form (Token.raw) via
                `fullmatch`. Defaults to ALPHANUM_DEFAULT_PATTERN, which matches tokens that either
                (a) contain at least one uppercase letter and are not the shape
                <single uppercase letter><lowercase letters>, or (b) consist of one or more letters
                followed by at least one digit. The pattern uses Unicode letter properties (\\p{Lu},
                \\p{Ll}, \\p{L}), so Greek letters and other scripts are classified by case.
        """

        pattern: str = ALPHANUM_DEFAULT_PATTERN

        def validate(self) -> None:
            try:
                self._compiled = re.compile(self.pattern)
            except re.error as e:
                raise ValueError(f"Invalid alphanumerical regex pattern: {e}") from e

    @metric_value
    def num_ref_terms(self) -> int:
        return sum(self.get_example_metric(example).num_ref_terms for example in self._src)

    @metric_value
    def num_hyp_terms(self) -> int:
        return sum(self.get_example_metric(example).num_hyp_terms for example in self._src)

    @metric_value
    def num_tp(self) -> int:
        return sum(self.get_example_metric(example).num_tp for example in self._src)

    @metric_value
    def num_fn(self) -> int:
        return sum(self.get_example_metric(example).num_fn for example in self._src)

    @metric_value
    def num_fp(self) -> int:
        return sum(self.get_example_metric(example).num_fp for example in self._src)
