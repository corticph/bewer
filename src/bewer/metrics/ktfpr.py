from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, dependency, metric_value

__all__ = ["KTFPR"]


class KTFPR_(ExampleMetric):
    @metric_value
    def num_fp(self) -> int:
        """Get the number of spurious key term token positions in the hypothesis text."""
        return self.parent_metric._kt_stats[self.example.index].num_fp

    @metric_value
    def num_ref_tokens(self) -> int:
        """Get the number of reference tokens (N_R) for this example."""
        alignment = self.example.metrics.levenshtein(
            normalized=self.params.normalized,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        ).alignment
        return alignment.num_matches + alignment.num_substitutions + alignment.num_deletions

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level key term false-positive rate."""
        if self.num_ref_tokens == 0:
            return 0.0
        return self.num_fp / self.num_ref_tokens


@METRIC_REGISTRY.register("ktfpr", tokenizer="key_term")
class KTFPR(Metric):
    short_name_base = "KTFPR"
    long_name_base = "Key Term False-Positive Rate"
    description = (
        "Key term false-positive rate (KTFPR) measures spurious key term detections relative to the total number "
        "of reference tokens: FPR = FP^[+] / N_R, where FP^[+] is the number of mismatched token positions inside "
        "hypothesis key term occurrences and N_R is the total reference token count. "
        "Only defined for the partial-credit view (partial_credit=True): the exact-match view counts FP at the "
        "occurrence level, not the token-position level, so dividing by N_R has no principled interpretation."
    )
    example_cls = KTFPR_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the KTFPR metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            normalized: Whether to use normalized tokens for alignment and key term matching.
            allow_subset_matches: Whether to allow subset matches.
            partial_credit: Must be True. KTFPR is only defined for the partial-credit view.
        """

        vocab: str
        normalized: bool = True
        allow_subset_matches: bool = False
        partial_credit: bool = True

        def validate(self) -> None:
            """Validate that the metric can be computed with the given parameters and source data."""
            if not self.partial_credit:
                raise ValueError(
                    "KTFPR requires partial_credit=True. "
                    "The exact-match view counts false positives at the occurrence level, not the token-position "
                    "level, so dividing by the reference token count N_R has no principled interpretation."
                )
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
    def num_fp(self) -> int:
        """Get the total number of spurious key term token positions across all hypothesis texts."""
        return sum(em.num_fp for em in self)

    @metric_value
    def num_ref_tokens(self) -> int:
        """Get the total number of reference tokens (N_R) across the dataset."""
        return sum(em.num_ref_tokens for em in self)

    @metric_value(main=True)
    def value(self) -> float:
        """Get the key term false-positive rate."""
        if self.num_ref_tokens == 0:
            return 0.0
        return self.num_fp / self.num_ref_tokens
