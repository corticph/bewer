from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, dependency, metric_value

__all__ = ["KTR"]


class KTR_(ExampleMetric):
    @metric_value
    def num_matches(self) -> int:
        """Get the number of key terms correctly transcribed in the hypothesis text."""
        return self.parent_metric._kt_stats.get_example_metric(self.example).num_tp

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the number of key terms in the reference text."""
        return self.parent_metric._kt_stats.get_example_metric(self.example).num_ref_terms

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level key term recall."""
        stats = self.parent_metric._kt_stats.get_example_metric(self.example)
        if (stats.num_tp + stats.num_fn) == 0:
            return 0.0
        return stats.num_tp / (stats.num_tp + stats.num_fn)


@METRIC_REGISTRY.register("ktr", tokenizer="key_term")
class KTR(Metric):
    short_name_base = "KTR"
    long_name_base = "Key Term Recall"
    description = (
        "Key term recall (KTR) is computed as TP / (TP + FN), where TP is the number of key terms correctly "
        "transcribed and FN is the number of key terms missed. A key term may consist of one or more tokens, but is "
        "treated as a single unit. KTR is the complement of KTER (Key Term Error Rate): when TP + FN > 0, "
        "KTR = 1 - KTER."
    )
    example_cls = KTR_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the KTR metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            normalized: Whether to use normalized tokens for alignment and key term matching.
            allow_subset_matches: Whether to allow subset matches.
            only_local_matches: If True, restrict matching to per-example local key terms only.
        """

        vocab: str
        normalized: bool = True
        allow_subset_matches: bool = False
        only_local_matches: bool = False

        def validate(self) -> None:
            """Validate that the metric can be computed with the given parameters and source data."""
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

    @metric_value
    def num_matches(self) -> int:
        """Get the number of key terms correctly transcribed in the hypothesis texts."""
        return self._kt_stats.num_tp

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the number of key terms in the reference texts."""
        return self._kt_stats.num_ref_terms

    @metric_value(main=True)
    def value(self) -> float:
        """Get the key term recall."""
        if (self._kt_stats.num_tp + self._kt_stats.num_fn) == 0:
            return 0.0
        return self._kt_stats.num_tp / (self._kt_stats.num_tp + self._kt_stats.num_fn)
