from __future__ import annotations

from dataclasses import dataclass

from bewer.alignment import Alignment
from bewer.metrics._rkt_stats import TermStat
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, dependency, metric_value

__all__ = ["RKTR"]


def _is_tp(ts: TermStat, threshold: float) -> bool:
    """Return True if a term's char CER is at or below the threshold."""
    cer = ts.char_edits / ts.ref_chars if ts.ref_chars else float(ts.char_edits)
    return cer <= threshold


class RKTR_(ExampleMetric):
    @metric_value
    def tp_alignments(self) -> list[Alignment]:
        """Get alignment segments for each key term classified as TP."""
        stats = self.parent_metric._rkt_stats.get_example_metric(self.example).term_stats
        return [ts.segment for ts in stats if _is_tp(ts, self.params.threshold)]

    @metric_value
    def fn_alignments(self) -> list[Alignment]:
        """Get alignment segments for each key term classified as FN."""
        stats = self.parent_metric._rkt_stats.get_example_metric(self.example).term_stats
        return [ts.segment for ts in stats if not _is_tp(ts, self.params.threshold)]

    @metric_value
    def num_relaxed_matches(self) -> int:
        """Get the number of key terms classified as TP in the hypothesis text."""
        return len(self.tp_alignments)

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the number of key terms in the reference text."""
        return self.parent_metric._rkt_stats.get_example_metric(self.example).num_ref_terms

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level relaxed key term recall."""
        if self.num_ref_terms == 0:
            return 0.0
        return self.num_relaxed_matches / self.num_ref_terms


@METRIC_REGISTRY.register("rktr", tokenizer="key_term")
class RKTR(Metric):
    short_name_base = "RKTR"
    long_name_base = "Relaxed Key Term Recall"
    description = (
        "Relaxed key term recall (RKTR) is computed as TP / (TP + FN), where a key term is counted as TP "
        "if its character error rate (CER) against the aligned hypothesis text is at or below the given "
        "threshold. Alignment is derived from error_align. With threshold=0.0, RKTR is equivalent to KTR."
    )
    example_cls = RKTR_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the RKTR metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            normalized: Whether to use normalized tokens for alignment and key term matching.
            allow_subset_matches: Whether to allow subset matches.
            only_local_matches: If True, restrict matching to per-example local key terms only.
            threshold: Maximum character error rate for a key term to be classified as TP. Default 0.0
                means exact match (equivalent to KTR).
        """

        vocab: str
        normalized: bool = True
        allow_subset_matches: bool = False
        only_local_matches: bool = False
        threshold: float = 0.0

        def validate(self) -> None:
            if not 0.0 <= self.threshold <= 1.0:
                raise ValueError(f"threshold must be between 0.0 and 1.0, got {self.threshold}.")
            is_global_vocab = self.vocab in self.metric.dataset._global_key_term_vocabs
            is_local_vocab = self.vocab in self.metric.dataset._local_key_term_vocabs
            if not is_global_vocab and not is_local_vocab:
                raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset key term vocabularies.")

    @dependency
    def _rkt_stats(self):
        """Get the shared _RKTStats metric instance."""
        return self.dataset.metrics._rkt_stats(
            vocab=self.params.vocab,
            normalized=self.params.normalized,
            allow_subset_matches=self.params.allow_subset_matches,
            only_local_matches=self.params.only_local_matches,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_value
    def num_relaxed_matches(self) -> int:
        """Get the number of key terms classified as TP across all hypothesis texts."""
        return sum(
            sum(1 for ts in self._rkt_stats.get_example_metric(ex).term_stats if _is_tp(ts, self.params.threshold))
            for ex in self._src
        )

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the number of key terms in the reference texts."""
        return self._rkt_stats.num_ref_terms

    @metric_value(main=True)
    def value(self) -> float:
        """Get the relaxed key term recall."""
        if self.num_ref_terms == 0:
            return 0.0
        return self.num_relaxed_matches / self.num_ref_terms
