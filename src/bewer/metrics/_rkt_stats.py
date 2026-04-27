from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from rapidfuzz.distance import Levenshtein

from bewer.alignment import Alignment
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, metric_value

__all__: list[str] = []


class TermStat(NamedTuple):
    """Per-term edit statistics and alignment segment for a ref key term match."""

    char_edits: int
    ref_chars: int
    segment: Alignment


def _join_op_refs(ops: Alignment) -> str:
    """Join reference text from alignment ops, preserving original spacing.

    Uses op.ref_span character positions to determine spacing, mirroring _join_op_hyps.
    INSERT ops (op.ref is None) are skipped.
    """
    joined = ""
    prev_end = 0
    for op in ops:
        if op.ref is not None:
            ref_start = op.ref_span.start if op.ref_span is not None else prev_end
            if joined and ref_start > prev_end:
                joined += " "
            joined += op.ref
            if op.ref_span is not None:
                prev_end = op.ref_span.stop
    return joined.strip()


def _join_op_hyps(ops: Alignment) -> str:
    """Join hypothesis text from alignment ops, preserving original spacing.

    Uses op.hyp_span character positions (from error_align) to determine whether
    a space should be inserted between adjacent hypothesis tokens, mirroring the
    logic of _join_tokens for reference tokens.
    """
    joined = ""
    prev_end = 0
    for op in ops:
        if op.hyp is not None:
            hyp_start = op.hyp_span.start if op.hyp_span is not None else prev_end
            if joined and hyp_start > prev_end:
                joined += " "
            joined += op.hyp
            if op.hyp_span is not None:
                prev_end = op.hyp_span.stop
    return joined.strip()


class _RKTStats_(ExampleMetric):
    def _get_alignment(self) -> Alignment:
        return self.example.metrics.error_align(
            normalized=self.params.normalized,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        ).alignment

    def _get_ref_matches(self) -> list[slice]:
        return self.example.ref.get_key_term_matches(
            vocab=self.params.vocab,
            normalized=self.params.normalized,
            allow_subset_matches=self.params.allow_subset_matches,
            only_local_matches=self.params.only_local_matches,
        )

    @metric_value
    def term_stats(self) -> list[TermStat]:
        """Compute TermStat for each ref key term match.

        Alignment is derived from error_align. Results are threshold-agnostic so
        this metric can be shared across RKTR instances with different thresholds.
        """
        key_term_matches = self._get_ref_matches()
        if not key_term_matches:
            return []
        alignment = self._get_alignment()
        stats: list[TermStat] = []
        for kt_match in key_term_matches:
            op_start = alignment.ref_index_mapping.get(kt_match.start)
            op_stop = alignment.ref_index_mapping.get(kt_match.stop - 1) + 1
            segment: Alignment = alignment[op_start:op_stop]
            ref_text = _join_op_refs(segment)
            hyp_text = _join_op_hyps(segment)
            char_edits = Levenshtein.distance(hyp_text, ref_text)
            char_edits += int(segment[0].hyp_left_partial) + int(segment[-1].hyp_right_partial)
            stats.append(TermStat(char_edits=char_edits, ref_chars=len(ref_text), segment=segment))
        return stats

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the number of key terms in the reference text."""
        return len(self._get_ref_matches())


@METRIC_REGISTRY.register("_rkt_stats", tokenizer="key_term")
class _RKTStats(Metric):
    short_name_base = "_RKTStats"
    long_name_base = "Relaxed Key Term Statistics"
    description = (
        "Private metric that computes per-term character edit distance statistics for relaxed key term "
        "metrics. Stores (char_edits, ref_chars) per key term match using error_align alignment, without "
        "applying any threshold. Intended to be shared across RKTR instances with different thresholds. "
        "Not intended for direct use."
    )
    example_cls = _RKTStats_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the _RKTStats metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            normalized: Whether to use normalized tokens for alignment and key term matching.
            allow_subset_matches: Whether to allow subset matches.
            only_local_matches: If True, match only per-example local key terms.
        """

        vocab: str
        normalized: bool = True
        allow_subset_matches: bool = False
        only_local_matches: bool = False

        def validate(self) -> None:
            is_global_vocab = self.vocab in self.metric.dataset._global_key_term_vocabs
            is_local_vocab = self.vocab in self.metric.dataset._local_key_term_vocabs
            if not is_global_vocab and not is_local_vocab:
                raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset key term vocabularies.")

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the total number of key terms in the reference texts."""
        return sum(self.get_example_metric(example).num_ref_terms for example in self._src)
