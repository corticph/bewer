from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from rapidfuzz.distance import Levenshtein

from bewer.alignment import Alignment
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, KeyTermParams, Metric, dependency, metric_value

if TYPE_CHECKING:
    from bewer.core.key_term import KeyTermMatch

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
    @metric_value
    def ref_matches(self) -> list[KeyTermMatch]:
        """Key term matches found in the reference text."""
        return self.parent_metric._matcher.find_matches(
            self.example.ref,
            allow_subset_matches=self.params.allow_subset_matches,
        )

    @metric_value
    def term_stats(self) -> list[TermStat]:
        """Compute TermStat for each ref key term match.

        Alignment is derived from error_align. Results are threshold-agnostic so
        this metric can be shared across RKTR instances with different thresholds.
        """
        key_term_matches = self.ref_matches
        if not key_term_matches:
            return []
        alignment = self.parent_metric._error_align[self.example.index].alignment
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
        return len(self.ref_matches)


@METRIC_REGISTRY.register("_rkt_stats", tokenizer="key_term")
class _RKTStats(Metric):
    short_name_base = "_RKTStats"
    long_name_base = "Relaxed Key-Term Statistics"
    description = (
        "Private metric that computes per-term character edit distance statistics for relaxed key term "
        "metrics. Stores (char_edits, ref_chars) per key term match using error_align alignment, without "
        "applying any threshold. Intended to be shared across RKTR instances with different thresholds. "
        "Not intended for direct use."
    )
    example_cls = _RKTStats_

    @dataclass
    class param_schema(KeyTermParams):
        """Parameters for the _RKTStats metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            normalized: Whether to use normalized tokens for alignment and key term matching.
            allow_subset_matches: Whether to allow subset matches.
        """

    @metric_value
    def _matcher(self):
        """Build (and cache) the key-term matcher for this metric's vocabulary + pipeline."""
        vocab = self.dataset.config.vocabularies[self.params.vocab]
        return vocab.build_matcher(
            self.dataset,
            normalized=self.params.normalized,
        )

    @dependency
    def _error_align(self):
        """The shared error_align metric instance."""
        return self.dataset.metrics.error_align(
            normalized=self.params.normalized,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the total number of key terms in the reference texts."""
        return sum(em.num_ref_terms for em in self)
