from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bewer.alignment import Alignment, OpType
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, metric_value

if TYPE_CHECKING:
    from bewer.core.key_term import KeyTermMatch

__all__: list[str] = []


class _KTStats_(ExampleMetric):
    def _get_alignment(self):
        """Get the alignment for the example."""
        return self.example.metrics.levenshtein(
            normalized=self.params.normalized,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        ).alignment

    def _get_ref_matches(self) -> list[KeyTermMatch]:
        return self.example.ref.get_key_term_matches(
            vocab=self.params.vocab,
            normalized=self.params.normalized,
            allow_subset_matches=self.params.allow_subset_matches,
        )

    def _get_hyp_matches(self) -> list[KeyTermMatch]:
        return self.example.hyp.get_key_term_matches(
            vocab=self.params.vocab,
            normalized=self.params.normalized,
            allow_subset_matches=self.params.allow_subset_matches,
        )

    @metric_value
    def _exact_match_alignments(self) -> dict[str, list[Alignment]]:
        """Exact-match TP/FN/FP as full-span alignment segments.

        TP/FN: one segment per ref key term occurrence (TP if all ops are MATCH, FN otherwise).
        FP: one segment per hyp key term occurrence that contains at least one non-MATCH op.
        All-MATCH hyp occurrences are excluded from FP regardless of whether they correspond
        to a ref key term.
        """
        ref_matches = self._get_ref_matches()
        hyp_matches = self._get_hyp_matches()

        if not ref_matches and not hyp_matches:
            return {"tp": [], "fn": [], "fp": []}

        alignment = self._get_alignment()
        tp: list[Alignment] = []
        fn: list[Alignment] = []
        for kt_match in ref_matches:
            op_start = alignment.ref_index_mapping.get(kt_match.start)
            op_stop = alignment.ref_index_mapping.get(kt_match.stop - 1) + 1
            segment: Alignment = alignment[op_start:op_stop]
            if segment.num_edits == 0:
                tp.append(segment)
            else:
                fn.append(segment)

        fp: list[Alignment] = []
        for hyp_match in hyp_matches:
            op_start = alignment.hyp_index_mapping[hyp_match.start]
            op_stop = alignment.hyp_index_mapping[hyp_match.stop - 1] + 1
            segment = alignment[op_start:op_stop]
            if segment.num_edits > 0:
                fp.append(segment)

        return {"tp": tp, "fn": fn, "fp": fp}

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the number of key terms in the reference text."""
        return len(self._get_ref_matches())

    @metric_value
    def num_hyp_terms(self) -> int:
        """Get the number of key terms in the hypothesis text."""
        return len(self._get_hyp_matches())

    @metric_value
    def _partial_credit_alignments(self) -> dict[str, list[Alignment]]:
        """Partial-credit TP/FN/FP as single-op alignment slices.

        Builds I_R (ref positions covered by any ref key term occurrence) and I_H (hyp
        positions covered by any hyp occurrence), then classifies each alignment op:
          - ref position in I_R + MATCH     → TP (one single-op slice)
          - ref position in I_R + non-MATCH → FN (one single-op slice)
          - hyp position in I_H + non-MATCH → FP (one single-op slice)
        Positions shared by overlapping or nested terms are counted only once.
        """
        ref_matches = self._get_ref_matches()
        hyp_matches = self._get_hyp_matches()

        I_R: set[int] = set()
        for m in ref_matches:
            I_R.update(range(m.start, m.stop))

        I_H: set[int] = set()
        for m in hyp_matches:
            I_H.update(range(m.start, m.stop))

        if not I_R and not I_H:
            return {"tp": [], "fn": [], "fp": []}

        alignment = self._get_alignment()
        tp: list[Alignment] = []
        fn: list[Alignment] = []
        fp: list[Alignment] = []
        for i, op in enumerate(alignment):
            is_match = op.type == OpType.MATCH
            if op.ref_token_idx is not None and op.ref_token_idx in I_R:
                if is_match:
                    tp.append(alignment[i : i + 1])
                else:
                    fn.append(alignment[i : i + 1])
            if op.hyp_token_idx is not None and op.hyp_token_idx in I_H and not is_match:
                fp.append(alignment[i : i + 1])
        return {"tp": tp, "fn": fn, "fp": fp}

    @metric_value
    def tp_alignments(self) -> list[Alignment]:
        """Alignment segments for each TP unit.

        Exact-match mode: one full-span segment per correctly transcribed key term occurrence.
        Partial-credit mode: one single-op slice per correctly transcribed token position
        inside a key term span.
        """
        source = self._partial_credit_alignments if self.params.partial_credit else self._exact_match_alignments
        return source["tp"]

    @metric_value
    def fn_alignments(self) -> list[Alignment]:
        """Alignment segments for each FN unit.

        Exact-match mode: one full-span segment per missed key term occurrence.
        Partial-credit mode: one single-op slice per incorrectly transcribed token position
        inside a reference key term span.
        """
        source = self._partial_credit_alignments if self.params.partial_credit else self._exact_match_alignments
        return source["fn"]

    @metric_value
    def fp_alignments(self) -> list[Alignment]:
        """Alignment segments for each FP unit.

        Exact-match mode: one full-span segment per spurious hypothesis key term occurrence
        that contains at least one error.
        Partial-credit mode: one single-op slice per incorrectly transcribed token position
        inside a hypothesis key term span.
        """
        source = self._partial_credit_alignments if self.params.partial_credit else self._exact_match_alignments
        return source["fp"]

    @metric_value
    def num_tp(self) -> int:
        """Get the number of TP units (occurrences in exact-match mode, token positions in partial-credit mode)."""
        return len(self.tp_alignments)

    @metric_value
    def num_fn(self) -> int:
        """Get the number of FN units (occurrences in exact-match mode, token positions in partial-credit mode)."""
        return len(self.fn_alignments)

    @metric_value
    def num_fp(self) -> int:
        """Get the number of FP units (occurrences in exact-match mode, token positions in partial-credit mode)."""
        return len(self.fp_alignments)


@METRIC_REGISTRY.register("_kt_stats", tokenizer="key_term")
class _KTStats(Metric):
    short_name_base = "_KTStats"
    long_name_base = "Key Term Statistics"
    description = (
        "Private metric that computes shared key term statistics (num_ref_terms, num_hyp_terms, TP, FN, FP) "
        "used by KTR, KTER, KTP, and KTF. Not intended for direct use. "
        "Note: the TP/FP/FN categories are practical approximations that do not map directly to their binary "
        "classification counterparts. A span where one key term is substituted for another is simultaneously an FN "
        "(missed ref term) and an FP (spurious hyp term). "
        "Conversely, a correctly transcribed hyp key term may count as neither TP nor FP when "
        "allow_subset_matches=False and the ref matches a longer superset phrase (e.g. ref matches "
        "'hello world' but hyp only matches the subset 'world')."
    )
    example_cls = _KTStats_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the _KTStats metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            normalized: Whether to use normalized tokens for alignment and key term matching.
            allow_subset_matches: Whether to allow subset matches.
            partial_credit: Whether to use the partial-credit view (position-level counts)
                instead of the exact-match view (occurrence-level counts).
        """

        vocab: str
        normalized: bool = True
        allow_subset_matches: bool = False
        partial_credit: bool = False

        def validate(self) -> None:
            """Validate that the metric can be computed with the given parameters and source data."""
            if self.vocab not in self.metric.dataset._vocabularies:
                raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset key term vocabularies.")

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the total number of key terms in the reference texts."""
        return sum(em.num_ref_terms for em in self)

    @metric_value
    def num_hyp_terms(self) -> int:
        """Get the total number of key terms in the hypothesis texts."""
        return sum(em.num_hyp_terms for em in self)

    @metric_value
    def num_tp(self) -> int:
        """Get the number of key terms correctly transcribed in the hypothesis texts."""
        return sum(em.num_tp for em in self)

    @metric_value
    def num_fn(self) -> int:
        """Get the number of key terms missed in the hypothesis texts."""
        return sum(em.num_fn for em in self)

    @metric_value
    def num_fp(self) -> int:
        """Get the number of key terms in the hypothesis texts that are not in the reference texts."""
        return sum(em.num_fp for em in self)
