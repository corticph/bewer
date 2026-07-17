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
    def _ref_match_classification(self) -> dict[str, list[Alignment]]:
        """Partition ref key term matches into tp and fn alignment segments."""
        key_term_matches = self._get_ref_matches()
        if not key_term_matches:
            return {"tp": [], "fn": []}
        alignment = self._get_alignment()
        tp: list[Alignment] = []
        fn: list[Alignment] = []
        for kt_match in key_term_matches:
            op_start = alignment.ref_index_mapping.get(kt_match.start)
            op_stop = alignment.ref_index_mapping.get(kt_match.stop - 1) + 1
            segment: Alignment = alignment[op_start:op_stop]
            if segment.num_edits == 0:
                tp.append(segment)
            else:
                fn.append(segment)
        return {"tp": tp, "fn": fn}

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the number of key terms in the reference text."""
        return len(self._get_ref_matches())

    @metric_value
    def num_hyp_terms(self) -> int:
        """Get the number of key terms in the hypothesis text."""
        return len(self._get_hyp_matches())

    @metric_value
    def tp_alignments(self) -> list[Alignment]:
        """Get alignment segments for each correctly transcribed key term (TP)."""
        return self._ref_match_classification["tp"]

    @metric_value
    def fn_alignments(self) -> list[Alignment]:
        """Get alignment segments for each missed key term (FN)."""
        return self._ref_match_classification["fn"]

    @metric_value
    def fp_alignments(self) -> list[Alignment]:
        """Get alignment segments for each spurious key term in the hypothesis (FP).

        A hyp key term match is a FP if not all of its alignment ops are MATCH.
        All-MATCH hyp matches were correctly transcribed and are excluded, whether
        or not they correspond to a ref key term (mirroring _ref_match_classification).
        """
        hyp_matches = self._get_hyp_matches()
        if not hyp_matches:
            return []
        alignment = self._get_alignment()
        result: list[Alignment] = []
        for hyp_match in hyp_matches:
            op_start = alignment.hyp_index_mapping[hyp_match.start]
            op_stop = alignment.hyp_index_mapping[hyp_match.stop - 1] + 1
            segment: Alignment = alignment[op_start:op_stop]
            # TODO: Should we allow spurious edits in the matched range, as long as the target tokens are correct?
            if segment.num_edits > 0:
                result.append(segment)
        return result

    @metric_value
    def _partial_credit_stats(self) -> dict[str, int]:
        """Compute K^[+] (partial-credit) TP/FN/FP counts at token-position level.

        Builds I_R (ref positions covered by any ref key term occurrence) and I_H (hyp
        positions covered by any hyp occurrence), then classifies each alignment op:
          - ref position in I_R + MATCH  → TP
          - ref position in I_R + non-MATCH → FN
          - hyp position in I_H + non-MATCH → FP
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
            return {"tp": 0, "fn": 0, "fp": 0}

        alignment = self._get_alignment()
        tp = fn = fp = 0
        for op in alignment:
            is_match = op.type == OpType.MATCH
            if op.ref_token_idx is not None and op.ref_token_idx in I_R:
                if is_match:
                    tp += 1
                else:
                    fn += 1
            if op.hyp_token_idx is not None and op.hyp_token_idx in I_H and not is_match:
                fp += 1
        return {"tp": tp, "fn": fn, "fp": fp}

    @metric_value
    def num_tp(self) -> int:
        """Get the number of key terms correctly transcribed in the hypothesis text."""
        if self.params.partial_credit:
            return self._partial_credit_stats["tp"]
        return len(self.tp_alignments)

    @metric_value
    def num_fn(self) -> int:
        """Get the number of key terms missed in the hypothesis text."""
        if self.params.partial_credit:
            return self._partial_credit_stats["fn"]
        return len(self.fn_alignments)

    @metric_value
    def num_fp(self) -> int:
        """Get the number of key terms in the hypothesis text that are not in the reference text."""
        if self.params.partial_credit:
            return self._partial_credit_stats["fp"]
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
            partial_credit: Whether to use the K^[+] partial-credit view (position-level counts)
                instead of the K^[=] exact-match view (occurrence-level counts).
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
