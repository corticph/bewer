from __future__ import annotations

from dataclasses import dataclass

from bewer.alignment import Alignment
from bewer.core.example import TextType
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, metric_value

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

    def _get_ref_matches(self) -> list[slice]:
        return self.example.get_key_term_matches(
            vocab=self.params.vocab,
            normalized=self.params.normalized,
            allow_subsets=self.params.allow_subsets,
            side=TextType.REF,
        )

    def _get_hyp_matches(self) -> list[slice]:
        return self.example.get_key_term_matches(
            vocab=self.params.vocab,
            normalized=self.params.normalized,
            allow_subsets=self.params.allow_subsets,
            side=TextType.HYP,
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
            if segment.num_edits > 0:
                result.append(segment)
        return result

    @metric_value
    def num_tp(self) -> int:
        """Get the number of key terms correctly transcribed in the hypothesis text."""
        return len(self.tp_alignments)

    @metric_value
    def num_fn(self) -> int:
        """Get the number of key terms missed in the hypothesis text."""
        return len(self.fn_alignments)

    @metric_value
    def num_fp(self) -> int:
        """Get the number of key terms in the hypothesis text that are not in the reference text."""
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
        "Conversely, a correctly transcribed hyp key term may count as neither TP nor FP when allow_subsets=False "
        "and the ref matches a longer superset phrase (e.g. ref matches 'hello world' but hyp only matches "
        "the subset 'world')."
    )
    example_cls = _KTStats_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the _KTStats metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            normalized: Whether to use normalized tokens for alignment and key term matching.
            allow_subsets: Whether to allow subset matches.
        """

        vocab: str
        normalized: bool = True
        allow_subsets: bool = True

        def validate(self) -> None:
            """Validate that the metric can be computed with the given parameters and source data."""
            is_dynamic_vocab = self.vocab in self.metric.dataset._dynamic_key_term_vocabs
            is_static_vocab = self.vocab in self.metric.dataset._static_key_term_vocabs
            if not is_dynamic_vocab and not is_static_vocab:
                raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset key term vocabularies.")

    @metric_value
    def num_ref_terms(self) -> int:
        """Get the total number of key terms in the reference texts."""
        return sum(self.get_example_metric(example).num_ref_terms for example in self._src)

    @metric_value
    def num_hyp_terms(self) -> int:
        """Get the total number of key terms in the hypothesis texts."""
        return sum(self.get_example_metric(example).num_hyp_terms for example in self._src)

    @metric_value
    def num_tp(self) -> int:
        """Get the number of key terms correctly transcribed in the hypothesis texts."""
        return sum(self.get_example_metric(example).num_tp for example in self._src)

    @metric_value
    def num_fn(self) -> int:
        """Get the number of key terms missed in the hypothesis texts."""
        return sum(self.get_example_metric(example).num_fn for example in self._src)

    @metric_value
    def num_fp(self) -> int:
        """Get the number of key terms in the hypothesis texts that are not in the reference texts."""
        return sum(self.get_example_metric(example).num_fp for example in self._src)
