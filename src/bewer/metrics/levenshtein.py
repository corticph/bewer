from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein as RFLevenshtein

from bewer.alignment import Alignment, Op, OpType
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, metric_value


class Levenshtein_(ExampleMetric):
    """Compute Levenshtein edit operations between hypothesis and reference text using RapidFuzz."""

    OPS_MAP = {
        "match": OpType.MATCH,
        "replace": OpType.SUBSTITUTE,
        "insert": OpType.INSERT,
        "delete": OpType.DELETE,
    }

    @metric_value
    def num_substitutions(self) -> int:
        """Get the number of substitutions."""
        return self.alignment.num_substitutions

    @metric_value
    def num_insertions(self) -> int:
        """Get the number of insertions."""
        return self.alignment.num_insertions

    @metric_value
    def num_deletions(self) -> int:
        """Get the number of deletions."""
        return self.alignment.num_deletions

    @metric_value
    def num_edits(self) -> int:
        """Get the number of edits."""
        return self.alignment.num_edits

    @metric_value
    def num_matches(self) -> int:
        """Get the number of matches."""
        return self.alignment.num_matches

    @metric_value(main=True)
    def alignment(self) -> Alignment:
        """Get the Levenshtein distance between the hypothesis and reference text."""
        return self._get_ops()

    def _get_ops(self) -> list[Op]:
        """
        Compute and convert RapidFuzz edit operations to BeWER operations.

        Args:
            rapidfuzz_ops (list): List of RapidFuzz edit operations.

        Returns:
            list[Op]: List of BeWER operations.
        """
        if self.params.normalized:
            ref_tokens = self.example.ref.tokens.normalized
            hyp_tokens = self.example.hyp.tokens.normalized
        else:
            ref_tokens = self.example.ref.tokens.raw
            hyp_tokens = self.example.hyp.tokens.raw

        rapidfuzz_ops = RFLevenshtein.editops(ref_tokens, hyp_tokens).as_list()

        bewer_ops = []
        ref_edit_idxs = {op[1] for op in rapidfuzz_ops if op[0] != "insert"}
        hyp_edit_idxs = {op[2] for op in rapidfuzz_ops if op[0] != "delete"}
        match_ref_indices = set(range(len(ref_tokens))) - ref_edit_idxs
        match_hyp_indices = set(range(len(hyp_tokens))) - hyp_edit_idxs

        assert len(match_ref_indices) == len(match_hyp_indices), "Mismatch in match indices"

        for ref_idx, hyp_idx in zip(match_ref_indices, match_hyp_indices):
            rapidfuzz_ops.append(("match", ref_idx, hyp_idx))
        rapidfuzz_ops = sorted(rapidfuzz_ops, key=lambda x: (x[1], x[2]))

        # Convert to Alignment objects
        bewer_ops = []
        for op_type, ref_idx, hyp_idx in rapidfuzz_ops:
            if op_type == "match" or op_type == "replace":
                op = Op(
                    type=self.OPS_MAP[op_type],
                    hyp=hyp_tokens[hyp_idx],
                    ref=ref_tokens[ref_idx],
                    hyp_token_idx=hyp_idx,
                    ref_token_idx=ref_idx,
                    ref_span=self.example.ref.tokens[ref_idx].slice,
                    hyp_span=self.example.hyp.tokens[hyp_idx].slice,
                )
            elif op_type == "delete":
                op = Op(
                    type=self.OPS_MAP[op_type],
                    ref=ref_tokens[ref_idx],
                    ref_token_idx=ref_idx,
                    ref_span=self.example.ref.tokens[ref_idx].slice,
                )
            elif op_type == "insert":
                op = Op(
                    type=self.OPS_MAP[op_type],
                    hyp=hyp_tokens[hyp_idx],
                    hyp_token_idx=hyp_idx,
                    hyp_span=self.example.hyp.tokens[hyp_idx].slice,
                )
            else:
                raise ValueError(f"Unknown operation type: {op_type}")
            bewer_ops.append(op)

        alignment = Alignment(bewer_ops, src=self.example)
        return alignment


@METRIC_REGISTRY.register("levenshtein")
class Levenshtein(Metric):
    short_name_base = "Levenshtein"
    long_name_base = "Levenshtein Alignment"
    description = "Levenshtein alignment between hypothesis and reference texts."
    example_cls = Levenshtein_

    @dataclass
    class param_schema(MetricParams):
        normalized: bool = True
