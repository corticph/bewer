from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from rapidfuzz.distance import Levenshtein as RFLevenshtein

from bewer.alignment.op import Alignment, Op, OpType
from bewer.metrics.base import ExampleMetric
from bewer.preprocessing.context import set_pipeline

if TYPE_CHECKING:
    pass


class Levenshtein(ExampleMetric):
    OPS_MAP = {
        "match": OpType.MATCH,
        "replace": OpType.SUBSTITUTE,
        "insert": OpType.INSERT,
        "delete": OpType.DELETE,
    }

    @cached_property
    def ops(self) -> int:
        """Get the Levenshtein distance between the hypothesis and reference text."""
        with set_pipeline(*self.src_metric.pipeline):
            return self._get_ops()

    @cached_property
    def num_substitutions(self) -> int:
        """Get the number of substitutions."""
        return len([op for op in self.ops if op.type == OpType.SUBSTITUTE])

    @cached_property
    def num_insertions(self) -> int:
        """Get the number of insertions."""
        return len([op for op in self.ops if op.type == OpType.INSERT])

    @cached_property
    def num_deletions(self) -> int:
        """Get the number of deletions."""
        return len([op for op in self.ops if op.type == OpType.DELETE])

    @cached_property
    def num_edits(self) -> int:
        """Get the number of edits."""
        return self.num_insertions + self.num_deletions + self.num_substitutions

    @cached_property
    def num_matches(self) -> int:
        """Get the number of matches."""
        return len(self.ops) - self.num_edits

    def _get_ops(self) -> list[Op]:
        """
        Compute and convert RapidFuzz edit operations to BeWER operations.

        Args:
            rapidfuzz_ops (list): List of RapidFuzz edit operations.

        Returns:
            list[Op]: List of BeWER operations.
        """
        ref_tokens = self.example.ref.tokens.normalized
        hyp_tokens = self.example.hyp.tokens.normalized
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
                    hyp_idx=hyp_idx,
                    ref_idx=ref_idx,
                )
            elif op_type == "delete":
                op = Op(
                    type=self.OPS_MAP[op_type],
                    hyp=None,
                    ref=ref_tokens[ref_idx],
                    hyp_idx=None,
                    ref_idx=ref_idx,
                )
            elif op_type == "insert":
                op = Op(
                    type=self.OPS_MAP[op_type],
                    hyp=hyp_tokens[hyp_idx],
                    ref=None,
                    hyp_idx=hyp_idx,
                    ref_idx=None,
                )
            else:
                raise ValueError(f"Unknown operation type: {op_type}")
            bewer_ops.append(op)

        return Alignment(bewer_ops)


class WER(ExampleMetric):
    @cached_property
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference text."""
        with set_pipeline(*self.src_metric.pipeline):
            return RFLevenshtein.distance(
                self.example.ref.tokens.normalized,
                self.example.hyp.tokens.normalized,
            )

    @cached_property
    def ref_length(self) -> int:
        """Get the number of tokens in the reference text."""
        with set_pipeline(*self.src_metric.pipeline):
            return len(self.example.ref.tokens.normalized)

    @cached_property
    def value(self) -> float:
        """Get the example-level word error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length


class CER(ExampleMetric):
    @cached_property
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference text."""
        with set_pipeline(*self.src_metric.pipeline):
            return RFLevenshtein.distance(
                self.example.hyp.joined(normalized=True),
                self.example.ref.joined(normalized=True),
            )

    @cached_property
    def ref_length(self) -> int:
        """Get the number of characters in the reference text."""
        with set_pipeline(*self.src_metric.pipeline):
            return len(self.example.ref.joined(normalized=True))

    @cached_property
    def value(self) -> float:
        """Get the example-level character error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length
