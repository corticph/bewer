from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from error_align import error_align
from error_align.utils import OpType as EAOpType
from error_align.utils import basic_normalizer, basic_tokenizer

from bewer.alignment import Alignment, Op, OpType
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, metric_value
from bewer.preprocessing.context import get_normalizer, get_tokenizer

__all__ = ["ErrorAlign", "ErrorAlign_"]


class ErrorAlign_(ExampleMetric):
    """Compute ErrorAlign edit operations between hypothesis and reference text."""

    OPS_MAP = {
        EAOpType.MATCH: OpType.MATCH,
        EAOpType.SUBSTITUTE: OpType.SUBSTITUTE,
        EAOpType.INSERT: OpType.INSERT,
        EAOpType.DELETE: OpType.DELETE,
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
        """Get the error alignment operations between the hypothesis and reference text."""
        return self._get_ops()

    @staticmethod
    def _normalize_conditionally(text: Optional[str], normalizer: Optional[callable]) -> Optional[str]:
        """Normalize text if normalizer is provided, otherwise return original text."""
        if text is None:
            return None
        return normalizer(text) if normalizer else text

    @staticmethod
    def _no_normalizer(text: str) -> str:
        """Return text unchanged (used when no normalizer is provided)."""
        return text

    def _get_ops(self) -> list[Op]:
        """
        Compute and convert ErrorAlign edit operations to BeWER operations.

        Returns:
            list[Op]: List of BeWER operations.
        """
        tokenizer = get_tokenizer(self.parent_metric.dataset)
        tokenizer = tokenizer or basic_tokenizer
        normalizer = get_normalizer(self.parent_metric.dataset) if self.params.normalized else None
        ea_ops = []
        ref_idx = 0
        for ea_op in error_align(
            self.example.ref.standardized,
            self.example.hyp.standardized,
            tokenizer=tokenizer,
            normalizer=basic_normalizer if self.params.normalized else self._no_normalizer,
        ):
            ref_empty = ea_op.ref is None
            op = Op(
                type=self.OPS_MAP[ea_op.op_type],
                ref=self._normalize_conditionally(ea_op.ref, normalizer),
                hyp=self._normalize_conditionally(ea_op.hyp, normalizer),
                ref_token_idx=None if ref_empty else ref_idx,
                hyp_token_idx=None,
                ref_span=ea_op.ref_slice,
                hyp_span=ea_op.hyp_slice,
                hyp_left_partial=ea_op.left_compound,
                hyp_right_partial=ea_op.right_compound,
            )
            if not ref_empty:
                ref_idx += 1
            ea_ops.append(op)

        alignment = Alignment(ea_ops, src=self.example)
        return alignment


@METRIC_REGISTRY.register("error_align")
class ErrorAlign(Metric):
    short_name_base = "EA"
    long_name_base = "Error Alignment"
    description = "Error alignment between hypothesis and reference texts."
    example_cls = ErrorAlign_

    @dataclass
    class param_schema(MetricParams):
        normalized: bool = True
