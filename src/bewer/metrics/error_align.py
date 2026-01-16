from __future__ import annotations

from error_align import error_align
from error_align.utils import OpType as EAOpType

from bewer.alignment.op import Alignment, Op, OpType
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_value


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
        return len([op for op in self.ops if op.type == OpType.SUBSTITUTE])

    @metric_value
    def num_insertions(self) -> int:
        """Get the number of insertions."""
        return len([op for op in self.ops if op.type == OpType.INSERT])

    @metric_value
    def num_deletions(self) -> int:
        """Get the number of deletions."""
        return len([op for op in self.ops if op.type == OpType.DELETE])

    @metric_value
    def num_edits(self) -> int:
        """Get the number of edits."""
        return self.num_insertions + self.num_deletions + self.num_substitutions

    @metric_value
    def num_matches(self) -> int:
        """Get the number of matches."""
        return len(self.ops) - self.num_edits

    @metric_value(main=True)
    def ops(self) -> Alignment:
        """Get the Levenshtein distance between the hypothesis and reference text."""
        return self._get_ops()

    def _get_ops(self) -> list[Op]:
        """
        Compute and convert ErrorAlign edit operations to BeWER operations.

        Returns:
            list[Op]: List of BeWER operations.
        """
        # import IPython; IPython.embed(using=False, header="Debugging error_align.py")
        ea_ops = []
        ref_idx = 0
        for ea_op in error_align(self.example.ref.standardized, self.example.hyp.standardized):
            ref_empty = ea_op.ref is None
            op = Op(
                type=self.OPS_MAP[ea_op.op_type],
                ref=ea_op.ref,
                hyp=ea_op.hyp,
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
        return Alignment(ea_ops)


@METRIC_REGISTRY.register("error_align")
class ErrorAlign(Metric):
    short_name = "EA"
    long_name = "Error Alignment"
    description = "Error alignment between hypothesis and reference texts."
    example_cls = ErrorAlign_
