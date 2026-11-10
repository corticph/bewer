from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from typing import Optional

from error_align import error_align
from error_align.utils import OpType as EAOpType
from error_align.utils import basic_normalizer, basic_tokenizer

from bewer.alignment import Alignment, Op, OpType
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_property
from bewer.preprocessing.context import get_normalizer, get_tokenizer

__all__ = ["ErrorAlign"]

logger = logging.getLogger(__name__)


class ErrorAlign_(ExampleMetric):
    """Compute ErrorAlign edit operations between hypothesis and reference text."""

    OPS_MAP = {
        EAOpType.MATCH: OpType.MATCH,
        EAOpType.SUBSTITUTE: OpType.SUBSTITUTE,
        EAOpType.INSERT: OpType.INSERT,
        EAOpType.DELETE: OpType.DELETE,
    }

    @metric_property(is_value=True)
    def num_substitutions(self) -> int:
        """Get the number of substitutions."""
        return self.alignment.num_substitutions

    @metric_property(is_value=True)
    def num_insertions(self) -> int:
        """Get the number of insertions."""
        return self.alignment.num_insertions

    @metric_property(is_value=True)
    def num_deletions(self) -> int:
        """Get the number of deletions."""
        return self.alignment.num_deletions

    @metric_property(is_value=True)
    def num_edits(self) -> int:
        """Get the number of edits."""
        return self.alignment.num_edits

    @metric_property(is_value=True)
    def num_matches(self) -> int:
        """Get the number of matches."""
        return self.alignment.num_matches

    @metric_property(is_main=True)
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
    def _length_preserving(normalizer: callable) -> callable:
        """Wrap a normalizer for use with error_align, which requires length-preserving normalizers.

        Tokens whose normalization changes the character length (e.g. ``ß`` → ``ss``) fall back
        to error_align's built-in ``basic_normalizer`` (lowercase-only) to avoid the ``ValueError``
        that would be raised otherwise. This means such tokens may not match during alignment
        even when the dataset normalizer considers them equal, but this is an edge case
        (i.e., ligatures) and never crashes.
        """

        def safe(text: str) -> str:
            result = normalizer(text)
            if len(result) == len(text):
                return result
            logger.warning(
                "Dataset normalizer is not length-preserving for %r. The error_align algorithm "
                "requires length-preserving normalization, so the token falls back to "
                "basic_normalizer and its alignment may be faulty. Use a length-preserving "
                "normalizer to avoid this warning.",
                text,
            )
            return basic_normalizer(text)

        return safe

    def _get_ops(self) -> list[Op]:
        """
        Compute and convert ErrorAlign edit operations to BeWER operations.

        The active normalizer (wrapped to be length-preserving) is used for both alignment
        and output — so ops show the same token form that was used for matching. Under
        ``normalizer="none"`` the identity normalizer applies, so alignment and output use
        the standardized text.

        error_align may split one hypothesis token across several ops (e.g. hypothesis
        ``bedroom`` against reference ``bed room``); every piece carries the index of the
        hypothesis token it starts in, so pieces of one token share a ``hyp_token_idx``. An op
        whose hypothesis text spans several tokens (a merge) covers all of them: its
        ``hyp_token_idx`` is the first and ``hyp_token_indices`` lists every one.
        Reference tokens are never split: each has exactly one op.

        Note:
            ``ref_span`` and ``hyp_span`` always index into the *standardized* (pre-normalized)
            text. The ``ref`` and ``hyp`` fields contain *normalized* text, so ``text[op.ref_span] != op.ref``
            for tokens where normalization changes characters (e.g. ``café`` → ``cafe``). Use the span to
            recover the surface form.

        Returns:
            list[Op]: List of BeWER operations.
        """
        tokenizer = get_tokenizer(self.parent_metric.dataset)
        tokenizer = tokenizer or basic_tokenizer
        dataset_normalizer = get_normalizer(self.parent_metric.dataset)
        normalizer = self._length_preserving(dataset_normalizer) if dataset_normalizer else basic_normalizer
        hyp_token_starts = [token.slice.start for token in self.example.hyp.tokens]
        ea_ops = []
        ref_idx = 0
        for ea_op in error_align(
            self.example.ref.standardized,
            self.example.hyp.standardized,
            tokenizer=tokenizer,
            normalizer=normalizer,
        ):
            ref_empty = ea_op.ref is None
            hyp_empty = ea_op.hyp is None
            op = Op(
                type=self.OPS_MAP[ea_op.op_type],
                ref=self._normalize_conditionally(ea_op.ref, normalizer),
                hyp=self._normalize_conditionally(ea_op.hyp, normalizer),
                ref_token_idx=None if ref_empty else ref_idx,
                hyp_token_idx=None if hyp_empty else bisect_right(hyp_token_starts, ea_op.hyp_slice.start) - 1,
                hyp_token_stop=None if hyp_empty else bisect_left(hyp_token_starts, ea_op.hyp_slice.stop),
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
