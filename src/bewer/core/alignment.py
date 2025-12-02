from functools import cached_property
from typing import TYPE_CHECKING

from rapidfuzz.distance import Levenshtein

from bewer.core.op import Op, OpList, OpType

if TYPE_CHECKING:
    from bewer.core.example import Example
    from bewer.core.text import TokenList


class Alignment:
    def __init__(
        self,
        example: "Example",
        standardizer: str | None = None,
        tokenizer: str | None = None,
        normalizer: str | None = None,
    ):
        """Initialize the Alignment object.

        Args:
            src (Example): The source Example object.
        """
        self._src_example = example
        self._standardizer = standardizer
        self._tokenizer = tokenizer
        self._normalizer = normalizer

    @cached_property
    def ops(self) -> list[Op]:
        """Get the list of operations."""
        return self._get_ops()

    @cached_property
    def num_edits(self) -> int:
        """Get the number of edits."""
        return len([op for op in self.ops if op.type != OpType.MATCH])

    @cached_property
    def num_matches(self) -> int:
        """Get the number of matches."""
        return len([op for op in self.ops if op.type == OpType.MATCH])

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
    def _token_to_op_index(self) -> dict[str, Op]:
        token_to_op_index = {}
        for op in self.ops:
            if op.ref_token is not None:
                token_to_op_index[op.ref_token] = op
            if op.hyp_token is not None:
                token_to_op_index[op.hyp_token] = op
        return token_to_op_index

    def _get_ops(self, ref_tokens, hyp_tokens):
        raise NotImplementedError("get_ops() method not implemented.")

    def __repr__(self):
        return f"Alignment(src={self._src_example.__repr__()})"


class LevenshteinAlignment(Alignment):
    OPS_MAP = {
        "match": OpType.MATCH,
        "replace": OpType.SUBSTITUTE,
        "insert": OpType.INSERT,
        "delete": OpType.DELETE,
    }

    def _get_ops(self) -> OpList:
        ref_tokens = self._src_example.ref.tokens
        hyp_tokens = self._src_example.hyp.tokens
        ops = Levenshtein.editops(ref_tokens.normalized, hyp_tokens.normalized).as_list()
        ops = self._rapidfuzz_to_bewer_op(ops, ref_tokens, hyp_tokens)
        return OpList(ops)

    def _rapidfuzz_to_bewer_op(
        self,
        rapidfuzz_ops: list[tuple[str, int, int]],
        ref_tokens: "TokenList",
        hyp_tokens: "TokenList",
    ) -> list[Op]:
        """
        Convert RapidFuzz edit operations to BeWER operations.

        Args:
            rapidfuzz_ops (list): List of RapidFuzz edit operations.

        Returns:
            list[Op]: List of BeWER operations.
        """
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
                    hyp_index=hyp_idx,
                    ref_index=ref_idx,
                    hyp_token=hyp_tokens[hyp_idx],
                    ref_token=ref_tokens[ref_idx],
                )
            elif op_type == "delete":
                op = Op(
                    type=self.OPS_MAP[op_type],
                    hyp_index=None,
                    ref_index=ref_idx,
                    hyp_token=None,
                    ref_token=ref_tokens[ref_idx],
                )
            elif op_type == "insert":
                op = Op(
                    type=self.OPS_MAP[op_type],
                    hyp_index=hyp_idx,
                    ref_index=None,
                    hyp_token=hyp_tokens[hyp_idx],
                    ref_token=None,
                )
            else:
                raise ValueError(f"Unknown operation type: {op_type}")
            bewer_ops.append(op)

        return bewer_ops
