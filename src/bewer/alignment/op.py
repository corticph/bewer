from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from bewer.alignment.op_type import OpType

if TYPE_CHECKING:
    from bewer.alignment.alignment import Alignment


class Op:
    """Class representing an operation with its type.

    Attributes:
        type (OpType): The type of operation.
        ref (str | None): The reference string involved in the operation.
        hyp (str | None): The hypothesis string involved in the operation.
        ref_token_idx (int | None): The index of the token in the reference text (for word-level alignments).
        hyp_token_idx (int | None): The index of the token in the hypothesis text (for word-level alignments).
        ref_span (slice | None): The index span in the reference text.
        hyp_span (slice | None): The index span in the hypothesis text.
        hyp_left_partial (bool): Whether the hypothesis token is a partial word, cropped on the left.
        hyp_right_partial (bool): Whether the hypothesis token is a partial word, cropped on the right.
        ref_left_partial (bool): Whether the reference token is a partial word, cropped on the left.
        ref_right_partial (bool): Whether the reference token is a partial word, cropped on the right.
    """

    def __init__(
        self,
        type: OpType,
        ref: str | None = None,
        hyp: str | None = None,
        ref_token_idx: int | None = None,
        hyp_token_idx: int | None = None,
        ref_span: slice | None = None,
        hyp_span: slice | None = None,
        hyp_left_partial: bool = False,
        hyp_right_partial: bool = False,
        ref_left_partial: bool = False,
        ref_right_partial: bool = False,
        _src_alignment: "Alignment" | None = None,
    ):
        self.type = type
        self.ref = ref
        self.hyp = hyp
        self.ref_token_idx = ref_token_idx
        self.hyp_token_idx = hyp_token_idx
        self._ref_span = ref_span
        self._hyp_span = hyp_span
        self.hyp_left_partial = hyp_left_partial
        self.hyp_right_partial = hyp_right_partial
        self.ref_left_partial = ref_left_partial
        self.ref_right_partial = ref_right_partial
        self._src_alignment = _src_alignment

        if self.type == OpType.MATCH:
            if self.ref is None or self.hyp is None:
                raise ValueError("MATCH operation must have non-empty ref or hyp.")
        elif self.type == OpType.INSERT:
            if self.hyp is None or self.ref is not None:
                raise ValueError("INSERT operation must have non-empty hyp and empty ref.")
        elif self.type == OpType.DELETE:
            if self.hyp is not None or self.ref is None:
                raise ValueError("DELETE operation must have non-empty ref and empty hyp.")
        elif self.type == OpType.SUBSTITUTE:
            if self.ref is None or self.hyp is None:
                raise ValueError("SUBSTITUTE operation must have both ref and hyp.")

    @property
    def _repr_hyp(self) -> str:
        """Return the hypothesis with partial markers if applicable."""
        if self.hyp is None:
            return None
        return f'{"-" if self.hyp_left_partial else ""}"{self.hyp}"{"-" if self.hyp_right_partial else ""}'

    @property
    def _repr_ref(self) -> str:
        """Return the reference with partial markers if applicable."""
        if self.ref is None:
            return None
        return f'{"-" if self.ref_left_partial else ""}"{self.ref}"{"-" if self.ref_right_partial else ""}'

    def set_source(self, src: "Alignment") -> None:
        """Set the source alignment for the operation."""
        self._src_alignment = src

    @cached_property
    def ref_span(self) -> slice | None:
        """Get the reference span for an operation."""
        if self._ref_span is not None:
            return self._ref_span
        if self.ref_token_idx is not None:
            if self._src_alignment is None or self._src_alignment._src_example is None:
                return None
            return self._src_alignment._src_example.ref.tokens[self.ref_token_idx].slice
        return None

    @cached_property
    def hyp_span(self) -> slice | None:
        """Get the hypothesis span for an operation."""
        if self._hyp_span is not None:
            return self._hyp_span
        if self.hyp_token_idx is not None:
            if self._src_alignment is None or self._src_alignment._src_example is None:
                return None
            return self._src_alignment._src_example.hyp.tokens[self.hyp_token_idx].slice
        return None

    def to_dict(self) -> dict:
        """Convert the Op instance to a dictionary."""
        return {
            "type": self.type.name,
            "ref": self.ref,
            "hyp": self.hyp,
            "ref_token_idx": self.ref_token_idx,
            "hyp_token_idx": self.hyp_token_idx,
            "ref_span": (self.ref_span.start, self.ref_span.stop) if self.ref_span else None,
            "hyp_span": (self.hyp_span.start, self.hyp_span.stop) if self.hyp_span else None,
            "hyp_left_partial": self.hyp_left_partial,
            "hyp_right_partial": self.hyp_right_partial,
            "ref_left_partial": self.ref_left_partial,
            "ref_right_partial": self.ref_right_partial,
        }

    def __repr__(self) -> str:
        if self.type == OpType.DELETE:
            return f"Op({self.type.name}: {self._repr_ref})"
        if self.type == OpType.INSERT:
            return f"Op({self.type.name}: {self._repr_hyp})"
        if self.type == OpType.SUBSTITUTE:
            return f"Op({self.type.name}: {self._repr_hyp} -> {self._repr_ref})"
        return f"Op({self.type.name}: {self._repr_hyp} == {self._repr_ref})"
