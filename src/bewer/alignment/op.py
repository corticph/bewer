from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Union


class OpType(IntEnum):
    MATCH = 0
    INSERT = 1
    DELETE = 2
    SUBSTITUTE = 3


@dataclass
class Op:
    """Class representing an operation with its type and cost."""

    type: OpType
    ref: str | None = None
    hyp: str | None = None
    ref_idx: slice | int | None = None
    hyp_idx: slice | int | None = None
    hyp_left_compound: bool = False
    hyp_right_compound: bool = False
    ref_left_compound: bool = False
    ref_right_compound: bool = False

    def __post_init__(self):
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
        """Return the hypothesis with compound markers if applicable."""
        if self.hyp is None:
            return None
        return f'{"-" if self.hyp_left_compound else ""}"{self.hyp}"{"-" if self.hyp_right_compound else ""}'

    @property
    def _repr_ref(self) -> str:
        """Return the reference with compound markers if applicable."""
        if self.ref is None:
            return None
        return f'{"-" if self.ref_left_compound else ""}"{self.ref}"{"-" if self.ref_right_compound else ""}'

    def __repr__(self) -> str:
        if self.type == OpType.DELETE:
            return f"Op({self.type.name}: {self._repr_ref})"
        if self.type == OpType.INSERT:
            return f"Op({self.type.name}: {self._repr_hyp})"
        if self.type == OpType.SUBSTITUTE:
            return f"Op({self.type.name}: {self._repr_hyp} -> {self._repr_ref})"
        return f"Op({self.type.name}: {self._repr_hyp} == {self._repr_ref})"


class Alignment(list[Op]):
    """
    List of operations with additional methods for processing.

    Attributes:
        ops (list[Op]): List of operations.
    """

    def __getitem__(self, index: int) -> Union[Op, "Alignment"]:
        if isinstance(index, slice):
            return Alignment(super().__getitem__(index))
        return super().__getitem__(index)

    def __add__(self, other: "Alignment") -> "Alignment":
        """Concatenate two Alignment objects.

        Args:
            other (Alignment): The other Alignment object.
        Returns:
            Alignment: The concatenated Alignment object.
        """
        return Alignment(super().__add__(other))

    def __repr__(self):
        ops = self[:60]
        ops_str = ",\n ".join([repr(op) for op in ops])
        if len(self) > 60:
            ops_str += ",\n ..."
        return f"Alignment([\n {ops_str}]\n)"
