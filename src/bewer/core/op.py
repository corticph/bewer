from enum import IntEnum
from typing import Union

from bewer.core.token import Token


class OpType(IntEnum):
    MATCH = 0
    INSERT = 1
    DELETE = 2
    SUBSTITUTE = 3


class Op:
    """
    Annotated operation with additional information.

    Attributes:
        type (OpType): Type of operation (e.g., MATCH, INSERT, DELETE, SUBSTITUTE).
        hyp_index (int | None): Index of the hypothesis token.
        ref_index (int | None): Index of the reference token.
        hyp_token (Token | None): Token from the hypothesis.
        ref_token (Token | None): Token from the reference.
    """

    def __init__(
        self,
        type: OpType,
        hyp_index: int,
        ref_index: int | None,
        hyp_token: Token | None,
        ref_token: Token | None,
    ):
        self.type = type
        self.hyp_index = hyp_index
        self.ref_index = ref_index
        self.hyp_token = hyp_token
        self.ref_token = ref_token

    def __repr__(self):
        return f"Op({self.type.name}, hyp_token={self.hyp_token}, ref_token={self.ref_token})"

    def __eq__(self, other):
        if not isinstance(other, Op):
            return False
        return (
            self.type == other.type
            and self.hyp_index == other.hyp_index
            and self.ref_index == other.ref_index
            and self.hyp_token == other.hyp_token
            and self.ref_token == other.ref_token
        )


class OpList(list[Op]):
    """
    List of operations with additional methods for processing.

    Attributes:
        ops (list[Op]): List of operations.
    """

    def __getitem__(self, index: int) -> Union[Op, "OpList"]:
        if isinstance(index, slice):
            return OpList(super().__getitem__(index))
        return super().__getitem__(index)

    def __add__(self, other: "OpList") -> "OpList":
        """Concatenate two TokenList objects.

        Args:
            other (TokenList): The other TokenList object.

        Returns:
            TokenList: The concatenated TokenList object.
        """
        return OpList(super().__add__(other))

    def __repr__(self):
        ops = self[:60]
        ops_str = ",\n ".join([repr(op) for op in ops])
        if len(self) > 60:
            ops_str += ",\n ..."
        return f"OpList([\n {ops_str}]\n)"
