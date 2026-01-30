from typing import TYPE_CHECKING

from bewer.alignment.op_type import OpType

if TYPE_CHECKING:
    from bewer.alignment.op import Op


class DefaultColorScheme:
    PAD = "bright_black"
    DEL = "bright_red"
    INS = "bright_cyan"
    SUB = "bright_yellow"


COLOR_SCHEMES = {
    "default": DefaultColorScheme,
}


def _format_match_op(op: Op) -> tuple[str, str, int]:
    """Format a match operation for display."""
    len_ref = len(op.ref)
    len_hyp = len(op.hyp)
    length = max(len_ref, len_hyp)
    # NOTE: For matches, the ref and hyp token should be identical, but might differ due to normalization.
    if len_ref < length:
        ref_str = op._repr_ref.ljust(length)
    if len_hyp < length:
        hyp_str = op._repr_hyp.ljust(length)
    return ref_str, hyp_str, length


def _format_alignment_op(op: Op) -> tuple[str, str, int]:
    """Format an alignment operation for display.

    Args:
        op (Op): The alignment operation.

    Returns:
        tuple[str, str, int]: A tuple containing the formatted ref and hyp tokens and their unformatted length.
    """
    if op.type == OpType.MATCH:
        return _format_match_op(op)
