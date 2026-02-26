"""Module for styling alignment display as HTML.

Alignments can be rendered as HTML with color coding for different operation types.
The generated HTML can be saved to a file and viewed in a browser.
"""

from html import escape
from typing import TYPE_CHECKING

from bewer.alignment.op_type import OpType
from bewer.reporting.html.color_schemes import (
    HTMLAlignmentColors,
    HTMLDefaultAlignmentColors,
)

if TYPE_CHECKING:
    from bewer.alignment.alignment import Alignment
    from bewer.alignment.op import Op

__all__ = ["generate_alignment_html_lines"]


def get_html_padding(length: int, color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors) -> str:
    """Get an HTML span representing padding spaces.

    Args:
        length: The number of spaces for padding.
        color_scheme: The color scheme to use.

    Returns:
        An HTML span element with the padding.
    """
    spaces = "&nbsp;" * length
    return f'<span style="background-color: {color_scheme.PAD};">{spaces}</span>'


def format_match_op_html(
    op: "Op", color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors
) -> tuple[str, str, int]:
    """Format a match operation for HTML display."""
    len_ref = len(op.ref)
    len_hyp = len(op.hyp)
    length = max(len_ref, len_hyp)

    ref_str = f'<span style="color: {color_scheme.MATCH};">{escape(op.ref)}</span>'
    hyp_str = f'<span style="color: {color_scheme.MATCH};">{escape(op.hyp)}</span>'

    if len_ref < length:
        ref_str += get_html_padding(length - len_ref, color_scheme=color_scheme)
    if len_hyp < length:
        hyp_str += get_html_padding(length - len_hyp, color_scheme=color_scheme)

    return ref_str, hyp_str, length


def format_substitute_op_html(
    op: "Op", color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors
) -> tuple[str, str, int]:
    """Format a substitute operation for HTML display."""
    len_ref = len(op.ref)
    len_hyp = len(op.hyp)
    length = max(len_ref, len_hyp)

    ref_str = f'<span style="color: {color_scheme.SUB};">{escape(op.ref)}</span>'
    hyp_str = f'<span style="color: {color_scheme.SUB};">{escape(op.hyp)}</span>'

    if len_ref < length:
        ref_str += get_html_padding(length - len_ref, color_scheme=color_scheme)
    if len_hyp < length:
        hyp_str += get_html_padding(length - len_hyp, color_scheme=color_scheme)

    return ref_str, hyp_str, length


def format_insert_op_html(
    op: "Op", color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors
) -> tuple[str, str, int]:
    """Format an insert operation for HTML display."""
    len_hyp = len(op.hyp)
    hyp_str = f'<span style="color: {color_scheme.INS};">{escape(op.hyp)}</span>'
    ref_str = get_html_padding(len_hyp, color_scheme=color_scheme)
    return ref_str, hyp_str, len_hyp


def format_delete_op_html(
    op: "Op", color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors
) -> tuple[str, str, int]:
    """Format a delete operation for HTML display."""
    len_ref = len(op.ref)
    ref_str = f'<span style="color: {color_scheme.DEL};">{escape(op.ref)}</span>'
    hyp_str = get_html_padding(len_ref, color_scheme=color_scheme)
    return ref_str, hyp_str, len_ref


def format_alignment_op_html(
    op: "Op", color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors
) -> tuple[str, str, int]:
    """Format an alignment operation for HTML display.

    Args:
        op: The alignment operation.
        color_scheme: The color scheme to use.

    Returns:
        A tuple containing the formatted ref and hyp HTML strings and the unformatted length.
    """
    if op.type == OpType.MATCH:
        return format_match_op_html(op, color_scheme=color_scheme)
    if op.type == OpType.SUBSTITUTE:
        return format_substitute_op_html(op, color_scheme=color_scheme)
    if op.type == OpType.INSERT:
        return format_insert_op_html(op, color_scheme=color_scheme)
    if op.type == OpType.DELETE:
        return format_delete_op_html(op, color_scheme=color_scheme)
    raise ValueError(f"Unknown operation type: {op.type}")


def _set_keyword_indicators(alignment: "Alignment") -> None:
    """Set indicators on alignment operations that correspond to keywords in the reference text."""

    example = alignment.src
    if example is None:
        return
    if not example.keywords:
        return
    for keywords in example.keywords.values():
        for keyword in keywords:
            matches = keyword.find_in_ref()
            for match in matches:
                start_op = alignment.start_index_to_op(match[0].start)
                end_op = alignment.end_index_to_op(match[-1].end)
                if start_op is None or end_op is None:
                    continue
                setattr(start_op, "keyword_start", True)
                setattr(end_op, "keyword_end", True)


def generate_alignment_html_lines(
    alignment: "Alignment",
    max_line_length: int = 100,
    color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
) -> list[tuple[str, str]]:
    """Render the alignment as an HTML table.

    This function generates only the inner content of the alignment container,
    without the full HTML document wrapper. Use this for embedding alignments
    in templates or combining multiple alignments.

    Args:
        alignment: The alignment to render.
        max_line_length: The maximum character length per line for wrapping.
        color_scheme: The color scheme to use for display.

    Returns:
        A list of tuples, each containing the reference and hypothesis HTML strings for each line.
    """
    ref_line, hyp_line = "", ""
    current_length = 0

    _set_keyword_indicators(alignment)  # Update alignment ops with keyword indicators before rendering

    lines = []
    for op in alignment:
        ref_str, hyp_str, op_length = format_alignment_op_html(op, color_scheme=color_scheme)

        if hasattr(op, "keyword_start") and op.keyword_start:
            ref_str = f'<span class="keyword-box">{ref_str}'
        if hasattr(op, "keyword_end") and op.keyword_end:
            ref_str = f"{ref_str}</span>"

        if current_length + op_length > max_line_length and current_length > 0:
            lines.append((ref_line, hyp_line))
            ref_line, hyp_line = "", ""
            current_length = 0

        ref_line += ref_str + "&nbsp;"
        hyp_line += hyp_str + (get_html_padding(1, color_scheme=color_scheme) if op.hyp_right_partial else "&nbsp;")
        current_length += op_length + 1

    lines.append((ref_line, hyp_line))
    return lines
