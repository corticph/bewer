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


def format_key_term(text: str, start: bool = False, end: bool = False) -> str:
    """Format a key term with HTML tags for highlighting.

    Args:
        text: The key term text to format.
        start: Whether this is the start of a key term span.
        end: Whether this is the end of a key term span.

    Returns:
        The formatted key term string with HTML span tags.
    """
    kw_class = "kw"
    if start:
        kw_class += " kw-start"
    if end:
        kw_class += " kw-end"
    return f'<span class="{kw_class}">{text}</span>'


def _get_key_term_indicators(alignment: "Alignment") -> tuple[set[int], set[int], set[int]]:
    """Compute key term span indicators for the given alignment.

    Args:
        alignment: The alignment whose reference side is inspected for key term spans.

    Returns:
        A tuple of three sets of operation indices:

        - start_indices: Indices of alignment operations that correspond to the first
          token of a key term span in the reference text.
        - stop_indices: Indices of alignment operations that correspond to the last
          token of a key term span in the reference text.
        - open_indices: Indices of alignment operations that fall inside any key term
          span (including the start index but excluding the stop index), i.e. where a
          key term span is considered "open"/ongoing.
    """
    example = alignment.src
    if example is None:
        return set(), set(), set()
    vocabs = example.vocabs
    if not vocabs:
        return set(), set(), set()

    start_indices, stop_indices, open_indices = set(), set(), set()
    for vocab in vocabs:
        matches = example.get_key_term_matches(vocab=vocab)
        for match in matches:
            start_op_idx = alignment.ref_index_mapping.get(match.start)
            end_op_idx = alignment.ref_index_mapping.get(match.stop - 1)
            if start_op_idx is None or end_op_idx is None:
                continue
            start_indices.add(start_op_idx)
            stop_indices.add(end_op_idx)
            for idx in range(start_op_idx, end_op_idx):
                open_indices.add(idx)

    return start_indices, stop_indices, open_indices


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

    start_indices, stop_indices, open_indices = _get_key_term_indicators(alignment)

    lines = []
    for op_idx, op in enumerate(alignment):
        ref_str, hyp_str, op_length = format_alignment_op_html(op, color_scheme=color_scheme)

        is_kt_start = op_idx in start_indices
        is_kt_end = op_idx in stop_indices
        is_kt_open = op_idx in open_indices
        is_kt = is_kt_start or is_kt_end or is_kt_open
        if is_kt:
            ref_str = format_key_term(ref_str, start=is_kt_start, end=is_kt_end)

        if current_length + op_length > max_line_length and current_length > 0:
            lines.append((ref_line, hyp_line))
            ref_line, hyp_line = "", ""
            current_length = 0

        ref_line += ref_str + (format_key_term("&nbsp;") if is_kt_open else "&nbsp;")
        hyp_line += hyp_str + (get_html_padding(1, color_scheme=color_scheme) if op.hyp_right_partial else "&nbsp;")
        current_length += op_length + 1

    lines.append((ref_line, hyp_line))
    return lines
