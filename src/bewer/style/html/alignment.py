"""Module for styling alignment display as HTML.

Alignments can be rendered as HTML with color coding for different operation types.
The generated HTML can be saved to a file and viewed in a browser.
"""

from html import escape
from typing import TYPE_CHECKING

from bewer.alignment.op_type import OpType
from bewer.style.html.color_schemes import (
    HTMLAlignmentColors,
    HTMLDefaultAlignmentColors,
)

if TYPE_CHECKING:
    from bewer.alignment.op import Alignment, Op


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


def get_ref_span(op: "Op", alignment: "Alignment") -> slice | None:
    """Get the reference span for an operation."""
    if op.ref_span is not None:
        return op.ref_span
    if op.ref_token_idx is not None:
        if alignment._src_example is None:
            return None
        return alignment._src_example.ref.tokens[op.ref_token_idx].slice
    return None


def set_keyword_indicators(alignment: "Alignment") -> None:
    """Set indicators on alignment operations that correspond to keywords in the reference text."""
    example = alignment._src_example

    if example.keywords is None:
        return
    if "medical_terms" not in example.keywords:
        return

    medical_terms = example.keywords["medical_terms"]
    for term in medical_terms:
        term_spans = term.get_keyword_span()
        if not term_spans:
            continue
        for term_span in term_spans:
            for op in alignment:
                ref_span = get_ref_span(op, alignment)
                if ref_span is None:
                    continue
                if ref_span.start == term_span.start:
                    setattr(op, "keyword_start", True)
                if ref_span.stop == term_span.stop:
                    setattr(op, "keyword_end", True)
                    break


def generate_alignment_html_lines(
    alignment: "Alignment",
    max_line_length: int = 100,
    title: str | None = None,
    color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
) -> str:
    """Render the alignment as an HTML table.

    This function generates only the inner content of the alignment container,
    without the full HTML document wrapper. Use this for embedding alignments
    in templates or combining multiple alignments.

    Args:
        alignment: The alignment to render.
        max_line_length: The maximum character length per line for wrapping.
        title: An optional title to display above the alignment.
        color_scheme: The color scheme to use for display.

    Returns:
        A string containing the rendered content.
    """
    ref_line, hyp_line = "", ""
    current_length = 0

    set_keyword_indicators(alignment)  # Update alignment ops with keyword indicators before rendering

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


# def render_alignment_html(
#     alignment: "Alignment",
#     max_line_length: int = 200,
#     title: str | None = None,
#     color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
# ) -> str:
#     """Render the alignment as a complete HTML document.

#     Args:
#         alignment: The alignment to render.
#         max_line_length: The maximum character length per line for wrapping.
#         title: An optional title to display above the alignment.
#         color_scheme: The color scheme to use for display.

#     Returns:
#         A complete HTML document string that can be saved to a file.
#     """
#     container = render_alignment_container_html(
#         alignment,
#         max_line_length=max_line_length,
#         title=title,
#         color_scheme=color_scheme,
#     )

#     legend = container.color_scheme._generate_legend()

#     html = f"""<!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Alignment Visualization</title>
#     <style>
#         body {{
#             font-family: 'Courier New', Consolas, monospace;
#             background-color: {HTMLBaseColors.BG_COLOR};
#             color: {HTMLBaseColors.TEXT_COLOR};
#             padding: 20px;
#             line-height: 0.7;
#         }}
#         .alignment-container {{
#             background-color: {HTMLBaseColors.DIV_BG_COLOR};
#             white-space: pre;
#             overflow-x: auto;
#             border-radius: 10px;
#             margin-bottom: 15px;
#         }}
#         .legend {{
#             margin-top: 20px;
#             margin-bottom: 20px;
#             padding: 10px;
#             border: 1px solid #ededed;
#             border-radius: 4px;
#         }}
#         .legend-item {{
#             display: inline-block;
#             margin-right: 20px;
#         }}
#     </style>
# </head>
# <body>
#     {legend}
#     <div class="alignment-container">
# {container.title}
# {container.content}
#     </div>
# </body>
# </html>"""

#     return html
