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

__all__ = ["generate_alignment_html_lines", "generate_alignment_html_lines_dual", "generate_alignment_html_lines_multi"]


def _eliminate_newlines(text: str) -> str:
    """Remove newline characters, replacing with a space when no adjacent non-newline whitespace."""
    result = []
    for i, c in enumerate(text):
        if c == "\n":
            prev_char = text[i - 1] if i > 0 else ""
            next_char = text[i + 1] if i < len(text) - 1 else ""
            has_adjacent_ws = (prev_char.isspace() and prev_char != "\n") or (next_char.isspace() and next_char != "\n")
            if has_adjacent_ws:
                continue
            result.append(" ")
        else:
            result.append(c)
    return "".join(result)


def _escape_and_nbsp(text: str) -> str:
    """HTML-escape text and convert all whitespace to &nbsp; entities."""
    return escape(text).replace(" ", "&nbsp;").replace("\t", "&nbsp;")


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
    op: "Op",
    color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
    ref_text: str | None = None,
    hyp_text: str | None = None,
) -> tuple[str, str, int]:
    """Format a match operation for HTML display."""
    ref = ref_text if ref_text is not None else op.ref
    hyp = hyp_text if hyp_text is not None else op.hyp
    len_ref = len(ref)
    len_hyp = len(hyp)
    length = max(len_ref, len_hyp)

    ref_str = f'<span style="color: {color_scheme.MATCH};">{escape(ref)}</span>'
    hyp_str = f'<span style="color: {color_scheme.MATCH};">{escape(hyp)}</span>'

    if len_ref < length:
        ref_str += get_html_padding(length - len_ref, color_scheme=color_scheme)
    if len_hyp < length:
        hyp_str += get_html_padding(length - len_hyp, color_scheme=color_scheme)

    return ref_str, hyp_str, length


def format_substitute_op_html(
    op: "Op",
    color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
    ref_text: str | None = None,
    hyp_text: str | None = None,
) -> tuple[str, str, int]:
    """Format a substitute operation for HTML display."""
    ref = ref_text if ref_text is not None else op.ref
    hyp = hyp_text if hyp_text is not None else op.hyp
    len_ref = len(ref)
    len_hyp = len(hyp)
    length = max(len_ref, len_hyp)

    ref_str = f'<span style="color: {color_scheme.SUB};">{escape(ref)}</span>'
    hyp_str = f'<span style="color: {color_scheme.SUB};">{escape(hyp)}</span>'

    if len_ref < length:
        ref_str += get_html_padding(length - len_ref, color_scheme=color_scheme)
    if len_hyp < length:
        hyp_str += get_html_padding(length - len_hyp, color_scheme=color_scheme)

    return ref_str, hyp_str, length


def format_insert_op_html(
    op: "Op",
    color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
    hyp_text: str | None = None,
) -> tuple[str, str, int]:
    """Format an insert operation for HTML display."""
    hyp = hyp_text if hyp_text is not None else op.hyp
    len_hyp = len(hyp)
    hyp_str = f'<span style="color: {color_scheme.INS};">{escape(hyp)}</span>'
    ref_str = get_html_padding(len_hyp, color_scheme=color_scheme)
    return ref_str, hyp_str, len_hyp


def format_delete_op_html(
    op: "Op",
    color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
    ref_text: str | None = None,
) -> tuple[str, str, int]:
    """Format a delete operation for HTML display."""
    ref = ref_text if ref_text is not None else op.ref
    len_ref = len(ref)
    ref_str = f'<span style="color: {color_scheme.DEL};">{escape(ref)}</span>'
    hyp_str = get_html_padding(len_ref, color_scheme=color_scheme)
    return ref_str, hyp_str, len_ref


def format_alignment_op_html(
    op: "Op",
    color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
    ref_text: str | None = None,
    hyp_text: str | None = None,
) -> tuple[str, str, int]:
    """Format an alignment operation for HTML display.

    Args:
        op: The alignment operation.
        color_scheme: The color scheme to use.
        ref_text: Optional surface-form text to display instead of ``op.ref``.
        hyp_text: Optional surface-form text to display instead of ``op.hyp``.

    Returns:
        A tuple containing the formatted ref and hyp HTML strings and the unformatted length.
    """
    if op.type == OpType.MATCH:
        return format_match_op_html(op, color_scheme=color_scheme, ref_text=ref_text, hyp_text=hyp_text)
    if op.type == OpType.SUBSTITUTE:
        return format_substitute_op_html(op, color_scheme=color_scheme, ref_text=ref_text, hyp_text=hyp_text)
    if op.type == OpType.INSERT:
        return format_insert_op_html(op, color_scheme=color_scheme, hyp_text=hyp_text)
    if op.type == OpType.DELETE:
        return format_delete_op_html(op, color_scheme=color_scheme, ref_text=ref_text)
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


def _get_key_term_indicators(
    alignment: "Alignment", allow_subset_matches: bool = False
) -> tuple[set[int], set[int], set[int]]:
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
        matches = example.ref.get_key_term_matches(vocab=vocab, allow_subset_matches=allow_subset_matches)
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
    allow_subset_matches: bool = False,
    surface: bool = False,
) -> list[tuple[str, str]]:
    """Render the alignment as an HTML table.

    This function generates only the inner content of the alignment container,
    without the full HTML document wrapper. Use this for embedding alignments
    in templates or combining multiple alignments.

    Args:
        alignment: The alignment to render.
        max_line_length: The maximum character length per line for wrapping.
        color_scheme: The color scheme to use for display.
        surface: If True, render the surface (standardized) form of tokens and include
            inter-token content (punctuation, whitespace) between ops. Requires
            ``alignment.src`` to be set. Falls back to normalized form if unavailable.

    Returns:
        A list of tuples, each containing the reference and hypothesis HTML strings for each line.
    """
    if surface and alignment.src is not None:
        ref_std = alignment.src.ref.standardized
        hyp_std = alignment.src.hyp.standardized
    else:
        surface = False
        ref_std = None
        hyp_std = None

    start_indices, stop_indices, open_indices = _get_key_term_indicators(
        alignment, allow_subset_matches=allow_subset_matches
    )

    n_ops = len(alignment)
    if n_ops == 0:
        return [("", "")]

    next_ref_span_after: list[int | None] = [None] * n_ops
    next_hyp_span_after: list[int | None] = [None] * n_ops
    last_ref: int | None = None
    last_hyp: int | None = None
    for i in range(n_ops - 1, -1, -1):
        next_ref_span_after[i] = last_ref
        next_hyp_span_after[i] = last_hyp
        if alignment[i].ref_span is not None:
            last_ref = i
        if alignment[i].hyp_span is not None:
            last_hyp = i

    ref_line, hyp_line = "", ""
    current_length = 0
    prev_ref_end = 0
    prev_hyp_end = 0

    lines = []
    for op_idx, op in enumerate(alignment):
        # --- Pre-separator (inter-token content before this op) ---
        if op_idx > 0:
            prev_op = alignment[op_idx - 1]
            prev_was_delete = prev_op.type == OpType.DELETE
            prev_was_insert = prev_op.type == OpType.INSERT
            prev_kt_open = (op_idx - 1) in open_indices

            if surface:
                # Ref inter-token
                if op.ref_span is not None and not prev_was_insert:
                    ref_inter_raw = ref_std[prev_ref_end : op.ref_span.start]
                elif op.type == OpType.INSERT and not prev_was_insert:
                    next_ref_idx = next_ref_span_after[op_idx]
                    end = alignment[next_ref_idx].ref_span.start if next_ref_idx is not None else prev_ref_end
                    ref_inter_raw = ref_std[prev_ref_end:end]
                else:
                    ref_inter_raw = ""

                # Hyp inter-token
                if op.hyp_span is not None and not prev_was_delete:
                    hyp_inter_raw = hyp_std[prev_hyp_end : op.hyp_span.start]
                elif op.type == OpType.DELETE and not prev_was_delete:
                    next_hyp_idx = next_hyp_span_after[op_idx]
                    end = alignment[next_hyp_idx].hyp_span.start if next_hyp_idx is not None else prev_hyp_end
                    hyp_inter_raw = hyp_std[prev_hyp_end:end]
                else:
                    hyp_inter_raw = ""

                ref_inter_text = _eliminate_newlines(ref_inter_raw)
                hyp_inter_text = _eliminate_newlines(hyp_inter_raw)
                ref_inter_len = len(ref_inter_text)
                hyp_inter_len = len(hyp_inter_text)
                sep_length = max(ref_inter_len, hyp_inter_len, 1)

                ref_sep = _escape_and_nbsp(ref_inter_text)
                hyp_sep = _escape_and_nbsp(hyp_inter_text)

                if ref_inter_len < sep_length:
                    ref_sep += get_html_padding(sep_length - ref_inter_len, color_scheme)
                if hyp_inter_len < sep_length:
                    hyp_sep += get_html_padding(sep_length - hyp_inter_len, color_scheme)

                if prev_kt_open:
                    ref_sep = format_key_term(ref_sep)

                if prev_op.hyp_right_partial:
                    hyp_sep = get_html_padding(sep_length, color_scheme)
            else:
                sep_length = 1
                ref_sep = format_key_term("&nbsp;") if prev_kt_open else "&nbsp;"
                hyp_sep = get_html_padding(1, color_scheme=color_scheme) if prev_op.hyp_right_partial else "&nbsp;"
        else:
            sep_length = 0
            ref_sep = ""
            hyp_sep = ""

        # --- Op content ---
        if surface:
            ref_text = ref_std[op.ref_span] if op.ref_span is not None else None
            hyp_text = hyp_std[op.hyp_span] if op.hyp_span is not None else None
        else:
            ref_text = None
            hyp_text = None

        ref_str, hyp_str, op_length = format_alignment_op_html(
            op, color_scheme=color_scheme, ref_text=ref_text, hyp_text=hyp_text
        )

        is_kt_start = op_idx in start_indices
        is_kt_end = op_idx in stop_indices
        is_kt = is_kt_start or is_kt_end or op_idx in open_indices
        if is_kt:
            ref_str = format_key_term(ref_str, start=is_kt_start, end=is_kt_end)

        # --- Add separator to current line (before wrap check so it stays at end) ---
        ref_line += ref_sep
        hyp_line += hyp_sep
        current_length += sep_length

        # --- Line wrap (op only — separator already on current line) ---
        if current_length + op_length > max_line_length and current_length > 0:
            lines.append((ref_line, hyp_line))
            ref_line, hyp_line = "", ""
            current_length = 0

        ref_line += ref_str
        hyp_line += hyp_str
        current_length += op_length

        # --- Update prev positions ---
        if op.ref_span is not None:
            prev_ref_end = op.ref_span.stop
        if op.hyp_span is not None:
            prev_hyp_end = op.hyp_span.stop

    lines.append((ref_line, hyp_line))
    return lines


def generate_alignment_html_lines_dual(
    alignment: "Alignment",
    max_line_length: int = 100,
    color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
    allow_subset_matches: bool = False,
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Render the alignment as both normalized and surface HTML views with synchronized line breaks.

    Line breaks are shared: a break occurs when either view would exceed ``max_line_length``.
    This ensures the same tokens appear on the same lines in both views for easy comparison.

    Args:
        alignment: The alignment to render.
        max_line_length: The maximum character length per line for wrapping.
        color_scheme: The color scheme to use for display.
        allow_subset_matches: If True, allow subset key term matches.

    Returns:
        A list of tuples ``((norm_ref, norm_hyp), (surf_ref, surf_hyp))`` per line.
    """
    has_surface = alignment.src is not None
    if has_surface:
        ref_std = alignment.src.ref.standardized
        hyp_std = alignment.src.hyp.standardized
    else:
        ref_std = None
        hyp_std = None

    start_indices, stop_indices, open_indices = _get_key_term_indicators(
        alignment, allow_subset_matches=allow_subset_matches
    )

    n_ops = len(alignment)
    if n_ops == 0:
        return [(("", ""), ("", ""))]

    next_ref_span_after: list[int | None] = [None] * n_ops
    next_hyp_span_after: list[int | None] = [None] * n_ops
    last_ref: int | None = None
    last_hyp: int | None = None
    for i in range(n_ops - 1, -1, -1):
        next_ref_span_after[i] = last_ref
        next_hyp_span_after[i] = last_hyp
        if alignment[i].ref_span is not None:
            last_ref = i
        if alignment[i].hyp_span is not None:
            last_hyp = i

    norm_ref_line, norm_hyp_line = "", ""
    surf_ref_line, surf_hyp_line = "", ""
    norm_length = 0
    surf_length = 0
    prev_ref_end = 0
    prev_hyp_end = 0

    # Leading prefix (inter-token content before the first op)
    if has_surface:
        first_ref_start = None
        first_hyp_start = None
        for op in alignment:
            if op.ref_span is not None and first_ref_start is None:
                first_ref_start = op.ref_span.start
            if op.hyp_span is not None and first_hyp_start is None:
                first_hyp_start = op.hyp_span.start
            if first_ref_start is not None and first_hyp_start is not None:
                break
        ref_prefix = ref_std[:first_ref_start] if first_ref_start else ""
        hyp_prefix = hyp_std[:first_hyp_start] if first_hyp_start else ""
        ref_prefix = _eliminate_newlines(ref_prefix)
        hyp_prefix = _eliminate_newlines(hyp_prefix)
        if ref_prefix or hyp_prefix:
            ref_len = len(ref_prefix)
            hyp_len = len(hyp_prefix)
            sep_len = max(ref_len, hyp_len, 1)
            surf_ref_line += _escape_and_nbsp(ref_prefix)
            surf_hyp_line += _escape_and_nbsp(hyp_prefix)
            if ref_len < sep_len:
                surf_ref_line += get_html_padding(sep_len - ref_len, color_scheme)
            if hyp_len < sep_len:
                surf_hyp_line += get_html_padding(sep_len - hyp_len, color_scheme)
            surf_length += sep_len

    lines: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for op_idx, op in enumerate(alignment):
        # --- Pre-separator (inter-token content before this op) ---
        if op_idx > 0:
            prev_op = alignment[op_idx - 1]
            prev_was_delete = prev_op.type == OpType.DELETE
            prev_was_insert = prev_op.type == OpType.INSERT
            prev_kt_open = (op_idx - 1) in open_indices

            # Normalized separator (always &nbsp;)
            norm_sep_length = 1
            norm_ref_sep = format_key_term("&nbsp;") if prev_kt_open else "&nbsp;"
            norm_hyp_sep = get_html_padding(1, color_scheme=color_scheme) if prev_op.hyp_right_partial else "&nbsp;"

            # Surface separator
            if has_surface:
                if op.ref_span is not None and not prev_was_insert:
                    ref_inter_raw = ref_std[prev_ref_end : op.ref_span.start]
                elif op.type == OpType.INSERT and not prev_was_insert:
                    next_ref_idx = next_ref_span_after[op_idx]
                    end = alignment[next_ref_idx].ref_span.start if next_ref_idx is not None else prev_ref_end
                    ref_inter_raw = ref_std[prev_ref_end:end]
                else:
                    ref_inter_raw = ""

                if op.hyp_span is not None and not prev_was_delete:
                    hyp_inter_raw = hyp_std[prev_hyp_end : op.hyp_span.start]
                elif op.type == OpType.DELETE and not prev_was_delete:
                    next_hyp_idx = next_hyp_span_after[op_idx]
                    end = alignment[next_hyp_idx].hyp_span.start if next_hyp_idx is not None else prev_hyp_end
                    hyp_inter_raw = hyp_std[prev_hyp_end:end]
                else:
                    hyp_inter_raw = ""

                ref_inter_text = _eliminate_newlines(ref_inter_raw)
                hyp_inter_text = _eliminate_newlines(hyp_inter_raw)
                ref_inter_len = len(ref_inter_text)
                hyp_inter_len = len(hyp_inter_text)
                surf_sep_length = max(ref_inter_len, hyp_inter_len, 1)

                surf_ref_sep = _escape_and_nbsp(ref_inter_text)
                surf_hyp_sep = _escape_and_nbsp(hyp_inter_text)

                if ref_inter_len < surf_sep_length:
                    surf_ref_sep += get_html_padding(surf_sep_length - ref_inter_len, color_scheme)
                if hyp_inter_len < surf_sep_length:
                    surf_hyp_sep += get_html_padding(surf_sep_length - hyp_inter_len, color_scheme)

                if prev_kt_open:
                    surf_ref_sep = format_key_term(surf_ref_sep)

                if prev_op.hyp_right_partial:
                    surf_hyp_sep = get_html_padding(surf_sep_length, color_scheme)
            else:
                surf_sep_length = 1
                surf_ref_sep = norm_ref_sep
                surf_hyp_sep = norm_hyp_sep
        else:
            norm_sep_length = 0
            surf_sep_length = 0
            norm_ref_sep = norm_hyp_sep = ""
            surf_ref_sep = surf_hyp_sep = ""

        # --- Op content ---
        norm_ref_str, norm_hyp_str, norm_op_length = format_alignment_op_html(op, color_scheme=color_scheme)

        if has_surface:
            ref_text = ref_std[op.ref_span] if op.ref_span is not None else None
            hyp_text = hyp_std[op.hyp_span] if op.hyp_span is not None else None
            surf_ref_str, surf_hyp_str, surf_op_length = format_alignment_op_html(
                op, color_scheme=color_scheme, ref_text=ref_text, hyp_text=hyp_text
            )
        else:
            surf_ref_str, surf_hyp_str, surf_op_length = norm_ref_str, norm_hyp_str, norm_op_length

        is_kt_start = op_idx in start_indices
        is_kt_end = op_idx in stop_indices
        is_kt = is_kt_start or is_kt_end or op_idx in open_indices
        if is_kt:
            norm_ref_str = format_key_term(norm_ref_str, start=is_kt_start, end=is_kt_end)
            surf_ref_str = format_key_term(surf_ref_str, start=is_kt_start, end=is_kt_end)

        # --- Add separators to current lines ---
        norm_ref_line += norm_ref_sep
        norm_hyp_line += norm_hyp_sep
        norm_length += norm_sep_length

        surf_ref_line += surf_ref_sep
        surf_hyp_line += surf_hyp_sep
        surf_length += surf_sep_length

        # --- Line wrap: break when EITHER view would exceed max ---
        # Guard: don't break if the line only has the separator (no token content yet),
        # to avoid producing separator-only lines in normalized mode.
        if (norm_length + norm_op_length > max_line_length or surf_length + surf_op_length > max_line_length) and (
            norm_length > norm_sep_length or surf_length > surf_sep_length
        ):
            lines.append(((norm_ref_line, norm_hyp_line), (surf_ref_line, surf_hyp_line)))
            norm_ref_line, norm_hyp_line = "", ""
            surf_ref_line, surf_hyp_line = "", ""
            norm_length = 0
            surf_length = 0

        norm_ref_line += norm_ref_str
        norm_hyp_line += norm_hyp_str
        norm_length += norm_op_length

        surf_ref_line += surf_ref_str
        surf_hyp_line += surf_hyp_str
        surf_length += surf_op_length

        if op.ref_span is not None:
            prev_ref_end = op.ref_span.stop
        if op.hyp_span is not None:
            prev_hyp_end = op.hyp_span.stop

    # Trailing inter-token content after the last op
    if has_surface:
        ref_trail = ref_std[prev_ref_end:] if prev_ref_end < len(ref_std) else ""
        hyp_trail = hyp_std[prev_hyp_end:] if prev_hyp_end < len(hyp_std) else ""
        ref_trail = _eliminate_newlines(ref_trail)
        hyp_trail = _eliminate_newlines(hyp_trail)
        if ref_trail or hyp_trail:
            ref_len = len(ref_trail)
            hyp_len = len(hyp_trail)
            sep_len = max(ref_len, hyp_len, 1)
            surf_ref_trail = _escape_and_nbsp(ref_trail)
            surf_hyp_trail = _escape_and_nbsp(hyp_trail)
            if ref_len < sep_len:
                surf_ref_trail += get_html_padding(sep_len - ref_len, color_scheme)
            if hyp_len < sep_len:
                surf_hyp_trail += get_html_padding(sep_len - hyp_len, color_scheme)
            surf_ref_line += surf_ref_trail
            surf_hyp_line += surf_hyp_trail
            surf_length += sep_len

    lines.append(((norm_ref_line, norm_hyp_line), (surf_ref_line, surf_hyp_line)))
    return lines


def _precompute_alignment_units(
    alignment: "Alignment",
    color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
    allow_subset_matches: bool = False,
) -> list[list[tuple[tuple[str, str, int, str, str, int], tuple[str, str, int, str, str, int]]]]:
    """Pre-compute render units for an alignment, grouped by ref token index.

    Each unit is a (sep, content) pair where each is
    (norm_ref, norm_hyp, norm_len, surf_ref, surf_hyp, surf_len).

    INSERT ops are assigned to the ref token index of the preceding non-INSERT op
    (or 0 for leading INSERTs).

    Returns a list indexed by ref token index, each entry a list of (sep, content) units.
    """
    has_surface = alignment.src is not None
    if has_surface:
        ref_std = alignment.src.ref.standardized
        hyp_std = alignment.src.hyp.standardized
    else:
        ref_std = None
        hyp_std = None

    start_indices, stop_indices, open_indices = _get_key_term_indicators(
        alignment, allow_subset_matches=allow_subset_matches
    )

    n_ops = len(alignment)
    if n_ops == 0:
        return []

    next_ref_span_after: list[int | None] = [None] * n_ops
    next_hyp_span_after: list[int | None] = [None] * n_ops
    last_ref: int | None = None
    last_hyp: int | None = None
    for i in range(n_ops - 1, -1, -1):
        next_ref_span_after[i] = last_ref
        next_hyp_span_after[i] = last_hyp
        if alignment[i].ref_span is not None:
            last_ref = i
        if alignment[i].hyp_span is not None:
            last_hyp = i

    op_to_ref_token: dict[int, int] = {}
    ref_token_idx = 0
    for op_idx, op in enumerate(alignment):
        if op.ref_span is not None:
            op_to_ref_token[op_idx] = ref_token_idx
            ref_token_idx += 1

    n_ref_tokens = ref_token_idx
    if n_ref_tokens == 0 and n_ops > 0:
        n_ref_tokens = 1
    units_by_ref_token: list[list[tuple[tuple[str, str, int, str, str, int], tuple[str, str, int, str, str, int]]]] = [
        [] for _ in range(n_ref_tokens)
    ]

    prev_ref_end = 0
    prev_hyp_end = 0
    current_ref_token = -1

    # Leading prefix (inter-token content before the first op)
    if has_surface:
        first_ref_start = None
        first_hyp_start = None
        for op in alignment:
            if op.ref_span is not None and first_ref_start is None:
                first_ref_start = op.ref_span.start
            if op.hyp_span is not None and first_hyp_start is None:
                first_hyp_start = op.hyp_span.start
            if first_ref_start is not None and first_hyp_start is not None:
                break
        ref_prefix = ref_std[:first_ref_start] if first_ref_start else ""
        hyp_prefix = hyp_std[:first_hyp_start] if first_hyp_start else ""
        ref_prefix = _eliminate_newlines(ref_prefix)
        hyp_prefix = _eliminate_newlines(hyp_prefix)
        if ref_prefix or hyp_prefix:
            ref_len = len(ref_prefix)
            hyp_len = len(hyp_prefix)
            sep_len = max(ref_len, hyp_len, 1)
            prefix_ref = _escape_and_nbsp(ref_prefix)
            prefix_hyp = _escape_and_nbsp(hyp_prefix)
            if ref_len < sep_len:
                prefix_ref += get_html_padding(sep_len - ref_len, color_scheme)
            if hyp_len < sep_len:
                prefix_hyp += get_html_padding(sep_len - hyp_len, color_scheme)
            prefix_sep = ("", "", 0, prefix_ref, prefix_hyp, sep_len)
            prefix_content = ("", "", 0, "", "", 0)
            if units_by_ref_token:
                units_by_ref_token[0].insert(0, (prefix_sep, prefix_content))

    for op_idx, op in enumerate(alignment):
        if op_idx in op_to_ref_token:
            current_ref_token = op_to_ref_token[op_idx]

        if op_idx > 0:
            prev_op = alignment[op_idx - 1]
            prev_was_delete = prev_op.type == OpType.DELETE
            prev_was_insert = prev_op.type == OpType.INSERT
            prev_kt_open = (op_idx - 1) in open_indices

            norm_sep_length = 1
            norm_ref_sep = format_key_term("&nbsp;") if prev_kt_open else "&nbsp;"
            norm_hyp_sep = get_html_padding(1, color_scheme=color_scheme) if prev_op.hyp_right_partial else "&nbsp;"

            if has_surface:
                if op.ref_span is not None and not prev_was_insert:
                    ref_inter_raw = ref_std[prev_ref_end : op.ref_span.start]
                elif op.type == OpType.INSERT and not prev_was_insert:
                    next_ref_idx = next_ref_span_after[op_idx]
                    end = alignment[next_ref_idx].ref_span.start if next_ref_idx is not None else prev_ref_end
                    ref_inter_raw = ref_std[prev_ref_end:end]
                else:
                    ref_inter_raw = ""

                if op.hyp_span is not None and not prev_was_delete:
                    hyp_inter_raw = hyp_std[prev_hyp_end : op.hyp_span.start]
                elif op.type == OpType.DELETE and not prev_was_delete:
                    next_hyp_idx = next_hyp_span_after[op_idx]
                    end = alignment[next_hyp_idx].hyp_span.start if next_hyp_idx is not None else prev_hyp_end
                    hyp_inter_raw = hyp_std[prev_hyp_end:end]
                else:
                    hyp_inter_raw = ""

                ref_inter_text = _eliminate_newlines(ref_inter_raw)
                hyp_inter_text = _eliminate_newlines(hyp_inter_raw)
                ref_inter_len = len(ref_inter_text)
                hyp_inter_len = len(hyp_inter_text)
                surf_sep_length = max(ref_inter_len, hyp_inter_len, 1)

                surf_ref_sep = _escape_and_nbsp(ref_inter_text)
                surf_hyp_sep = _escape_and_nbsp(hyp_inter_text)

                if ref_inter_len < surf_sep_length:
                    surf_ref_sep += get_html_padding(surf_sep_length - ref_inter_len, color_scheme)
                if hyp_inter_len < surf_sep_length:
                    surf_hyp_sep += get_html_padding(surf_sep_length - hyp_inter_len, color_scheme)

                if prev_kt_open:
                    surf_ref_sep = format_key_term(surf_ref_sep)

                if prev_op.hyp_right_partial:
                    surf_hyp_sep = get_html_padding(surf_sep_length, color_scheme)
            else:
                surf_sep_length = 1
                surf_ref_sep = norm_ref_sep
                surf_hyp_sep = norm_hyp_sep
        else:
            norm_sep_length = 0
            surf_sep_length = 0
            norm_ref_sep = norm_hyp_sep = ""
            surf_ref_sep = surf_hyp_sep = ""

        norm_ref_str, norm_hyp_str, norm_op_length = format_alignment_op_html(op, color_scheme=color_scheme)

        if has_surface:
            ref_text = ref_std[op.ref_span] if op.ref_span is not None else None
            hyp_text = hyp_std[op.hyp_span] if op.hyp_span is not None else None
            surf_ref_str, surf_hyp_str, surf_op_length = format_alignment_op_html(
                op, color_scheme=color_scheme, ref_text=ref_text, hyp_text=hyp_text
            )
        else:
            surf_ref_str, surf_hyp_str, surf_op_length = norm_ref_str, norm_hyp_str, norm_op_length

        is_kt_start = op_idx in start_indices
        is_kt_end = op_idx in stop_indices
        is_kt = is_kt_start or is_kt_end or op_idx in open_indices
        if is_kt:
            norm_ref_str = format_key_term(norm_ref_str, start=is_kt_start, end=is_kt_end)
            surf_ref_str = format_key_term(surf_ref_str, start=is_kt_start, end=is_kt_end)

        sep = (
            norm_ref_sep,
            norm_hyp_sep,
            norm_sep_length,
            surf_ref_sep,
            surf_hyp_sep,
            surf_sep_length,
        )
        content = (
            norm_ref_str,
            norm_hyp_str,
            norm_op_length,
            surf_ref_str,
            surf_hyp_str,
            surf_op_length,
        )

        token_idx = max(current_ref_token, 0)
        units_by_ref_token[token_idx].append((sep, content))

        if op.ref_span is not None:
            prev_ref_end = op.ref_span.stop
        if op.hyp_span is not None:
            prev_hyp_end = op.hyp_span.stop

    # Trailing separator (inter-token content after the last op)
    if has_surface:
        ref_trail = ref_std[prev_ref_end:] if prev_ref_end < len(ref_std) else ""
        hyp_trail = hyp_std[prev_hyp_end:] if prev_hyp_end < len(hyp_std) else ""
        ref_trail = _eliminate_newlines(ref_trail)
        hyp_trail = _eliminate_newlines(hyp_trail)
        if ref_trail or hyp_trail:
            ref_len = len(ref_trail)
            hyp_len = len(hyp_trail)
            sep_len = max(ref_len, hyp_len, 1)
            trail_ref = _escape_and_nbsp(ref_trail)
            trail_hyp = _escape_and_nbsp(hyp_trail)
            if ref_len < sep_len:
                trail_ref += get_html_padding(sep_len - ref_len, color_scheme)
            if hyp_len < sep_len:
                trail_hyp += get_html_padding(sep_len - hyp_len, color_scheme)
            trail_sep = ("", "", 0, trail_ref, trail_hyp, sep_len)
            trail_content = ("", "", 0, "", "", 0)
            if units_by_ref_token:
                units_by_ref_token[-1].append((trail_sep, trail_content))

    return units_by_ref_token


def generate_alignment_html_lines_multi(
    alignments: list["Alignment"],
    max_line_length: int = 100,
    color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
    allow_subset_matches: bool = False,
) -> list[list[tuple[tuple[str, str], tuple[str, str]]]]:
    """Render multiple alignments with synchronized line breaks.

    Line breaks are shared across all alignments and both views (normalized and
    surface): a break occurs when any alignment's any view would exceed
    ``max_line_length``. Requires all alignments to share the same reference text
    for meaningful synchronization.

    Args:
        alignments: List of alignments (one per dataset) for the same example.
        max_line_length: The maximum character length per line for wrapping.
        color_scheme: The color scheme to use for display.
        allow_subset_matches: If True, allow subset key term matches.

    Returns:
        Per line, per alignment, ``((norm_ref, norm_hyp), (surf_ref, surf_hyp))``.
    """
    n_alignments = len(alignments)
    if n_alignments == 0:
        return []
    if n_alignments == 1:
        dual = generate_alignment_html_lines_dual(
            alignments[0],
            max_line_length=max_line_length,
            color_scheme=color_scheme,
            allow_subset_matches=allow_subset_matches,
        )
        return [[line] for line in dual]

    all_units = [_precompute_alignment_units(a, color_scheme, allow_subset_matches) for a in alignments]
    max_ref_tokens = max(len(u) for u in all_units) if all_units else 0
    if max_ref_tokens == 0:
        return [[(("", ""), ("", "")) for _ in range(n_alignments)]]

    # Precompute content-only lengths per ref token for break checks
    token_norm_content_lens = [
        [sum(unit[1][2] for unit in tokens) for tokens in all_units[a]] if all_units[a] else []
        for a in range(n_alignments)
    ]
    token_surf_content_lens = [
        [sum(unit[1][5] for unit in tokens) for tokens in all_units[a]] if all_units[a] else []
        for a in range(n_alignments)
    ]

    norm_ref_lines = [""] * n_alignments
    norm_hyp_lines = [""] * n_alignments
    surf_ref_lines = [""] * n_alignments
    surf_hyp_lines = [""] * n_alignments
    norm_lengths = [0] * n_alignments
    surf_lengths = [0] * n_alignments

    lines: list[list[tuple[tuple[str, str], tuple[str, str]]]] = []

    for ref_token_idx in range(max_ref_tokens):
        # Add first separator (inter-token between ref tokens) to current line before break check
        if ref_token_idx > 0:
            first_sep_norm_lens = [0] * n_alignments
            first_sep_surf_lens = [0] * n_alignments
            for a in range(n_alignments):
                if ref_token_idx >= len(all_units[a]):
                    continue
                if all_units[a][ref_token_idx]:
                    sep = all_units[a][ref_token_idx][0][0]
                    norm_ref_lines[a] += sep[0]
                    norm_hyp_lines[a] += sep[1]
                    norm_lengths[a] += sep[2]
                    surf_ref_lines[a] += sep[3]
                    surf_hyp_lines[a] += sep[4]
                    surf_lengths[a] += sep[5]
                    first_sep_norm_lens[a] = sep[2]
                    first_sep_surf_lens[a] = sep[5]

            needs_break = False
            for a in range(n_alignments):
                if ref_token_idx >= len(all_units[a]):
                    continue
                next_norm = token_norm_content_lens[a][ref_token_idx]
                next_surf = token_surf_content_lens[a][ref_token_idx]
                if (
                    norm_lengths[a] + next_norm > max_line_length or surf_lengths[a] + next_surf > max_line_length
                ) and (norm_lengths[a] > first_sep_norm_lens[a] or surf_lengths[a] > first_sep_surf_lens[a]):
                    needs_break = True
                    break

            if needs_break:
                line = []
                for a in range(n_alignments):
                    line.append(
                        (
                            (norm_ref_lines[a], norm_hyp_lines[a]),
                            (surf_ref_lines[a], surf_hyp_lines[a]),
                        )
                    )
                lines.append(line)
                norm_ref_lines = [""] * n_alignments
                norm_hyp_lines = [""] * n_alignments
                surf_ref_lines = [""] * n_alignments
                surf_hyp_lines = [""] * n_alignments
                norm_lengths = [0] * n_alignments
                surf_lengths = [0] * n_alignments

        # Add content interleaved with remaining separators (sep[1:] within each unit)
        for a in range(n_alignments):
            if ref_token_idx >= len(all_units[a]):
                continue
            for unit_idx, (sep, content) in enumerate(all_units[a][ref_token_idx]):
                if unit_idx > 0:
                    norm_ref_lines[a] += sep[0]
                    norm_hyp_lines[a] += sep[1]
                    norm_lengths[a] += sep[2]
                    surf_ref_lines[a] += sep[3]
                    surf_hyp_lines[a] += sep[4]
                    surf_lengths[a] += sep[5]
                norm_ref_lines[a] += content[0]
                norm_hyp_lines[a] += content[1]
                norm_lengths[a] += content[2]
                surf_ref_lines[a] += content[3]
                surf_hyp_lines[a] += content[4]
                surf_lengths[a] += content[5]

    if any(norm_lengths[a] > 0 or surf_lengths[a] > 0 for a in range(n_alignments)):
        line = []
        for a in range(n_alignments):
            line.append(
                (
                    (norm_ref_lines[a], norm_hyp_lines[a]),
                    (surf_ref_lines[a], surf_hyp_lines[a]),
                )
            )
        lines.append(line)

    return lines
