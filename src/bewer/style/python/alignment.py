"""Module for styling alignment display with colors.

Alignments can be displayed in the terminal with color coding for different operation types.

In terms of detail level, three main display modes are supported:
- Basic: The tokens are displayed as provided by the alignment. Tokens are separated by a single space.
- Contextualized: The tokens are displayed with surrounding context (new lines may be added/removed for the
                  hypothesis). This mode requires that either the span attributes are set for the operations, or that
                  the alignment token_idx attributes are set and the original texts are accessible.
"""

import shutil
from typing import TYPE_CHECKING

from rich.console import Console
from rich.text import Text

from bewer.alignment.op_type import OpType

if TYPE_CHECKING:
    from bewer.alignment.op import Alignment, Op


class ColorScheme:
    """Base class for color schemes used in alignment display."""

    PAD: str
    DEL: str
    INS: str
    SUB: str
    MATCH: str


class DefaultColorScheme(ColorScheme):
    PAD = "bright_black"
    DEL = "bright_red"
    INS = "bright_cyan"
    SUB = "bright_yellow"
    MATCH = "default"


def get_padding(length: int, color_scheme: ColorScheme = DefaultColorScheme) -> Text:
    """Get a padding string of spaces of the specified length.

    Args:
        length (int): The length of the padding.

    Returns:
        str: A string of spaces of the specified length.
    """
    return Text(" " * length, style=f"on {color_scheme.PAD}")


def format_match_op(op: "Op", color_scheme: ColorScheme = DefaultColorScheme) -> tuple[Text, Text, int]:
    """Format a match operation for display."""
    len_ref = len(op.ref)
    len_hyp = len(op.hyp)
    length = max(len_ref, len_hyp)
    # NOTE: For matches, the ref and hyp token should be identical, but might differ due to normalization.
    ref_str = Text(op.ref, style=color_scheme.MATCH)
    hyp_str = Text(op.hyp, style=color_scheme.MATCH)
    if len_ref < length:
        ref_str.append(get_padding(length - len_ref, color_scheme=color_scheme))
    if len_hyp < length:
        hyp_str.append(get_padding(length - len_hyp, color_scheme=color_scheme))
    return ref_str, hyp_str, length


def format_substitute_op(op: "Op", color_scheme: ColorScheme = DefaultColorScheme) -> tuple[Text, Text, int]:
    """Format a substitute operation for display."""
    len_ref = len(op.ref)
    len_hyp = len(op.hyp)
    length = max(len_ref, len_hyp)
    ref_str = Text(op.ref, style=color_scheme.SUB)
    hyp_str = Text(op.hyp, style=color_scheme.SUB)
    if len_ref < length:
        ref_str.append(get_padding(length - len_ref, color_scheme=color_scheme))
    if len_hyp < length:
        hyp_str.append(get_padding(length - len_hyp, color_scheme=color_scheme))
    return ref_str, hyp_str, length


def format_insert_op(op: "Op", color_scheme: ColorScheme = DefaultColorScheme) -> tuple[Text, Text, int]:
    """Format an insert operation for display."""
    len_hyp = len(op.hyp)
    hyp_str = Text(op.hyp, style=color_scheme.INS)
    ref_str = get_padding(len_hyp, color_scheme=color_scheme)
    return ref_str, hyp_str, len_hyp


def format_delete_op(op: "Op", color_scheme: ColorScheme = DefaultColorScheme) -> tuple[Text, Text, int]:
    """Format a delete operation for display."""
    len_ref = len(op.ref)
    ref_str = Text(op.ref, style=color_scheme.DEL)
    hyp_str = get_padding(len_ref, color_scheme=color_scheme)
    return ref_str, hyp_str, len_ref


def format_alignment_op(op: "Op", color_scheme: ColorScheme = DefaultColorScheme) -> tuple[Text, Text, int]:
    """Format an alignment operation for display.

    Args:
        op (Op): The alignment operation.

    Returns:
        tuple[str, str, int]: A tuple containing the formatted ref and hyp tokens and their unformatted length.
    """
    if op.type == OpType.MATCH:
        return format_match_op(op, color_scheme=color_scheme)
    if op.type == OpType.SUBSTITUTE:
        return format_substitute_op(op, color_scheme=color_scheme)
    if op.type == OpType.INSERT:
        return format_insert_op(op, color_scheme=color_scheme)
    if op.type == OpType.DELETE:
        return format_delete_op(op, color_scheme=color_scheme)
    raise ValueError(f"Unknown operation type: {op.type}")


def get_line_prefixes(line_number: int) -> tuple[Text, Text]:
    """Get the line prefixes for reference and hypothesis lines."""
    str_num = str(line_number).rjust(4)

    ref_prefix = Text(f"{str_num}  Ref.  ", style="bright_black")
    hyp_prefix = Text("      Hyp.  ", style="bright_black")
    return ref_prefix, hyp_prefix


def display_basic_aligned(
    alignment: "Alignment",
    max_line_length: int | float = 0.5,
    title: str | None = None,
    color_scheme: ColorScheme = DefaultColorScheme,
) -> None:
    """Display the alignment in a basic format.

    Args:
        alignment (Alignment): The alignment to display.
        max_line_length (int | float): The maximum line length for wrapping. If a float, it is interpreted as a
            fraction of the terminal width.
        title (str | None): An optional title to display above the alignment.
        color_scheme (ColorScheme): The color scheme to use for display.

    Prints:
        The formatted alignment to the console.
    """
    if isinstance(max_line_length, float):
        if not (0.0 < max_line_length <= 1.0):
            raise ValueError("If max_line_length is a float, it must be in the range (0.0, 1.0].")
        max_line_length = round(shutil.get_terminal_size().columns * max_line_length)

    # Construct and format lines.
    lines: list[tuple[Text, Text]] = []
    ref_line, hyp_line = get_line_prefixes(line_number=1)
    max_line_length -= len(ref_line)  # Adjust max_line_length to account for prefixes
    for op in alignment:
        ref_str, hyp_str, op_length = format_alignment_op(op, color_scheme=color_scheme)

        if len(ref_line) + op_length > max_line_length:
            lines.append((ref_line, hyp_line))
            ref_line, hyp_line = get_line_prefixes(line_number=len(lines) + 1)

        ref_line.append(ref_str).append(" ")
        hyp_line.append(hyp_str)
        # NOTE: The ref cannot be partial with any of the current alignment algorithms.
        if op.hyp_right_partial:
            hyp_line.append(get_padding(1, color_scheme=color_scheme))
        else:
            hyp_line.append(" ")

    lines.append((ref_line, hyp_line))

    # Build and print output.
    output = Text()
    separator = Text("", style="bright_black")
    if title:
        output.append(Text(title + "\n\n", style="bold bright_black"))  # type: ignore
    for i, (ref, hyp) in enumerate(lines):
        ref.rstrip()
        hyp.rstrip()
        output.append(ref).append("\n").append(hyp).append("\n")
        if i < len(lines) - 1:
            output.append(separator).append("\n")

    Console().print(output)
