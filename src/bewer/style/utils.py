from io import StringIO
from typing import TYPE_CHECKING

from rich.console import Console
from rich.text import Text

if TYPE_CHECKING:
    from bewer.core.token import TokenList


def highlight_span(text: str, start: int, end: int, style: str) -> str:
    # Use a StringIO buffer to capture ANSI output
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, color_system="truecolor")

    # Create styled text
    text = Text(text)
    text.stylize(style, start, end)

    # Print to the buffer instead of the terminal
    console.print(text, end="")

    # Get the ANSI string
    return buffer.getvalue()


def highlight_tokens(text: str, tokens: "TokenList", style1: str = "green", style2: str = "yellow") -> str:
    # Use a StringIO buffer to capture ANSI output
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, color_system="truecolor")

    # Create styled text
    text = Text(text)
    for token in tokens:
        text.stylize(style1, token.start, token.end)
        style1, style2 = style2, style1

    # Print to the buffer instead of the terminal
    console.print(text, end="")

    # Get the ANSI string
    return buffer.getvalue()
