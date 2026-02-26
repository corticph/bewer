__all__ = ["HTMLAlignmentColors", "HTMLDefaultAlignmentColors", "HTMLBaseColors"]


class HTMLAlignmentColors:
    """Base class for HTML color schemes used in alignment display."""

    PAD: str
    DEL: str
    INS: str
    SUB: str
    MATCH: str
    KEYWORD: str


class HTMLDefaultAlignmentColors(HTMLAlignmentColors):
    """Default color scheme for HTML alignment display."""

    PAD = "#ededed"
    DEL = "#dc3545"
    INS = "#17a2b8"
    SUB = "#c46f01"
    MATCH = "#212529"
    KEYWORD = "#00a2ff"


class HTMLBaseColors:
    """Color scheme constants for HTML reports."""

    TABLE_HEADER_BG = "#595959"
    TEXT_COLOR = "#000000"
    BG_COLOR = "#faf9f5"
    DIV_TEXT_COLOR = "#acacac"
    DIV_BG_COLOR = "#ffffff"
