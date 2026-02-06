class HTMLAlignmentColors:
    """Base class for HTML color schemes used in alignment display."""

    PAD: str
    DEL: str
    INS: str
    SUB: str
    MATCH: str

    @classmethod
    def to_html(cls) -> str:
        return f"""\
    <div class="legend-container">
        Alignment color codes:
        <span class="legend-container-item"><span style="color: {cls.MATCH};">■</span> Match</span>
        <span class="legend-container-item"><span style="color: {cls.SUB};">■</span> Substitution</span>
        <span class="legend-container-item"><span style="color: {cls.INS};">■</span> Insertion</span>
        <span class="legend-container-item"><span style="color: {cls.DEL};">■</span> Deletion</span>
        <span class="legend-container-item"><span style="color: {cls.PAD};">■</span> Padding</span>
    </div>\
    """


class HTMLDefaultAlignmentColors(HTMLAlignmentColors):
    """Default color scheme for HTML alignment display."""

    PAD = "#ededed"
    DEL = "#dc3545"
    INS = "#17a2b8"
    SUB = "#c46f01"
    MATCH = "#212529"


class HTMLBaseColors:
    """Color scheme constants for HTML reports."""

    TEXT_COLOR = "#212529"
    BG_COLOR = "#ffffff"
    DIV_TEXT_COLOR = "#acacac"
    DIV_BG_COLOR = "#f7f7f7"
