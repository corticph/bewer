"""Configurable labels and tooltips for alignment display in HTML reports."""

__all__ = ["HTMLAlignmentLabels"]


class HTMLAlignmentLabels:
    """Configurable labels and tooltips for alignment display in HTML reports.

    Subclass and override any attribute to customize the labels shown in the report.
    Follows the same subclassing pattern as HTMLAlignmentColors.
    """

    # Line indicators
    REF = "Ref."
    HYP = "Hyp."

    # Legend labels
    MATCH = "Match"
    SUBSTITUTION = "Substitution"
    INSERTION = "Insertion"
    DELETION = "Deletion"
    PADDING = "Padding"
    KEYWORD = "Keyword"

    # Legend tooltips (None = no tooltip rendered)
    MATCH_TOOLTIP: str | None = None
    SUBSTITUTION_TOOLTIP: str | None = None
    INSERTION_TOOLTIP: str | None = None
    DELETION_TOOLTIP: str | None = None
    PADDING_TOOLTIP: str | None = None
    KEYWORD_TOOLTIP: str | None = None
