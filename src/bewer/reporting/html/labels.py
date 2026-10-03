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
    MATCH_TOOLTIP: str | None = "Correct: hypothesis matches reference."
    SUBSTITUTION_TOOLTIP: str | None = "Hypothesis differs from reference."
    INSERTION_TOOLTIP: str | None = "Extra word in hypothesis not in reference."
    DELETION_TOOLTIP: str | None = "Missing word from reference not in hypothesis."
    PADDING_TOOLTIP: str | None = "Alignment padding to keep both sides in sync."
    KEYWORD_TOOLTIP: str | None = "Key term highlighted for targeted metrics."
