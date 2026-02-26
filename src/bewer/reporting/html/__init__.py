__all__ = [
    "generate_report",
    "render_report_html",
    "ReportMetric",
    "ReportAlignment",
    "ReportSummaryItem",
    "DEFAULT_REPORT_METRICS",
    "DEFAULT_REPORT_SUMMARY_ITEMS",
    "DEFAULT_REPORT_ALIGNMENT",
    "HTMLAlignmentColors",
    "HTMLDefaultAlignmentColors",
    "HTMLBaseColors",
    "HTMLAlignmentLabels",
]

_REPORT_NAMES = {
    "generate_report",
    "render_report_html",
    "ReportMetric",
    "ReportAlignment",
    "ReportSummaryItem",
    "DEFAULT_REPORT_METRICS",
    "DEFAULT_REPORT_SUMMARY_ITEMS",
    "DEFAULT_REPORT_ALIGNMENT",
}

_COLOR_SCHEME_NAMES = {"HTMLAlignmentColors", "HTMLDefaultAlignmentColors", "HTMLBaseColors"}


def __getattr__(name: str):
    if name in _REPORT_NAMES:
        from bewer.reporting.html import report

        return getattr(report, name)
    if name in _COLOR_SCHEME_NAMES:
        from bewer.reporting.html import color_schemes

        return getattr(color_schemes, name)
    if name == "HTMLAlignmentLabels":
        from bewer.reporting.html.labels import HTMLAlignmentLabels

        return HTMLAlignmentLabels
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
