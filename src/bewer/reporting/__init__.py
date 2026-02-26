__all__ = [
    "generate_report",
    "render_report_html",
    "ReportMetric",
    "ReportAlignment",
    "ReportSummaryItem",
    "DEFAULT_REPORT_METRICS",
    "DEFAULT_REPORT_SUMMARY_ITEMS",
    "DEFAULT_REPORT_ALIGNMENT",
    "display_basic_aligned",
    "ColorScheme",
    "DefaultColorScheme",
]

_HTML_REPORT_NAMES = {
    "generate_report",
    "render_report_html",
    "ReportMetric",
    "ReportAlignment",
    "ReportSummaryItem",
    "DEFAULT_REPORT_METRICS",
    "DEFAULT_REPORT_SUMMARY_ITEMS",
    "DEFAULT_REPORT_ALIGNMENT",
}

_PYTHON_ALIGNMENT_NAMES = {"display_basic_aligned", "ColorScheme", "DefaultColorScheme"}


def __getattr__(name: str):
    if name in _HTML_REPORT_NAMES:
        from bewer.reporting.html.report import (
            DEFAULT_REPORT_ALIGNMENT,
            DEFAULT_REPORT_METRICS,
            DEFAULT_REPORT_SUMMARY_ITEMS,
            ReportAlignment,
            ReportMetric,
            ReportSummaryItem,
            generate_report,
            render_report_html,
        )

        _attrs = {
            "generate_report": generate_report,
            "render_report_html": render_report_html,
            "ReportMetric": ReportMetric,
            "ReportAlignment": ReportAlignment,
            "ReportSummaryItem": ReportSummaryItem,
            "DEFAULT_REPORT_METRICS": DEFAULT_REPORT_METRICS,
            "DEFAULT_REPORT_SUMMARY_ITEMS": DEFAULT_REPORT_SUMMARY_ITEMS,
            "DEFAULT_REPORT_ALIGNMENT": DEFAULT_REPORT_ALIGNMENT,
        }
        return _attrs[name]
    if name in _PYTHON_ALIGNMENT_NAMES:
        from bewer.reporting.python.alignment import (
            ColorScheme,
            DefaultColorScheme,
            display_basic_aligned,
        )

        _attrs = {
            "display_basic_aligned": display_basic_aligned,
            "ColorScheme": ColorScheme,
            "DefaultColorScheme": DefaultColorScheme,
        }
        return _attrs[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
