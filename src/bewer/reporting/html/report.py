"""Module for generating HTML reports from datasets."""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, PackageLoader

from bewer.reporting.html.color_schemes import HTMLAlignmentColors, HTMLBaseColors, HTMLDefaultAlignmentColors
from bewer.reporting.html.labels import HTMLAlignmentLabels

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset


class ReportMetric:
    """Specification for a metric to include in the report."""

    def __init__(self, name: str, label: str | None = None, format: str = ".2%", **metric_kwargs):
        self.name = name  # metric registry name (e.g. "wer")
        self.label = label  # display label override (default: metric.long_name)
        self.format = format  # format spec for the value
        self.metric_kwargs = metric_kwargs  # optional kwargs to pass when resolving the metric from the dataset


class ReportAlignment:
    """Specification for the alignment to include in the report."""

    def __init__(self, name: str, **kwargs):
        self.name = name  # metric registry name (e.g. "levenshtein")
        self.metric_kwargs = kwargs  # kwargs passed to the metric factory


class ReportSummaryItem:
    """Specification for a summary item to include in the report."""

    def __init__(self, name: str, label: str | None = None, format: str = ",.0f"):
        self.name = name  # summary attribute name (e.g. "num_examples")
        self.label = label  # display label override
        self.format = format  # format spec


DEFAULT_REPORT_ALIGNMENT = ReportAlignment("levenshtein")

DEFAULT_REPORT_METRICS = [
    ReportMetric("wer"),
    ReportMetric("cer"),
    ReportMetric("legacy_medical_word_accuracy", label="Medical Term Recall"),
]

DEFAULT_REPORT_SUMMARY_ITEMS = [
    ReportSummaryItem("num_examples", label="Number of examples"),
    ReportSummaryItem("num_ref_words", label="Number of reference words"),
    ReportSummaryItem("num_ref_chars", label="Number of reference characters"),
    ReportSummaryItem("num_hyp_words", label="Number of hypothesis words"),
    ReportSummaryItem("num_hyp_chars", label="Number of hypothesis characters"),
]


def indent_tabs(text: str, width: int = 1) -> str:
    """Indent each line of the given text with a specified number of tabs."""
    padding = "\t" * width
    return "\n".join(padding + line for line in text.split("\n"))


def render_report_html(
    dataset: "Dataset",
    template: str = "report_basic",
    title: str | None = None,
    base_color_scheme: type[HTMLBaseColors] = HTMLBaseColors,
    alignment_color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
    alignment_labels: type[HTMLAlignmentLabels] = HTMLAlignmentLabels,
    report_metrics: list[ReportMetric] | None = None,
    report_summary: list[ReportSummaryItem] | None = None,
    report_alignment: ReportAlignment | None = None,
    metadata: dict[str, str] | None = None,
) -> str:
    """Render an HTML report with alignment visualizations for all examples in a dataset.

    Args:
        dataset: The dataset to generate the report for.
        template: The template name to use (e.g., "report_basic"). Templates are looked up in the bewer.templates
            package.
        title: An optional title for the report.
        base_color_scheme: The base color scheme to use for the report.
        alignment_color_scheme: The color scheme to use for alignment display.
        alignment_labels: The labels and tooltips to use for alignment display.
        report_metrics: List of ReportMetric specs controlling which metrics appear. Defaults to
            DEFAULT_REPORT_METRICS.
        report_summary: List of ReportSummaryItem specs controlling the summary section. Defaults to
            DEFAULT_REPORT_SUMMARY_ITEMS.
        report_alignment: ReportAlignment spec controlling which alignment to display. Defaults to
            DEFAULT_REPORT_ALIGNMENT.
        metadata: Optional dict of key-value pairs to display in the report metadata line.

    Returns:
        The rendered HTML report string.
    """
    if report_metrics is None:
        report_metrics = DEFAULT_REPORT_METRICS
    if report_summary is None:
        report_summary = DEFAULT_REPORT_SUMMARY_ITEMS
    if report_alignment is None:
        report_alignment = DEFAULT_REPORT_ALIGNMENT
    if metadata is None:
        metadata = {}

    # Resolve metrics against the dataset
    resolved_metrics = []
    for spec in report_metrics:
        metric = dataset.metrics.get(spec.name)(**spec.metric_kwargs)
        label = spec.label if spec.label is not None else metric.long_name
        resolved_metrics.append({"name": label, "value": f"{metric.value:{spec.format}}"})

    # Resolve summary items against the dataset summary
    resolved_summary = []
    for spec in report_summary:
        value = getattr(dataset.metrics.summary(), spec.name)
        label = spec.label if spec.label is not None else spec.name
        resolved_summary.append({"name": label, "value": f"{value:{spec.format}}"})

    # Resolve alignments for each example
    resolved_alignments = []
    for example in dataset:
        alignment = example.metrics.get(report_alignment.name)(**report_alignment.metric_kwargs).alignment
        resolved_alignments.append(alignment)

    # Load and render the Jinja template
    env = Environment(loader=PackageLoader("bewer", "templates"), autoescape=True)
    env.filters["indent_tabs"] = indent_tabs
    jinja_template = env.get_template(f"{template}.html.j2")

    html = jinja_template.render(
        dataset=dataset,
        title=title,
        creation_date=datetime.now().strftime("%B %d, %Y"),
        base_color_scheme=base_color_scheme,
        metrics=resolved_metrics,
        summary=resolved_summary,
        alignments=resolved_alignments,
        alignment_color_scheme=alignment_color_scheme,
        alignment_labels=alignment_labels,
        metadata=metadata,
    )

    return html


def generate_report(
    dataset: "Dataset",
    path: str | Path | None = None,
    allow_overwrite: bool = False,
    template: str = "report_basic",
    title: str | None = None,
    base_color_scheme: type[HTMLBaseColors] = HTMLBaseColors,
    alignment_color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
    alignment_labels: type[HTMLAlignmentLabels] = HTMLAlignmentLabels,
    report_metrics: list[ReportMetric] | None = None,
    report_summary: list[ReportSummaryItem] | None = None,
    report_alignment: ReportAlignment | None = None,
    metadata: dict[str, str] | None = None,
) -> str:
    """Generate an HTML report with alignment visualizations for all examples.

    Args:
        dataset: The dataset to generate the report for.
        path: If provided, write the HTML to this file.
        allow_overwrite: If True, overwrite the file if it exists.
        template: The template name to use (e.g., "report_basic"). Templates are looked up
            in the bewer.templates package.
        title: An optional title for the report.
        base_color_scheme: The base color scheme to use for the report.
        alignment_color_scheme: The color scheme to use for alignment display.
        alignment_labels: The labels and tooltips to use for alignment display.
        report_metrics: List of ReportMetric specs controlling which metrics appear. Defaults to
            DEFAULT_REPORT_METRICS.
        report_summary: List of ReportSummaryItem specs controlling the summary section. Defaults to
            DEFAULT_REPORT_SUMMARY_ITEMS.
        report_alignment: ReportAlignment spec controlling which alignment to display. Defaults to
            DEFAULT_REPORT_ALIGNMENT.
        metadata: Optional dict of key-value pairs to display in the report metadata line.

    Returns:
        The rendered HTML report string.
    """
    html = render_report_html(
        dataset,
        template=template,
        title=title,
        base_color_scheme=base_color_scheme,
        alignment_color_scheme=alignment_color_scheme,
        alignment_labels=alignment_labels,
        report_metrics=report_metrics,
        report_summary=report_summary,
        report_alignment=report_alignment,
        metadata=metadata,
    )

    if path is not None:
        path = Path(path)
        if path.is_dir():
            raise ValueError("Provided path is a directory, expected a file path.")
        if path.exists() and not allow_overwrite:
            raise FileExistsError(f"File {path} already exists.")
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    return html
