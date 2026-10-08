"""Module for generating HTML reports from datasets."""

import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union

from jinja2 import Environment, PackageLoader

from bewer.reporting.html.color_schemes import HTMLAlignmentColors, HTMLBaseColors, HTMLDefaultAlignmentColors
from bewer.reporting.html.labels import HTMLAlignmentLabels

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset

__all__ = [
    "generate_report",
    "render_report_html",
    "ReportMetric",
    "ReportAlignment",
    "ReportSummaryItem",
    "DEFAULT_REPORT_METRICS",
    "DEFAULT_REPORT_SUMMARY_ITEMS",
    "DEFAULT_REPORT_ALIGNMENT",
]


class ReportMetric:
    """Specification for a metric to include in the report."""

    def __init__(self, name: str, label: str | None = None, format: str = ".2%", **metric_kwargs):
        self.name = name  # metric registry name (e.g. "wer")
        self.label = label  # display label override (default: metric.short_name_base)
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

# Only core, non-domain-specific metrics are included by default. Key-term metrics
# (e.g. "ktr", "rktr") require a dataset-specific ``vocab`` and are opt-in via the
# ``report_metrics`` argument, e.g. ReportMetric("ktr", label="Key-Term Recall", vocab="...").
DEFAULT_REPORT_METRICS = [
    ReportMetric("wer"),
    ReportMetric("cer"),
]

DEFAULT_REPORT_SUMMARY_ITEMS = [
    ReportSummaryItem("num_examples", label="# Examples"),
    ReportSummaryItem("num_ref_words", label="# Ref. words"),
    ReportSummaryItem("num_ref_chars", label="# Ref. chars"),
    ReportSummaryItem("num_hyp_words", label="# Hyp. words"),
    ReportSummaryItem("num_hyp_chars", label="# Hyp. chars"),
]


def indent_tabs(text: str, width: int = 1) -> str:
    """Indent each line of the given text with a specified number of tabs."""
    padding = "\t" * width
    return "\n".join(padding + line for line in text.split("\n"))


def _sanitize_css_class(name: str) -> str:
    """Sanitize a dataset name for use as a CSS class."""
    return re.sub(r"[^a-zA-Z0-9_-]", "-", name)


def _refs_match(datasets: list["Dataset"]) -> bool:
    """Check if all datasets have identical reference texts."""
    if len(datasets) <= 1:
        return True
    first_refs = [ex.ref.raw for ex in datasets[0]]
    for ds in datasets[1:]:
        if len(ds) != len(first_refs):
            return False
        for i, ex in enumerate(ds):
            if ex.ref.raw != first_refs[i]:
                return False
    return True


def render_report_html(
    dataset: Union["Dataset", dict[str, "Dataset"]],
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
        dataset: A dataset or a dict of named datasets to compare. When a dict is given,
            the keys are used as display names and a comparison toggle is added to the
            Options box.
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

    if isinstance(dataset, dict):
        datasets = dataset
        multi_dataset = len(datasets) > 1
    else:
        datasets = {"Default": dataset}
        multi_dataset = False

    dataset_names = list(datasets.keys())
    dataset_list = [datasets[name] for name in dataset_names]

    refs_match = _refs_match(dataset_list) if multi_dataset else False
    if multi_dataset and not refs_match:
        warnings.warn(
            "Reference texts differ across datasets. Line breaks will not be synchronized across datasets.",
            stacklevel=2,
        )

    n_examples = max(len(ds) for ds in dataset_list) if dataset_list else 0

    if multi_dataset:
        resolved_metrics = []
        for spec in report_metrics:
            values = []
            label = spec.label
            for ds in dataset_list:
                metric = ds.metrics.get(spec.name)(**spec.metric_kwargs)
                if label is None:
                    label = metric.short_name_base
                values.append(f"{metric.value:{spec.format}}")
            resolved_metrics.append({"name": label, "values": values})

        resolved_summary = []
        for spec in report_summary:
            values = []
            for ds in dataset_list:
                value = getattr(ds.metrics.summary(), spec.name)
                values.append(f"{value:{spec.format}}")
            label = spec.label if spec.label is not None else spec.name
            resolved_summary.append({"name": label, "values": values})
    else:
        ds = dataset_list[0]
        resolved_metrics = []
        for spec in report_metrics:
            metric = ds.metrics.get(spec.name)(**spec.metric_kwargs)
            label = spec.label if spec.label is not None else metric.short_name_base
            resolved_metrics.append({"name": label, "value": f"{metric.value:{spec.format}}"})

        resolved_summary = []
        for spec in report_summary:
            value = getattr(ds.metrics.summary(), spec.name)
            label = spec.label if spec.label is not None else spec.name
            resolved_summary.append({"name": label, "value": f"{value:{spec.format}}"})

    if multi_dataset:
        resolved_alignments = []
        for ex_idx in range(n_examples):
            ex_alignments = []
            for ds in dataset_list:
                if ex_idx < len(ds):
                    alignment = (
                        ds[ex_idx].metrics.get(report_alignment.name)(**report_alignment.metric_kwargs).alignment
                    )
                    ex_alignments.append(alignment)
                else:
                    ex_alignments.append(None)
            resolved_alignments.append(ex_alignments)
    else:
        ds = dataset_list[0]
        resolved_alignments = []
        for example in ds:
            alignment = example.metrics.get(report_alignment.name)(**report_alignment.metric_kwargs).alignment
            resolved_alignments.append(alignment)

    env = Environment(loader=PackageLoader("bewer", "templates"), autoescape=True)
    env.filters["indent_tabs"] = indent_tabs
    jinja_template = env.get_template(f"{template}.html.j2")

    html = jinja_template.render(
        dataset=dataset_list[0],
        datasets=datasets,
        dataset_names=dataset_names,
        dataset_css_names=[str(i) for i in range(len(dataset_names))],
        multi_dataset=multi_dataset,
        refs_match=refs_match,
        n_examples=n_examples,
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
    dataset: Union["Dataset", dict[str, "Dataset"]],
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
        dataset: A dataset or a dict of named datasets to compare. When a dict is given,
            the keys are used as display names and a comparison toggle is added to the
            Options box.
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
        report_summary: List ofReportSummaryItem specs controlling the summary section. Defaults to
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
