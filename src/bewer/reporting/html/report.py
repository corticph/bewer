"""Module for generating HTML reports from datasets."""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, PackageLoader

from bewer.reporting.html.color_schemes import HTMLAlignmentColors, HTMLBaseColors, HTMLDefaultAlignmentColors

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset


def indent_tabs(text: str, width: int = 1) -> str:
    """Indent each line of the given text with a specified number of tabs."""
    padding = "\t" * width
    return "\n".join(padding + line for line in text.split("\n"))


def render_report_html(
    dataset: "Dataset",
    template: str = "report_basic",
    title: str | None = None,
    base_color_scheme: type[HTMLBaseColors] = HTMLBaseColors,
    alignment_type: str = "levenshtein",
    alignment_color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
) -> str:
    """Render an HTML report with alignment visualizations for all examples in a dataset.

    Args:
        dataset: The dataset to generate the report for.
        template: The template name to use (e.g., "report_basic"). Templates are looked up in the bewer.templates
            package.
        title: An optional title for the report.
        base_color_scheme: The base color scheme to use for the report.
        alignment_type: The alignment metric to use (default: "levenshtein").
        alignment_color_scheme: The color scheme to use for alignment display.

    Returns:
        The rendered HTML report string.
    """
    # Load and render the Jinja template
    env = Environment(loader=PackageLoader("bewer", "templates"), autoescape=True)
    env.filters["indent_tabs"] = indent_tabs
    jinja_template = env.get_template(f"{template}.html.j2")

    html = jinja_template.render(
        dataset=dataset,
        title=title,
        creation_date=datetime.now().strftime("%B %d, %Y"),
        base_color_scheme=base_color_scheme,
        metrics=[
            {"name": dataset.metrics.wer.long_name, "value": f"{dataset.metrics.wer.value:.2%}"},
            {"name": dataset.metrics.cer.long_name, "value": f"{dataset.metrics.cer.value:.2%}"},
            {"name": "Medical Term Recall", "value": f"{dataset.metrics.legacy_medical_word_accuracy.value:.2%}"},
        ],
        summary=[
            {"name": "Number of examples", "value": f"{dataset.metrics.summary.num_examples:,.0f}"},
            {"name": "Number of reference words", "value": f"{dataset.metrics.summary.num_ref_words:,.0f}"},
            {"name": "Number of reference characters", "value": f"{dataset.metrics.summary.num_ref_chars:,.0f}"},
            {"name": "Number of hypothesis words", "value": f"{dataset.metrics.summary.num_hyp_words:,.0f}"},
            {"name": "Number of hypothesis characters", "value": f"{dataset.metrics.summary.num_hyp_chars:,.0f}"},
        ],
        alignment_type=alignment_type,
        alignment_color_scheme=alignment_color_scheme,
    )

    return html


def generate_report(
    dataset: "Dataset",
    path: str | Path | None,
    allow_overwrite: bool,
    template: str = "report_basic",
    title: str | None = None,
    base_color_scheme: type[HTMLBaseColors] = HTMLBaseColors,
    alignment_type: str = "levenshtein",
    alignment_color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
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
        alignment_type: The alignment metric to use (default: "levenshtein").
        alignment_color_scheme: The color scheme to use for alignment display.

    Returns:
        The rendered HTML report string.
    """
    html = render_report_html(
        dataset,
        template=template,
        title=title,
        base_color_scheme=base_color_scheme,
        alignment_type=alignment_type,
        alignment_color_scheme=alignment_color_scheme,
    )

    if path is not None:
        path = Path(path)
        if path.is_dir():
            raise ValueError("Provided path is a directory, expected a file path.")
        if path.exists() and not allow_overwrite:
            raise FileExistsError(f"File {path} already exists.")
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(html)

    return html
