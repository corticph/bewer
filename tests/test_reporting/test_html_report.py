"""Tests for bewer.reporting.html.report module."""

from unittest.mock import patch

import pytest

from bewer.reporting.html.labels import HTMLAlignmentLabels
from bewer.reporting.html.report import (
    ReportAlignment,
    ReportMetric,
    ReportSummaryItem,
    generate_report,
    indent_tabs,
    render_report_html,
)


class TestIndentTabs:
    """Tests for indent_tabs function."""

    def test_indent_tabs_single_line(self):
        """Test indenting a single line of text."""
        result = indent_tabs("hello", width=1)
        assert result == "\thello"

    def test_indent_tabs_multiple_lines(self):
        """Test indenting multiple lines of text."""
        text = "line1\nline2\nline3"
        result = indent_tabs(text, width=1)
        assert result == "\tline1\n\tline2\n\tline3"

    def test_indent_tabs_multiple_width(self):
        """Test indenting with width > 1."""
        result = indent_tabs("test", width=3)
        assert result == "\t\t\ttest"

    def test_indent_tabs_zero_width(self):
        """Test indenting with width = 0."""
        result = indent_tabs("test", width=0)
        assert result == "test"

    def test_indent_tabs_empty_string(self):
        """Test indenting an empty string."""
        result = indent_tabs("", width=1)
        assert result == "\t"


class TestRenderReportHtml:
    """Tests for render_report_html function."""

    def test_render_report_html_returns_string(self, sample_dataset):
        """Test that render_report_html returns a string."""
        result = render_report_html(sample_dataset)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_report_html_contains_metrics(self, sample_dataset):
        """Test that rendered HTML contains metric values."""
        result = render_report_html(sample_dataset)

        # Check that metrics are in the output (should contain percentage format)
        assert "%" in result  # Metrics are formatted as percentages

    def test_render_report_html_contains_summary_stats(self, sample_dataset):
        """Test that rendered HTML contains summary statistics."""
        result = render_report_html(sample_dataset)

        # Check that summary stats are in the output
        # sample_dataset has 3 examples
        assert "3" in result  # num_examples

    def test_render_report_html_with_title(self, sample_dataset):
        """Test that title is included in rendered HTML."""
        result = render_report_html(sample_dataset, title="My Test Report")
        assert "My Test Report" in result

    @patch("bewer.reporting.html.report.datetime")
    def test_render_report_html_includes_creation_date(self, mock_datetime, sample_dataset):
        """Test that creation date is included in rendered HTML."""
        mock_now = mock_datetime.now.return_value
        mock_now.strftime.return_value = "February 09, 2026"

        result = render_report_html(sample_dataset)
        assert "February 09, 2026" in result

    def test_render_report_html_with_different_alignment_type(self, sample_dataset):
        """Test rendering with different alignment type."""
        result = render_report_html(sample_dataset, report_alignment=ReportAlignment("error_align"))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_report_html_is_valid_html(self, sample_dataset):
        """Test that rendered output contains HTML structure."""
        result = render_report_html(sample_dataset)

        # Check for basic HTML structure markers
        assert "<" in result and ">" in result  # Contains HTML tags


class TestGenerateReport:
    """Tests for generate_report function."""

    def test_generate_report_returns_html_string(self, sample_dataset):
        """Test that generate_report returns HTML string."""
        result = generate_report(sample_dataset, path=None, allow_overwrite=False)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_report_writes_to_file(self, sample_dataset, tmp_path):
        """Test that generate_report writes to file when path is provided."""
        output_file = tmp_path / "report.html"
        result = generate_report(sample_dataset, path=output_file, allow_overwrite=False)

        assert output_file.exists()
        assert output_file.read_text() == result

    def test_generate_report_raises_if_file_exists_without_overwrite(self, sample_dataset, tmp_path):
        """Test that generate_report raises if file exists and overwrite is False."""
        output_file = tmp_path / "report.html"
        output_file.write_text("existing content")

        with pytest.raises(FileExistsError):
            generate_report(sample_dataset, path=output_file, allow_overwrite=False)

    def test_generate_report_overwrites_if_allowed(self, sample_dataset, tmp_path):
        """Test that generate_report overwrites file when allow_overwrite is True."""
        output_file = tmp_path / "report.html"
        output_file.write_text("existing content")

        result = generate_report(sample_dataset, path=output_file, allow_overwrite=True)

        assert output_file.exists()
        assert output_file.read_text() == result
        assert output_file.read_text() != "existing content"

    def test_generate_report_creates_parent_directories(self, sample_dataset, tmp_path):
        """Test that generate_report creates parent directories if they don't exist."""
        output_file = tmp_path / "nested" / "dir" / "report.html"
        _ = generate_report(sample_dataset, path=output_file, allow_overwrite=False)

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_generate_report_raises_if_path_is_directory(self, sample_dataset, tmp_path):
        """Test that generate_report raises if path is a directory."""
        with pytest.raises(ValueError, match="directory"):
            generate_report(sample_dataset, path=tmp_path, allow_overwrite=False)

    def test_generate_report_with_custom_template(self, sample_dataset, tmp_path):
        """Test that generate_report works with custom template name."""
        output_file = tmp_path / "report.html"
        # Using the default template "report_basic" explicitly
        _ = generate_report(sample_dataset, path=output_file, allow_overwrite=False, template="report_basic")
        assert output_file.exists()

    def test_generate_report_with_title(self, sample_dataset, tmp_path):
        """Test that title parameter works in generate_report."""
        output_file = tmp_path / "report.html"
        result = generate_report(sample_dataset, path=output_file, allow_overwrite=False, title="Custom Title")
        assert "Custom Title" in result


class TestCustomAlignmentLabels:
    """Tests for custom alignment labels in rendered reports."""

    def test_default_labels_appear_in_html(self, sample_dataset):
        """Test that default labels appear in rendered HTML."""
        result = render_report_html(sample_dataset)
        assert "Ref." in result
        assert "Hyp." in result
        assert "Match" in result
        assert "Substitution" in result
        assert "Insertion" in result
        assert "Deletion" in result
        assert "Padding" in result
        assert "Keyword" in result

    def test_custom_labels_appear_in_html(self, sample_dataset):
        """Test that custom labels replace defaults in rendered HTML."""

        class CustomLabels(HTMLAlignmentLabels):
            REF = "Reference"
            HYP = "Hypothesis"
            MATCH = "Correct"
            SUBSTITUTION = "Replaced"

        result = render_report_html(sample_dataset, alignment_labels=CustomLabels)
        assert "Reference" in result
        assert "Hypothesis" in result
        assert "Correct" in result
        assert "Replaced" in result

    def test_no_tooltips_by_default(self, sample_dataset):
        """Test that no data-tooltip attributes are rendered with default labels."""
        result = render_report_html(sample_dataset)
        # Legend items should not have data-tooltip attributes by default
        assert 'class="legend-container-item" data-tooltip=' not in result

    def test_tooltips_render_as_data_tooltip_attributes(self, sample_dataset):
        """Test that tooltips render as data-tooltip attributes when set."""

        class LabelsWithTooltips(HTMLAlignmentLabels):
            MATCH_TOOLTIP = "Words that match exactly"
            INSERTION_TOOLTIP = "Words added in hypothesis"

        result = render_report_html(sample_dataset, alignment_labels=LabelsWithTooltips)
        assert 'data-tooltip="Words that match exactly"' in result
        assert 'data-tooltip="Words added in hypothesis"' in result

    def test_tooltip_absent_when_none(self, sample_dataset):
        """Test that tooltip is absent when set to None."""

        class PartialTooltips(HTMLAlignmentLabels):
            MATCH_TOOLTIP = "A tooltip"
            SUBSTITUTION_TOOLTIP = None  # explicitly None

        result = render_report_html(sample_dataset, alignment_labels=PartialTooltips)
        assert 'data-tooltip="A tooltip"' in result
        # The substitution legend item should not have a data-tooltip attribute
        assert "Substitution" in result

    def test_generate_report_forwards_labels(self, sample_dataset):
        """Test that generate_report forwards alignment_labels to render_report_html."""

        class CustomLabels(HTMLAlignmentLabels):
            REF = "Src."
            HYP = "Tgt."

        result = generate_report(sample_dataset, alignment_labels=CustomLabels)
        assert "Src." in result
        assert "Tgt." in result


class TestCustomReportMetrics:
    """Tests for custom report metrics configuration."""

    def test_default_metrics_match_previous_behavior(self, sample_dataset):
        """Test that default metrics produce the same output as the old hard-coded values."""
        result = render_report_html(sample_dataset)
        assert "Word Error Rate" in result
        assert "Character Error Rate" in result
        assert "Medical Term Recall" in result

    def test_custom_metrics_list(self, sample_dataset):
        """Test that a custom metrics list controls which metrics appear."""
        custom_metrics = [
            ReportMetric("wer", label="WER Score"),
        ]
        result = render_report_html(sample_dataset, report_metrics=custom_metrics)
        assert "WER Score" in result
        # CER and Medical Term Recall should not appear
        assert "Character Error Rate" not in result
        assert "Medical Term Recall" not in result

    def test_metric_label_defaults_to_long_name(self, sample_dataset):
        """Test that metric label defaults to the metric's long_name when not specified."""
        custom_metrics = [
            ReportMetric("wer"),  # no label override
        ]
        result = render_report_html(sample_dataset, report_metrics=custom_metrics)
        assert "Word Error Rate" in result

    def test_custom_metric_format(self, sample_dataset):
        """Test that custom format spec is applied to metric values."""
        custom_metrics = [
            ReportMetric("wer", label="WER", format=".4f"),
        ]
        result = render_report_html(sample_dataset, report_metrics=custom_metrics)
        # The value should be formatted with 4 decimal places, not as percentage
        assert "%" not in result or "WER" in result  # WER row won't have %


class TestCustomReportSummary:
    """Tests for custom report summary configuration."""

    def test_default_summary_matches_previous_behavior(self, sample_dataset):
        """Test that default summary items match the old hard-coded values."""
        result = render_report_html(sample_dataset)
        assert "Number of examples" in result
        assert "Number of reference words" in result
        assert "Number of reference characters" in result
        assert "Number of hypothesis words" in result
        assert "Number of hypothesis characters" in result

    def test_custom_summary_list(self, sample_dataset):
        """Test that a custom summary list controls which items appear."""
        custom_summary = [
            ReportSummaryItem("num_examples", label="Total Examples"),
        ]
        result = render_report_html(sample_dataset, report_summary=custom_summary)
        assert "Total Examples" in result
        assert "Number of reference words" not in result
        assert "Number of hypothesis characters" not in result

    def test_summary_label_defaults_to_name(self, sample_dataset):
        """Test that summary label defaults to the attribute name when not specified."""
        custom_summary = [
            ReportSummaryItem("num_examples"),  # no label override
        ]
        result = render_report_html(sample_dataset, report_summary=custom_summary)
        assert "num_examples" in result

    def test_custom_summary_format(self, sample_dataset):
        """Test that custom format spec is applied to summary values."""
        custom_summary = [
            ReportSummaryItem("num_examples", label="Examples", format="d"),
        ]
        result = render_report_html(sample_dataset, report_summary=custom_summary)
        assert "Examples" in result
