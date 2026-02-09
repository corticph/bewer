"""Tests for bewer.reporting.html.report module."""

from unittest.mock import patch

import pytest

from bewer.reporting.html.report import generate_report, indent_tabs, render_report_html


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
        result = render_report_html(sample_dataset, alignment_type="error_align")
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
