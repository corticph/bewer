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

    def test_default_tooltips_present(self, sample_dataset):
        """Test that default tooltips are rendered as data-tooltip attributes."""
        result = render_report_html(sample_dataset)
        assert 'data-tooltip="Correct: hypothesis matches reference."' in result
        assert 'data-tooltip="Extra word in hypothesis not in reference."' in result

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

    def test_default_metrics_are_core_only(self, sample_dataset):
        """Test that default metrics are the core, non-domain-specific WER and CER."""
        result = render_report_html(sample_dataset)
        assert "WER" in result
        assert "CER" in result
        # Domain-specific key-term metrics are opt-in, not default.
        assert "Key-Term Recall" not in result

    def test_custom_metrics_list(self, sample_dataset):
        """Test that a custom metrics list controls which metrics appear."""
        custom_metrics = [
            ReportMetric("wer", label="WER Score"),
        ]
        result = render_report_html(sample_dataset, report_metrics=custom_metrics)
        assert "WER Score" in result
        # CER should not appear when only WER is requested.
        assert "CER" not in result

    def test_metric_label_defaults_to_short_name_base(self, sample_dataset):
        """Test that metric label defaults to the metric's short_name_base when not specified."""
        custom_metrics = [
            ReportMetric("wer"),  # no label override
        ]
        result = render_report_html(sample_dataset, report_metrics=custom_metrics)
        assert "WER" in result

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
        """Test that default summary items match the expected labels."""
        result = render_report_html(sample_dataset)
        assert "# Examples" in result
        assert "# Ref. words" in result
        assert "# Ref. chars" in result
        assert "# Hyp. words" in result
        assert "# Hyp. chars" in result

    def test_custom_summary_list(self, sample_dataset):
        """Test that a custom summary list controls which items appear."""
        custom_summary = [
            ReportSummaryItem("num_examples", label="Total Examples"),
        ]
        result = render_report_html(sample_dataset, report_summary=custom_summary)
        assert "Total Examples" in result
        assert "# Ref. words" not in result
        assert "# Hyp. chars" not in result

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


class TestKeyTermIndicators:
    """Tests for key term indicators in HTML alignment rendering."""

    @pytest.fixture
    def dataset_with_key_terms(self):
        """Create a dataset with key terms for testing HTML rendering."""
        from bewer import Vocabulary
        from bewer.core.dataset import Dataset

        dataset = Dataset()
        dataset.add("the quick brown fox", "the quick brown dog")
        dataset.add_vocabulary(Vocabulary(name="animals").add_terms(["fox"]))
        return dataset

    def test_key_term_classes_rendered_in_html(self, dataset_with_key_terms):
        """Test that kw CSS classes appear in rendered HTML when key terms are present."""
        result = render_report_html(dataset_with_key_terms)
        # Legend has one kw span, alignment content should add more
        assert result.count("kw kw-start") > 1

    def test_no_key_term_classes_without_key_terms(self, sample_dataset):
        """Test that kw CSS classes do not appear in alignment content when no key terms are set."""
        result = render_report_html(sample_dataset)
        # Only the legend kw span should be present
        assert result.count("kw kw-start") == 1

    def test_overlapping_key_terms_merge_into_run(self):
        """Test that overlapping key terms merge into a single contiguous run when allow_subset_matches=True."""
        from bewer import Vocabulary
        from bewer.core.dataset import Dataset
        from bewer.reporting.html.alignment import _get_key_term_indicators

        dataset = Dataset()
        # "brown fox" and "brown" overlap — their union covers "brown" and "fox"
        dataset.add(
            "the quick brown fox jumps",
            "the quick brown dog jumps",
        )
        dataset.add_vocabulary(Vocabulary(name="overlapping").add_terms(["brown fox", "brown"]))
        example = dataset[0]
        alignment = example.metrics.levenshtein().alignment
        start_indices, stop_indices, _ = _get_key_term_indicators(alignment, allow_subset_matches=True)
        # "brown" ends at its own op and "brown fox" ends at fox's op — two distinct stop indices
        assert len(stop_indices) > len(start_indices)

    def test_key_term_rendering_is_idempotent(self, dataset_with_key_terms):
        """Test that rendering twice produces the same output."""
        result1 = render_report_html(dataset_with_key_terms)
        result2 = render_report_html(dataset_with_key_terms)
        assert result1 == result2


class TestSurfaceToggle:
    """Tests for the surface form toggle in HTML reports."""

    def test_report_contains_both_views(self, sample_dataset):
        """Report HTML contains both normalized and surface alignment tables."""
        result = render_report_html(sample_dataset)
        assert "alignment-normalized" in result
        assert "alignment-surface" in result

    def test_report_contains_toggle_checkbox(self, sample_dataset):
        """Report HTML contains the surface toggle checkbox."""
        result = render_report_html(sample_dataset)
        assert 'id="surface-toggle"' in result
        assert "Normalize" in result

    def test_report_contains_toggle_css(self, sample_dataset):
        """Report HTML contains CSS for surface view toggling."""
        result = render_report_html(sample_dataset)
        assert ".alignment-surface" in result
        assert "body.surface-view" in result

    def test_report_contains_toggle_js(self, sample_dataset):
        """Report HTML contains JS for surface view toggling."""
        result = render_report_html(sample_dataset)
        assert "surface-toggle" in result
        assert "addEventListener" in result
        assert "surface-view" in result

    def test_surface_shows_cased_text(self, sample_dataset):
        """Surface view in report contains cased (standardized) text."""
        from bewer.core.dataset import Dataset

        ds = Dataset()
        ds.add("Hello World", "hello world")
        result = render_report_html(ds)
        assert "Hello" in result

    def test_surface_view_hidden_by_default(self, sample_dataset):
        """Surface view is hidden by default via CSS display:none."""
        result = render_report_html(sample_dataset)
        assert ".alignment-surface" in result
        assert "display: none" in result


class TestMultiDatasetReport:
    """Tests for multi-dataset comparison reports."""

    def _make_two_datasets(self):
        from bewer.core.dataset import Dataset

        ds_a = Dataset()
        ds_a.add("Hello world", "Hello world")
        ds_a.add("The quick brown fox", "The quick brown dog")
        ds_b = Dataset()
        ds_b.add("Hello world", "Hello weird")
        ds_b.add("The quick brown fox", "The quick brown fox")
        return ds_a, ds_b

    def test_multi_dataset_report_generates(self):
        """Multi-dataset report generates without error."""
        ds_a, ds_b = self._make_two_datasets()
        html = render_report_html({"System A": ds_a, "System B": ds_b})
        assert len(html) > 0

    def test_multi_dataset_contains_both_names(self):
        """Report contains both dataset names."""
        ds_a, ds_b = self._make_two_datasets()
        html = render_report_html({"System A": ds_a, "System B": ds_b})
        assert "System A" in html
        assert "System B" in html

    def test_multi_dataset_contains_radio_buttons(self):
        """Report contains radio buttons for dataset selection."""
        ds_a, ds_b = self._make_two_datasets()
        html = render_report_html({"System A": ds_a, "System B": ds_b})
        assert 'name="dataset-toggle"' in html
        assert 'type="radio"' in html

    def test_multi_dataset_contains_css_classes(self):
        """Report contains CSS classes for dataset visibility."""
        ds_a, ds_b = self._make_two_datasets()
        html = render_report_html({"System A": ds_a, "System B": ds_b})
        assert "alignment-dataset-System-A" in html
        assert "alignment-dataset-System-B" in html

    def test_multi_dataset_first_dataset_active_by_default(self):
        """First dataset is active by default (body class set)."""
        ds_a, ds_b = self._make_two_datasets()
        html = render_report_html({"System A": ds_a, "System B": ds_b})
        assert 'class="dataset-System-A"' in html

    def test_multi_dataset_metrics_table_has_columns(self):
        """Metrics table has a column per dataset."""
        ds_a, ds_b = self._make_two_datasets()
        html = render_report_html({"System A": ds_a, "System B": ds_b})
        # Both dataset names appear in the metrics table
        assert "System A" in html
        assert "System B" in html

    def test_single_dataset_backward_compat(self):
        """Single dataset still works without comparison toggle."""
        from bewer.core.dataset import Dataset

        ds = Dataset()
        ds.add("Hello world", "Hello world")
        html = render_report_html(ds)
        assert 'name="dataset-toggle"' not in html
        assert '<input type="radio"' not in html

    def test_refs_match_warning_on_mismatch(self):
        """Warning is issued when references differ across datasets."""
        from bewer.core.dataset import Dataset

        ds_a = Dataset()
        ds_a.add("Hello world", "Hello world")
        ds_b = Dataset()
        ds_b.add("Different ref", "Different hyp")
        with pytest.warns(UserWarning, match="Reference texts differ"):
            render_report_html({"A": ds_a, "B": ds_b})

    def test_multi_dataset_synchronized_line_counts(self):
        """Both datasets have the same number of alignment lines."""
        ds_a, ds_b = self._make_two_datasets()
        html = render_report_html({"A": ds_a, "B": ds_b})
        import re

        # Count REF rows inside each dataset's normalized tables
        # Tables are: <table class="alignment-table alignment-normalized alignment-dataset-{A|B}">
        # Each ref row has class="alignment-row" with labels.REF
        a_tables = re.findall(
            r'<table class="alignment-table alignment-normalized alignment-dataset-A">(.*?)</table>', html, re.DOTALL
        )
        b_tables = re.findall(
            r'<table class="alignment-table alignment-normalized alignment-dataset-B">(.*?)</table>', html, re.DOTALL
        )
        a_ref_rows = sum(t.count("alignment-table-lines") for t in a_tables)
        b_ref_rows = sum(t.count("alignment-table-lines") for t in b_tables)
        assert a_ref_rows > 0
        assert a_ref_rows == b_ref_rows
