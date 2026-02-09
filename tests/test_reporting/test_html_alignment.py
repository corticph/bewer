"""Tests for bewer.reporting.html.alignment module."""

from html import escape
from unittest.mock import Mock

import pytest

from bewer.alignment import Alignment, Op, OpType
from bewer.reporting.html.alignment import (
    format_alignment_op_html,
    format_delete_op_html,
    format_insert_op_html,
    format_match_op_html,
    format_substitute_op_html,
    generate_alignment_html_lines,
    get_html_padding,
)
from bewer.reporting.html.color_schemes import HTMLDefaultAlignmentColors


class TestGetHtmlPadding:
    """Tests for get_html_padding function."""

    def test_get_html_padding_returns_string(self):
        """Test that get_html_padding returns a string."""
        result = get_html_padding(5)
        assert isinstance(result, str)

    def test_get_html_padding_correct_length(self):
        """Test that padding has correct number of nbsp entities."""
        result = get_html_padding(5)
        assert result.count("&nbsp;") == 5

    def test_get_html_padding_zero_length(self):
        """Test padding with zero length."""
        result = get_html_padding(0)
        assert result.count("&nbsp;") == 0

    def test_get_html_padding_contains_color(self):
        """Test that padding contains background-color style."""
        result = get_html_padding(3)
        assert "background-color" in result
        assert HTMLDefaultAlignmentColors.PAD in result

    def test_get_html_padding_is_wrapped_in_span(self):
        """Test that padding is wrapped in a span element."""
        result = get_html_padding(3)
        assert result.startswith("<span")
        assert result.endswith("</span>")

    def test_get_html_padding_uses_custom_color_scheme(self):
        """Test that padding uses custom color scheme."""

        class CustomColorScheme(HTMLDefaultAlignmentColors):
            PAD = "#123456"

        result = get_html_padding(2, color_scheme=CustomColorScheme)
        assert "#123456" in result


class TestFormatMatchOpHtml:
    """Tests for format_match_op_html function."""

    def test_format_match_op_html_returns_tuple(self):
        """Test that format_match_op_html returns a tuple of (ref_str, hyp_str, length)."""
        op = Op(type=OpType.MATCH, ref="hello", hyp="hello")
        ref_str, hyp_str, length = format_match_op_html(op)
        assert isinstance(ref_str, str)
        assert isinstance(hyp_str, str)
        assert isinstance(length, int)

    def test_format_match_op_html_equal_length(self):
        """Test match op with equal length ref and hyp."""
        op = Op(type=OpType.MATCH, ref="test", hyp="test")
        ref_str, hyp_str, length = format_match_op_html(op)
        assert length == 4
        assert escape("test") in ref_str
        assert escape("test") in hyp_str

    def test_format_match_op_html_ref_shorter(self):
        """Test match op where ref is shorter."""
        op = Op(type=OpType.MATCH, ref="hi", hyp="HI!")
        ref_str, hyp_str, length = format_match_op_html(op)
        assert length == 3
        assert "&nbsp;" in ref_str  # Contains padding

    def test_format_match_op_html_hyp_shorter(self):
        """Test match op where hyp is shorter."""
        op = Op(type=OpType.MATCH, ref="HELLO", hyp="hi")
        ref_str, hyp_str, length = format_match_op_html(op)
        assert length == 5
        assert "&nbsp;" in hyp_str  # Contains padding

    def test_format_match_op_html_uses_match_color(self):
        """Test that match op uses MATCH color."""
        op = Op(type=OpType.MATCH, ref="test", hyp="test")
        ref_str, hyp_str, _ = format_match_op_html(op)
        assert HTMLDefaultAlignmentColors.MATCH in ref_str
        assert HTMLDefaultAlignmentColors.MATCH in hyp_str

    def test_format_match_op_html_escapes_html_characters(self):
        """Test that HTML characters are properly escaped."""
        op = Op(type=OpType.MATCH, ref="<div>", hyp="<div>")
        ref_str, hyp_str, _ = format_match_op_html(op)
        assert "&lt;div&gt;" in ref_str
        assert "&lt;div&gt;" in hyp_str
        assert "<div>" not in ref_str  # Should not contain unescaped HTML


class TestFormatSubstituteOpHtml:
    """Tests for format_substitute_op_html function."""

    def test_format_substitute_op_html_returns_tuple(self):
        """Test that format_substitute_op_html returns a tuple."""
        op = Op(type=OpType.SUBSTITUTE, ref="cat", hyp="dog")
        ref_str, hyp_str, length = format_substitute_op_html(op)
        assert isinstance(ref_str, str)
        assert isinstance(hyp_str, str)
        assert isinstance(length, int)

    def test_format_substitute_op_html_equal_length(self):
        """Test substitute op with equal length tokens."""
        op = Op(type=OpType.SUBSTITUTE, ref="cat", hyp="dog")
        ref_str, hyp_str, length = format_substitute_op_html(op)
        assert length == 3
        assert escape("cat") in ref_str
        assert escape("dog") in hyp_str

    def test_format_substitute_op_html_ref_longer(self):
        """Test substitute op where ref is longer."""
        op = Op(type=OpType.SUBSTITUTE, ref="hello", hyp="hi")
        ref_str, hyp_str, length = format_substitute_op_html(op)
        assert length == 5
        assert "&nbsp;" in hyp_str  # Contains padding

    def test_format_substitute_op_html_hyp_longer(self):
        """Test substitute op where hyp is longer."""
        op = Op(type=OpType.SUBSTITUTE, ref="hi", hyp="hello")
        ref_str, hyp_str, length = format_substitute_op_html(op)
        assert length == 5
        assert "&nbsp;" in ref_str  # Contains padding

    def test_format_substitute_op_html_uses_sub_color(self):
        """Test that substitute op uses SUB color."""
        op = Op(type=OpType.SUBSTITUTE, ref="cat", hyp="dog")
        ref_str, hyp_str, _ = format_substitute_op_html(op)
        assert HTMLDefaultAlignmentColors.SUB in ref_str
        assert HTMLDefaultAlignmentColors.SUB in hyp_str


class TestFormatInsertOpHtml:
    """Tests for format_insert_op_html function."""

    def test_format_insert_op_html_returns_tuple(self):
        """Test that format_insert_op_html returns a tuple."""
        op = Op(type=OpType.INSERT, ref=None, hyp="inserted")
        ref_str, hyp_str, length = format_insert_op_html(op)
        assert isinstance(ref_str, str)
        assert isinstance(hyp_str, str)
        assert isinstance(length, int)

    def test_format_insert_op_html_ref_is_padding(self):
        """Test that ref is padding for insert op."""
        op = Op(type=OpType.INSERT, ref=None, hyp="new")
        ref_str, hyp_str, length = format_insert_op_html(op)
        assert length == 3
        assert "&nbsp;" in ref_str  # Ref should be padding
        assert escape("new") in hyp_str

    def test_format_insert_op_html_uses_insert_color(self):
        """Test that insert op uses INS color."""
        op = Op(type=OpType.INSERT, ref=None, hyp="new")
        ref_str, hyp_str, _ = format_insert_op_html(op)
        assert HTMLDefaultAlignmentColors.INS in hyp_str


class TestFormatDeleteOpHtml:
    """Tests for format_delete_op_html function."""

    def test_format_delete_op_html_returns_tuple(self):
        """Test that format_delete_op_html returns a tuple."""
        op = Op(type=OpType.DELETE, ref="deleted", hyp=None)
        ref_str, hyp_str, length = format_delete_op_html(op)
        assert isinstance(ref_str, str)
        assert isinstance(hyp_str, str)
        assert isinstance(length, int)

    def test_format_delete_op_html_hyp_is_padding(self):
        """Test that hyp is padding for delete op."""
        op = Op(type=OpType.DELETE, ref="old", hyp=None)
        ref_str, hyp_str, length = format_delete_op_html(op)
        assert length == 3
        assert escape("old") in ref_str
        assert "&nbsp;" in hyp_str  # Hyp should be padding

    def test_format_delete_op_html_uses_delete_color(self):
        """Test that delete op uses DEL color."""
        op = Op(type=OpType.DELETE, ref="old", hyp=None)
        ref_str, hyp_str, _ = format_delete_op_html(op)
        assert HTMLDefaultAlignmentColors.DEL in ref_str


class TestFormatAlignmentOpHtml:
    """Tests for format_alignment_op_html dispatch function."""

    def test_format_alignment_op_html_match(self):
        """Test format_alignment_op_html dispatches to match handler."""
        op = Op(type=OpType.MATCH, ref="test", hyp="test")
        ref_str, hyp_str, _ = format_alignment_op_html(op)
        assert escape("test") in ref_str
        assert escape("test") in hyp_str

    def test_format_alignment_op_html_substitute(self):
        """Test format_alignment_op_html dispatches to substitute handler."""
        op = Op(type=OpType.SUBSTITUTE, ref="dog", hyp="doggy")
        ref_str, hyp_str, length = format_alignment_op_html(op)
        assert length == 5
        assert escape("dog") in ref_str
        assert escape("doggy") in hyp_str

    def test_format_alignment_op_html_insert(self):
        """Test format_alignment_op_html dispatches to insert handler."""
        op = Op(type=OpType.INSERT, ref=None, hyp="new")
        ref_str, hyp_str, _ = format_alignment_op_html(op)
        assert "&nbsp;" in ref_str
        assert escape("new") in hyp_str

    def test_format_alignment_op_html_delete(self):
        """Test format_alignment_op_html dispatches to delete handler."""
        op = Op(type=OpType.DELETE, ref="old", hyp=None)
        ref_str, hyp_str, _ = format_alignment_op_html(op)
        assert escape("old") in ref_str
        assert "&nbsp;" in hyp_str

    def test_format_alignment_op_html_invalid_type_raises(self):
        """Test that invalid operation type raises ValueError."""
        # Create an op with invalid type (not one of the OpType values)
        op = Op(type=OpType.MATCH, ref="test", hyp="test")
        op.type = "INVALID"  # Override with invalid type

        with pytest.raises(ValueError, match="Unknown operation type"):
            format_alignment_op_html(op)


class TestGenerateAlignmentHtmlLines:
    """Tests for generate_alignment_html_lines function."""

    def _create_mock_example(self):
        """Helper to create a mock example with keywords attribute."""
        mock_example = Mock()
        mock_example.keywords = None
        return mock_example

    def test_generate_alignment_html_lines_returns_list_of_tuples(self):
        """Test that generate_alignment_html_lines returns a list of tuples."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops, src_example=self._create_mock_example())
        result = generate_alignment_html_lines(alignment)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)

    def test_generate_alignment_html_lines_simple_alignment(self):
        """Test with a simple alignment."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello"),
            Op(type=OpType.MATCH, ref="world", hyp="world"),
        ]
        alignment = Alignment(ops, src_example=self._create_mock_example())
        result = generate_alignment_html_lines(alignment)
        assert len(result) >= 1
        ref_line, hyp_line = result[0]
        assert "hello" in ref_line
        assert "world" in ref_line

    def test_generate_alignment_html_lines_all_op_types(self):
        """Test with all operation types."""
        ops = [
            Op(type=OpType.MATCH, ref="the", hyp="the"),
            Op(type=OpType.SUBSTITUTE, ref="cat", hyp="dog"),
            Op(type=OpType.INSERT, ref=None, hyp="big"),
            Op(type=OpType.DELETE, ref="sat", hyp=None),
        ]
        alignment = Alignment(ops, src_example=self._create_mock_example())
        result = generate_alignment_html_lines(alignment)
        assert len(result) >= 1
        ref_line, hyp_line = result[0]

        # Check that all operations are present
        assert "the" in ref_line
        assert "cat" in ref_line
        assert "sat" in ref_line
        assert "the" in hyp_line
        assert "dog" in hyp_line
        assert "big" in hyp_line

    def test_generate_alignment_html_lines_wraps_long_lines(self):
        """Test that long alignments wrap across multiple lines."""
        # Create a long alignment that should wrap
        ops = [Op(type=OpType.MATCH, ref=f"word{i}", hyp=f"word{i}") for i in range(20)]
        alignment = Alignment(ops, src_example=self._create_mock_example())
        result = generate_alignment_html_lines(alignment, max_line_length=30)

        # Should have multiple lines
        assert len(result) > 1

    def test_generate_alignment_html_lines_empty_alignment(self):
        """Test with empty alignment."""
        alignment = Alignment(src_example=self._create_mock_example())
        result = generate_alignment_html_lines(alignment)
        assert len(result) == 1  # Should have at least one line even if empty

    def test_generate_alignment_html_lines_handles_right_partial(self):
        """Test that right partial tokens are handled correctly."""
        ops = [
            Op(type=OpType.MATCH, ref="test", hyp="test", hyp_right_partial=True),
            Op(type=OpType.MATCH, ref="ing", hyp="ing"),
        ]
        alignment = Alignment(ops, src_example=self._create_mock_example())
        result = generate_alignment_html_lines(alignment)
        assert len(result) >= 1

        # Right partial should have different spacing
        _, hyp_line = result[0]
        assert "test" in hyp_line
        assert "ing" in hyp_line

    def test_generate_alignment_html_lines_uses_color_scheme(self):
        """Test that custom color scheme is used."""

        class CustomColorScheme(HTMLDefaultAlignmentColors):
            MATCH = "#custom123"

        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops, src_example=self._create_mock_example())
        result = generate_alignment_html_lines(alignment, color_scheme=CustomColorScheme)

        ref_line, _ = result[0]
        assert "#custom123" in ref_line

    def test_generate_alignment_html_lines_escapes_html(self):
        """Test that HTML characters are properly escaped."""
        ops = [Op(type=OpType.MATCH, ref="<script>", hyp="<script>")]
        alignment = Alignment(ops, src_example=self._create_mock_example())
        result = generate_alignment_html_lines(alignment)

        ref_line, hyp_line = result[0]
        assert "&lt;script&gt;" in ref_line
        assert "&lt;script&gt;" in hyp_line
        assert "<script>" not in ref_line
