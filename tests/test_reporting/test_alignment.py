"""Tests for bewer.reporting.alignment module."""

from unittest.mock import patch

import pytest
from rich.text import Text

from bewer.alignment.op import Alignment, Op
from bewer.alignment.op_type import OpType
from bewer.reporting.python.alignment import (
    ColorScheme,
    DefaultColorScheme,
    display_basic_aligned,
    format_alignment_op,
    format_delete_op,
    format_insert_op,
    format_match_op,
    format_substitute_op,
    get_line_prefixes,
    get_padding,
)


class TestColorScheme:
    """Tests for ColorScheme classes."""

    def test_default_color_scheme_has_required_attributes(self):
        """Test DefaultColorScheme has all required color attributes."""
        assert hasattr(DefaultColorScheme, "PAD")
        assert hasattr(DefaultColorScheme, "DEL")
        assert hasattr(DefaultColorScheme, "INS")
        assert hasattr(DefaultColorScheme, "SUB")
        assert hasattr(DefaultColorScheme, "MATCH")

    def test_default_color_scheme_values_are_strings(self):
        """Test that color scheme values are strings."""
        assert isinstance(DefaultColorScheme.PAD, str)
        assert isinstance(DefaultColorScheme.DEL, str)
        assert isinstance(DefaultColorScheme.INS, str)
        assert isinstance(DefaultColorScheme.SUB, str)
        assert isinstance(DefaultColorScheme.MATCH, str)

    def test_color_scheme_base_class_defines_annotations(self):
        """Test that ColorScheme base class defines required type annotations."""
        annotations = ColorScheme.__annotations__
        assert "PAD" in annotations
        assert "DEL" in annotations
        assert "INS" in annotations
        assert "SUB" in annotations
        assert "MATCH" in annotations


class TestGetPadding:
    """Tests for get_padding function."""

    def test_get_padding_returns_text_object(self):
        """Test that get_padding returns a Rich Text object."""
        result = get_padding(5)
        assert isinstance(result, Text)

    def test_get_padding_correct_length(self):
        """Test that padding has correct number of spaces."""
        result = get_padding(5)
        assert len(result.plain) == 5
        assert result.plain == "     "

    def test_get_padding_zero_length(self):
        """Test padding with zero length."""
        result = get_padding(0)
        assert len(result.plain) == 0
        assert result.plain == ""

    def test_get_padding_uses_color_scheme(self):
        """Test that padding uses the color scheme's PAD color."""
        result = get_padding(3, color_scheme=DefaultColorScheme)
        assert result.plain == "   "


class TestFormatMatchOp:
    """Tests for format_match_op function."""

    def test_format_match_op_returns_tuple(self):
        """Test that format_match_op returns a tuple of (ref_str, hyp_str, length)."""
        op = Op(type=OpType.MATCH, ref="hello", hyp="hello")
        ref_str, hyp_str, length = format_match_op(op)
        assert isinstance(ref_str, Text)
        assert isinstance(hyp_str, Text)
        assert isinstance(length, int)

    def test_format_match_op_equal_length(self):
        """Test match op with equal length ref and hyp."""
        op = Op(type=OpType.MATCH, ref="test", hyp="test")
        ref_str, hyp_str, length = format_match_op(op)
        assert length == 4
        assert ref_str.plain == "test"
        assert hyp_str.plain == "test"

    def test_format_match_op_ref_shorter(self):
        """Test match op where ref is shorter (due to normalization)."""
        op = Op(type=OpType.MATCH, ref="hi", hyp="HI!")
        ref_str, hyp_str, length = format_match_op(op)
        assert length == 3
        assert len(ref_str.plain) == 3  # Padded

    def test_format_match_op_hyp_shorter(self):
        """Test match op where hyp is shorter (due to normalization)."""
        op = Op(type=OpType.MATCH, ref="HELLO", hyp="hi")
        ref_str, hyp_str, length = format_match_op(op)
        assert length == 5
        assert len(hyp_str.plain) == 5  # Padded


class TestFormatSubstituteOp:
    """Tests for format_substitute_op function."""

    def test_format_substitute_op_returns_tuple(self):
        """Test that format_substitute_op returns a tuple."""
        op = Op(type=OpType.SUBSTITUTE, ref="cat", hyp="dog")
        ref_str, hyp_str, length = format_substitute_op(op)
        assert isinstance(ref_str, Text)
        assert isinstance(hyp_str, Text)
        assert isinstance(length, int)

    def test_format_substitute_op_equal_length(self):
        """Test substitute op with equal length tokens."""
        op = Op(type=OpType.SUBSTITUTE, ref="cat", hyp="dog")
        ref_str, hyp_str, length = format_substitute_op(op)
        assert length == 3
        assert ref_str.plain == "cat"
        assert hyp_str.plain == "dog"

    def test_format_substitute_op_ref_longer(self):
        """Test substitute op where ref is longer."""
        op = Op(type=OpType.SUBSTITUTE, ref="hello", hyp="hi")
        ref_str, hyp_str, length = format_substitute_op(op)
        assert length == 5
        assert ref_str.plain == "hello"
        assert len(hyp_str.plain) == 5  # Padded

    def test_format_substitute_op_hyp_longer(self):
        """Test substitute op where hyp is longer."""
        op = Op(type=OpType.SUBSTITUTE, ref="hi", hyp="hello")
        ref_str, hyp_str, length = format_substitute_op(op)
        assert length == 5
        assert len(ref_str.plain) == 5  # Padded
        assert hyp_str.plain == "hello"


class TestFormatInsertOp:
    """Tests for format_insert_op function."""

    def test_format_insert_op_returns_tuple(self):
        """Test that format_insert_op returns a tuple."""
        op = Op(type=OpType.INSERT, ref=None, hyp="inserted")
        ref_str, hyp_str, length = format_insert_op(op)
        assert isinstance(ref_str, Text)
        assert isinstance(hyp_str, Text)
        assert isinstance(length, int)

    def test_format_insert_op_ref_is_padding(self):
        """Test that ref is padding for insert op."""
        op = Op(type=OpType.INSERT, ref=None, hyp="new")
        ref_str, hyp_str, length = format_insert_op(op)
        assert length == 3
        assert ref_str.plain == "   "  # Padding
        assert hyp_str.plain == "new"


class TestFormatDeleteOp:
    """Tests for format_delete_op function."""

    def test_format_delete_op_returns_tuple(self):
        """Test that format_delete_op returns a tuple."""
        op = Op(type=OpType.DELETE, ref="deleted", hyp=None)
        ref_str, hyp_str, length = format_delete_op(op)
        assert isinstance(ref_str, Text)
        assert isinstance(hyp_str, Text)
        assert isinstance(length, int)

    def test_format_delete_op_hyp_is_padding(self):
        """Test that hyp is padding for delete op."""
        op = Op(type=OpType.DELETE, ref="old", hyp=None)
        ref_str, hyp_str, length = format_delete_op(op)
        assert length == 3
        assert ref_str.plain == "old"
        assert hyp_str.plain == "   "  # Padding


class TestFormatAlignmentOp:
    """Tests for format_alignment_op dispatch function."""

    def test_format_alignment_op_match(self):
        """Test format_alignment_op dispatches to match handler."""
        op = Op(type=OpType.MATCH, ref="test", hyp="test")
        ref_str, hyp_str, _ = format_alignment_op(op)
        assert ref_str.plain == "test"
        assert hyp_str.plain == "test"

    def test_format_alignment_op_substitute(self):
        """Test format_alignment_op dispatches to substitute handler."""
        op = Op(type=OpType.SUBSTITUTE, ref="dog", hyp="doggy")
        ref_str, hyp_str, _ = format_alignment_op(op)
        assert ref_str.plain == "dog  "
        assert hyp_str.plain == "doggy"

    def test_format_alignment_op_insert(self):
        """Test format_alignment_op dispatches to insert handler."""
        op = Op(type=OpType.INSERT, ref=None, hyp="new")
        ref_str, hyp_str, _ = format_alignment_op(op)
        assert ref_str.plain == "   "
        assert hyp_str.plain == "new"

    def test_format_alignment_op_delete(self):
        """Test format_alignment_op dispatches to delete handler."""
        op = Op(type=OpType.DELETE, ref="old", hyp=None)
        ref_str, hyp_str, _ = format_alignment_op(op)
        assert ref_str.plain == "old"
        assert hyp_str.plain == "   "


class TestGetLinePrefixes:
    """Tests for get_line_prefixes function."""

    def test_get_line_prefixes_returns_tuple(self):
        """Test that get_line_prefixes returns a tuple of Text objects."""
        ref_prefix, hyp_prefix = get_line_prefixes(1)
        assert isinstance(ref_prefix, Text)
        assert isinstance(hyp_prefix, Text)

    def test_get_line_prefixes_contains_ref_label(self):
        """Test that ref prefix contains REF label."""
        ref_prefix, _ = get_line_prefixes(1)
        assert "Ref." in ref_prefix.plain

    def test_get_line_prefixes_contains_hyp_label(self):
        """Test that hyp prefix contains HYP label."""
        _, hyp_prefix = get_line_prefixes(1)
        assert "Hyp." in hyp_prefix.plain

    def test_get_line_prefixes_contains_line_number(self):
        """Test that ref prefix contains line number."""
        ref_prefix, _ = get_line_prefixes(42)
        assert "42" in ref_prefix.plain

    def test_get_line_prefixes_equal_length(self):
        """Test that both prefixes have equal length."""
        ref_prefix, hyp_prefix = get_line_prefixes(1)
        assert len(ref_prefix.plain) == len(hyp_prefix.plain)


class TestDisplayBasicAligned:
    """Tests for display_basic_aligned function."""

    def test_display_basic_aligned_with_simple_alignment(self):
        """Test display_basic_aligned with a simple alignment."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello"),
            Op(type=OpType.MATCH, ref="world", hyp="world"),
        ]
        alignment = Alignment(ops)
        # Should not raise
        with patch("bewer.reporting.python.alignment.Console") as mock_console:
            display_basic_aligned(alignment)
            mock_console.return_value.print.assert_called_once()

    def test_display_basic_aligned_with_all_op_types(self):
        """Test display_basic_aligned handles all operation types."""
        ops = [
            Op(type=OpType.MATCH, ref="the", hyp="the"),
            Op(type=OpType.SUBSTITUTE, ref="cat", hyp="dog"),
            Op(type=OpType.INSERT, ref=None, hyp="big"),
            Op(type=OpType.DELETE, ref="sat", hyp=None),
        ]
        alignment = Alignment(ops)
        with patch("bewer.reporting.python.alignment.Console") as mock_console:
            display_basic_aligned(alignment)
            mock_console.return_value.print.assert_called_once()

    def test_display_basic_aligned_with_title(self):
        """Test display_basic_aligned with a title."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)
        with patch("bewer.reporting.python.alignment.Console") as mock_console:
            display_basic_aligned(alignment, title="Test Title")
            mock_console.return_value.print.assert_called_once()
            # Verify the output contains the title
            call_args = mock_console.return_value.print.call_args
            output_text = call_args[0][0]
            assert "Test Title" in output_text.plain

    def test_display_basic_aligned_with_integer_max_line_length(self):
        """Test display_basic_aligned with integer max_line_length."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)
        with patch("bewer.reporting.python.alignment.Console") as mock_console:
            display_basic_aligned(alignment, max_line_length=80)
            mock_console.return_value.print.assert_called_once()

    def test_display_basic_aligned_with_float_max_line_length(self):
        """Test display_basic_aligned with float max_line_length (fraction of terminal)."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)
        with patch("bewer.reporting.python.alignment.Console") as mock_console:
            display_basic_aligned(alignment, max_line_length=0.5)
            mock_console.return_value.print.assert_called_once()

    def test_display_basic_aligned_invalid_float_max_line_length(self):
        """Test display_basic_aligned raises for invalid float max_line_length."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)
        with pytest.raises(ValueError, match="must be in the range"):
            display_basic_aligned(alignment, max_line_length=0.0)
        with pytest.raises(ValueError, match="must be in the range"):
            display_basic_aligned(alignment, max_line_length=1.5)

    def test_display_basic_aligned_wraps_long_alignments(self):
        """Test display_basic_aligned wraps when exceeding max line length."""
        # Create a long alignment that should wrap
        ops = [Op(type=OpType.MATCH, ref=f"word{i}", hyp=f"word{i}") for i in range(20)]
        alignment = Alignment(ops)
        with patch("bewer.reporting.python.alignment.Console") as mock_console:
            display_basic_aligned(alignment, max_line_length=50)
            mock_console.return_value.print.assert_called_once()

    def test_display_basic_aligned_handles_right_partial(self):
        """Test display_basic_aligned handles right partial tokens."""
        ops = [
            Op(type=OpType.MATCH, ref="test", hyp="test", hyp_right_partial=True),
            Op(type=OpType.MATCH, ref="ing", hyp="ing"),
        ]
        alignment = Alignment(ops)
        with patch("bewer.reporting.python.alignment.Console") as mock_console:
            display_basic_aligned(alignment)
            mock_console.return_value.print.assert_called_once()

    def test_display_basic_aligned_empty_alignment(self):
        """Test display_basic_aligned with empty alignment."""
        alignment = Alignment()
        with patch("bewer.reporting.python.alignment.Console") as mock_console:
            display_basic_aligned(alignment)
            mock_console.return_value.print.assert_called_once()


class TestAlignmentDisplayMethod:
    """Tests for Alignment.display() method."""

    def test_alignment_display_calls_display_basic_aligned(self):
        """Test that Alignment.display() calls display_basic_aligned."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)
        with patch("bewer.alignment.op.display_basic_aligned") as mock_display:
            alignment.display()
            mock_display.assert_called_once()

    def test_alignment_display_passes_max_line_length(self):
        """Test that Alignment.display() passes max_line_length parameter."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)
        with patch("bewer.alignment.op.display_basic_aligned") as mock_display:
            alignment.display(max_line_length=100)
            mock_display.assert_called_once()
            call_kwargs = mock_display.call_args[1]
            assert call_kwargs["max_line_length"] == 100

    def test_alignment_display_passes_color_scheme(self):
        """Test that Alignment.display() passes color_scheme parameter."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)
        with patch("bewer.alignment.op.display_basic_aligned") as mock_display:
            alignment.display(color_scheme=DefaultColorScheme)
            mock_display.assert_called_once()

    def test_alignment_display_with_source_example(self):
        """Test Alignment.display() includes title when source is set."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)

        # Create a mock example
        class MockExample:
            index = 42

        alignment.set_source(MockExample())
        with patch("bewer.alignment.op.display_basic_aligned") as mock_display:
            alignment.display()
            call_kwargs = mock_display.call_args[1]
            assert "42" in call_kwargs["title"]


class TestAlignmentSetSource:
    """Tests for Alignment.set_source() method."""

    def test_set_source_stores_example(self):
        """Test that set_source stores the example."""
        alignment = Alignment()

        class MockExample:
            index = 5

        example = MockExample()
        alignment.set_source(example)
        assert alignment._src_example is example

    def test_set_source_can_be_overwritten(self):
        """Test that set_source can be called multiple times."""
        alignment = Alignment()

        class MockExample:
            def __init__(self, idx):
                self.index = idx

        alignment.set_source(MockExample(1))
        alignment.set_source(MockExample(2))
        assert alignment._src_example.index == 2
