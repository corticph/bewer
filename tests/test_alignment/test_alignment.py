"""Tests for bewer.alignment.alignment module."""

import json
from unittest.mock import Mock, patch

import pytest

from bewer.alignment import Alignment, Op, OpType


class TestAlignmentProperties:
    """Tests for Alignment class properties."""

    def test_num_matches(self):
        """Test num_matches property."""
        ops = [
            Op(type=OpType.MATCH, ref="a", hyp="a"),
            Op(type=OpType.MATCH, ref="b", hyp="b"),
            Op(type=OpType.SUBSTITUTE, ref="c", hyp="d"),
        ]
        alignment = Alignment(ops)
        assert alignment.num_matches == 2

    def test_num_substitutions(self):
        """Test num_substitutions property."""
        ops = [
            Op(type=OpType.SUBSTITUTE, ref="a", hyp="b"),
            Op(type=OpType.SUBSTITUTE, ref="c", hyp="d"),
            Op(type=OpType.MATCH, ref="e", hyp="e"),
        ]
        alignment = Alignment(ops)
        assert alignment.num_substitutions == 2

    def test_num_insertions(self):
        """Test num_insertions property."""
        ops = [
            Op(type=OpType.INSERT, ref=None, hyp="a"),
            Op(type=OpType.INSERT, ref=None, hyp="b"),
            Op(type=OpType.MATCH, ref="c", hyp="c"),
        ]
        alignment = Alignment(ops)
        assert alignment.num_insertions == 2

    def test_num_deletions(self):
        """Test num_deletions property."""
        ops = [
            Op(type=OpType.DELETE, ref="a", hyp=None),
            Op(type=OpType.DELETE, ref="b", hyp=None),
            Op(type=OpType.MATCH, ref="c", hyp="c"),
        ]
        alignment = Alignment(ops)
        assert alignment.num_deletions == 2

    def test_num_edits(self):
        """Test num_edits property (total of subs + ins + dels)."""
        ops = [
            Op(type=OpType.SUBSTITUTE, ref="a", hyp="b"),
            Op(type=OpType.INSERT, ref=None, hyp="c"),
            Op(type=OpType.DELETE, ref="d", hyp=None),
            Op(type=OpType.MATCH, ref="e", hyp="e"),
        ]
        alignment = Alignment(ops)
        assert alignment.num_edits == 3

    def test_counts_updated_on_append(self):
        """Test that operation counts are updated when appending."""
        alignment = Alignment()
        alignment.append(Op(type=OpType.MATCH, ref="a", hyp="a"))
        assert alignment.num_matches == 1

        alignment.append(Op(type=OpType.SUBSTITUTE, ref="b", hyp="c"))
        assert alignment.num_substitutions == 1

    def test_counts_updated_on_extend(self):
        """Test that operation counts are updated when extending."""
        alignment = Alignment()
        new_ops = [
            Op(type=OpType.MATCH, ref="a", hyp="a"),
            Op(type=OpType.INSERT, ref=None, hyp="b"),
        ]
        alignment.extend(new_ops)
        assert alignment.num_matches == 1
        assert alignment.num_insertions == 1


class TestAlignmentModification:
    """Tests for modifying Alignment objects."""

    def test_append_when_source_not_set(self):
        """Test that append works when source is not set."""
        alignment = Alignment()
        op = Op(type=OpType.MATCH, ref="test", hyp="test")
        alignment.append(op)
        assert len(alignment) == 1

    def test_append_when_source_set_raises(self):
        """Test that append raises when source is set."""
        alignment = Alignment()
        mock_example = Mock()
        alignment.set_source(mock_example)

        op = Op(type=OpType.MATCH, ref="test", hyp="test")
        with pytest.raises(ValueError, match="Cannot modify"):
            alignment.append(op)

    def test_extend_when_source_not_set(self):
        """Test that extend works when source is not set."""
        alignment = Alignment()
        ops = [
            Op(type=OpType.MATCH, ref="a", hyp="a"),
            Op(type=OpType.MATCH, ref="b", hyp="b"),
        ]
        alignment.extend(ops)
        assert len(alignment) == 2

    def test_extend_when_source_set_raises(self):
        """Test that extend raises when source is set."""
        alignment = Alignment()
        mock_example = Mock()
        alignment.set_source(mock_example)

        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        with pytest.raises(ValueError, match="Cannot modify"):
            alignment.extend(ops)


class TestAlignmentSetSource:
    """Tests for set_source method."""

    def test_set_source_stores_example(self):
        """Test that set_source stores the example."""
        alignment = Alignment()
        mock_example = Mock()
        alignment.set_source(mock_example)
        assert alignment.src is mock_example

    def test_set_source_raises_on_reassignment(self):
        """Test that set_source raises ValueError on reassignment (single assignment only)."""
        alignment = Alignment()
        mock_example1 = Mock()
        mock_example2 = Mock()

        alignment.set_source(mock_example1)
        assert alignment.src is mock_example1

        with pytest.raises(ValueError, match="Source already set"):
            alignment.set_source(mock_example2)


class TestAlignmentToDicts:
    """Tests for to_dicts method."""

    def test_to_dicts_returns_list(self):
        """Test that to_dicts returns a list."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)
        result = alignment.to_dicts()
        assert isinstance(result, list)

    def test_to_dicts_calls_op_to_dict(self):
        """Test that to_dicts calls to_dict on each operation."""
        mock_op = Mock()
        mock_op.to_dict.return_value = {"type": "match"}

        alignment = Alignment([mock_op])
        result = alignment.to_dicts()

        mock_op.to_dict.assert_called_once()
        assert result == [{"type": "match"}]

    def test_to_dicts_empty_alignment(self):
        """Test to_dicts with empty alignment."""
        alignment = Alignment()
        result = alignment.to_dicts()
        assert result == []


class TestAlignmentToJson:
    """Tests for to_json method."""

    def test_to_json_returns_string(self):
        """Test that to_json returns a JSON string."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)

        with patch.object(Op, "to_dict", return_value={"type": "match"}):
            result = alignment.to_json()
            assert isinstance(result, str)
            # Verify it's valid JSON
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_to_json_writes_to_file(self, tmp_path):
        """Test that to_json writes to file when path is provided."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)

        output_file = tmp_path / "alignment.json"

        with patch.object(Op, "to_dict", return_value={"type": "match", "ref": "test", "hyp": "test"}):
            result = alignment.to_json(path=str(output_file))

            assert output_file.exists()
            assert output_file.read_text() == result

    def test_to_json_raises_if_file_exists_without_overwrite(self, tmp_path):
        """Test that to_json raises if file exists and overwrite is False."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)

        output_file = tmp_path / "alignment.json"
        output_file.write_text("existing content")

        with pytest.raises(FileExistsError):
            alignment.to_json(path=str(output_file), allow_overwrite=False)

    def test_to_json_overwrites_if_allowed(self, tmp_path):
        """Test that to_json overwrites file when allow_overwrite is True."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)

        output_file = tmp_path / "alignment.json"
        output_file.write_text("existing content")

        with patch.object(Op, "to_dict", return_value={"type": "match"}):
            result = alignment.to_json(path=str(output_file), allow_overwrite=True)

            assert output_file.exists()
            assert output_file.read_text() == result
            assert output_file.read_text() != "existing content"

    def test_to_json_creates_parent_directories(self, tmp_path):
        """Test that to_json creates parent directories if they don't exist."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)

        output_file = tmp_path / "nested" / "dir" / "alignment.json"

        with patch.object(Op, "to_dict", return_value={"type": "match"}):
            alignment.to_json(path=str(output_file))

            assert output_file.exists()
            assert output_file.parent.exists()

    def test_to_json_raises_if_path_is_directory(self, tmp_path):
        """Test that to_json raises if path is a directory."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)

        with pytest.raises(ValueError, match="directory"):
            alignment.to_json(path=str(tmp_path))


class TestAlignmentToHtmlLines:
    """Tests for _to_html_lines method."""

    def _create_mock_example(self):
        """Helper to create a mock example with keywords attribute."""
        mock_example = Mock()
        mock_example.keywords = None
        return mock_example

    def test_to_html_lines_returns_list_of_tuples(self):
        """Test that _to_html_lines returns a list of tuples."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops, src=self._create_mock_example())
        result = alignment._to_html_lines()

        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)

    def test_to_html_lines_with_custom_color_scheme(self):
        """Test that _to_html_lines accepts custom color scheme."""
        from bewer.reporting.html.color_schemes import HTMLDefaultAlignmentColors

        class CustomColorScheme(HTMLDefaultAlignmentColors):
            MATCH = "#custom"

        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops, src=self._create_mock_example())
        result = alignment._to_html_lines(color_scheme=CustomColorScheme)

        # Verify custom color is used
        ref_line, _ = result[0]
        assert "#custom" in ref_line


class TestAlignmentSlicing:
    """Tests for Alignment slicing behavior."""

    def test_getitem_single_index_returns_op(self):
        """Test that single index returns an Op."""
        ops = [
            Op(type=OpType.MATCH, ref="a", hyp="a"),
            Op(type=OpType.MATCH, ref="b", hyp="b"),
        ]
        alignment = Alignment(ops)
        result = alignment[0]
        assert isinstance(result, Op)
        assert result.ref == "a"

    def test_getitem_slice_returns_alignment(self):
        """Test that slice returns an Alignment."""
        ops = [
            Op(type=OpType.MATCH, ref="a", hyp="a"),
            Op(type=OpType.MATCH, ref="b", hyp="b"),
            Op(type=OpType.MATCH, ref="c", hyp="c"),
        ]
        alignment = Alignment(ops)
        result = alignment[0:2]
        assert isinstance(result, Alignment)
        assert len(result) == 2


class TestAlignmentConcatenation:
    """Tests for Alignment concatenation."""

    def test_add_returns_alignment(self):
        """Test that adding two alignments returns an Alignment."""
        ops1 = [Op(type=OpType.MATCH, ref="a", hyp="a")]
        ops2 = [Op(type=OpType.MATCH, ref="b", hyp="b")]

        alignment1 = Alignment(ops1)
        alignment2 = Alignment(ops2)

        result = alignment1 + alignment2
        assert isinstance(result, Alignment)
        assert len(result) == 2

    def test_add_raises_if_source_set(self):
        """Test that adding raises if either alignment has source set."""
        ops1 = [Op(type=OpType.MATCH, ref="a", hyp="a")]
        ops2 = [Op(type=OpType.MATCH, ref="b", hyp="b")]

        alignment1 = Alignment(ops1)
        alignment2 = Alignment(ops2)

        mock_example = Mock()
        alignment1.set_source(mock_example)

        with pytest.raises(ValueError, match="Cannot concatenate"):
            _ = alignment1 + alignment2


class TestAlignmentRepr:
    """Tests for Alignment __repr__ method."""

    def test_repr_short_alignment(self):
        """Test repr with short alignment."""
        ops = [Op(type=OpType.MATCH, ref="test", hyp="test")]
        alignment = Alignment(ops)
        result = repr(alignment)
        assert "Alignment" in result
        assert "test" in result

    def test_repr_long_alignment_truncates(self):
        """Test repr truncates long alignments."""
        ops = [Op(type=OpType.MATCH, ref=f"word{i}", hyp=f"word{i}") for i in range(100)]
        alignment = Alignment(ops)
        result = repr(alignment)
        assert "..." in result  # Should be truncated
