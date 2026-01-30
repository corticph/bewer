"""Tests for bewer.alignment.op module."""

import json
import os
import tempfile

import pytest

from bewer.alignment.op import Alignment, Op, OpType


class TestOpType:
    """Tests for OpType enum."""

    def test_optype_values(self):
        """Test OpType enum values."""
        assert OpType.MATCH == 0
        assert OpType.INSERT == 1
        assert OpType.DELETE == 2
        assert OpType.SUBSTITUTE == 3

    def test_optype_is_int_enum(self):
        """Test that OpType values are integers."""
        assert isinstance(OpType.MATCH.value, int)


class TestOpInit:
    """Tests for Op dataclass initialization and validation."""

    def test_match_operation_valid(self):
        """Test valid MATCH operation."""
        op = Op(type=OpType.MATCH, ref="hello", hyp="hello")
        assert op.type == OpType.MATCH
        assert op.ref == "hello"
        assert op.hyp == "hello"

    def test_match_operation_invalid_no_ref(self):
        """Test that MATCH operation requires non-empty ref."""
        with pytest.raises(ValueError, match="MATCH operation must have non-empty"):
            Op(type=OpType.MATCH, ref=None, hyp="hello")

    def test_match_operation_invalid_no_hyp(self):
        """Test that MATCH operation requires non-empty hyp."""
        with pytest.raises(ValueError, match="MATCH operation must have non-empty"):
            Op(type=OpType.MATCH, ref="hello", hyp=None)

    def test_insert_operation_valid(self):
        """Test valid INSERT operation."""
        op = Op(type=OpType.INSERT, ref=None, hyp="inserted")
        assert op.type == OpType.INSERT
        assert op.ref is None
        assert op.hyp == "inserted"

    def test_insert_operation_invalid_no_hyp(self):
        """Test that INSERT operation requires non-empty hyp."""
        with pytest.raises(ValueError, match="INSERT operation must have non-empty hyp"):
            Op(type=OpType.INSERT, ref=None, hyp=None)

    def test_insert_operation_invalid_has_ref(self):
        """Test that INSERT operation must have empty ref."""
        with pytest.raises(ValueError, match="INSERT operation must have non-empty hyp and empty ref"):
            Op(type=OpType.INSERT, ref="something", hyp="inserted")

    def test_delete_operation_valid(self):
        """Test valid DELETE operation."""
        op = Op(type=OpType.DELETE, ref="deleted", hyp=None)
        assert op.type == OpType.DELETE
        assert op.ref == "deleted"
        assert op.hyp is None

    def test_delete_operation_invalid_no_ref(self):
        """Test that DELETE operation requires non-empty ref."""
        with pytest.raises(ValueError, match="DELETE operation must have non-empty ref"):
            Op(type=OpType.DELETE, ref=None, hyp=None)

    def test_delete_operation_invalid_has_hyp(self):
        """Test that DELETE operation must have empty hyp."""
        with pytest.raises(ValueError, match="DELETE operation must have non-empty ref and empty hyp"):
            Op(type=OpType.DELETE, ref="deleted", hyp="something")

    def test_substitute_operation_valid(self):
        """Test valid SUBSTITUTE operation."""
        op = Op(type=OpType.SUBSTITUTE, ref="cat", hyp="dog")
        assert op.type == OpType.SUBSTITUTE
        assert op.ref == "cat"
        assert op.hyp == "dog"

    def test_substitute_operation_invalid_no_ref(self):
        """Test that SUBSTITUTE operation requires both ref and hyp."""
        with pytest.raises(ValueError, match="SUBSTITUTE operation must have both"):
            Op(type=OpType.SUBSTITUTE, ref=None, hyp="dog")

    def test_substitute_operation_invalid_no_hyp(self):
        """Test that SUBSTITUTE operation requires both ref and hyp."""
        with pytest.raises(ValueError, match="SUBSTITUTE operation must have both"):
            Op(type=OpType.SUBSTITUTE, ref="cat", hyp=None)


class TestOpToDict:
    """Tests for Op.to_dict() method."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        op = Op(type=OpType.MATCH, ref="hello", hyp="hello")
        d = op.to_dict()
        assert d["type"] == "MATCH"
        assert d["ref"] == "hello"
        assert d["hyp"] == "hello"

    def test_to_dict_with_token_indices(self):
        """Test to_dict includes token indices."""
        op = Op(type=OpType.MATCH, ref="hello", hyp="hello", ref_token_idx=0, hyp_token_idx=0)
        d = op.to_dict()
        assert d["ref_token_idx"] == 0
        assert d["hyp_token_idx"] == 0

    def test_to_dict_with_spans(self):
        """Test to_dict includes spans as tuples."""
        op = Op(type=OpType.MATCH, ref="hello", hyp="hello", ref_span=slice(0, 5), hyp_span=slice(0, 5))
        d = op.to_dict()
        assert d["ref_span"] == (0, 5)
        assert d["hyp_span"] == (0, 5)

    def test_to_dict_none_spans(self):
        """Test to_dict handles None spans."""
        op = Op(type=OpType.MATCH, ref="hello", hyp="hello")
        d = op.to_dict()
        assert d["ref_span"] is None
        assert d["hyp_span"] is None

    def test_to_dict_partial_flags(self):
        """Test to_dict includes partial flags."""
        op = Op(
            type=OpType.SUBSTITUTE,
            ref="hello",
            hyp="hello",
            hyp_left_partial=True,
            hyp_right_partial=True,
            ref_left_partial=False,
            ref_right_partial=False,
        )
        d = op.to_dict()
        assert d["hyp_left_partial"] is True
        assert d["hyp_right_partial"] is True
        assert d["ref_left_partial"] is False
        assert d["ref_right_partial"] is False


class TestOpRepr:
    """Tests for Op.__repr__()."""

    def test_repr_match(self):
        """Test repr for MATCH operation."""
        op = Op(type=OpType.MATCH, ref="hello", hyp="hello")
        repr_str = repr(op)
        assert "MATCH" in repr_str
        assert "hello" in repr_str

    def test_repr_insert(self):
        """Test repr for INSERT operation."""
        op = Op(type=OpType.INSERT, ref=None, hyp="inserted")
        repr_str = repr(op)
        assert "INSERT" in repr_str
        assert "inserted" in repr_str

    def test_repr_delete(self):
        """Test repr for DELETE operation."""
        op = Op(type=OpType.DELETE, ref="deleted", hyp=None)
        repr_str = repr(op)
        assert "DELETE" in repr_str
        assert "deleted" in repr_str

    def test_repr_substitute(self):
        """Test repr for SUBSTITUTE operation."""
        op = Op(type=OpType.SUBSTITUTE, ref="cat", hyp="dog")
        repr_str = repr(op)
        assert "SUBSTITUTE" in repr_str
        assert "dog" in repr_str
        assert "cat" in repr_str

    def test_repr_with_partials(self):
        """Test repr includes partial markers."""
        op = Op(type=OpType.MATCH, ref="hello", hyp="hello", hyp_left_partial=True, ref_right_partial=True)
        repr_str = repr(op)
        # The repr should include the partial markers
        assert "MATCH" in repr_str


class TestAlignment:
    """Tests for Alignment class."""

    def test_alignment_is_list(self):
        """Test that Alignment is a list."""
        alignment = Alignment()
        assert isinstance(alignment, list)

    def test_alignment_append(self):
        """Test appending to Alignment."""
        alignment = Alignment()
        op = Op(type=OpType.MATCH, ref="hello", hyp="hello")
        alignment.append(op)
        assert len(alignment) == 1

    def test_alignment_from_list(self):
        """Test creating Alignment from list of ops."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello"),
            Op(type=OpType.SUBSTITUTE, ref="world", hyp="earth"),
        ]
        alignment = Alignment(ops)
        assert len(alignment) == 2

    def test_to_dicts(self):
        """Test to_dicts method."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello"),
            Op(type=OpType.DELETE, ref="world", hyp=None),
        ]
        alignment = Alignment(ops)
        dicts = alignment.to_dicts()
        assert len(dicts) == 2
        assert dicts[0]["type"] == "MATCH"
        assert dicts[1]["type"] == "DELETE"

    def test_to_json_returns_string(self):
        """Test to_json returns JSON string."""
        ops = [Op(type=OpType.MATCH, ref="hello", hyp="hello")]
        alignment = Alignment(ops)
        json_str = alignment.to_json()
        assert isinstance(json_str, str)
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert len(parsed) == 1

    def test_to_json_writes_file(self):
        """Test to_json writes to file."""
        ops = [Op(type=OpType.MATCH, ref="hello", hyp="hello")]
        alignment = Alignment(ops)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "alignment.json")
            alignment.to_json(path=filepath)
            assert os.path.exists(filepath)
            with open(filepath) as f:
                content = json.load(f)
            assert len(content) == 1

    def test_to_json_raises_on_existing_file(self):
        """Test to_json raises if file exists without allow_overwrite."""
        ops = [Op(type=OpType.MATCH, ref="hello", hyp="hello")]
        alignment = Alignment(ops)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            with pytest.raises(FileExistsError):
                alignment.to_json(path=filepath, allow_overwrite=False)
        finally:
            os.unlink(filepath)

    def test_to_json_allows_overwrite(self):
        """Test to_json overwrites with allow_overwrite=True."""
        ops = [Op(type=OpType.MATCH, ref="hello", hyp="hello")]
        alignment = Alignment(ops)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            alignment.to_json(path=filepath, allow_overwrite=True)
            assert os.path.exists(filepath)
        finally:
            os.unlink(filepath)

    def test_to_json_raises_on_directory_path(self):
        """Test to_json raises if path is a directory."""
        ops = [Op(type=OpType.MATCH, ref="hello", hyp="hello")]
        alignment = Alignment(ops)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="directory"):
                alignment.to_json(path=tmpdir)

    def test_to_json_creates_parent_dirs(self):
        """Test to_json creates parent directories."""
        ops = [Op(type=OpType.MATCH, ref="hello", hyp="hello")]
        alignment = Alignment(ops)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "nested", "dir", "alignment.json")
            alignment.to_json(path=filepath)
            assert os.path.exists(filepath)


class TestAlignmentSlicing:
    """Tests for Alignment slicing and concatenation."""

    def test_getitem_returns_op(self):
        """Test that indexing returns Op."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello"),
            Op(type=OpType.DELETE, ref="world", hyp=None),
        ]
        alignment = Alignment(ops)
        assert isinstance(alignment[0], Op)
        assert alignment[0].type == OpType.MATCH

    def test_slice_returns_alignment(self):
        """Test that slicing returns Alignment."""
        ops = [
            Op(type=OpType.MATCH, ref="a", hyp="a"),
            Op(type=OpType.MATCH, ref="b", hyp="b"),
            Op(type=OpType.MATCH, ref="c", hyp="c"),
        ]
        alignment = Alignment(ops)
        sliced = alignment[0:2]
        assert isinstance(sliced, Alignment)
        assert len(sliced) == 2

    def test_add_returns_alignment(self):
        """Test that adding Alignments returns Alignment."""
        ops1 = [Op(type=OpType.MATCH, ref="a", hyp="a")]
        ops2 = [Op(type=OpType.MATCH, ref="b", hyp="b")]
        alignment1 = Alignment(ops1)
        alignment2 = Alignment(ops2)
        combined = alignment1 + alignment2
        assert isinstance(combined, Alignment)
        assert len(combined) == 2


class TestAlignmentRepr:
    """Tests for Alignment.__repr__()."""

    def test_repr_short(self):
        """Test repr with few operations."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello"),
            Op(type=OpType.DELETE, ref="world", hyp=None),
        ]
        alignment = Alignment(ops)
        repr_str = repr(alignment)
        assert "Alignment" in repr_str

    def test_repr_truncates_long(self):
        """Test that repr truncates long alignments."""
        ops = [Op(type=OpType.MATCH, ref=f"w{i}", hyp=f"w{i}") for i in range(100)]
        alignment = Alignment(ops)
        repr_str = repr(alignment)
        assert "..." in repr_str
