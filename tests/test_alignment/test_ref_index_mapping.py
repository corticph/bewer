"""Tests for Alignment.ref_index_mapping and ops_from_ref_index methods."""

from unittest.mock import Mock

import pytest

from bewer.alignment import Alignment, Op, OpType


class TestRefIndexMapping:
    """Tests for ref_index_mapping cached property."""

    def test_mapping_with_all_ref_tokens(self):
        """Test mapping when all ops have ref_token_idx set."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello", ref_token_idx=0),
            Op(type=OpType.MATCH, ref="world", hyp="world", ref_token_idx=1),
        ]
        alignment = Alignment(ops, src=Mock())
        assert alignment.ref_index_mapping == {0: 0, 1: 1}

    def test_mapping_skips_insertions(self):
        """Test that ops without ref_token_idx (insertions) are skipped."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello", ref_token_idx=0),
            Op(type=OpType.INSERT, ref=None, hyp="extra"),
            Op(type=OpType.MATCH, ref="world", hyp="world", ref_token_idx=1),
        ]
        alignment = Alignment(ops, src=Mock())
        assert alignment.ref_index_mapping == {0: 0, 1: 2}

    def test_mapping_includes_deletions(self):
        """Test that deletions (which have ref_token_idx) are included."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello", ref_token_idx=0),
            Op(type=OpType.DELETE, ref="missing", hyp=None, ref_token_idx=1),
            Op(type=OpType.MATCH, ref="world", hyp="world", ref_token_idx=2),
        ]
        alignment = Alignment(ops, src=Mock())
        assert alignment.ref_index_mapping == {0: 0, 1: 1, 2: 2}

    def test_mapping_includes_substitutions(self):
        """Test that substitutions are included in the mapping."""
        ops = [
            Op(type=OpType.MATCH, ref="the", hyp="the", ref_token_idx=0),
            Op(type=OpType.SUBSTITUTE, ref="fox", hyp="dog", ref_token_idx=1),
        ]
        alignment = Alignment(ops, src=Mock())
        assert alignment.ref_index_mapping == {0: 0, 1: 1}

    def test_mapping_empty_alignment(self):
        """Test mapping for an empty alignment."""
        alignment = Alignment(src=Mock())
        assert alignment.ref_index_mapping == {}


class TestHypIndexMapping:
    """Tests for hyp_index_mapping cached property."""

    def test_mapping_with_all_hyp_tokens(self):
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello", ref_token_idx=0, hyp_token_idx=0),
            Op(type=OpType.MATCH, ref="world", hyp="world", ref_token_idx=1, hyp_token_idx=1),
        ]
        alignment = Alignment(ops, src=Mock())
        assert alignment.hyp_index_mapping == {0: 0, 1: 1}

    def test_mapping_skips_deletions(self):
        """DELETE ops have no hyp_token_idx and should be excluded."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello", ref_token_idx=0, hyp_token_idx=0),
            Op(type=OpType.DELETE, ref="extra", hyp=None, ref_token_idx=1),
            Op(type=OpType.MATCH, ref="world", hyp="world", ref_token_idx=2, hyp_token_idx=1),
        ]
        alignment = Alignment(ops, src=Mock())
        assert alignment.hyp_index_mapping == {0: 0, 1: 2}

    def test_mapping_includes_insertions(self):
        """INSERT ops have hyp_token_idx but no ref_token_idx and should be included."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello", ref_token_idx=0, hyp_token_idx=0),
            Op(type=OpType.INSERT, ref=None, hyp="extra", hyp_token_idx=1),
            Op(type=OpType.MATCH, ref="world", hyp="world", ref_token_idx=1, hyp_token_idx=2),
        ]
        alignment = Alignment(ops, src=Mock())
        assert alignment.hyp_index_mapping == {0: 0, 1: 1, 2: 2}

    def test_mapping_empty_alignment(self):
        alignment = Alignment(src=Mock())
        assert alignment.hyp_index_mapping == {}

    def test_mapping_no_hyp_token_idx(self):
        """Alignments without hyp_token_idx set (e.g. error-align) return empty dict."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello"),
            Op(type=OpType.MATCH, ref="world", hyp="world"),
        ]
        alignment = Alignment(ops, src=Mock())
        assert alignment.hyp_index_mapping == {}


class TestOpsFromRefIndex:
    """Tests for ops_from_ref_index method."""

    @pytest.fixture
    def alignment_with_mixed_ops(self):
        """Create an alignment with various op types and ref_token_idx values."""
        ops = [
            Op(type=OpType.MATCH, ref="the", hyp="the", ref_token_idx=0),
            Op(type=OpType.MATCH, ref="quick", hyp="quick", ref_token_idx=1),
            Op(type=OpType.SUBSTITUTE, ref="brown", hyp="red", ref_token_idx=2),
            Op(type=OpType.MATCH, ref="fox", hyp="fox", ref_token_idx=3),
        ]
        return Alignment(ops, src=Mock())

    def test_single_index_returns_single_op(self, alignment_with_mixed_ops):
        """Test that a single start index returns a list with one op."""
        result = alignment_with_mixed_ops.ops_from_ref_index(0)
        assert len(result) == 1
        assert result[0].ref == "the"
        assert result[0].type == OpType.MATCH

    def test_single_index_substitute_op(self, alignment_with_mixed_ops):
        """Test retrieving a substitution op by ref index."""
        result = alignment_with_mixed_ops.ops_from_ref_index(2)
        assert len(result) == 1
        assert result[0].ref == "brown"
        assert result[0].type == OpType.SUBSTITUTE

    def test_range_returns_slice(self, alignment_with_mixed_ops):
        """Test that start/stop returns inclusive slice of ops."""
        result = alignment_with_mixed_ops.ops_from_ref_index(1, 3)
        assert len(result) == 3
        assert result[0].ref == "quick"
        assert result[1].ref == "brown"
        assert result[2].ref == "fox"

    def test_range_same_start_stop(self, alignment_with_mixed_ops):
        """Test that start == stop returns a single op."""
        result = alignment_with_mixed_ops.ops_from_ref_index(1, 1)
        assert len(result) == 1
        assert result[0].ref == "quick"

    def test_range_includes_interleaved_insertions(self):
        """Test that range includes insertion ops between ref-anchored ops."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello", ref_token_idx=0),
            Op(type=OpType.INSERT, ref=None, hyp="extra"),
            Op(type=OpType.MATCH, ref="world", hyp="world", ref_token_idx=1),
        ]
        alignment = Alignment(ops, src=Mock())
        result = alignment.ops_from_ref_index(0, 1)
        assert len(result) == 3
        assert result[1].type == OpType.INSERT

    def test_start_not_found_raises(self, alignment_with_mixed_ops):
        """Test that a missing start index raises ValueError."""
        with pytest.raises(ValueError, match="Start index 99 not found"):
            alignment_with_mixed_ops.ops_from_ref_index(99)

    def test_stop_not_found_raises(self, alignment_with_mixed_ops):
        """Test that a missing stop index raises ValueError."""
        with pytest.raises(ValueError, match="Stop index 99 not found"):
            alignment_with_mixed_ops.ops_from_ref_index(0, 99)

    def test_stop_less_than_start_raises(self, alignment_with_mixed_ops):
        """Test that stop < start raises ValueError."""
        with pytest.raises(ValueError, match="Stop index must be greater than or equal"):
            alignment_with_mixed_ops.ops_from_ref_index(3, 1)

    def test_returns_alignment_type(self, alignment_with_mixed_ops):
        """Test that range queries return an Alignment (via __getitem__ slice)."""
        result = alignment_with_mixed_ops.ops_from_ref_index(0, 2)
        assert isinstance(result, Alignment)


class TestOpsFromHypIndex:
    """Tests for ops_from_hyp_index, which covers hypothesis tokens split across ops."""

    @pytest.fixture
    def split_alignment(self):
        """`bedroom` (hyp token 1) is split across two ops, as error_align does."""
        ops = [
            Op(type=OpType.MATCH, ref="the", hyp="the", ref_token_idx=0, hyp_token_idx=0),
            Op(type=OpType.SUBSTITUTE, ref="bed", hyp="bed", ref_token_idx=1, hyp_token_idx=1),
            Op(type=OpType.SUBSTITUTE, ref="room", hyp="room", ref_token_idx=2, hyp_token_idx=1),
            Op(type=OpType.DELETE, ref="is", hyp=None, ref_token_idx=3),
            Op(type=OpType.MATCH, ref="big", hyp="big", ref_token_idx=4, hyp_token_idx=2),
        ]
        return Alignment(ops, src=Mock())

    def test_single_token_covers_all_its_ops(self, split_alignment):
        assert [op.ref for op in split_alignment.ops_from_hyp_index(1)] == ["bed", "room"]

    def test_range_spans_first_to_last_op(self, split_alignment):
        assert [op.ref for op in split_alignment.ops_from_hyp_index(1, 2)] == ["bed", "room", "is", "big"]

    def test_hyp_index_mapping_holds_last_op(self, split_alignment):
        assert split_alignment.hyp_index_mapping == {0: 0, 1: 2, 2: 4}

    def test_unknown_start_raises(self, split_alignment):
        with pytest.raises(ValueError, match="Start index 9"):
            split_alignment.ops_from_hyp_index(9)

    def test_unknown_stop_raises(self, split_alignment):
        with pytest.raises(ValueError, match="Stop index 9"):
            split_alignment.ops_from_hyp_index(0, 9)

    def test_merged_op_counts_for_each_token(self):
        """An op covering hyp tokens 1-2 (error_align merge) is found from either token."""
        ops = [
            Op(type=OpType.INSERT, hyp="c", hyp_token_idx=0),
            Op(type=OpType.SUBSTITUTE, ref="ba", hyp="ab cab", ref_token_idx=0, hyp_token_idx=1, hyp_token_stop=3),
        ]
        alignment = Alignment(ops, src=Mock())
        assert alignment.hyp_index_mapping == {0: 0, 1: 1, 2: 1}
        assert [op.hyp for op in alignment.ops_from_hyp_index(2)] == ["ab cab"]

    def test_stop_before_start_raises(self, split_alignment):
        with pytest.raises(ValueError, match="greater than or equal"):
            split_alignment.ops_from_hyp_index(2, 1)
