"""Tests for Alignment._ref_index_mapping and ops_from_ref_index methods."""

import pytest

from bewer.alignment import Alignment, Op, OpType


class TestRefIndexMapping:
    """Tests for _ref_index_mapping cached property."""

    def test_mapping_with_all_ref_tokens(self):
        """Test mapping when all ops have ref_token_idx set."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello", ref_token_idx=0),
            Op(type=OpType.MATCH, ref="world", hyp="world", ref_token_idx=1),
        ]
        alignment = Alignment(ops)
        assert alignment._ref_index_mapping == {0: 0, 1: 1}

    def test_mapping_skips_insertions(self):
        """Test that ops without ref_token_idx (insertions) are skipped."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello", ref_token_idx=0),
            Op(type=OpType.INSERT, ref=None, hyp="extra"),
            Op(type=OpType.MATCH, ref="world", hyp="world", ref_token_idx=1),
        ]
        alignment = Alignment(ops)
        assert alignment._ref_index_mapping == {0: 0, 1: 2}

    def test_mapping_includes_deletions(self):
        """Test that deletions (which have ref_token_idx) are included."""
        ops = [
            Op(type=OpType.MATCH, ref="hello", hyp="hello", ref_token_idx=0),
            Op(type=OpType.DELETE, ref="missing", hyp=None, ref_token_idx=1),
            Op(type=OpType.MATCH, ref="world", hyp="world", ref_token_idx=2),
        ]
        alignment = Alignment(ops)
        assert alignment._ref_index_mapping == {0: 0, 1: 1, 2: 2}

    def test_mapping_includes_substitutions(self):
        """Test that substitutions are included in the mapping."""
        ops = [
            Op(type=OpType.MATCH, ref="the", hyp="the", ref_token_idx=0),
            Op(type=OpType.SUBSTITUTE, ref="fox", hyp="dog", ref_token_idx=1),
        ]
        alignment = Alignment(ops)
        assert alignment._ref_index_mapping == {0: 0, 1: 1}

    def test_mapping_empty_alignment(self):
        """Test mapping for an empty alignment."""
        alignment = Alignment()
        assert alignment._ref_index_mapping == {}


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
        return Alignment(ops)

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
        alignment = Alignment(ops)
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
