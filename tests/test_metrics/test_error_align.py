"""Tests for bewer.metrics.error_align module."""

from bewer.alignment import OpType
from bewer.metrics.error_align import ErrorAlign, ErrorAlign_


class TestErrorAlignExampleMetric:
    """Tests for ErrorAlign_ (ExampleMetric) class."""

    def test_perfect_match_counts(self, dataset_perfect_match):
        """Test that a perfect match has zero edits and correct match count."""
        example = dataset_perfect_match[0]  # "hello world" vs "hello world"
        ea = example.metrics.error_align()
        assert ea.num_edits == 0
        assert ea.num_substitutions == 0
        assert ea.num_insertions == 0
        assert ea.num_deletions == 0
        assert ea.num_matches == 2

    def test_substitution(self, sample_dataset):
        """Test substitution detection."""
        # "the quick brown fox" vs "the quick brown dog"
        example = sample_dataset[1]
        ea = example.metrics.error_align()
        assert ea.num_substitutions == 1
        assert ea.num_insertions == 0
        assert ea.num_deletions == 0
        assert ea.num_matches == 3

    def test_deletion(self, sample_dataset):
        """Test deletion detection."""
        # "testing one two three" vs "testing one two"
        example = sample_dataset[2]
        ea = example.metrics.error_align()
        assert ea.num_deletions == 1
        assert ea.num_substitutions == 0
        assert ea.num_insertions == 0
        assert ea.num_matches == 3

    def test_num_edits_is_sum(self, sample_dataset):
        """Test that num_edits equals sum of substitutions, insertions, and deletions."""
        for example in sample_dataset:
            ea = example.metrics.error_align()
            assert ea.num_edits == ea.num_substitutions + ea.num_insertions + ea.num_deletions

    def test_insertion(self, empty_dataset):
        """Test insertion detection."""
        empty_dataset.add("hello", "hello world")
        example = empty_dataset[0]
        ea = example.metrics.error_align()
        assert ea.num_insertions == 1
        assert ea.num_matches == 1

    def test_complete_mismatch(self, dataset_with_errors):
        """Test complete mismatch."""
        # "hello" vs "goodbye"
        example = dataset_with_errors[0]
        ea = example.metrics.error_align()
        assert ea.num_edits > 0
        assert ea.num_matches == 0


class TestErrorAlignAlignment:
    """Tests for the alignment object returned by ErrorAlign."""

    def test_alignment_returns_alignment_type(self, sample_dataset):
        """Test that alignment returns an Alignment object."""
        from bewer.alignment import Alignment

        example = sample_dataset[0]
        ea = example.metrics.error_align()
        assert isinstance(ea.alignment, Alignment)

    def test_alignment_op_types(self, sample_dataset):
        """Test that ops have correct types for a substitution case."""
        # "the quick brown fox" vs "the quick brown dog"
        example = sample_dataset[1]
        ea = example.metrics.error_align()
        types = [op.type for op in ea.alignment]
        assert types.count(OpType.MATCH) == 3
        assert types.count(OpType.SUBSTITUTE) == 1

    def test_alignment_op_content_substitution(self, sample_dataset):
        """Test that substitution op has correct ref and hyp values."""
        example = sample_dataset[1]
        ea = example.metrics.error_align()
        sub_ops = [op for op in ea.alignment if op.type == OpType.SUBSTITUTE]
        assert len(sub_ops) == 1
        assert sub_ops[0].ref == "fox"
        assert sub_ops[0].hyp == "dog"

    def test_alignment_op_content_deletion(self, sample_dataset):
        """Test that deletion op has correct ref and None hyp."""
        example = sample_dataset[2]
        ea = example.metrics.error_align()
        del_ops = [op for op in ea.alignment if op.type == OpType.DELETE]
        assert len(del_ops) == 1
        assert del_ops[0].ref == "three"
        assert del_ops[0].hyp is None

    def test_alignment_op_content_match(self, dataset_perfect_match):
        """Test that match ops have equal ref and hyp."""
        example = dataset_perfect_match[0]
        ea = example.metrics.error_align()
        for op in ea.alignment:
            assert op.type == OpType.MATCH
            assert op.ref == op.hyp

    def test_alignment_ref_token_idx_sequential(self, sample_dataset):
        """Test that ref_token_idx is sequential for non-insertion ops."""
        example = sample_dataset[1]
        ea = example.metrics.error_align()
        ref_indices = [op.ref_token_idx for op in ea.alignment if op.ref_token_idx is not None]
        assert ref_indices == list(range(len(ref_indices)))

    def test_alignment_ref_span(self, sample_dataset):
        """Test that ref_span correctly maps back to the reference text."""
        example = sample_dataset[1]
        ea = example.metrics.error_align()
        ref_text = example.ref.standardized
        for op in ea.alignment:
            if op.ref_span is not None and op.ref is not None:
                assert ref_text[op.ref_span] == op.ref

    def test_alignment_hyp_span(self, sample_dataset):
        """Test that hyp_span correctly maps back to the hypothesis text."""
        example = sample_dataset[1]
        ea = example.metrics.error_align()
        hyp_text = example.hyp.standardized
        for op in ea.alignment:
            if op.hyp_span is not None and op.hyp is not None:
                assert hyp_text[op.hyp_span] == op.hyp


class TestErrorAlignNormalization:
    """Tests for the normalized parameter."""

    def test_default_is_normalized(self, sample_dataset):
        """Test that the default parameter is normalized=True."""
        example = sample_dataset[0]
        ea = example.metrics.error_align()
        assert ea.params.normalized is True

    def test_normalized_false(self, sample_dataset):
        """Test that normalized=False can be set."""
        example = sample_dataset[0]
        ea = example.metrics.error_align(normalized=False)
        assert ea.params.normalized is False

    def test_normalized_does_not_change_edit_counts(self, sample_dataset):
        """Test that normalization does not change the edit counts."""
        example = sample_dataset[1]
        ea_norm = example.metrics.error_align(normalized=True)
        ea_no_norm = example.metrics.error_align(normalized=False)
        assert ea_norm.num_edits == ea_no_norm.num_edits
        assert ea_norm.num_matches == ea_no_norm.num_matches


class TestErrorAlignDatasetMetric:
    """Tests for ErrorAlign (dataset-level Metric) class."""

    def test_all_examples_computed(self, sample_dataset):
        """Test that error_align can be computed for all examples."""
        for example in sample_dataset:
            ea = example.metrics.error_align()
            assert ea.num_edits + ea.num_matches > 0

    def test_perfect_match_dataset(self, dataset_perfect_match):
        """Test ErrorAlign on a dataset with all perfect matches."""
        for example in dataset_perfect_match:
            ea = example.metrics.error_align()
            assert ea.num_edits == 0
            assert ea.num_matches > 0

    def test_empty_dataset(self, empty_dataset):
        """Test ErrorAlign on empty dataset has no examples."""
        assert len(empty_dataset) == 0


class TestErrorAlignMetricAttributes:
    """Tests for ErrorAlign metric attributes."""

    def test_short_name(self, sample_dataset):
        """Test ErrorAlign short_name."""
        ea = sample_dataset.metrics.error_align()
        assert ea.short_name_base == "EA"

    def test_long_name(self, sample_dataset):
        """Test ErrorAlign long_name."""
        ea = sample_dataset.metrics.error_align()
        assert ea.long_name_base == "Error Alignment"

    def test_description(self, sample_dataset):
        """Test ErrorAlign has description."""
        ea = sample_dataset.metrics.error_align()
        assert len(ea.description) > 0

    def test_example_cls(self, sample_dataset):
        """Test ErrorAlign has example_cls set."""
        ea = sample_dataset.metrics.error_align()
        assert ea.example_cls == ErrorAlign_


class TestErrorAlignMetricValues:
    """Tests for ErrorAlign metric_values method."""

    def test_example_metric_values_main(self):
        """Test that alignment is the main metric on ErrorAlign_."""
        values = ErrorAlign_.metric_values()
        assert values["main"] == "alignment"

    def test_example_metric_values_other(self):
        """Test that edit counts are other metrics on ErrorAlign_."""
        values = ErrorAlign_.metric_values()
        other = values["other"]
        assert "num_substitutions" in other
        assert "num_insertions" in other
        assert "num_deletions" in other
        assert "num_edits" in other
        assert "num_matches" in other

    def test_dataset_metric_has_no_main(self):
        """Test that ErrorAlign (dataset-level) has no main metric."""
        values = ErrorAlign.metric_values()
        assert values["main"] is None
