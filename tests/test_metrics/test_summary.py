"""Tests for bewer.metrics.summary module."""

from bewer.metrics.summary import DatasetSummary, DatasetSummary_


class TestDatasetSummaryExample:
    """Tests for DatasetSummary_ (example-level) metric."""

    def test_num_ref_words(self, sample_dataset):
        """Test num_ref_words returns correct count."""
        # "hello world" has 2 tokens
        example = sample_dataset[0]
        summary = example.metrics.summary
        assert summary.num_ref_words == 2

    def test_num_ref_chars(self, sample_dataset):
        """Test num_ref_chars returns correct count."""
        # "hello world" has 11 characters
        example = sample_dataset[0]
        summary = example.metrics.summary
        assert summary.num_ref_chars == 11

    def test_num_hyp_words(self, sample_dataset):
        """Test num_hyp_words returns correct count."""
        # "hello world" has 2 tokens
        example = sample_dataset[0]
        summary = example.metrics.summary
        assert summary.num_hyp_words == 2

    def test_num_hyp_chars(self, sample_dataset):
        """Test num_hyp_chars returns correct count."""
        # "hello world" has 11 characters
        example = sample_dataset[0]
        summary = example.metrics.summary
        assert summary.num_hyp_chars == 11

    def test_metrics_with_different_lengths(self, sample_dataset):
        """Test metrics with different ref and hyp lengths."""
        # "testing one two three" vs "testing one two" - different lengths
        example = sample_dataset[2]
        summary = example.metrics.summary
        assert summary.num_ref_words == 4  # "testing one two three"
        assert summary.num_hyp_words == 3  # "testing one two"
        assert summary.num_ref_chars > summary.num_hyp_chars

    def test_metrics_with_empty_dataset(self, empty_dataset):
        """Test metrics with empty dataset."""
        # Empty dataset has no examples
        assert len(empty_dataset) == 0


class TestDatasetSummary:
    """Tests for DatasetSummary (dataset-level) metric."""

    def test_short_name(self):
        """Test that short_name is set."""
        assert DatasetSummary._short_name_base == "Summary"

    def test_long_name(self):
        """Test that long_name is set."""
        assert DatasetSummary._long_name_base == "Dataset Summary"

    def test_description_exists(self):
        """Test that description is set."""
        assert len(DatasetSummary.description) > 0

    def test_example_cls_is_set(self):
        """Test that example_cls points to DatasetSummary_."""
        assert DatasetSummary.example_cls == DatasetSummary_

    def test_num_examples(self, sample_dataset):
        """Test num_examples returns correct count."""
        summary = sample_dataset.metrics.summary
        assert summary.num_examples == 3  # sample_dataset has 3 examples

    def test_num_ref_words_aggregates_from_examples(self, sample_dataset):
        """Test num_ref_words aggregates from all examples."""
        # "hello world" (2) + "the quick brown fox" (4) + "testing one two three" (4) = 10
        summary = sample_dataset.metrics.summary
        assert summary.num_ref_words == 10

    def test_num_ref_chars_aggregates_from_examples(self, sample_dataset):
        """Test num_ref_chars aggregates from all examples."""
        summary = sample_dataset.metrics.summary
        # "hello world" (11) + "the quick brown fox" (19) + "testing one two three" (21) = 51
        assert summary.num_ref_chars == 51

    def test_num_hyp_words_aggregates_from_examples(self, sample_dataset):
        """Test num_hyp_words aggregates from all examples."""
        # "hello world" (2) + "the quick brown dog" (4) + "testing one two" (3) = 9
        summary = sample_dataset.metrics.summary
        assert summary.num_hyp_words == 9

    def test_num_hyp_chars_aggregates_from_examples(self, sample_dataset):
        """Test num_hyp_chars aggregates from all examples."""
        summary = sample_dataset.metrics.summary
        # "hello world" (11) + "the quick brown dog" (19) + "testing one two" (15) = 45
        assert summary.num_hyp_chars == 45

    def test_empty_dataset(self, empty_dataset):
        """Test with empty dataset."""
        summary = empty_dataset.metrics.summary

        assert summary.num_examples == 0
        assert summary.num_ref_words == 0
        assert summary.num_ref_chars == 0
        assert summary.num_hyp_words == 0
        assert summary.num_hyp_chars == 0

    def test_perfect_match_dataset(self, dataset_perfect_match):
        """Test with dataset where all matches are perfect."""
        summary = dataset_perfect_match.metrics.summary

        # Should have same counts for ref and hyp
        assert summary.num_ref_words == summary.num_hyp_words
        assert summary.num_ref_chars == summary.num_hyp_chars
        assert summary.num_examples == 2

    def test_metric_registered(self):
        """Test that DatasetSummary is registered in METRIC_REGISTRY."""
        from bewer.metrics.base import METRIC_REGISTRY

        assert "summary" in METRIC_REGISTRY.metric_classes
        assert METRIC_REGISTRY.metric_classes["summary"] == DatasetSummary
