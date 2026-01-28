"""Tests for bewer.metrics.wer module."""

from bewer.metrics.wer import WER, WER_


class TestWERExampleMetric:
    """Tests for WER_ (ExampleMetric) class."""

    def test_num_edits_perfect_match(self, dataset_perfect_match):
        """Test num_edits is 0 for perfect match."""
        example = dataset_perfect_match[0]
        wer = example.metrics.wer
        assert wer.num_edits == 0

    def test_num_edits_with_errors(self, sample_dataset):
        """Test num_edits counts errors correctly."""
        # "the quick brown fox" vs "the quick brown dog" - 1 substitution
        example = sample_dataset[1]
        wer = example.metrics.wer
        assert wer.num_edits == 1

    def test_num_edits_deletion(self, sample_dataset):
        """Test num_edits counts deletions."""
        # "testing one two three" vs "testing one two" - 1 deletion
        example = sample_dataset[2]
        wer = example.metrics.wer
        assert wer.num_edits == 1

    def test_ref_length(self, sample_dataset):
        """Test ref_length returns correct token count."""
        # "hello world" - 2 tokens
        example = sample_dataset[0]
        wer = example.metrics.wer
        assert wer.ref_length == 2

    def test_ref_length_longer(self, sample_dataset):
        """Test ref_length for longer text."""
        # "the quick brown fox" - 4 tokens
        example = sample_dataset[1]
        wer = example.metrics.wer
        assert wer.ref_length == 4

    def test_value_perfect_match(self, dataset_perfect_match):
        """Test WER value is 0 for perfect match."""
        example = dataset_perfect_match[0]
        wer = example.metrics.wer
        assert wer.value == 0.0

    def test_value_with_error(self, sample_dataset):
        """Test WER value calculation with errors."""
        # "the quick brown fox" vs "the quick brown dog" - 1/4 = 0.25
        example = sample_dataset[1]
        wer = example.metrics.wer
        assert wer.value == 0.25

    def test_value_complete_mismatch(self, dataset_with_errors):
        """Test WER value for complete mismatch."""
        # "hello" vs "goodbye" - 1 word, 1 substitution = 1.0
        example = dataset_with_errors[0]
        wer = example.metrics.wer
        assert wer.value == 1.0


class TestWEREmptyReference:
    """Tests for WER edge case: empty reference."""

    def test_empty_ref_returns_num_edits(self, empty_dataset):
        """Test that empty reference returns num_edits as float."""
        empty_dataset.add("", "hello world")
        example = empty_dataset[0]
        wer = example.metrics.wer
        # Empty ref with non-empty hyp: should return num_edits as value
        # Since we can't divide by 0
        assert wer.ref_length == 0
        assert wer.value == float(wer.num_edits)


class TestWERDatasetMetric:
    """Tests for WER (dataset-level Metric) class."""

    def test_num_edits_aggregates(self, sample_dataset):
        """Test that dataset num_edits aggregates example values."""
        wer = sample_dataset.metrics.wer
        # Sum of all example num_edits
        expected = sum(ex.metrics.wer.num_edits for ex in sample_dataset)
        assert wer.num_edits == expected

    def test_ref_length_aggregates(self, sample_dataset):
        """Test that dataset ref_length aggregates example values."""
        wer = sample_dataset.metrics.wer
        # Sum of all example ref_lengths
        expected = sum(ex.metrics.wer.ref_length for ex in sample_dataset)
        assert wer.ref_length == expected

    def test_value_calculation(self, sample_dataset):
        """Test dataset-level WER value calculation."""
        wer = sample_dataset.metrics.wer
        # WER = total_edits / total_ref_tokens
        expected = wer.num_edits / wer.ref_length
        assert wer.value == expected

    def test_value_perfect_match_dataset(self, dataset_perfect_match):
        """Test WER is 0 for dataset with all perfect matches."""
        wer = dataset_perfect_match.metrics.wer
        assert wer.value == 0.0

    def test_empty_dataset(self, empty_dataset):
        """Test WER on empty dataset."""
        wer = empty_dataset.metrics.wer
        assert wer.ref_length == 0
        assert wer.num_edits == 0


class TestWERMetricAttributes:
    """Tests for WER metric attributes."""

    def test_short_name(self, sample_dataset):
        """Test WER short_name."""
        wer = sample_dataset.metrics.wer
        assert wer.short_name == "WER"

    def test_long_name(self, sample_dataset):
        """Test WER long_name."""
        wer = sample_dataset.metrics.wer
        assert wer.long_name == "Word Error Rate"

    def test_description(self, sample_dataset):
        """Test WER has description."""
        wer = sample_dataset.metrics.wer
        assert len(wer.description) > 0

    def test_example_cls(self, sample_dataset):
        """Test WER has example_cls set."""
        wer = sample_dataset.metrics.wer
        assert wer.example_cls == WER_


class TestWERMetricValues:
    """Tests for WER metric_values method."""

    def test_metric_values_main(self):
        """Test that value is the main metric."""
        values = WER.metric_values()
        assert values["main"] == "value"

    def test_metric_values_other(self):
        """Test that num_edits and ref_length are other metrics."""
        values = WER.metric_values()
        assert "num_edits" in values["other"]
        assert "ref_length" in values["other"]

    def test_example_metric_values(self):
        """Test WER_ metric_values."""
        values = WER_.metric_values()
        assert values["main"] == "value"
        assert "num_edits" in values["other"]
        assert "ref_length" in values["other"]
