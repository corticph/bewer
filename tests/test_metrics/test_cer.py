"""Tests for bewer.metrics.cer module."""

from bewer.metrics.cer import CER, CER_


class TestCERExampleMetric:
    """Tests for CER_ (ExampleMetric) class."""

    def test_num_edits_perfect_match(self, dataset_perfect_match):
        """Test num_edits is 0 for perfect match."""
        example = dataset_perfect_match[0]
        cer = example.metrics.cer()
        assert cer.num_edits == 0

    def test_num_edits_with_errors(self, sample_dataset):
        """Test num_edits counts character errors."""
        # "the quick brown fox" vs "the quick brown dog"
        # fox -> dog = 2 substitutions (f->d, o->o, x->g) = 2 char edits
        example = sample_dataset[1]
        cer = example.metrics.cer()
        assert cer.num_edits > 0

    def test_ref_length(self, sample_dataset):
        """Test ref_length returns correct character count."""
        example = sample_dataset[0]  # "hello world"
        cer = example.metrics.cer()
        # Joined normalized text character count
        assert cer.ref_length > 0

    def test_value_perfect_match(self, dataset_perfect_match):
        """Test CER value is 0 for perfect match."""
        example = dataset_perfect_match[0]
        cer = example.metrics.cer()
        assert cer.value == 0.0

    def test_value_with_error(self, sample_dataset):
        """Test CER value calculation with errors."""
        # "the quick brown fox" vs "the quick brown dog"
        example = sample_dataset[1]
        cer = example.metrics.cer()
        # Value should be between 0 and 1 for partial errors
        assert 0 < cer.value < 1


class TestCEREmptyReference:
    """Tests for CER edge case: empty reference."""

    def test_empty_ref_returns_num_edits(self, empty_dataset):
        """Test that empty reference returns num_edits as float."""
        empty_dataset.add("", "hello")
        example = empty_dataset[0]
        cer = example.metrics.cer()
        # Empty ref with non-empty hyp
        assert cer.ref_length == 0
        assert cer.value == float(cer.num_edits)


class TestCERDatasetMetric:
    """Tests for CER (dataset-level Metric) class."""

    def test_num_edits_aggregates(self, sample_dataset):
        """Test that dataset num_edits aggregates example values."""
        cer = sample_dataset.metrics.cer()
        # Sum of all example num_edits
        expected = sum(ex.metrics.cer().num_edits for ex in sample_dataset)
        assert cer.num_edits == expected

    def test_ref_length_aggregates(self, sample_dataset):
        """Test that dataset ref_length aggregates example values."""
        cer = sample_dataset.metrics.cer()
        # Sum of all example ref_lengths
        expected = sum(ex.metrics.cer().ref_length for ex in sample_dataset)
        assert cer.ref_length == expected

    def test_value_calculation(self, sample_dataset):
        """Test dataset-level CER value calculation."""
        cer = sample_dataset.metrics.cer()
        # CER = total_edits / total_ref_chars
        expected = cer.num_edits / cer.ref_length
        assert cer.value == expected

    def test_value_perfect_match_dataset(self, dataset_perfect_match):
        """Test CER is 0 for dataset with all perfect matches."""
        cer = dataset_perfect_match.metrics.cer()
        assert cer.value == 0.0

    def test_empty_dataset(self, empty_dataset):
        """Test CER on empty dataset."""
        cer = empty_dataset.metrics.cer()
        assert cer.ref_length == 0
        assert cer.num_edits == 0


class TestCERMetricAttributes:
    """Tests for CER metric attributes."""

    def test_short_name(self, sample_dataset):
        """Test CER short_name."""
        cer = sample_dataset.metrics.cer()
        assert cer.short_name == "CER"

    def test_long_name(self, sample_dataset):
        """Test CER long_name."""
        cer = sample_dataset.metrics.cer()
        assert cer.long_name == "Character Error Rate"

    def test_description(self, sample_dataset):
        """Test CER has description."""
        cer = sample_dataset.metrics.cer()
        assert len(cer.description) > 0

    def test_example_cls(self, sample_dataset):
        """Test CER has example_cls set."""
        cer = sample_dataset.metrics.cer()
        assert cer.example_cls == CER_


class TestCERMetricValues:
    """Tests for CER metric_values method."""

    def test_metric_values_main(self):
        """Test that value is the main metric."""
        values = CER.metric_values()
        assert values["main"] == "value"

    def test_metric_values_other(self):
        """Test that num_edits and ref_length are other metrics."""
        values = CER.metric_values()
        assert "num_edits" in values["other"]
        assert "ref_length" in values["other"]

    def test_example_metric_values(self):
        """Test CER_ metric_values."""
        values = CER_.metric_values()
        assert values["main"] == "value"
        assert "num_edits" in values["other"]
        assert "ref_length" in values["other"]


class TestCERVsWER:
    """Tests comparing CER and WER behavior."""

    def test_cer_typically_lower_than_wer(self, sample_dataset):
        """Test that CER is typically lower than WER for minor errors."""
        # For a single word substitution, WER counts it as 1 word error
        # but CER only counts the character differences
        example = sample_dataset[1]  # fox -> dog
        cer = example.metrics.cer().value
        wer = example.metrics.wer().value
        # CER should be lower because most characters match
        assert cer < wer

    def test_cer_wer_both_zero_for_match(self, dataset_perfect_match):
        """Test both CER and WER are 0 for perfect match."""
        example = dataset_perfect_match[0]
        assert example.metrics.cer().value == 0.0
        assert example.metrics.wer().value == 0.0
