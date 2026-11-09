"""Tests for bewer.metrics.general.wip module."""

import pytest

from bewer.core.dataset import Dataset
from bewer.metrics.general.wip import WIP, WIP_


@pytest.fixture
def dataset_empty_texts():
    """Dataset with empty reference and/or hypothesis edge cases."""
    dataset = Dataset()
    dataset.add("", "silence")  # empty ref, non-empty hyp
    dataset.add("hello", "")  # non-empty ref, empty hyp
    dataset.add("", "")  # both empty
    return dataset


class TestWIPExampleMetric:
    """Tests for WIP_ (ExampleMetric) class."""

    def test_num_matches_perfect_match(self, sample_dataset):
        """Test num_matches returns matched token count."""
        # "hello world" - 2 matched tokens
        example = sample_dataset[0]
        wip = example.metrics.wip()
        assert wip.num_matches == 2

    def test_num_matches_with_substitution(self, sample_dataset):
        """Test num_matches excludes substituted tokens."""
        # "the quick brown fox" vs "the quick brown dog" - 3 matches
        example = sample_dataset[1]
        wip = example.metrics.wip()
        assert wip.num_matches == 3

    def test_ref_length(self, sample_dataset):
        """Test ref_length returns correct token count."""
        # "hello world" - 2 tokens
        example = sample_dataset[0]
        wip = example.metrics.wip()
        assert wip.ref_length == 2

    def test_hyp_length_with_deletion(self, sample_dataset):
        """Test hyp_length returns the hypothesis token count."""
        # "testing one two three" vs "testing one two" - 3 hyp tokens
        example = sample_dataset[2]
        wip = example.metrics.wip()
        assert wip.hyp_length == 3

    def test_value_perfect_match(self, dataset_perfect_match):
        """Test WIP value is 1 for perfect match."""
        example = dataset_perfect_match[0]
        wip = example.metrics.wip()
        assert wip.value == 1.0

    def test_value_with_substitution(self, sample_dataset):
        """Test WIP value with a substitution."""
        # (3/4) * (3/4) = 0.5625
        example = sample_dataset[1]
        wip = example.metrics.wip()
        assert wip.value == 0.5625

    def test_value_with_deletion(self, sample_dataset):
        """Test WIP value with a deletion."""
        # (3/4) * (3/3) = 0.75
        example = sample_dataset[2]
        wip = example.metrics.wip()
        assert wip.value == 0.75

    def test_value_complete_mismatch(self, dataset_with_errors):
        """Test WIP value for complete mismatch."""
        # "hello" vs "goodbye" - 0 matches = 0.0
        example = dataset_with_errors[0]
        wip = example.metrics.wip()
        assert wip.value == 0.0

    def test_value_empty_ref(self, dataset_empty_texts):
        """Test WIP value is 0 when the reference is empty and the hypothesis is not."""
        example = dataset_empty_texts[0]
        wip = example.metrics.wip()
        assert wip.value == 0.0

    def test_value_empty_hyp(self, dataset_empty_texts):
        """Test WIP value is 0 when the hypothesis is empty and the reference is not."""
        example = dataset_empty_texts[1]
        wip = example.metrics.wip()
        assert wip.value == 0.0

    def test_value_both_empty(self, dataset_empty_texts):
        """Test WIP value is 1 when both texts are empty."""
        example = dataset_empty_texts[2]
        wip = example.metrics.wip()
        assert wip.value == 1.0


class TestWIPDatasetMetric:
    """Tests for WIP (dataset-level Metric) class."""

    def test_num_matches_aggregates(self, sample_dataset):
        """Test that dataset num_matches aggregates example values."""
        wip = sample_dataset.metrics.wip()
        expected = sum(ex.metrics.wip().num_matches for ex in sample_dataset)
        assert wip.num_matches == expected

    def test_ref_length_aggregates(self, sample_dataset):
        """Test that dataset ref_length aggregates example values."""
        wip = sample_dataset.metrics.wip()
        expected = sum(ex.metrics.wip().ref_length for ex in sample_dataset)
        assert wip.ref_length == expected

    def test_hyp_length_aggregates(self, sample_dataset):
        """Test that dataset hyp_length aggregates example values."""
        wip = sample_dataset.metrics.wip()
        expected = sum(ex.metrics.wip().hyp_length for ex in sample_dataset)
        assert wip.hyp_length == expected

    def test_value_calculation(self, sample_dataset):
        """Test dataset-level WIP value is the product of matched shares."""
        # (8/10) * (8/9), matching jiwer's corpus-level WIP
        wip = sample_dataset.metrics.wip()
        assert wip.value == pytest.approx((8 / 10) * (8 / 9))

    def test_value_perfect_match_dataset(self, dataset_perfect_match):
        """Test WIP is 1 for dataset with all perfect matches."""
        wip = dataset_perfect_match.metrics.wip()
        assert wip.value == 1.0

    def test_empty_dataset(self, empty_dataset):
        """Test WIP on empty dataset follows the both-empty convention."""
        wip = empty_dataset.metrics.wip()
        assert wip.num_matches == 0
        assert wip.ref_length == 0
        assert wip.hyp_length == 0
        assert wip.value == 1.0

    def test_value_bounds(self, sample_dataset, dataset_with_errors, dataset_perfect_match):
        """Test WIP never exceeds 1."""
        for dataset in (sample_dataset, dataset_with_errors, dataset_perfect_match):
            assert 0.0 <= dataset.metrics.wip().value <= 1.0


class TestWIPNormalizedFalse:
    """Tests for WIP with normalized=False."""

    def test_value_unnormalized(self, sample_dataset):
        """Test WIP value with unnormalized tokens."""
        wip = sample_dataset.metrics.wip(normalized=False)
        num_matches = sum(ex.metrics.wip(normalized=False).num_matches for ex in sample_dataset)
        ref_length = sum(ex.metrics.wip(normalized=False).ref_length for ex in sample_dataset)
        hyp_length = sum(ex.metrics.wip(normalized=False).hyp_length for ex in sample_dataset)
        assert wip.value == pytest.approx((num_matches / ref_length) * (num_matches / hyp_length))


class TestWIPMetricAttributes:
    """Tests for WIP metric attributes."""

    def test_short_name(self, sample_dataset):
        """Test WIP short_name."""
        wip = sample_dataset.metrics.wip()
        assert wip.short_name_base == "WIP"

    def test_long_name(self, sample_dataset):
        """Test WIP long_name."""
        wip = sample_dataset.metrics.wip()
        assert wip.long_name_base == "Word Information Preserved"

    def test_description(self, sample_dataset):
        """Test WIP has description."""
        wip = sample_dataset.metrics.wip()
        assert len(wip.description) > 0

    def test_example_cls(self, sample_dataset):
        """Test WIP has example_cls set."""
        wip = sample_dataset.metrics.wip()
        assert wip.example_cls == WIP_


class TestWIPMetricValues:
    """Tests for WIP metric_values method."""

    def test_metric_values_main(self):
        """Test that value is the main metric."""
        values = WIP.metric_values()
        assert values["main"] == "value"

    def test_metric_values_other(self):
        """Test that num_matches, ref_length and hyp_length are other metrics."""
        values = WIP.metric_values()
        assert "num_matches" in values["other"]
        assert "ref_length" in values["other"]
        assert "hyp_length" in values["other"]

    def test_example_metric_values(self):
        """Test WIP_ metric_values."""
        values = WIP_.metric_values()
        assert values["main"] == "value"
        assert "num_matches" in values["other"]
        assert "ref_length" in values["other"]
        assert "hyp_length" in values["other"]
