"""Tests for bewer.metrics.general.mer module."""

import pytest

from bewer.core.dataset import Dataset
from bewer.metrics.general.mer import MER, MER_


@pytest.fixture
def dataset_insertions():
    """Dataset with insertion-heavy hypotheses, where MER diverges from WER."""
    dataset = Dataset()
    dataset.add("hello", "hello there friend")  # 2 insertions: WER 2.0, MER 2/3
    dataset.add("hello world", "hello there world")  # 1 insertion: WER 0.5, MER 1/3
    return dataset


@pytest.fixture
def dataset_empty_texts():
    """Dataset with empty reference and/or hypothesis edge cases."""
    dataset = Dataset()
    dataset.add("", "silence")  # empty ref, non-empty hyp
    dataset.add("hello", "")  # non-empty ref, empty hyp
    dataset.add("", "")  # both empty
    return dataset


class TestMERExampleMetric:
    """Tests for MER_ (ExampleMetric) class."""

    def test_num_edits_perfect_match(self, dataset_perfect_match):
        """Test num_edits is 0 for perfect match."""
        example = dataset_perfect_match[0]
        mer = example.metrics.mer()
        assert mer.num_edits == 0

    def test_num_edits_with_errors(self, sample_dataset):
        """Test num_edits counts errors correctly."""
        # "the quick brown fox" vs "the quick brown dog" - 1 substitution
        example = sample_dataset[1]
        mer = example.metrics.mer()
        assert mer.num_edits == 1

    def test_num_edits_deletion(self, sample_dataset):
        """Test num_edits counts deletions."""
        # "testing one two three" vs "testing one two" - 1 deletion
        example = sample_dataset[2]
        mer = example.metrics.mer()
        assert mer.num_edits == 1

    def test_num_matches_perfect_match(self, sample_dataset):
        """Test num_matches returns matched token count."""
        # "hello world" - 2 matched tokens
        example = sample_dataset[0]
        mer = example.metrics.mer()
        assert mer.num_matches == 2

    def test_num_matches_with_substitution(self, sample_dataset):
        """Test num_matches excludes substituted tokens."""
        # "the quick brown fox" vs "the quick brown dog" - 3 matches
        example = sample_dataset[1]
        mer = example.metrics.mer()
        assert mer.num_matches == 3

    def test_value_perfect_match(self, dataset_perfect_match):
        """Test MER value is 0 for perfect match."""
        example = dataset_perfect_match[0]
        mer = example.metrics.mer()
        assert mer.value == 0.0

    def test_value_with_substitution(self, sample_dataset):
        """Test MER value with a substitution."""
        # 1 edit / (1 edit + 3 matches) = 0.25
        example = sample_dataset[1]
        mer = example.metrics.mer()
        assert mer.value == 0.25

    def test_value_with_deletion(self, sample_dataset):
        """Test MER value with a deletion."""
        # 1 edit / (1 edit + 3 matches) = 0.25
        example = sample_dataset[2]
        mer = example.metrics.mer()
        assert mer.value == 0.25

    def test_value_complete_mismatch(self, dataset_with_errors):
        """Test MER value for complete mismatch."""
        # "hello" vs "goodbye" - 0 matches, 1 substitution = 1.0
        example = dataset_with_errors[0]
        mer = example.metrics.mer()
        assert mer.value == 1.0

    def test_value_with_insertions(self, dataset_insertions):
        """Test MER value counts insertions in the denominator, unlike WER."""
        # "hello" vs "hello there friend" - 2 insertions: 2 / (2 + 1) = 2/3
        example = dataset_insertions[0]
        mer = example.metrics.mer()
        assert mer.value == pytest.approx(2 / 3)
        # WER exceeds 1 here while MER cannot
        assert example.metrics.wer().value == 2.0

    def test_value_empty_ref(self, dataset_empty_texts):
        """Test MER value is 1 when the reference is empty and the hypothesis is not."""
        example = dataset_empty_texts[0]
        mer = example.metrics.mer()
        assert mer.value == 1.0

    def test_value_empty_hyp(self, dataset_empty_texts):
        """Test MER value is 1 when the hypothesis is empty and the reference is not."""
        example = dataset_empty_texts[1]
        mer = example.metrics.mer()
        assert mer.value == 1.0

    def test_value_both_empty(self, dataset_empty_texts):
        """Test MER value is 0 when both texts are empty."""
        example = dataset_empty_texts[2]
        mer = example.metrics.mer()
        assert mer.value == 0.0


class TestMERDatasetMetric:
    """Tests for MER (dataset-level Metric) class."""

    def test_num_edits_aggregates(self, sample_dataset):
        """Test that dataset num_edits aggregates example values."""
        mer = sample_dataset.metrics.mer()
        expected = sum(ex.metrics.mer().num_edits for ex in sample_dataset)
        assert mer.num_edits == expected

    def test_num_matches_aggregates(self, sample_dataset):
        """Test that dataset num_matches aggregates example values."""
        mer = sample_dataset.metrics.mer()
        expected = sum(ex.metrics.mer().num_matches for ex in sample_dataset)
        assert mer.num_matches == expected

    def test_value_calculation(self, sample_dataset):
        """Test dataset-level MER value is the ratio of summed counts."""
        # 2 edits / (2 edits + 8 matches) = 0.2, matching jiwer's corpus-level MER
        mer = sample_dataset.metrics.mer()
        assert mer.value == 0.2
        assert mer.value == mer.num_edits / (mer.num_edits + mer.num_matches)

    def test_value_is_micro_averaged(self, sample_dataset):
        """Test dataset MER is not the mean of example MER values."""
        mer = sample_dataset.metrics.mer()
        example_mean = sum(ex.metrics.mer().value for ex in sample_dataset) / len(sample_dataset)
        assert mer.value != example_mean

    def test_value_perfect_match_dataset(self, dataset_perfect_match):
        """Test MER is 0 for dataset with all perfect matches."""
        mer = dataset_perfect_match.metrics.mer()
        assert mer.value == 0.0

    def test_empty_dataset(self, empty_dataset):
        """Test MER on empty dataset."""
        mer = empty_dataset.metrics.mer()
        assert mer.num_edits == 0
        assert mer.num_matches == 0
        assert mer.value == 0.0

    def test_value_matches_wer_identity(self, sample_dataset):
        """Test MER equals the WER-based identity num_edits / (ref_length + insertions)."""
        mer = sample_dataset.metrics.mer()
        wer = sample_dataset.metrics.wer()
        num_insertions = sum(ex.metrics.levenshtein().num_insertions for ex in sample_dataset)
        assert mer.value == pytest.approx(wer.num_edits / (wer.ref_length + num_insertions))


class TestMERNoneNormalizer:
    """Tests for MER with normalizer="none" (standardized tokens)."""

    def test_value_unnormalized(self, sample_dataset):
        """Test MER value with unnormalized tokens."""
        mer = sample_dataset.metrics.mer(normalizer="none")
        expected = sum(ex.metrics.mer(normalizer="none").num_edits for ex in sample_dataset)
        matches = sum(ex.metrics.mer(normalizer="none").num_matches for ex in sample_dataset)
        assert mer.num_edits == expected
        assert mer.num_matches == matches
        assert mer.value == expected / (expected + matches)


class TestMERMetricAttributes:
    """Tests for MER metric attributes."""

    def test_short_name(self, sample_dataset):
        """Test MER short_name."""
        mer = sample_dataset.metrics.mer()
        assert mer.short_name_base == "MER"

    def test_long_name(self, sample_dataset):
        """Test MER long_name."""
        mer = sample_dataset.metrics.mer()
        assert mer.long_name_base == "Match Error Rate"

    def test_description(self, sample_dataset):
        """Test MER has description."""
        mer = sample_dataset.metrics.mer()
        assert len(mer.description) > 0

    def test_example_cls(self, sample_dataset):
        """Test MER has example_cls set."""
        mer = sample_dataset.metrics.mer()
        assert mer.example_cls == MER_


class TestMERMetricValues:
    """Tests for MER metric_values method."""

    def test_metric_values_main(self):
        """Test that value is the main metric."""
        values = MER.metric_values()
        assert values["main"] == "value"

    def test_metric_values_other(self):
        """Test that num_edits and num_matches are other metrics."""
        values = MER.metric_values()
        assert "num_edits" in values["other"]
        assert "num_matches" in values["other"]

    def test_example_metric_values(self):
        """Test MER_ metric_values."""
        values = MER_.metric_values()
        assert values["main"] == "value"
        assert "num_edits" in values["other"]
        assert "num_matches" in values["other"]


class TestMERAlignment:
    def test_counts_from_selected_alignment(self, empty_dataset):
        """error_align splits `bedroom` across `bed room` and keeps `is` matched; Levenshtein does not."""
        empty_dataset.add("the bed room is big", "the bedroom is a big")
        assert empty_dataset.metrics.mer().value == pytest.approx(3 / 5)  # 3 subs, 2 matches
        assert empty_dataset.metrics.mer(alignment="error_align").value == pytest.approx(
            3 / 6
        )  # 2 subs + 1 ins, 3 matches
