"""Tests for bewer.metrics.hallucination.example_insertion_rate module."""

import pytest

from bewer.metrics.hallucination.example_insertion_rate import ExampleInsertionRate, ExampleInsertionRate_


@pytest.fixture
def dataset_insertions():
    """Dataset with various insertion patterns."""
    dataset = __import__("bewer").Dataset()
    dataset.add("hello world", "hello world")  # perfect match, 0 insertions
    dataset.add("hello world", "hello there world")  # 1 insertion (run of 1)
    dataset.add("hello world", "hello foo bar world")  # 2 insertions (run of 2)
    dataset.add("a b c d", "a x y b z c d")  # runs [2, 1]
    return dataset


@pytest.fixture
def dataset_all_insertions():
    """Dataset where hyp is pure insertion (empty ref)."""
    dataset = __import__("bewer").Dataset()
    dataset.add("", "hello world")
    return dataset


class TestExampleInsertionRateExampleMetric:
    """Tests for ExampleInsertionRate_ (ExampleMetric) class."""

    def test_num_insertions_perfect_match(self, dataset_perfect_match):
        """Test num_insertions is 0 for perfect match."""
        em = dataset_perfect_match[0].metrics.example_insertion_rate()
        assert em.num_insertions == 0

    def test_num_insertions_single_insertion(self, dataset_insertions):
        """Test num_insertions counts a single insertion."""
        em = dataset_insertions[1].metrics.example_insertion_rate()
        assert em.num_insertions == 1

    def test_num_insertions_contiguous_run(self, dataset_insertions):
        """Test num_insertions counts a contiguous run of 2."""
        em = dataset_insertions[2].metrics.example_insertion_rate()
        assert em.num_insertions == 2

    def test_has_insertions_perfect_match(self, dataset_perfect_match):
        """Test has_insertions is False for perfect match."""
        em = dataset_perfect_match[0].metrics.example_insertion_rate()
        assert em.has_insertions is False

    def test_has_insertions_single_insertion(self, dataset_insertions):
        """Test has_insertions is True for single insertion."""
        em = dataset_insertions[1].metrics.example_insertion_rate()
        assert em.has_insertions is True

    def test_has_insertions_contiguous_run(self, dataset_insertions):
        """Test has_insertions is True for contiguous run."""
        em = dataset_insertions[2].metrics.example_insertion_rate()
        assert em.has_insertions is True

    def test_has_insertions_min_run_length_2_excludes_singletons(self, dataset_insertions):
        """Test has_insertions with min_run_length=2 excludes single insertions."""
        em = dataset_insertions[1].metrics.example_insertion_rate(min_run_length=2)
        assert em.has_insertions is False

    def test_has_insertions_min_run_length_2_includes_run(self, dataset_insertions):
        """Test has_insertions with min_run_length=2 includes run of 2."""
        em = dataset_insertions[2].metrics.example_insertion_rate(min_run_length=2)
        assert em.has_insertions is True

    def test_has_insertions_min_run_length_3_excludes_run_of_2(self, dataset_insertions):
        """Test has_insertions with min_run_length=3 excludes run of 2."""
        em = dataset_insertions[2].metrics.example_insertion_rate(min_run_length=3)
        assert em.has_insertions is False


class TestExampleInsertionRateDatasetMetric:
    """Tests for ExampleInsertionRate (dataset-level Metric) class."""

    def test_value_default(self, dataset_insertions):
        """Test EIR value: 3 of 4 examples have insertions."""
        eir = dataset_insertions.metrics.example_insertion_rate()
        assert eir.value == pytest.approx(3 / 4)

    def test_value_min_run_length_2(self, dataset_insertions):
        """Test EIR with min_run_length=2: only runs >= 2 count."""
        eir = dataset_insertions.metrics.example_insertion_rate(min_run_length=2)
        # Example 2 (run of 2) and example 3 (run of 2) qualify = 2/4
        assert eir.value == pytest.approx(2 / 4)

    def test_value_min_run_length_3(self, dataset_insertions):
        """Test EIR with min_run_length=3: no runs >= 3."""
        eir = dataset_insertions.metrics.example_insertion_rate(min_run_length=3)
        assert eir.value == 0.0

    def test_value_perfect_match_dataset(self, dataset_perfect_match):
        """Test EIR is 0 for dataset with all perfect matches."""
        eir = dataset_perfect_match.metrics.example_insertion_rate()
        assert eir.value == 0.0

    def test_value_all_insertions(self, dataset_all_insertions):
        """Test EIR is 1.0 when all examples have insertions."""
        eir = dataset_all_insertions.metrics.example_insertion_rate()
        assert eir.value == 1.0

    def test_num_examples_with_insertions(self, dataset_insertions):
        """Test num_examples_with_insertions."""
        eir = dataset_insertions.metrics.example_insertion_rate()
        assert eir.num_examples_with_insertions == 3

    def test_num_examples_with_insertions_min_run_length_2(self, dataset_insertions):
        """Test num_examples_with_insertions with min_run_length=2."""
        eir = dataset_insertions.metrics.example_insertion_rate(min_run_length=2)
        assert eir.num_examples_with_insertions == 2

    def test_num_examples(self, dataset_insertions):
        """Test num_examples returns total count."""
        eir = dataset_insertions.metrics.example_insertion_rate()
        assert eir.num_examples == 4

    def test_empty_dataset(self, empty_dataset):
        """Test EIR on empty dataset."""
        eir = empty_dataset.metrics.example_insertion_rate()
        assert eir.num_examples == 0
        assert eir.num_examples_with_insertions == 0
        assert eir.value == 0.0


class TestExampleInsertionRateErrorAlign:
    """Tests for ExampleInsertionRate with alignment='error_align'."""

    def test_value_matches_levenshtein(self, dataset_insertions):
        """Test error_align gives same result as levenshtein."""
        eir_ea = dataset_insertions.metrics.example_insertion_rate(alignment="error_align")
        eir_lev = dataset_insertions.metrics.example_insertion_rate(alignment="levenshtein")
        assert eir_ea.value == eir_lev.value
        assert eir_ea.num_examples_with_insertions == eir_lev.num_examples_with_insertions

    def test_value_matches_levenshtein_min_run_length_2(self, dataset_insertions):
        """Test error_align with min_run_length=2 matches levenshtein."""
        eir_ea = dataset_insertions.metrics.example_insertion_rate(alignment="error_align", min_run_length=2)
        eir_lev = dataset_insertions.metrics.example_insertion_rate(alignment="levenshtein", min_run_length=2)
        assert eir_ea.value == eir_lev.value


class TestExampleInsertionRateMetricAttributes:
    """Tests for ExampleInsertionRate metric attributes."""

    def test_short_name(self, sample_dataset):
        """Test EIR short_name."""
        eir = sample_dataset.metrics.example_insertion_rate()
        assert eir.short_name_base == "EIR"

    def test_long_name(self, sample_dataset):
        """Test EIR long_name."""
        eir = sample_dataset.metrics.example_insertion_rate()
        assert eir.long_name_base == "Example Insertion Rate"

    def test_description(self, sample_dataset):
        """Test EIR has description."""
        eir = sample_dataset.metrics.example_insertion_rate()
        assert len(eir.description) > 0

    def test_example_cls(self, sample_dataset):
        """Test EIR has example_cls set."""
        eir = sample_dataset.metrics.example_insertion_rate()
        assert eir.example_cls == ExampleInsertionRate_

    def test_short_name_includes_params(self, sample_dataset):
        """Test short_name includes parameters."""
        eir = sample_dataset.metrics.example_insertion_rate(min_run_length=3)
        assert "min_run_length=3" in eir.short_name


class TestExampleInsertionRateMetricValues:
    """Tests for ExampleInsertionRate metric_values method."""

    def test_metric_values_main(self):
        """Test that value is the main metric."""
        values = ExampleInsertionRate.metric_values()
        assert values["main"] == "value"

    def test_metric_values_other(self):
        """Test that other metric values are present."""
        values = ExampleInsertionRate.metric_values()
        assert "num_examples_with_insertions" in values["other"]
        assert "num_examples" in values["other"]

    def test_example_metric_values(self):
        """Test ExampleInsertionRate_ metric_values."""
        values = ExampleInsertionRate_.metric_values()
        assert values["main"] is None
        assert "num_insertions" in values["other"]
        assert "has_insertions" in values["other"]


class TestExampleInsertionRateValidation:
    """Tests for ExampleInsertionRate parameter validation."""

    def test_min_run_length_zero_raises(self, sample_dataset):
        """Test that min_run_length=0 raises ValueError."""
        with pytest.raises(ValueError, match="min_run_length must be >= 1"):
            sample_dataset.metrics.example_insertion_rate(min_run_length=0)

    def test_min_run_length_negative_raises(self, sample_dataset):
        """Test that negative min_run_length raises ValueError."""
        with pytest.raises(ValueError, match="min_run_length must be >= 1"):
            sample_dataset.metrics.example_insertion_rate(min_run_length=-1)

    def test_invalid_alignment_raises(self, sample_dataset):
        """Test that invalid alignment value raises ValueError."""
        with pytest.raises(ValueError, match="alignment must be 'levenshtein' or 'error_align'"):
            sample_dataset.metrics.example_insertion_rate(alignment="foo")
