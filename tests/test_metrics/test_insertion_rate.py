"""Tests for bewer.metrics.insertion_rate module."""

import pytest

from bewer.metrics.insertion_rate import InsertionRate, InsertionRate_


@pytest.fixture
def dataset_insertions():
    """Dataset with various insertion patterns."""
    dataset = __import__("bewer").Dataset()
    dataset.add("hello world", "hello world")  # perfect match
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


class TestInsertionRateExampleMetric:
    """Tests for InsertionRate_ (ExampleMetric) class."""

    def test_num_insertions_perfect_match(self, dataset_perfect_match):
        """Test num_insertions is 0 for perfect match."""
        em = dataset_perfect_match[0].metrics.insertion_rate()
        assert em.num_insertions == 0

    def test_num_insertions_single_insertion(self, dataset_insertions):
        """Test num_insertions counts a single insertion."""
        em = dataset_insertions[1].metrics.insertion_rate()
        assert em.num_insertions == 1

    def test_num_insertions_contiguous_run(self, dataset_insertions):
        """Test num_insertions counts a contiguous run of 2."""
        em = dataset_insertions[2].metrics.insertion_rate()
        assert em.num_insertions == 2

    def test_num_insertions_multiple_runs(self, dataset_insertions):
        """Test num_insertions counts across multiple runs."""
        em = dataset_insertions[3].metrics.insertion_rate()
        assert em.num_insertions == 3

    def test_num_insertions_run_length_2_excludes_singletons(self, dataset_insertions):
        """Test num_insertions with run_length=2 excludes single insertions."""
        em = dataset_insertions[1].metrics.insertion_rate(run_length=2)
        assert em.num_insertions == 0

    def test_num_insertions_run_length_2_includes_run(self, dataset_insertions):
        """Test num_insertions with run_length=2 includes run of 2."""
        em = dataset_insertions[2].metrics.insertion_rate(run_length=2)
        assert em.num_insertions == 2

    def test_num_insertions_run_length_2_multiple_runs(self, dataset_insertions):
        """Test num_insertions with run_length=2 counts only runs >= 2."""
        em = dataset_insertions[3].metrics.insertion_rate(run_length=2)
        # runs [2, 1] -> only the run of 2 counts = 2
        assert em.num_insertions == 2

    def test_ref_length_perfect_match(self, dataset_perfect_match):
        """Test ref_length for perfect match."""
        em = dataset_perfect_match[0].metrics.insertion_rate()
        assert em.ref_length == 2

    def test_ref_length_with_insertions(self, dataset_insertions):
        """Test ref_length is the number of reference tokens."""
        em = dataset_insertions[1].metrics.insertion_rate()
        assert em.ref_length == 2

    def test_value_perfect_match(self, dataset_perfect_match):
        """Test value is 0.0 for perfect match."""
        em = dataset_perfect_match[0].metrics.insertion_rate()
        assert em.value == 0.0

    def test_value_single_insertion_default(self, dataset_insertions):
        """Test value with run_length=1 (default) counts all insertions."""
        em = dataset_insertions[1].metrics.insertion_rate()
        # 1 insertion / 2 ref tokens = 0.5
        assert em.value == 0.5

    def test_value_single_insertion_run_length_2(self, dataset_insertions):
        """Test value with run_length=2 excludes single insertions."""
        em = dataset_insertions[1].metrics.insertion_rate(run_length=2)
        # 0 qualifying insertions / 2 ref tokens = 0.0
        assert em.value == 0.0

    def test_value_contiguous_run_default(self, dataset_insertions):
        """Test value with run_length=1 counts all insertions."""
        em = dataset_insertions[2].metrics.insertion_rate()
        # 2 insertions / 2 ref tokens = 1.0
        assert em.value == 1.0

    def test_value_contiguous_run_run_length_2(self, dataset_insertions):
        """Test value with run_length=2 includes run of 2."""
        em = dataset_insertions[2].metrics.insertion_rate(run_length=2)
        # 2 qualifying / 2 ref tokens = 1.0
        assert em.value == 1.0

    def test_value_contiguous_run_run_length_3(self, dataset_insertions):
        """Test value with run_length=3 excludes run of 2."""
        em = dataset_insertions[2].metrics.insertion_rate(run_length=3)
        assert em.value == 0.0

    def test_value_multiple_runs_default(self, dataset_insertions):
        """Test value with run_length=1 counts all insertions."""
        em = dataset_insertions[3].metrics.insertion_rate()
        # 3 insertions / 4 ref tokens = 3/4
        assert em.value == pytest.approx(3 / 4)

    def test_value_multiple_runs_run_length_2(self, dataset_insertions):
        """Test value with run_length=2 counts only run of 2."""
        em = dataset_insertions[3].metrics.insertion_rate(run_length=2)
        # 2 qualifying / 4 ref tokens = 1/2
        assert em.value == 0.5

    def test_value_all_insertions(self, dataset_all_insertions):
        """Test value when ref is empty (pure insertion)."""
        em = dataset_all_insertions[0].metrics.insertion_rate()
        # 2 insertions / 0 ref tokens -> return raw count
        assert em.value == 2.0


class TestInsertionRateEmptyReference:
    """Tests for IR edge case: empty reference and hypothesis."""

    def test_empty_ref_and_hyp(self, empty_dataset):
        """Test that empty ref and hyp gives 0.0."""
        empty_dataset.add("", "")
        em = empty_dataset[0].metrics.insertion_rate()
        assert em.ref_length == 0
        assert em.value == 0.0


class TestInsertionRateDatasetMetric:
    """Tests for InsertionRate (dataset-level Metric) class."""

    def test_num_insertions_aggregates(self, dataset_insertions):
        """Test that dataset num_insertions aggregates example values."""
        ir = dataset_insertions.metrics.insertion_rate()
        expected = sum(ex.metrics.insertion_rate().num_insertions for ex in dataset_insertions)
        assert ir.num_insertions == expected

    def test_num_insertions_run_length_2(self, dataset_insertions):
        """Test num_insertions with run_length=2 excludes singletons."""
        ir = dataset_insertions.metrics.insertion_rate(run_length=2)
        # Only runs >= 2: example 2 (run of 2) + example 3 (run of 2) = 2 + 2 = 4
        assert ir.num_insertions == 4

    def test_ref_length_aggregates(self, dataset_insertions):
        """Test that dataset ref_length aggregates example values."""
        ir = dataset_insertions.metrics.insertion_rate()
        expected = sum(ex.metrics.insertion_rate().ref_length for ex in dataset_insertions)
        assert ir.ref_length == expected

    def test_value_calculation(self, dataset_insertions):
        """Test dataset-level IR value calculation."""
        ir = dataset_insertions.metrics.insertion_rate()
        assert ir.value == ir.num_insertions / ir.ref_length

    def test_value_perfect_match_dataset(self, dataset_perfect_match):
        """Test IR is 0 for dataset with all perfect matches."""
        ir = dataset_perfect_match.metrics.insertion_rate()
        assert ir.value == 0.0

    def test_value_run_length_2(self, dataset_insertions):
        """Test dataset-level value with run_length=2."""
        ir = dataset_insertions.metrics.insertion_rate(run_length=2)
        # num_insertions=4, ref_length=2+2+2+4=10
        assert ir.value == pytest.approx(4 / 10)

    def test_empty_dataset(self, empty_dataset):
        """Test IR on empty dataset."""
        ir = empty_dataset.metrics.insertion_rate()
        assert ir.ref_length == 0
        assert ir.num_insertions == 0
        assert ir.value == 0.0


class TestInsertionRateMetricAttributes:
    """Tests for InsertionRate metric attributes."""

    def test_short_name(self, sample_dataset):
        """Test IR short_name."""
        ir = sample_dataset.metrics.insertion_rate()
        assert ir.short_name_base == "IR"

    def test_long_name(self, sample_dataset):
        """Test IR long_name."""
        ir = sample_dataset.metrics.insertion_rate()
        assert ir.long_name_base == "Insertion Rate"

    def test_description(self, sample_dataset):
        """Test IR has description."""
        ir = sample_dataset.metrics.insertion_rate()
        assert len(ir.description) > 0

    def test_example_cls(self, sample_dataset):
        """Test IR has example_cls set."""
        ir = sample_dataset.metrics.insertion_rate()
        assert ir.example_cls == InsertionRate_

    def test_short_name_includes_params(self, sample_dataset):
        """Test short_name includes parameters."""
        ir = sample_dataset.metrics.insertion_rate(run_length=3)
        assert "run_length=3" in ir.short_name

    def test_long_name_includes_params(self, sample_dataset):
        """Test long_name includes parameters."""
        ir = sample_dataset.metrics.insertion_rate(run_length=3)
        assert "run_length=3" in ir.long_name


class TestInsertionRateMetricValues:
    """Tests for InsertionRate metric_values method."""

    def test_metric_values_main(self):
        """Test that value is the main metric."""
        values = InsertionRate.metric_values()
        assert values["main"] == "value"

    def test_metric_values_other(self):
        """Test that other metric values are present."""
        values = InsertionRate.metric_values()
        assert "num_insertions" in values["other"]
        assert "ref_length" in values["other"]

    def test_example_metric_values(self):
        """Test InsertionRate_ metric_values."""
        values = InsertionRate_.metric_values()
        assert values["main"] == "value"
        assert "num_insertions" in values["other"]
        assert "ref_length" in values["other"]


class TestInsertionRateValidation:
    """Tests for InsertionRate parameter validation."""

    def test_run_length_zero_raises(self, sample_dataset):
        """Test that run_length=0 raises ValueError."""
        with pytest.raises(ValueError, match="run_length must be >= 1"):
            sample_dataset.metrics.insertion_rate(run_length=0)

    def test_run_length_negative_raises(self, sample_dataset):
        """Test that negative run_length raises ValueError."""
        with pytest.raises(ValueError, match="run_length must be >= 1"):
            sample_dataset.metrics.insertion_rate(run_length=-1)
