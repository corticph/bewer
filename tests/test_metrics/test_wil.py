"""Tests for bewer.metrics.general.wil module."""

import pytest

from bewer.core.dataset import Dataset
from bewer.metrics.general.wil import WIL, WIL_


@pytest.fixture
def dataset_empty_texts():
    """Dataset with empty reference and/or hypothesis edge cases."""
    dataset = Dataset()
    dataset.add("", "silence")  # empty ref, non-empty hyp
    dataset.add("hello", "")  # non-empty ref, empty hyp
    dataset.add("", "")  # both empty
    return dataset


class TestWILExampleMetric:
    """Tests for WIL_ (ExampleMetric) class."""

    def test_value_perfect_match(self, dataset_perfect_match):
        """Test WIL value is 0 for perfect match."""
        example = dataset_perfect_match[0]
        wil = example.metrics.wil()
        assert wil.value == 0.0

    def test_value_with_substitution(self, sample_dataset):
        """Test WIL value with a substitution."""
        # 1 - 0.5625 = 0.4375
        example = sample_dataset[1]
        wil = example.metrics.wil()
        assert wil.value == 0.4375

    def test_value_with_deletion(self, sample_dataset):
        """Test WIL value with a deletion."""
        # 1 - 0.75 = 0.25
        example = sample_dataset[2]
        wil = example.metrics.wil()
        assert wil.value == 0.25

    def test_value_complete_mismatch(self, dataset_with_errors):
        """Test WIL value for complete mismatch."""
        # "hello" vs "goodbye" - 0 matches = 1.0
        example = dataset_with_errors[0]
        wil = example.metrics.wil()
        assert wil.value == 1.0

    def test_value_empty_ref(self, dataset_empty_texts):
        """Test WIL value is 1 when the reference is empty and the hypothesis is not."""
        example = dataset_empty_texts[0]
        wil = example.metrics.wil()
        assert wil.value == 1.0

    def test_value_empty_hyp(self, dataset_empty_texts):
        """Test WIL value is 1 when the hypothesis is empty and the reference is not."""
        example = dataset_empty_texts[1]
        wil = example.metrics.wil()
        assert wil.value == 1.0

    def test_value_both_empty(self, dataset_empty_texts):
        """Test WIL value is 0 when both texts are empty."""
        example = dataset_empty_texts[2]
        wil = example.metrics.wil()
        assert wil.value == 0.0


class TestWILIdentity:
    """Tests for the WIL + WIP = 1 identity."""

    def test_identity_example_level(self, sample_dataset, dataset_with_errors, dataset_empty_texts):
        """Test example-level WIL is 1 minus example-level WIP."""
        for dataset in (sample_dataset, dataset_with_errors, dataset_empty_texts):
            for example in dataset:
                wil = example.metrics.wil()
                wip = example.metrics.wip()
                assert wil.value == pytest.approx(1.0 - wip.value)

    def test_identity_dataset_level(self, sample_dataset, dataset_with_errors, dataset_empty_texts):
        """Test dataset-level WIL is 1 minus dataset-level WIP."""
        for dataset in (sample_dataset, dataset_with_errors, dataset_empty_texts):
            wil = dataset.metrics.wil()
            wip = dataset.metrics.wip()
            assert wil.value == pytest.approx(1.0 - wip.value)

    def test_shared_params(self, sample_dataset):
        """Test that WIL forwards its params and pipeline to the borrowed WIP metric."""
        wil = sample_dataset.metrics.wil(normalizer="none")
        wip = wil._wip
        assert wip.normalizer == "none"
        assert wip.pipeline == wil.pipeline

    def test_borrowed_wip_is_cached(self, sample_dataset):
        """Test that repeated WIL requests share the same borrowed WIP instance."""
        wil = sample_dataset.metrics.wil()
        assert wil._wip is sample_dataset.metrics.wip()
        assert wil._wip is sample_dataset.metrics.wil()._wip


class TestWILDatasetMetric:
    """Tests for WIL (dataset-level Metric) class."""

    def test_value_calculation(self, sample_dataset):
        """Test dataset-level WIL value."""
        # 1 - (8/10) * (8/9), matching jiwer's corpus-level WIL
        wil = sample_dataset.metrics.wil()
        assert wil.value == pytest.approx(1.0 - (8 / 10) * (8 / 9))

    def test_value_perfect_match_dataset(self, dataset_perfect_match):
        """Test WIL is 0 for dataset with all perfect matches."""
        wil = dataset_perfect_match.metrics.wil()
        assert wil.value == 0.0

    def test_empty_dataset(self, empty_dataset):
        """Test WIL on empty dataset follows the both-empty convention."""
        wil = empty_dataset.metrics.wil()
        assert wil.value == 0.0

    def test_value_bounds(self, sample_dataset, dataset_with_errors, dataset_perfect_match):
        """Test WIL never exceeds 1."""
        for dataset in (sample_dataset, dataset_with_errors, dataset_perfect_match):
            assert 0.0 <= dataset.metrics.wil().value <= 1.0


class TestWILMetricAttributes:
    """Tests for WIL metric attributes."""

    def test_short_name(self, sample_dataset):
        """Test WIL short_name."""
        wil = sample_dataset.metrics.wil()
        assert wil.short_name_base == "WIL"

    def test_long_name(self, sample_dataset):
        """Test WIL long_name."""
        wil = sample_dataset.metrics.wil()
        assert wil.long_name_base == "Word Information Lost"

    def test_description(self, sample_dataset):
        """Test WIL has description."""
        wil = sample_dataset.metrics.wil()
        assert len(wil.description) > 0

    def test_example_cls(self, sample_dataset):
        """Test WIL has example_cls set."""
        wil = sample_dataset.metrics.wil()
        assert wil.example_cls == WIL_


class TestWILMetricValues:
    """Tests for WIL metric_values method."""

    def test_metric_values_main(self):
        """Test that value is the main metric."""
        values = WIL.metric_values()
        assert values["main"] == "value"

    def test_metric_values_other(self):
        """Test that WIL has no other values besides the main one."""
        values = WIL.metric_values()
        assert values["other"] == []

    def test_example_metric_values(self):
        """Test WIL_ metric_values."""
        values = WIL_.metric_values()
        assert values["main"] == "value"
        assert values["other"] == []


class TestWILAlignment:
    def test_forwards_alignment_to_wip(self, empty_dataset):
        empty_dataset.add("the bed room is big", "the bedroom is a big")
        wil = empty_dataset.metrics.wil(alignment="error_align")
        assert wil.value == pytest.approx(1 - empty_dataset.metrics.wip(alignment="error_align").value)
        assert wil.value != pytest.approx(empty_dataset.metrics.wil().value)
