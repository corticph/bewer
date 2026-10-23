"""Tests for the orthographically-complex term metric (a predefined regex metric)."""

import pytest

from bewer import Dataset

METRICS = (
    "orthographically_complex_term_recall",
    "orthographically_complex_term_precision",
    "orthographically_complex_term_fscore",
)


class TestRecall:
    """Fraction of reference orthographically-complex terms transcribed correctly."""

    def test_perfect_recall(self):
        dataset = Dataset()
        dataset.add("the MRI and CO2 results", "the MRI and CO2 results")
        recall = dataset.metrics.orthographically_complex_term_recall()
        assert recall.num_ref_terms == 2
        assert recall.value == 1.0

    def test_missed_term(self):
        dataset = Dataset()
        dataset.add("ordered an MRI", "ordered an MIR")
        assert dataset.metrics.orthographically_complex_term_recall().value == 0.0

    def test_partial_recall(self):
        dataset = Dataset()
        dataset.add("the MRI and CO2 readings", "the MRI and CO twins")
        recall = dataset.metrics.orthographically_complex_term_recall()
        assert recall.num_ref_terms == 2
        assert recall.num_matches == 1
        assert recall.value == 0.5

    def test_hyphen_compound_scored_strictly(self):
        """A spaced 'CT scan' does not match a 'CT-scan' term."""
        dataset = Dataset()
        dataset.add("got a CT-scan", "got a CT scan")
        recall = dataset.metrics.orthographically_complex_term_recall()
        assert recall.num_ref_terms == 1
        assert recall.value == 0.0

    def test_case_scored_strictly(self):
        """A lowercased 'mri' does not match an 'MRI' term."""
        dataset = Dataset()
        dataset.add("an MRI scan", "an mri scan")
        assert dataset.metrics.orthographically_complex_term_recall().value == 0.0

    def test_works_after_dataset_already_frozen(self):
        dataset = Dataset()
        dataset.add("an MRI scan", "an MRI scan")
        dataset.metrics.wer().value  # freezes the dataset
        assert dataset.is_frozen
        assert dataset.metrics.orthographically_complex_term_recall().value == 1.0


class TestPrecision:
    def test_perfect_precision(self):
        dataset = Dataset()
        dataset.add("the MRI and CO2 results", "the MRI and CO2 results")
        assert dataset.metrics.orthographically_complex_term_precision().value == 1.0

    def test_spurious_term_lowers_precision(self):
        """A reference term surfacing erroneously in another hypothesis is a false positive."""
        dataset = Dataset()
        dataset.add("the MRI scan", "the MRI scan")  # MRI correct -> TP
        dataset.add("the ECG result", "the MRI result")  # ECG missed -> FN; spurious MRI -> FP
        precision = dataset.metrics.orthographically_complex_term_precision()
        assert precision.num_matches == 1  # TP
        assert precision.value == 0.5  # TP / (TP + FP) = 1 / (1 + 1)


class TestFScore:
    def test_f1_is_harmonic_mean(self):
        dataset = Dataset()
        dataset.add("the MRI and CO2 readings", "the MRI and CO twins")
        p = dataset.metrics.orthographically_complex_term_precision().value
        r = dataset.metrics.orthographically_complex_term_recall().value
        f = dataset.metrics.orthographically_complex_term_fscore().value
        assert f == pytest.approx(2 * p * r / (p + r))

    def test_beta_must_be_positive(self):
        dataset = Dataset()
        dataset.add("an MRI scan", "an MRI scan")
        with pytest.raises(ValueError, match="beta must be positive"):
            dataset.metrics.orthographically_complex_term_fscore(beta=0).value


class TestMetricAttributes:
    """Generated metric metadata."""

    @pytest.mark.parametrize("name", METRICS)
    def test_registered(self, name):
        from bewer.metrics.base import METRIC_REGISTRY

        assert name in METRIC_REGISTRY.metric_metadata

    @pytest.mark.parametrize("name", METRICS)
    def test_main_value(self, name):
        from bewer.metrics.base import METRIC_REGISTRY

        metric_cls = METRIC_REGISTRY.metric_metadata[name]["metric_cls"]
        assert metric_cls.metric_values()["main"] == "value"

    def test_default_vocab(self):
        from bewer.metrics.key_term.orthographically_complex_term import DEFAULT_ORTHOGRAPHICALLY_COMPLEX_TERM_VOCAB

        dataset = Dataset()
        recall = dataset.metrics.orthographically_complex_term_recall()
        assert recall.params.vocab == DEFAULT_ORTHOGRAPHICALLY_COMPLEX_TERM_VOCAB

    def test_runs_under_complex_term_pipeline(self):
        dataset = Dataset()
        dataset.add("an MRI scan", "an MRI scan")
        recall = dataset.metrics.orthographically_complex_term_recall()
        assert recall.tokenizer == "orthographically_complex_term"
        assert recall.normalizer == "cased"
