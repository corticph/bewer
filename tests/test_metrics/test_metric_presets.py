"""Tests for the metric presets module and its orthographically-complex term preset."""

import pytest

from bewer import Dataset, get_config
from bewer.metric_presets import ORTHOGRAPHICALLY_COMPLEX_TERM_KWARGS as PRESET


class TestPresetContent:
    def test_carries_vocabulary_and_pipeline(self):
        assert dict(PRESET) == {
            "vocab": "orthographically_complex_terms",
            "tokenizer": "orthographically_complex_term",
            "normalizer": "cased",
        }

    def test_is_immutable(self):
        with pytest.raises(TypeError):
            PRESET["vocab"] = "medical"

    def test_importable_from_the_package_root(self):
        import bewer

        assert bewer.metric_presets.ORTHOGRAPHICALLY_COMPLEX_TERM_KWARGS is PRESET


class TestPresetMatchesBaseConfig:
    def test_names_exist_in_the_base_config(self):
        config = get_config("base")
        assert PRESET["vocab"] in config.vocabularies
        assert PRESET["tokenizer"] in config.tokenizers
        assert PRESET["normalizer"] in config.normalizers


class TestRecall:
    """Key-term recall on orthographically-complex terms via the preset."""

    def test_perfect_recall(self):
        dataset = Dataset()
        dataset.add("the MRI and CO2 results", "the MRI and CO2 results")
        recall = dataset.metrics.key_term_recall(**PRESET)
        assert recall.num_ref_terms == 2
        assert recall.value == 1.0

    def test_missed_term(self):
        dataset = Dataset()
        dataset.add("ordered an MRI", "ordered an MIR")
        assert dataset.metrics.key_term_recall(**PRESET).value == 0.0

    def test_partial_recall(self):
        dataset = Dataset()
        dataset.add("the MRI and CO2 readings", "the MRI and CO twins")
        recall = dataset.metrics.key_term_recall(**PRESET)
        assert recall.num_ref_terms == 2
        assert recall.num_matches == 1
        assert recall.value == 0.5

    def test_hyphen_compound_scored_strictly(self):
        """A spaced 'CT scan' does not match a 'CT-scan' term."""
        dataset = Dataset()
        dataset.add("got a CT-scan", "got a CT scan")
        recall = dataset.metrics.key_term_recall(**PRESET)
        assert recall.num_ref_terms == 1
        assert recall.value == 0.0

    def test_case_scored_strictly(self):
        """A lowercased 'mri' does not match an 'MRI' term."""
        dataset = Dataset()
        dataset.add("an MRI scan", "an mri scan")
        assert dataset.metrics.key_term_recall(**PRESET).value == 0.0

    def test_works_after_dataset_already_frozen(self):
        dataset = Dataset()
        dataset.add("an MRI scan", "an MRI scan")
        dataset.metrics.wer().value  # freezes the dataset
        assert dataset.is_frozen
        assert dataset.metrics.key_term_recall(**PRESET).value == 1.0


class TestPrecision:
    def test_perfect_precision(self):
        dataset = Dataset()
        dataset.add("the MRI and CO2 results", "the MRI and CO2 results")
        assert dataset.metrics.key_term_precision(**PRESET).value == 1.0

    def test_spurious_term_lowers_precision(self):
        """A reference term surfacing erroneously in another hypothesis is a false positive."""
        dataset = Dataset()
        dataset.add("the MRI scan", "the MRI scan")  # MRI correct -> TP
        dataset.add("the ECG result", "the MRI result")  # ECG missed -> FN; spurious MRI -> FP
        precision = dataset.metrics.key_term_precision(**PRESET)
        assert precision.num_matches == 1  # TP
        assert precision.value == 0.5  # TP / (TP + FP) = 1 / (1 + 1)

    def test_hallucinated_term_absent_from_references_is_false_positive(self):
        """A hypothesis class term that no reference contains still counts against precision."""
        dataset = Dataset()
        dataset.add("the MRI was clear", "the MRI and HbA1c was clear")
        precision = dataset.metrics.key_term_precision(**PRESET)
        assert precision.num_matches == 1  # MRI -> TP
        assert precision.num_fp == 1  # HbA1c, found only in the hypothesis
        assert precision.value == 0.5

    def test_hallucinated_term_does_not_change_recall(self):
        dataset = Dataset()
        dataset.add("the MRI was clear", "the MRI and HbA1c was clear")
        recall = dataset.metrics.key_term_recall(**PRESET)
        assert recall.num_ref_terms == 1
        assert recall.value == 1.0


class TestFScore:
    def test_f1_is_harmonic_mean(self):
        dataset = Dataset()
        dataset.add("the MRI and CO2 readings", "the MRI and CO twins")
        p = dataset.metrics.key_term_precision(**PRESET).value
        r = dataset.metrics.key_term_recall(**PRESET).value
        f = dataset.metrics.key_term_fscore(**PRESET).value
        assert f == pytest.approx(2 * p * r / (p + r))

    def test_beta_must_be_positive(self):
        dataset = Dataset()
        dataset.add("an MRI scan", "an MRI scan")
        with pytest.raises(ValueError, match="beta must be positive"):
            dataset.metrics.key_term_fscore(**PRESET, beta=0).value


class TestPipelineAttributes:
    def test_runs_under_the_preset_pipeline(self):
        dataset = Dataset()
        dataset.add("an MRI scan", "an MRI scan")
        recall = dataset.metrics.key_term_recall(**PRESET)
        assert recall.tokenizer == "orthographically_complex_term"
        assert recall.normalizer == "cased"
        assert recall.params.vocab == "orthographically_complex_terms"


class TestOverrides:
    def test_tokenizer_override_splits_the_compound(self):
        """Splitting 'CT-scan' lets the bare 'CT' match instead."""
        dataset = Dataset(config="en")
        dataset.add("got a CT-scan", "got a CT scan")

        strict = dataset.metrics.key_term_recall(**PRESET).value
        loose = dataset.metrics.key_term_recall(**{**PRESET, "tokenizer": "default"}).value

        assert strict == 0.0
        assert loose == 1.0


class TestPresetOnOtherKeyTermMetrics:
    def test_key_term_fpr_computes(self):
        dataset = Dataset()
        dataset.add("the MRI was clear", "the MRI and HbA1c was clear")
        fpr = dataset.metrics.key_term_fpr(**PRESET)
        assert fpr.value == 0.25

    def test_b_wer_computes(self):
        """The hypothesis-only term is an in-vocabulary insertion, so all biased edits count."""
        dataset = Dataset()
        dataset.add("the MRI was clear", "the MRI and HbA1c was clear")
        b_wer = dataset.metrics.b_wer(**PRESET)
        assert b_wer.value == 1.0

    def test_relaxed_key_term_recall_computes(self):
        dataset = Dataset()
        dataset.add("the MRI was clear", "the MRI and HbA1c was clear")
        relaxed_recall = dataset.metrics.relaxed_key_term_recall(**PRESET)
        assert relaxed_recall.value == 1.0

    def test_key_term_error_rate_computes(self):
        dataset = Dataset()
        dataset.add("the MRI was clear", "the MRI and HbA1c was clear")
        error_rate = dataset.metrics.key_term_error_rate(**PRESET)
        assert error_rate.value == 0.0
