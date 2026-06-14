"""Tests for bewer.metrics.complex_term (CTR / CTP / CTF)."""

import pytest

from bewer import Dataset
from bewer.metrics.complex_term import CTF, CTP, CTR, DEFAULT_COMPLEX_TERM_VOCAB


class TestComplexTermAutoVocabulary:
    """The complex-term vocabulary is registered on first use, out of the box."""

    def test_first_use_registers_vocabulary(self):
        dataset = Dataset()
        dataset.add("an MRI scan", "an MRI scan")
        assert DEFAULT_COMPLEX_TERM_VOCAB not in dataset._vocabularies
        dataset.metrics.ctr().value
        assert DEFAULT_COMPLEX_TERM_VOCAB in dataset._vocabularies

    def test_works_after_dataset_already_frozen(self):
        """A complex-term metric works even when another metric froze the dataset first."""
        dataset = Dataset()
        dataset.add("an MRI scan", "an MRI scan")
        dataset.metrics.wer().value  # freezes the dataset
        assert dataset.is_frozen
        assert dataset.metrics.ctr().value == 1.0


class TestComplexTermRecall:
    """CTR: fraction of reference complex terms transcribed correctly."""

    def test_perfect_recall(self):
        dataset = Dataset()
        dataset.add("the MRI and CO2 results", "the MRI and CO2 results")
        ctr = dataset.metrics.ctr()
        assert ctr.num_ref_terms == 2
        assert ctr.value == 1.0

    def test_missed_term(self):
        dataset = Dataset()
        dataset.add("ordered an MRI", "ordered an MIR")
        assert dataset.metrics.ctr().value == 0.0

    def test_partial_recall(self):
        dataset = Dataset()
        dataset.add("the MRI and CO2 readings", "the MRI and CO twins")
        ctr = dataset.metrics.ctr()
        assert ctr.num_ref_terms == 2
        assert ctr.num_matches == 1
        assert ctr.value == 0.5

    def test_example_without_complex_terms_contributes_nothing(self):
        dataset = Dataset()
        dataset.add("an MRI scan", "an MRI scan")
        dataset.add("the patient is well", "the patient is well")
        ctr = dataset.metrics.ctr()
        assert ctr.num_ref_terms == 1
        assert ctr.value == 1.0

    def test_hyphen_compound_scored_strictly(self):
        """A spaced 'CT scan' does not match a 'CT-scan' complex term."""
        dataset = Dataset()
        dataset.add("got a CT-scan", "got a CT scan")
        ctr = dataset.metrics.ctr()
        assert ctr.num_ref_terms == 1
        assert ctr.value == 0.0

    def test_case_scored_strictly(self):
        """A lowercased 'mri' does not match an 'MRI' complex term."""
        dataset = Dataset()
        dataset.add("an MRI scan", "an mri scan")
        assert dataset.metrics.ctr().value == 0.0


class TestComplexTermPrecision:
    """CTP: how precisely the produced complex terms were transcribed."""

    def test_perfect_precision(self):
        dataset = Dataset()
        dataset.add("the MRI and CO2 results", "the MRI and CO2 results")
        assert dataset.metrics.ctp().value == 1.0

    def test_spurious_complex_term_lowers_precision(self):
        """A reference complex term surfacing erroneously in another hypothesis is a false positive.

        The vocabulary is global and reference-derived, so 'MRI' (from example 0) is a complex
        term everywhere; emitting it where the reference said 'ECG' is a spurious occurrence.
        """
        dataset = Dataset()
        dataset.add("the MRI scan", "the MRI scan")  # MRI correct -> TP
        dataset.add("the ECG result", "the MRI result")  # ECG missed -> FN; spurious MRI -> FP
        ctp = dataset.metrics.ctp()
        assert ctp.num_matches == 1  # TP
        assert ctp.value == 0.5  # TP / (TP + FP) = 1 / (1 + 1)


class TestComplexTermFScore:
    """CTF: weighted harmonic mean of CTP and CTR."""

    def test_f1_is_harmonic_mean(self):
        dataset = Dataset()
        dataset.add("the MRI and CO2 readings", "the MRI and CO twins")
        ctp = dataset.metrics.ctp().value
        ctr = dataset.metrics.ctr().value
        ctf = dataset.metrics.ctf().value
        expected = 2 * ctp * ctr / (ctp + ctr)
        assert ctf == pytest.approx(expected)

    def test_beta_must_be_positive(self):
        dataset = Dataset()
        dataset.add("an MRI scan", "an MRI scan")
        with pytest.raises(ValueError, match="beta must be positive"):
            dataset.metrics.ctf(beta=0).value


class TestComplexTermMetricAttributes:
    """Metric metadata and inheritance."""

    @pytest.mark.parametrize(
        "cls, short, long",
        [
            (CTR, "CTR", "Complex Term Recall"),
            (CTP, "CTP", "Complex Term Precision"),
            (CTF, "CTF", "Complex Term F-Score"),
        ],
    )
    def test_names(self, cls, short, long):
        assert cls.short_name_base == short
        assert cls.long_name_base == long
        assert len(cls.description) > 0

    @pytest.mark.parametrize("cls", [CTR, CTP, CTF])
    def test_main_value(self, cls):
        assert cls.metric_values()["main"] == "value"

    def test_default_vocab_is_complex_terms(self):
        assert CTR.param_schema().vocab == DEFAULT_COMPLEX_TERM_VOCAB
        assert CTF.param_schema().vocab == DEFAULT_COMPLEX_TERM_VOCAB

    def test_runs_under_complex_term_pipeline(self):
        dataset = Dataset()
        dataset.add("an MRI scan", "an MRI scan")
        ctr = dataset.metrics.ctr()
        assert ctr.tokenizer == "complex_term"
        assert ctr.normalizer == "cased"
